"""Lean training loop for the policy net (replaces pytorch_lightning.Trainer.fit).

Lightning costs several ms of Python per step (hooks, closures, progress and result
bookkeeping) plus a host sync per step (loss .item()), which dominates the ~7 ms GPU
step of this small net. This loop keeps exactly the same TensorBoard metrics by
reusing the model's own code (training_step, _shared_step, on_before_optimizer_step,
_log_weight_stats) and replacing model.log with a collector:

  train/*, train/grad_norm, epoch  every log_every steps (step = batch index, i.e.
                                    49, 99, ... like Lightning)
  val/*, val_by/*                   once per epoch at step global_step-1, batch-size
                                    weighted means (Lightning's on_epoch reduction)
  weights/* scalars + histograms    once per epoch

Checkpoints use Lightning's keys (state_dict, optimizer_states, lr_schedulers, epoch,
global_step), so --resume also works from checkpoints written by the Lightning loop.
"""
import glob
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter


class _MetricCollector:
    """Stands in for LightningModule.log."""

    def __init__(self, writer):
        self.writer = writer
        self.mode = None          # "train", "val" or "epoch"
        self.pending = []         # (name, tensor/float) of the current logging step
        self.acc = {}             # val: name -> (weighted sum, weight)

    def __call__(self, name, value, on_step=None, on_epoch=None, prog_bar=False,
                 batch_size=None, **kwargs):
        if self.mode == "val":
            w = float(batch_size) if batch_size is not None else 1.0
            v = value.detach().float() if torch.is_tensor(value) else float(value)
            s, n = self.acc.get(name, (0.0, 0.0))
            self.acc[name] = (s + v * w, n + w)
        elif self.mode in ("train", "epoch"):
            self.pending.append((name, value))
        # mode None: not a logging step -> ignored

    def flush(self, step):
        if self.pending:
            vals = [v.detach().float() if torch.is_tensor(v) else torch.tensor(float(v))
                    for _, v in self.pending]
            floats = torch.stack([v.cpu() for v in vals]).tolist()   # one sync
            for (name, _), f in zip(self.pending, floats):
                self.writer.add_scalar(name, f, step)
        self.pending = []

    def flush_val(self, step):
        for name, (s, n) in self.acc.items():
            self.writer.add_scalar(name, float(s / n) if n else 0.0, step)
        self.acc = {}


def _next_version(log_root):
    versions = [int(p.rsplit("_", 1)[1]) for p in glob.glob(os.path.join(log_root, "version_*"))
                if p.rsplit("_", 1)[1].isdigit()]
    return max(versions) + 1 if versions else 0


def fit(model, train_loader, val_loader, run, max_epochs, ckpt_dir, resume=False,
        log_every=50, precision="tf32", print_every=2000, limit_train_batches=None,
        limit_val_batches=None):
    dev = torch.device("cuda")
    model.to(dev)
    optimizers, schedulers = model.configure_optimizers()
    opt, sched = optimizers[0], schedulers[0]
    interval = "epoch"
    if isinstance(sched, dict):                   # {"scheduler": ..., "interval": "step"}
        interval = sched.get("interval", "epoch")
        sched = sched["scheduler"]

    start_epoch, global_step = 0, 0
    os.makedirs(ckpt_dir, exist_ok=True)
    last = os.path.join(ckpt_dir, "last.ckpt")
    if resume:
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer_states"][0])
        sched.load_state_dict(ck["lr_schedulers"][0])
        start_epoch, global_step = ck["epoch"] + 1, ck["global_step"]
        print(f"resumed from {last}: epoch {start_epoch}, global step {global_step}", flush=True)

    log_root = os.path.join("lightning_logs", run)
    writer = SummaryWriter(os.path.join(log_root, f"version_{_next_version(log_root)}"))
    log = _MetricCollector(writer)
    model.log = log                 # instance attribute shadows LightningModule.log
    model._lean = True
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16"
                else torch.autocast("cuda", enabled=False))

    n_train = len(train_loader) if limit_train_batches is None else min(len(train_loader), limit_train_batches)
    for epoch in range(start_epoch, max_epochs):
        # iter() per epoch like Lightning: ThreadedBatchLoader yields len() batches per
        # iter() and keeps its thread/stream across epochs (prefetched batches carry over)
        train_iter = iter(train_loader)
        model.train()
        t0, t_last, s_last = time.time(), time.time(), 0
        for i in range(n_train):
            batch = next(train_iter)
            # new step for CUDA graph trees (no-op with --no-cudagraphs)
            torch.compiler.cudagraph_mark_step_begin()
            logging_step = (global_step + 1) % log_every == 0
            model._lean_log_now = logging_step
            log.mode = "train" if logging_step else None
            with autocast:
                loss = model.training_step(batch, i)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            model.on_before_optimizer_step(opt)        # grad norm on logging steps
            opt.step()
            if interval == "step":
                sched.step()
            if logging_step:
                log.pending.append(("train/lr", float(opt.param_groups[0]["lr"])))
                log.pending.append(("epoch", float(epoch)))
                log.flush(global_step)
            global_step += 1
            if (i + 1) % print_every == 0:
                torch.cuda.synchronize()
                now = time.time()
                print(f"epoch {epoch} step {i + 1}/{n_train}  {(i + 1 - s_last) / (now - t_last):.1f} it/s",
                      flush=True)
                t_last, s_last = now, i + 1
        torch.cuda.synchronize()
        train_time = time.time() - t0

        # ---- validation (same metrics as Lightning's on_epoch logging)
        model.eval()
        model._lean_log_now = False
        log.mode = "val"
        n_val = 0
        with torch.no_grad(), autocast:
            for j, vb in enumerate(val_loader):
                if limit_val_batches is not None and j >= limit_val_batches:
                    break
                model.validation_step(vb, j)
                n_val += 1
        val_loss = log.acc.get("val/loss", (0.0, 1.0))
        log.flush_val(global_step - 1)

        # ---- epoch end: weight statistics, LR step, .quant export, checkpoint
        log.mode = "epoch"
        model._log_weight_stats()                       # weights/*_saturated, *_delta_ratio
        for name, layer in (("l1", model.layer_one), ("l2", model.layer_sec), ("out", model.output)):
            writer.add_histogram(f"weights/{name}", layer.weight.detach().float().cpu(), epoch)
        log.pending.append(("epoch", float(epoch)))
        log.flush(global_step - 1)
        log.mode = None
        if interval == "epoch":
            sched.step()
        model.save_quantized_bucket(f"{model.run_name}_{epoch}.quant")
        model.to(dev)
        ck = {"epoch": epoch, "global_step": global_step, "state_dict": model.state_dict(),
              "optimizer_states": [opt.state_dict()], "lr_schedulers": [sched.state_dict()],
              "hyper_parameters": dict(model.hparams), "loop": "lean"}
        torch.save(ck, os.path.join(ckpt_dir, f"epoch={epoch}.ckpt"))
        torch.save(ck, last)
        writer.flush()
        vl = float(val_loss[0] / val_loss[1]) if val_loss[1] else float("nan")
        print(f"epoch {epoch} done: {n_train} batches in {train_time:.0f} s "
              f"({n_train / train_time:.1f} it/s), {n_val} val batches, val/loss {vl:.4f}", flush=True)
    writer.close()
    for loader in (train_loader, val_loader):     # clean thread shutdown (see close())
        if hasattr(loader, "close"):
            loader.close()
