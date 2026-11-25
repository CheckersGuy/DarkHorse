"""Train a policy network on rescored child scores.

Architecture <l1>,32,32,128 = C++ Network<l1, 32, 32, 128, 128>:
accumulator half L1, hidden layers 32 and 32, 128 policy outputs, 128 buckets.
The engine's policy net is Network<1024, 32, 32, 128, 128> (default --l1 1024).

Run from the Training directory, e.g.
    ~/training/bin/python train_policy.py --train T.samples --val V.samples
    ~/training/bin/python train_policy.py --train T.samples --val V.samples --fast-dev
    ~/training/bin/python train_policy.py --train T.samples --val V.samples --resume

The run name contains the number of training positions (from the POL1 header of
the train file), e.g. policy1024_hard_49.8M. It is used for the TensorBoard run
(lightning_logs/<name>), the checkpoint dir and the .quant files
(Networks/<name>_<epoch>.quant), so runs on different data sizes can be compared
side by side in TensorBoard.
"""
import argparse
import os
import struct

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import Experimental

TRAIN_DATA = "/mnt/c/TrainingData/reservoir-test.10M.train.children.samples"
VAL_DATA = "/mnt/c/TrainingData/reservoir-test.10M.val.children.samples"
BATCH_SIZE = 8192


def sample_count(path):
    """Number of samples from the 12-byte POL1 header (u32 magic, u64 count)."""
    with open(path, "rb") as f:
        magic, count = struct.unpack("<IQ", f.read(12))
    if magic != 0x504F4C31:
        raise ValueError(f"{path} is not a policy-sample file")
    return count


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN_DATA, help="training file (POL1)")
    ap.add_argument("--val", default=VAL_DATA, help="validation file (POL1)")
    ap.add_argument("--l1", type=int, default=1024, help="accumulator half size")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--run", default=None,
                    help="run name prefix (default policy<l1>_<label-mode>)")
    ap.add_argument("--fast-dev", action="store_true",
                    help="smoke test: a few batches, one epoch")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last checkpoint")
    ap.add_argument("--label-mode", choices=("soft", "hard"), default="hard",
                    help="soft: softmax(score/T) targets; hard: best move only")
    ap.add_argument("--temperature", type=float, default=20.0,
                    help="soft-label temperature in eval units (softmax(score/T))")
    ap.add_argument("--precision", choices=("fp32", "tf32", "bf16"), default="tf32",
                    help="fp32: exact; tf32: TF32 matmuls (~1.4x, logits within ~3e-5 "
                         "of fp32); bf16: bf16-mixed autocast (~2.5x, ~1e-3)")
    ap.add_argument("--lightning", action="store_true",
                    help="use the pytorch_lightning Trainer instead of the lean loop "
                         "(lean_loop.py; same TensorBoard metrics, much less overhead)")
    ap.add_argument("--no-rotate-epochs", dest="rotate_epochs", action="store_false",
                    help="read every pass from the file start (default: every pass over "
                         "the training file starts at a random record, so the shuffle-buffer "
                         "blocks differ between epochs)")
    ap.add_argument("--lr", type=float, default=5.0e-3, help="peak learning rate")
    ap.add_argument("--optimizer", choices=("rangerlite", "radam", "adamw"), default="rangerlite")
    ap.add_argument("--schedule", choices=("step", "cosine"), default="cosine",
                    help="cosine (default, best in the 2026-11-25 stage-1 runs): per step, ends at 0.93**29 of the peak; step: x gamma per epoch (same total decay)")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="cosine: linear warm-up as a fraction of all steps")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--no-cudagraphs", dest="cudagraphs", action="store_false",
                    help="compile without CUDA graphs (default: mode reduce-overhead, "
                         "~30%% faster, same training up to GPU rounding noise)")
    ap.add_argument("--no-compile", action="store_true",
                    help="do not torch.compile the training loss (eager mode)")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="benchmark: train only this many batches per epoch, no validation")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("highest" if args.precision == "fp32" else "high")

    positions = sample_count(args.train)
    prefix = args.run or (f"policy{args.l1}_{args.label_mode}"
                          + (f"_T{args.temperature:g}" if args.label_mode == "soft" else ""))
    run = f"{prefix}_{positions / 1e6:.1f}M" + ("_fastdev" if args.fast_dev else "")
    ckpt_dir = os.path.join("Networks", run + "_ckpt")
    print(f"run {run}: {positions:,} training positions from {args.train}")

    # .quant files are written as Networks/<run>_<epoch>.quant at each epoch end
    model = Experimental.PolicyNetwork(run_name=os.path.join("Networks", run),
                                       l1=args.l1, label_mode=args.label_mode,
                                       temperature=args.temperature)
    model.opt_name = args.optimizer
    model.base_lr = args.lr
    model.schedule = args.schedule
    model.warmup_frac = args.warmup
    lim = args.limit_batches or (20 if args.fast_dev else None)
    steps_per_epoch = min(positions // args.batch_size, lim) if lim else positions // args.batch_size
    model.total_steps = steps_per_epoch * (1 if args.fast_dev else args.max_epochs)
    model.use_compile = not args.no_compile
    model.use_cudagraphs = args.cudagraphs and not args.no_compile
    # LR schedule: StepLR per epoch. The total decay over the run is fixed at
    # 0.93**29 (the original 30-epoch schedule, lr 5e-3 -> 6.1e-4); with fewer
    # epochs gamma gets smaller so the run still ends at the same low LR.
    # 30 epochs -> gamma 0.93 exactly (unchanged).
    model.gamma = 0.93 ** (29 / max(1, args.max_epochs - 1))
    print(f"lr schedule: {args.schedule}, {args.max_epochs} epochs, peak lr {args.lr:.1e}, "
          f"final lr {args.lr * 0.93 ** 29:.2e}" + (f", gamma {model.gamma:.4f}" if args.schedule == "step" else ""))
    data = Experimental.PolicyLitDataModule(train_data=args.train,
                                            val_data=args.val,
                                            batch_size=args.batch_size,
                                            rotate=args.rotate_epochs)

    if not args.lightning:
        import lean_loop
        limits = {}
        if args.fast_dev:
            limits = dict(limit_train_batches=20, limit_val_batches=5, log_every=10)
        if args.limit_batches:
            limits = dict(limit_train_batches=args.limit_batches, limit_val_batches=0)
        lean_loop.fit(model, data.train_dataloader(), data.val_dataloader(), run,
                      max_epochs=1 if args.fast_dev else args.max_epochs,
                      ckpt_dir=ckpt_dir, resume=args.resume, precision=args.precision,
                      **limits)
        raise SystemExit(0)

    logger = TensorBoardLogger(save_dir="lightning_logs", name=run)
    checkpoint = ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch}",
                                 every_n_epochs=1, save_top_k=-1,
                                 save_last=True)

    extra = {}
    if args.fast_dev:
        extra = dict(limit_train_batches=20, limit_val_batches=5)
    if args.limit_batches:
        extra = dict(limit_train_batches=args.limit_batches, limit_val_batches=0)
    if args.precision == "bf16":
        extra["precision"] = "bf16-mixed"

    trainer = pl.Trainer(accelerator="gpu", devices=1,
                         max_epochs=1 if args.fast_dev else args.max_epochs,
                         logger=logger,
                         # redrawing the bar every step costs ~3 ms per step
                         callbacks=[checkpoint, TQDMProgressBar(refresh_rate=50)],
                         log_every_n_steps=10 if args.fast_dev else 50,
                         **extra)

    ckpt_path = os.path.join(ckpt_dir, "last.ckpt") if args.resume else None
    trainer.fit(model, data, ckpt_path=ckpt_path)
