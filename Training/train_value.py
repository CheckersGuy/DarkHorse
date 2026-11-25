"""Train a value network on the rescored POL1 files ("root + children"), target = the
stored root value of each sample (current engine), with the policy-net pipeline:
compact loader (Rust fill_value, GIL released, background refill), GPU input
expansion, blocked bucket layers, torch.compile, FastRangerLite and the lean loop.

Architecture <l1>,32,32,1 = C++ Network<l1, 32, 32, 1, 128>; the engine's value net is
Network<2048, 32, 32, 1, 128> (default --l1 2048). Loss as the old trainer.py:
MSE between sigmoid(output) and sigmoid(eval / 127), eval clamped to +-500.
Quantisation-aware: the forward pass is the engine's int8 arithmetic (bit-exact).

Run from the Training directory, e.g.
    ~/training/bin/python train_value.py --train T.children.samples --val V.children.samples
    ~/training/bin/python train_value.py --fast-dev
    ~/training/bin/python train_value.py --resume

Run name value<l1>_<positions>M: TensorBoard lightning_logs/<name>, checkpoints
Networks/<name>_ckpt, .quant files Networks/<name>_<epoch>.quant.
New embedded nets need their neuron permutation (scripts/compute_permutation.py).
"""
import argparse
import os

import torch

import Experimental
import lean_loop
from train_policy import sample_count

TRAIN_DATA = "/mnt/c/TrainingData/reservoir-test.10M.train.children.samples"
VAL_DATA = "/mnt/c/TrainingData/reservoir-full.val.children.samples"
BATCH_SIZE = 8192


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN_DATA, help="training file (POL1)")
    ap.add_argument("--val", default=VAL_DATA, help="validation file (POL1)")
    ap.add_argument("--l1", type=int, default=2048, help="accumulator half size")
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--run", default=None, help="run name prefix (default value<l1>)")
    ap.add_argument("--fast-dev", action="store_true", help="smoke test: a few batches, one epoch")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    ap.add_argument("--precision", choices=("fp32", "tf32", "bf16"), default="tf32")
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
    ap.add_argument("--buffer-size", type=int, default=5_000_000,
                    help="shuffle buffer of the training loader (samples)")
    ap.add_argument("--no-cudagraphs", dest="cudagraphs", action="store_false",
                    help="compile without CUDA graphs (default: mode reduce-overhead, "
                         "~30%% faster, same training up to GPU rounding noise)")
    ap.add_argument("--no-compile", action="store_true",
                    help="do not torch.compile the training loss (eager mode)")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="benchmark: train only this many batches per epoch, no validation")
    ap.add_argument("--limit-val-batches", type=int, default=None)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("highest" if args.precision == "fp32" else "high")

    positions = sample_count(args.train)
    prefix = args.run or f"value{args.l1}"
    run = f"{prefix}_{positions / 1e6:.1f}M" + ("_fastdev" if args.fast_dev else "")
    ckpt_dir = os.path.join("Networks", run + "_ckpt")
    print(f"run {run}: {positions:,} training positions from {args.train}")

    model = Experimental.ValueNetwork(run_name=os.path.join("Networks", run), l1=args.l1)
    model.opt_name = args.optimizer
    model.base_lr = args.lr
    model.schedule = args.schedule
    model.warmup_frac = args.warmup
    lim = args.limit_batches or (20 if args.fast_dev else None)
    steps_per_epoch = min(positions // args.batch_size, lim) if lim else positions // args.batch_size
    model.total_steps = steps_per_epoch * (1 if args.fast_dev else args.max_epochs)
    model.use_compile = not args.no_compile
    model.use_cudagraphs = args.cudagraphs and not args.no_compile
    # same compressed step schedule as the policy net: total decay 0.93**29
    model.gamma = 0.93 ** (29 / max(1, args.max_epochs - 1))
    print(f"lr schedule: {args.schedule}, {args.max_epochs} epochs, peak lr {args.lr:.1e}, "
          f"final lr {args.lr * 0.93 ** 29:.2e}" + (f", gamma {model.gamma:.4f}" if args.schedule == "step" else ""))
    data = Experimental.PolicyLitDataModule(train_data=args.train, val_data=args.val,
                                            batch_size=args.batch_size,
                                            buffer_size=args.buffer_size, kind="value",
                                            rotate=args.rotate_epochs)
    limits = {}
    if args.fast_dev:
        limits = dict(limit_train_batches=20, limit_val_batches=5, log_every=10)
    if args.limit_batches:
        limits = dict(limit_train_batches=args.limit_batches, limit_val_batches=0)
    if args.limit_val_batches is not None:
        limits["limit_val_batches"] = args.limit_val_batches
    lean_loop.fit(model, data.train_dataloader(), data.val_dataloader(), run,
                  max_epochs=1 if args.fast_dev else args.max_epochs,
                  ckpt_dir=ckpt_dir, resume=args.resume, precision=args.precision, **limits)
