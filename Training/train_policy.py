"""Train a policy network on the soft targets (rescored child scores).

Architecture 256,32,32,128 = C++ Network<256, 32, 32, 128, 128>:
accumulator half L1=256, hidden layers 32 and 32, 128 policy outputs, 128 buckets.

Run from the Training directory:
    ~/training/bin/python train_policy.py              # full training
    ~/training/bin/python train_policy.py --fast-dev   # quick smoke test
    ~/training/bin/python train_policy.py --resume     # continue from last.ckpt
    ~/training/bin/python train_policy.py --label-mode hard   # best-move labels

Each label mode gets its own run name (policy256_soft / policy256_hard), so
logs, checkpoints and .quant files of both runs can be compared side by side.
"""
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import Experimental

TRAIN_DATA = "/mnt/c/TrainingData/reservoir-test.10M.train.children.samples"
VAL_DATA = "/mnt/c/TrainingData/reservoir-test.10M.val.children.samples"

RUN = "policy256"
L1 = 256
BATCH_SIZE = 8192
MAX_EPOCHS = 100


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-dev", action="store_true",
                    help="smoke test: a few batches, one epoch")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last checkpoint")
    ap.add_argument("--label-mode", choices=("soft", "hard"), default="soft",
                    help="soft: softmax(score/T) targets; hard: best move only")
    args = ap.parse_args()

    run = f"{RUN}_{args.label_mode}" + ("_fastdev" if args.fast_dev else "")
    ckpt_dir = os.path.join("Networks", run + "_ckpt")

    # .quant files are written as Networks/<run>_<epoch>.quant at each epoch end
    model = Experimental.PolicyNetwork(run_name=os.path.join("Networks", run),
                                       l1=L1, label_mode=args.label_mode,
                                       temperature=50.0)
    data = Experimental.PolicyLitDataModule(train_data=TRAIN_DATA,
                                            val_data=VAL_DATA,
                                            batch_size=BATCH_SIZE)

    logger = TensorBoardLogger(save_dir="lightning_logs", name=run)
    checkpoint = ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch}",
                                 every_n_epochs=1, save_top_k=-1,
                                 save_last=True)

    extra = {}
    if args.fast_dev:
        extra = dict(limit_train_batches=20, limit_val_batches=5)

    trainer = pl.Trainer(accelerator="gpu", devices=1,
                         max_epochs=1 if args.fast_dev else MAX_EPOCHS,
                         logger=logger, callbacks=[checkpoint],
                         log_every_n_steps=10 if args.fast_dev else 50,
                         **extra)

    ckpt_path = os.path.join(ckpt_dir, "last.ckpt") if args.resume else None
    trainer.fit(model, data, ckpt_path=ckpt_path)
