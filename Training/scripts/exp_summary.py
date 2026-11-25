"""exp_summary.py RUN ARGS CSV: appends the per-epoch-best and last validation metrics of
lightning_logs/RUN_*/ (latest version) to CSV."""
import csv, glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run, args, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = sorted(glob.glob(f"/home/leaguerobin/DarkHorse/Training/lightning_logs/{run}_*/version_*"), key=os.path.getmtime)[-1]
ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
get = lambda t: [e.value for e in ea.Scalars(t)] if t in ea.Tags()["scalars"] else []
tie, reg, loss = get("val/top1_tie_aware"), get("val/regret_mean"), get("val/loss")
best = max(range(len(tie)), key=lambda i: (round(tie[i], 4), i)) if tie else -1
import re
speeds = []
logf = f"/home/leaguerobin/DarkHorse/Training/exp_logs/{run}.log"
if os.path.exists(logf):
    speeds = [float(x) for x in re.findall(r"epoch \d+ done: \d+ batches in \d+ s \(([\d.]+) it/s\)", open(logf).read())]
it_s = round(sum(speeds[1:]) / len(speeds[1:]), 1) if len(speeds) > 1 else (speeds[0] if speeds else "")
row = {"run": run, "args": args, "epochs": len(tie),
       "best_epoch": best, "best_tie": round(tie[best], 4) if tie else "", "best_regret": round(reg[best], 2) if reg else "",
       "last_tie": round(tie[-1], 4) if tie else "", "last_regret": round(reg[-1], 2) if reg else "",
       "last_val_loss": round(loss[-1], 4) if loss else "",
       "it_per_s": it_s,       # mean over epochs 1.. (epoch 0 includes compiling)
       "tie_per_epoch": " ".join(f"{v:.4f}" for v in tie)}
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
print(f"{run}: best tie {row['best_tie']} (epoch {best}), last tie {row['last_tie']} regret {row['last_regret']} loss {row['last_val_loss']}, {it_s} it/s")
