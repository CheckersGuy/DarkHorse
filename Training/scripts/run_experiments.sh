#!/bin/bash
# run_experiments.sh LISTFILE CSV : trains the runs of LISTFILE one after another
# (lines "name|extra args"; all on the 100M set, same validation set) and appends
# the final validation metrics of each run to CSV (exp_summary.py).
LIST=$1; CSV=$2
R=/home/leaguerobin/DarkHorse/Training; T=/mnt/c/TrainingData
cd $R
while IFS='|' read -r name args; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  echo "$(date +%H:%M) start $name ($args)"
  /home/leaguerobin/training/bin/python train_policy.py \
    --train $T/reservoir-full.train.cum4.children.samples --val $T/reservoir-full.val.children.samples \
    --l1 1024 --label-mode soft --temperature 20 --run $name $args > $R/exp_logs/$name.log 2>&1
  echo "$(date +%H:%M) end $name exit $?"
  /home/leaguerobin/training/bin/python3 $R/scripts/exp_summary.py "$name" "$args" "$CSV"
done < "$LIST"
echo "$(date +%H:%M) queue $LIST finished"
