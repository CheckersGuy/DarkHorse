#!/bin/bash
# after_chunk.sh N [--no-train] : run when rescoring of chunk N has finished.
#  default:
#   1. builds the cumulative training file train.cum<N> from the most recent
#      existing cumulative file cum<K> (K < N) plus chunks K+1..N, globally shuffled
#   2. plays the audible signal, starts the soft-label (T=20) policy training on the GPU
#   3. starts rescoring chunk N+1 with 14 workers
#  --no-train: only starts rescoring chunk N+1 (16 workers, nothing else uses the
#   CPU/GPU); the chunk is merged later by the next run without --no-train.
set -e
N=$1
NOTRAIN=$2
T=/mnt/c/TrainingData
R=/home/leaguerobin/DarkHorse/Training
CH=$T/reservoir-full.train.c$N.children.samples
[ -f "$CH" ] || { echo "missing $CH"; exit 1; }
NEXT=$((N+1))

start_rescore() { # $1 = workers
  if [ "$NEXT" -le 10 ]; then
    cd $R/Rust && setsid nohup nice -n 10 ./target/release/movegen rescore \
      $T/reservoir-full.train.c$NEXT.samples $T/reservoir-full.train.c$NEXT.children.samples $1 \
      > $R/rescore_c$NEXT.log 2>&1 < /dev/null &
  fi
}

if [ "$NOTRAIN" = "--no-train" ]; then
  start_rescore 16
  echo "chunk $N done: no training (skipped), rescoring chunk $NEXT started with 16 workers"
  exit 0
fi

if [ "$N" -eq 1 ]; then
  CUM=$CH
else
  CUM=$T/reservoir-full.train.cum$N.children.samples
  # most recent cumulative file below N (cum1 = chunk 1 itself)
  K=$((N-1))
  while [ $K -gt 1 ] && [ ! -f $T/reservoir-full.train.cum$K.children.samples ]; do K=$((K-1)); done
  if [ $K -eq 1 ]; then PREV=$T/reservoir-full.train.c1.children.samples
  else PREV=$T/reservoir-full.train.cum$K.children.samples; fi
  PARTS=("$PREV")
  for i in $(seq $((K+1)) $N); do PARTS+=("$T/reservoir-full.train.c$i.children.samples"); done
  python3 $R/scripts/merge_policy_files.py "$CUM" "${PARTS[@]}"
  # global shuffle: the dedup keeps first occurrences, so the files drift from
  # endgame-heavy to middlegame-heavy; the loader only shuffles within 5M
  $R/Rust/target/release/movegen pshuffle "$CUM" "$CUM.shuf"
  mv "$CUM.shuf" "$CUM"
fi
$R/scripts/beep.sh 2
# soft labels, T = 20; 15 epochs with the compressed LR schedule (user 2026-11-24;
# EPOCHS=30 for the old length). The running script reads this file only when it gets here.
cd $R && setsid nohup /home/leaguerobin/training/bin/python train_policy.py \
  --train "$CUM" --val $T/reservoir-full.val.children.samples --l1 1024 \
  --max-epochs ${EPOCHS:-15} --label-mode soft --temperature 20 > $R/train_policy_soft_c$N.log 2>&1 < /dev/null &
start_rescore 14
echo "chunk $N done: training on $CUM started, rescoring chunk $NEXT started"
