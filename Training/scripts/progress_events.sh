#!/bin/bash
# Emits one line only when a task ends (state kept in a file, so re-arming does not repeat):
#  - rescoring chunk C: 25/50/75 % and end
#  - training run RUN: epochs 8/15/23 and end (with the last progress line)
C=$1; RUN=$2
T=/mnt/c/TrainingData; R=/home/leaguerobin/DarkHorse/Training
ST=$R/.progress_state_c$C; touch $ST
N=24993215
emit() { grep -qx "$1" $ST || { echo "$2"; echo "$1" >> $ST; }; }
while true; do
  if pgrep -f "[m]ovegen rescore.*c$C\." >/dev/null; then
    B=$(cat $T/reservoir-full.train.c$C.children.samples[0-9]* 2>/dev/null | wc -c); P=$((B*100/49/N))
  elif [ -f $T/reservoir-full.train.c$C.children.samples ]; then
    emit "rend" "chunk $C rescoring finished"
  fi
  if [ -n "$RUN" ]; then
    E=$(ls $R/Networks/${RUN}_*.quant 2>/dev/null | wc -l)
    if [ $E -ge 30 ] || { [ $E -gt 0 ] && ! pgrep -f "[t]rain_policy.py" >/dev/null; }; then
      emit "tend" "training $RUN ended after $E epochs"
    fi
  fi
  grep -qx rend $ST && { [ -z "$RUN" ] || grep -qx tend $ST; } && break
  sleep 60
done
