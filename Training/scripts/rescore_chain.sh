#!/bin/bash
# After the 100M intermediate net: no more intermediate trainings.
# Chunks 5..9: when chunk i is rescored, start rescoring chunk i+1 (16 workers, no training).
# Chunk 10: merge everything (cum4 + c5..c10), shuffle, train the final soft T20 net.
T=/mnt/c/TrainingData; S=/home/leaguerobin/DarkHorse/Training/scripts
for i in 5 6 7 8 9 10; do
  until [ -f $T/reservoir-full.train.c$i.children.samples ] && ! pgrep -f "[m]ovegen rescore.*c$i\." >/dev/null; do sleep 30; done
  sleep 5
  if [ $i -lt 10 ]; then $S/after_chunk.sh $i --no-train; else $S/after_chunk.sh 10; fi
done
