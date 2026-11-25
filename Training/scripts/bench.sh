#!/bin/bash
# bench.sh <outfile> [engine options...]  -> depth-18 node counts of MainEngine on fens40
S=/home/leaguerobin/DarkHorse/Training/scripts
out=$1; shift
: > $out
while read -r fen; do
  n=$(/home/leaguerobin/DarkHorse/cmake-build-debug/MainEngine "$@" --search --time 100000000 --hash_size 64 --depth 18 --position "$fen" 2>&1 | grep "^info depth 18 " | tail -1 | sed -E 's/.* nodes ([0-9]+).*/\1/')
  echo "${n:-NA}" >> $out
done < $S/fens40.txt
