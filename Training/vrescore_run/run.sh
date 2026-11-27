#!/bin/bash
# label experiment: the same positions as cum8 / val, rescored value-only (root without MLH,
# node budget 2500,2500,1500,750, fail-low rule) with the current engine
set -e
D=/home/leaguerobin/DarkHorse/Training/vrescore_run; T=/mnt/c/TrainingData; cd $D
echo "$(date '+%F %T') val: extract + rescore"
./movegen pol2samples $T/reservoir-full.val.children.samples $T/reservoir-full.val.pos.samples
./movegen vrescore $T/reservoir-full.val.pos.samples $T/reservoir-full.val.vlabels.samples 16 > vrescore_val.log 2>&1
echo "$(date '+%F %T') val done; train: extract"
./movegen pol2samples $T/reservoir-full.train.cum8.children.samples $T/reservoir-full.train.cum8.pos.samples
echo "$(date '+%F %T') train: rescore"
./movegen vrescore $T/reservoir-full.train.cum8.pos.samples $T/reservoir-full.train.cum8.vlabels.samples 16 > vrescore_train.log 2>&1
echo "$(date '+%F %T') RESCORE_DONE"
