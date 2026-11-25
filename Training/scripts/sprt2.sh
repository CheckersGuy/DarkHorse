#!/bin/bash
# sprt2.sh <name> <opt=val>...  stage-2 re-test with [0,2] using the existing Engines/<name>-dev
name=$1; shift
cd /home/leaguerobin/DarkHorse/cmake-build-debug || exit 1
setsid nohup ./Train --match --engines $name-dev $name-dev --time 20 20 --threads 14 \
  --book ../Training/Positions/unbalanced_70_150.book --sprt 0 2 --opts1 "$@" \
  > ../Training/${name}_sprt2_20ms.log 2>&1 < /dev/null &
sleep 3; pgrep -x Train -a
