#!/bin/bash
# sprt.sh <name> <opt=val>...  copies MainEngine to Engines/<name>-dev and starts the SPRT detached
# NOTE: --time 20 is calibrated for this machine; on other machines run Training/bench_parallel.sh first and scale it.
name=$1; shift
cd /home/leaguerobin/DarkHorse/cmake-build-debug || exit 1
cp MainEngine ../Training/Engines/$name-dev
setsid nohup ./Train --match --engines $name-dev $name-dev --time 20 20 --threads 14 \
  --book ../Training/Positions/unbalanced_70_150.book --sprt 0 3 --opts1 "$@" \
  > ../Training/${name}_sprt_20ms.log 2>&1 < /dev/null &
sleep 3; pgrep -x Train -a
