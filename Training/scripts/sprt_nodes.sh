#!/bin/bash
# sprt_nodes.sh <name> <opt=val>...  node-limited SPRT [0,3] for value nets of the SAME
# architecture (same speed, so a fixed node count per move measures only the eval).
# Copies MainEngine to Engines/<name>-dev and starts the SPRT detached; both sides
# search NODES nodes per move (default 6000; 20 ms on this machine = ~24k nodes with
# 14 parallel games), --time 1000 is only a safety cap. Machine independent.
# Example: sprt_nodes.sh v2b value_file=/abs/path/net.quant
# Larger/smaller nets (other speed) must be tested with time (sprt.sh).
name=$1; shift
NODES=${NODES:-6000}
cd /home/leaguerobin/DarkHorse/cmake-build-debug || exit 1
cp MainEngine ../Training/Engines/$name-dev
setsid nohup ./Train --match --engines $name-dev $name-dev --time 1000 1000 --threads 14 \
  --book ../Training/Positions/unbalanced_70_150.book --sprt 0 3 \
  --opts1 nodes=$NODES "$@" --opts2 nodes=$NODES \
  > ../Training/${name}_sprt_n${NODES}.log 2>&1 < /dev/null &
sleep 3; pgrep -x Train -a
