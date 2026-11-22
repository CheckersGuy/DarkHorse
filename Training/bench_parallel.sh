#!/bin/bash
# Measures the engine speed under load, for calibrating time controls across
# machines (SPRT/SPSA on a cloud machine vs. the local one).
#
#   Training/bench_parallel.sh N [engine] [depth]
#
# Starts N copies of `<engine> --bench` at the same time (N = number of games
# the match tool will play in parallel, i.e. its --threads value) and prints
# the average NPS per copy. Every copy must report the same node count; the
# node count is a signature of the engine version.
#
# Calibration: run it here with the local --threads value (e.g. 14) and on the
# other machine with its value (e.g. 150), then
#     time_other = time_local * nps_local / nps_other
# e.g. 20 ms locally and an NPS ratio of 0.75 -> about 27 ms per move.
# Measure under full load: shared caches, memory bandwidth, all-core turbo and
# hyperthreads (cloud "vCPUs") all lower the per-copy speed.
set -e
N=${1:?usage: bench_parallel.sh N [engine] [depth]}
ENGINE=${2:-$(dirname "$0")/../cmake-build-debug/MainEngine}
DEPTH=${3:-16}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
for i in $(seq 1 "$N"); do
  "$ENGINE" --bench --depth "$DEPTH" > "$TMP/$i.out" 2>&1 &
done
wait
grep -h "^bench" "$TMP"/*.out | awk -v n="$N" '
  { nodes[$3]++; nps += $7; t += $5; c++ }
  END {
    if (c != n) { print "only " c " of " n " copies reported a result"; exit 1 }
    sig = 0; for (k in nodes) { sig++; node_count = k }
    if (sig != 1) print "WARNING: copies report different node counts";
    printf "copies %d  nodes %s  avg time %.0f ms  avg nps per copy %.0f  total nps %.0f\n",
           c, node_count, t / c, nps / c, nps
  }'
