#!/bin/bash
# cloud_rescore.sh INPUT WORKDIR WORKERS [CHUNK_M] [BLOCK_CHUNKS]
#
# Rescores a (deduplicated) sample file in chunks and builds ready-to-train files:
#   0. splits off a NEW validation set of VAL_POS positions (default 2,500,000; split by
#      a hash of the position, so a position is never in both sets) and rescores it
#      first: WORKDIR/val.children.samples + WORKDIR/val.ready. All nets (old and new)
#      are compared on this set from now on (owner's decision 2026-11-25).
#   1. splits INPUT into chunks of about CHUNK_M million positions (default 50); every
#      position goes to a RANDOM chunk, so each chunk (and the early block below) is a
#      random sample of the whole input: WORKDIR/data.c<i>.samples
#   2. rescores the chunks one after another (movegen rescore = root value without MLH
#      + child scores, node budget NODES_PHASE) and keeps them:
#      WORKDIR/data.c<i>.children.samples
#   3. once, after BLOCK_CHUNKS chunks (default 10 = 500M; 0 = no block): merges and
#      shuffles them into the early training file WORKDIR/block.children.samples and
#      writes WORKDIR/block.ready -> download it for first checks while rescoring
#      continues. Not needed for step 4 (delete it after the download).
#   4. at the end: merges and shuffles ALL chunks into the training file
#      WORKDIR/train.children.samples and writes WORKDIR/train.ready.
#   Optional: COMPRESS=LEVEL (e.g. 9) also writes <file>.zst for each ready file
#   (needs the zstd tool; about 45 % smaller at level 9, measured 2026-11-25).
#
# Crash-safe and resumable: run the same command again. Finished chunks/files are
# skipped; an interrupted chunk is rescored again from scratch (at most one chunk is
# lost). Outputs are written under temporary names and renamed when complete+checked.
#
# Environment (optional):
#   MOVEGEN=path         movegen binary (default Training/Rust/target/release/movegen)
#   ENGINE_DIR=dir       directory with the engine as ./MainEngine (default Training/Rust)
#   NODES_PHASE=A,B,C,D  node budget (default: movegen's default 2500,2500,1500,750)
#   DELETE_INPUTS=1      delete each unrescored chunk after it is rescored
#   DELETE_CHUNKS=1      delete the rescored chunks once train.children.samples is ready
#   CHUNK_POS=N          chunk size in positions (tests)
#   VAL_POS=N            size of the new validation set (default 2500000, 0 = none)
#
# The engine must be the CURRENT build (--no_mlh, node budgets; the script checks it).
# Disk for 3B positions: input ~54 GB + input chunks ~54 GB (or DELETE_INPUTS=1)
# + rescored chunks ~147 GB + final merge/shuffle peak ~300 GB + block ~25 GB.
# Start detached, e.g.:
#   setsid nohup Training/scripts/cloud_rescore.sh /data/gen.dedup.samples /data/rescore 350 \
#     > /data/rescore.out 2>&1 < /dev/null &
set -u
IN=$(readlink -f "$1"); W=$2; WORKERS=$3; CHUNK_M=${4:-50}; BLOCK=${5:-10}
R=$(cd "$(dirname "$0")/.." && pwd)              # Training
MOVEGEN=$(readlink -f "${MOVEGEN:-$R/Rust/target/release/movegen}")
ENGINE_DIR=$(readlink -f "${ENGINE_DIR:-$R/Rust}")   # must contain MainEngine
mkdir -p "$W"; W=$(readlink -f "$W")
LOG=$W/cloud_rescore.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() { log "ERROR: $*"; exit 1; }
count_pol1() { od -An -t u8 -j 4 -N 8 "$1" | tr -d ' '; }   # POL1 header sample count

[ -x "$MOVEGEN" ] || fail "movegen not found: $MOVEGEN"
"$ENGINE_DIR/MainEngine" --help 2>/dev/null | grep -q nodes_phase ||
  fail "$ENGINE_DIR/MainEngine is missing or too old (no --nodes_phase); copy the current build"
if [ -n "${COMPRESS:-}" ]; then command -v zstd >/dev/null || fail "COMPRESS set but zstd not installed"; fi

# merge + shuffle chunk files into a training file: build_file NAME CHUNK_FILES...
build_file() {
  local name=$1; shift
  local f=$W/$name.children.samples
  [ -f "$W/$name.ready" ] && return 0
  log "$name: merging $# chunks"
  rm -f "$f" "$f.tmp" "$f.shuf" "$f.shuf.part"*
  python3 "$R/scripts/merge_policy_files.py" "$f.tmp" "$@" >> "$LOG" 2>&1 || fail "$name: merge failed"
  "$MOVEGEN" pshuffle "$f.tmp" "$f.shuf" >> "$LOG" 2>&1 || fail "$name: shuffle failed"
  [ "$(count_pol1 "$f.shuf")" = "$(count_pol1 "$f.tmp")" ] || fail "$name: shuffle lost samples"
  rm -f "$f.tmp"
  mv "$f.shuf" "$f"
  if [ -n "${COMPRESS:-}" ]; then
    log "$name: compressing (zstd -$COMPRESS)"
    rm -f "$f.zst"
    zstd -q -T0 "-$COMPRESS" "$f" -o "$f.zst.tmp" && mv "$f.zst.tmp" "$f.zst" || fail "$name: zstd failed"
  fi
  echo "$(count_pol1 "$f") positions, $(stat -c %s "$f") bytes" > "$W/$name.ready"
  log "$name: ready for download ($(cat "$W/$name.ready"))"
}

# rescore one sample file into a POL1 file: rescore_file SRC OUT LABEL
rescore_file() {
  local src=$1 out=$2 label=$3 want got
  rm -f "$out.tmp"*                               # leftovers of an interrupted run
  want=$(( $(stat -c %s "$src") / 18 ))
  log "$label: rescoring $want positions"
  ( cd "$ENGINE_DIR" && "$MOVEGEN" rescore "$src" "$out.tmp" "$WORKERS" ${NODES_PHASE:-} ) >> "$LOG" 2>&1 ||
    fail "$label: movegen rescore failed (rerun the script to retry)"
  got=$(count_pol1 "$out.tmp")
  # rescore drops a few positions (captures, decided evals): expect >= 90 %
  [ "$got" -ge $(( want * 9 / 10 )) ] || fail "$label: only $got of $want positions rescored"
  mv "$out.tmp" "$out"
  log "$label: done, $got positions"
}

# ---- 0. new validation set (once), rescored first
VAL_POS=${VAL_POS:-2500000}
TRAIN_IN=$IN
if [ "$VAL_POS" -gt 0 ]; then
  TRAIN_IN=$W/train_input.samples
  if [ ! -f "$W/split.done" ]; then
    [ -s "$IN" ] || fail "input $IN is missing or empty"
    total=$(( $(stat -c %s "$IN") / 18 ))
    frac=$(python3 -c "print(min(1.0, $VAL_POS / $total))")
    log "splitting off a validation set of about $VAL_POS positions (fraction $frac)"
    rm -f "$TRAIN_IN" "$W/val.samples"
    "$MOVEGEN" split "$IN" "$TRAIN_IN" "$W/val.samples" "$frac" >> "$LOG" 2>&1 || fail "validation split failed"
    touch "$W/split.done"
  fi
  if [ ! -f "$W/val.ready" ]; then
    rescore_file "$W/val.samples" "$W/val.children.samples" "validation set"
    echo "$(count_pol1 "$W/val.children.samples") positions" > "$W/val.ready"
    log "validation set ready for download ($(cat "$W/val.ready"))"
  fi
fi

# ---- 1. split into random chunks (once)
if [ ! -f "$W/chunks.done" ]; then
  [ -s "$TRAIN_IN" ] || fail "input $TRAIN_IN is missing or empty"
  total=$(( $(stat -c %s "$TRAIN_IN") / 18 ))
  per=${CHUNK_POS:-$(( CHUNK_M * 1000000 ))}
  n=$(( (total + per - 1) / per ))
  log "splitting $IN ($total positions) into $n random chunks"
  rm -f "$W"/data.c*.samples
  "$MOVEGEN" chunks "$TRAIN_IN" "$W/data." "$n" random >> "$LOG" 2>&1 || fail "chunk split failed"
  echo "$n" > "$W/chunks.done"
  [ "${DELETE_INPUTS:-0}" = 1 ] && [ "$TRAIN_IN" != "$IN" ] && rm -f "$TRAIN_IN"
fi
N=$(cat "$W/chunks.done")
[ "${N:-0}" -ge 1 ] 2>/dev/null ||
  fail "$W/chunks.done says '$N' chunks (left over from a failed run?); delete $W and start again"
log "start: $N chunks, $WORKERS workers, early block after $BLOCK chunks"

# ---- 2./3. rescore chunk by chunk, early block once
for i in $(seq 1 "$N"); do
  src=$W/data.c$i.samples
  out=$W/data.c$i.children.samples
  if [ ! -f "$out" ] && [ ! -f "$W/train.ready" ]; then
    [ -f "$src" ] || fail "missing input chunk $src"
    rescore_file "$src" "$out" "chunk $i/$N"
    [ "${DELETE_INPUTS:-0}" = 1 ] && rm -f "$src"
  fi
  if [ "$BLOCK" -gt 0 ] && [ "$i" -eq "$BLOCK" ] && [ "$BLOCK" -lt "$N" ]; then
    files=(); for j in $(seq 1 "$BLOCK"); do files+=("$W/data.c$j.children.samples"); done
    build_file block "${files[@]}"
  fi
done

# ---- 4. the full training file
if [ ! -f "$W/train.ready" ]; then
  files=(); for j in $(seq 1 "$N"); do files+=("$W/data.c$j.children.samples"); done
  build_file train "${files[@]}"
fi
[ "${DELETE_CHUNKS:-0}" = 1 ] && rm -f "$W"/data.c*.children.samples
log "all $N chunks rescored, training file ready"
