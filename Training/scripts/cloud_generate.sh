#!/bin/bash
# cloud_generate.sh OUTPUT WORKERS TOTAL [generate options ...]
#
# Self-play data generation (positions sampled from the search trees), resumable:
#   1. builds the opening book Training/Positions/tiny_plus4.book if it does not exist
#      yet (tiny.book expanded by 2 + 2 plies, depth-1 score within +-55, deduplicated;
#      ~5.5M openings in ~4 min on 16 threads; deterministic). Not in git.
#   2. generates positions in pieces of GEN_PIECE_M million (default 50):
#      OUTPUT.pieces/piece<i>.samples, each written under a temporary name and kept
#      once complete. After an interruption (spot instance) rerun the same command:
#      finished pieces are skipped, at most one piece is lost.
#      Default options nodes=1500 prob=0.05 cap=300 dedup=1 (dedup within a piece);
#      options on the command line replace the defaults (keys: movegen main.rs).
#   3. deduplicates across all pieces; if fewer than TOTAL unique positions remain, it
#      generates further pieces and deduplicates again until TOTAL is reached, then
#      writes exactly TOTAL unique positions to OUTPUT.
#   Then rescore with cloud_rescore.sh OUTPUT ... (or run everything with
#   cloud_pipeline.sh).
#
# Environment (optional):
#   MOVEGEN=path    movegen binary (default Training/Rust/target/release/movegen)
#   ENGINE_DIR=dir  directory with the CURRENT engine build as ./MainEngine
#                   (default Training/Rust)
#   BOOK=path       use this book instead of building tiny_plus4.book
#   GEN_PIECE_M=N   piece size in million positions (default 50)
#   GEN_PIECE_POS=N piece size in positions (tests)
#   KEEP_PIECES=1   keep the pieces after the final dedup (default: deleted)
set -u -o pipefail
OUT=$1; WORKERS=$2; TOTAL=$3; shift 3
OPTS=("$@"); [ ${#OPTS[@]} -eq 0 ] && OPTS=(nodes=1500 prob=0.05 cap=300 dedup=1)
R=$(cd "$(dirname "$0")/.." && pwd)              # Training
MOVEGEN=$(readlink -f "${MOVEGEN:-$R/Rust/target/release/movegen}")
ENGINE_DIR=$(readlink -f "${ENGINE_DIR:-$R/Rust}")
OUT=$(readlink -f "$OUT")
PIECES=$OUT.pieces
log() { echo "$(date '+%F %T') $*"; }
fail() { log "ERROR: $*"; exit 1; }

if [ -s "$OUT" ]; then
  log "$OUT already complete ($(( $(stat -c %s "$OUT") / 18 )) positions)"; exit 0
fi
[ -x "$MOVEGEN" ] || fail "movegen not found: $MOVEGEN"
"$ENGINE_DIR/MainEngine" --help 2>/dev/null | grep -q sample_prob ||
  fail "$ENGINE_DIR/MainEngine is missing or too old (no --sample_prob); copy the current build"

# ---- 1. opening book (only if missing)
P=$R/Positions
if [ -z "${BOOK:-}" ]; then
  BOOK=$P/tiny_plus4.book
  if [ ! -s "$BOOK" ]; then
    log "building $BOOK"
    cd "$ENGINE_DIR" || fail "no engine dir"
    if [ ! -s "$P/tiny_plus2.book" ]; then
      "$MOVEGEN" book "$P/tiny.book" "$P/tiny_plus2.book.tmp" "$WORKERS" 2 > /dev/null ||
        fail "book stage 1 failed"
      mv "$P/tiny_plus2.book.tmp" "$P/tiny_plus2.book"
    fi
    "$MOVEGEN" book "$P/tiny_plus2.book" "$BOOK.tmp" "$WORKERS" 2 > /dev/null ||
      fail "book stage 2 failed"
    mv "$BOOK.tmp" "$BOOK"
  fi
fi
log "book $BOOK: $(wc -l < "$BOOK") openings"

# ---- 2./3. pieces + dedup across pieces, until TOTAL unique positions exist
# Pieces are only deduplicated internally; across pieces many positions repeat
# (endgames above all: 1.5B generated -> ~1.0B unique in the first cloud run). So after
# the dedup the script generates further pieces for the deficit and deduplicates again,
# until at least TOTAL unique positions exist, then keeps exactly TOTAL of them.
# Resumable: finished pieces are kept, the planned piece count is in pieces.target.
PIECE=${GEN_PIECE_POS:-$(( ${GEN_PIECE_M:-50} * 1000000 ))}
mkdir -p "$PIECES"
TF=$PIECES/pieces.target
[ -s "$TF" ] || echo $(( (TOTAL + PIECE - 1) / PIECE )) > "$TF"
cd "$ENGINE_DIR" || fail "no engine dir"
for round in $(seq 1 20); do
  N=$(cat "$TF")
  log "round $round: $N pieces of $PIECE positions (${OPTS[*]})"
  for i in $(seq 1 "$N"); do
    f=$PIECES/piece$i.samples
    [ -s "$f" ] && continue
    rm -f "$f.tmp"
    log "piece $i/$N: $PIECE positions"
    "$MOVEGEN" generate "$BOOK" "$f.tmp" "$WORKERS" "$PIECE" "${OPTS[@]}" 2>&1 |
      grep -a "We got back\|stopping\|ERROR\|panick" || fail "piece $i: generate failed"
    [ -s "$f.tmp" ] || fail "piece $i: empty output"
    mv "$f.tmp" "$f"
  done
  log "deduplicating $N pieces"
  rm -f "$OUT.all.tmp" "$OUT.tmp"
  cat $(ls "$PIECES"/piece*.samples | sort -V) > "$OUT.all.tmp" || fail "concatenation failed"
  "$MOVEGEN" dedup "$OUT.all.tmp" "$OUT.tmp" 2>&1 | grep -a "dedup done" || fail "dedup failed"
  rm -f "$OUT.all.tmp"
  [ -s "$OUT.tmp" ] || fail "dedup produced no positions"
  U=$(( $(stat -c %s "$OUT.tmp") / 18 ))
  if [ "$U" -ge "$TOTAL" ]; then
    # keep exactly TOTAL (the file order is piece order; pieces are independent)
    truncate -s $(( TOTAL * 18 )) "$OUT.tmp"
    log "$U unique positions >= $TOTAL; keeping $TOTAL"
    break
  fi
  # deficit: further pieces, assuming new pieces add unique positions at 80 % of the
  # average rate so far (later pieces repeat more of what already exists)
  RATE_PPM=$(( U * 1000000 / (N * PIECE) ))
  EXTRA=$(( ( (TOTAL - U) * 1000000 * 10 / (RATE_PPM * 8) + PIECE - 1) / PIECE ))
  [ "$EXTRA" -ge 1 ] || EXTRA=1
  log "only $U unique positions of $TOTAL (unique rate $(( RATE_PPM / 10000 )) %); adding $EXTRA pieces"
  echo $(( N + EXTRA )) > "$TF"
  rm -f "$OUT.tmp"
done
[ -s "$OUT.tmp" ] || fail "target not reached after 20 rounds"
mv "$OUT.tmp" "$OUT"
[ "${KEEP_PIECES:-0}" = 1 ] || rm -rf "$PIECES"
log "done: $OUT ($(( $(stat -c %s "$OUT") / 18 )) positions)"
