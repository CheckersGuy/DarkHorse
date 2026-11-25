#!/bin/bash
# cloud_generate.sh OUTPUT WORKERS MAX_SAMPLES [generate options ...]
#
# Self-play data generation (positions sampled from the search trees):
#   1. builds the opening book Training/Positions/tiny_plus4.book if it does not exist
#      yet (tiny.book expanded by 2 + 2 plies, depth-1 score within +-55, deduplicated;
#      ~5.5M openings in ~3 min on 16 threads). The book is not in git.
#   2. runs `movegen generate` with that book into OUTPUT (default options
#      nodes=1500 prob=0.05 cap=300 dedup=1; options given on the command line replace
#      these defaults completely, see `movegen` main.rs for the keys)
#   Then rescore with cloud_rescore.sh OUTPUT ...
#
# Environment (optional):
#   MOVEGEN=path   movegen binary (default Training/Rust/target/release/movegen)
#   ENGINE_DIR=dir directory with the CURRENT engine build as ./MainEngine
#                  (default Training/Rust)
#   BOOK=path      use this book instead of building tiny_plus4.book
set -u -o pipefail
OUT=$1; WORKERS=$2; MAX=$3; shift 3
OPTS=("$@"); [ ${#OPTS[@]} -eq 0 ] && OPTS=(nodes=1500 prob=0.05 cap=300 dedup=1)
R=$(cd "$(dirname "$0")/.." && pwd)              # Training
MOVEGEN=$(readlink -f "${MOVEGEN:-$R/Rust/target/release/movegen}")
ENGINE_DIR=$(readlink -f "${ENGINE_DIR:-$R/Rust}")
OUT=$(readlink -f "$OUT")
log() { echo "$(date '+%F %T') $*"; }
fail() { log "ERROR: $*"; exit 1; }

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

# ---- 2. generate
log "generating $MAX positions into $OUT with $WORKERS workers (${OPTS[*]})"
cd "$ENGINE_DIR" || fail "no engine dir"
"$MOVEGEN" generate "$BOOK" "$OUT.tmp" "$WORKERS" "$MAX" "${OPTS[@]}" 2>&1 | grep -av "^\[" ||
  fail "generate failed"
mv "$OUT.tmp" "$OUT"
log "done: $OUT ($(( $(stat -c %s "$OUT") / 18 )) positions)"
