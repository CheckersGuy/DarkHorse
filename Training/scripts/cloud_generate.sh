#!/bin/bash
# cloud_generate.sh OUTPUT WORKERS TOTAL [generate options ...]
#
# Self-play data generation (positions sampled from the search trees), resumable:
#   1. builds the opening book Training/Positions/tiny_plus4.book if it does not exist
#      yet (tiny.book expanded by 2 + 2 plies, depth-1 score within +-55, deduplicated;
#      ~5.5M openings in ~4 min on 16 threads; deterministic). Not in git.
#   2. generates TOTAL positions in pieces of GEN_PIECE_M million (default 50):
#      OUTPUT.pieces/piece<i>.samples, each written under a temporary name and kept
#      once complete. After an interruption (spot instance) rerun the same command:
#      finished pieces are skipped, at most one piece is lost.
#      Default options nodes=1500 prob=0.05 cap=300 dedup=1 (dedup within a piece);
#      options on the command line replace the defaults (keys: movegen main.rs).
#   3. deduplicates across all pieces into OUTPUT (movegen dedup, Bloom filter sized
#      for the input). Cross-piece duplicates make OUTPUT slightly smaller than TOTAL.
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

# ---- 2. pieces
PIECE=${GEN_PIECE_POS:-$(( ${GEN_PIECE_M:-50} * 1000000 ))}
N=$(( (TOTAL + PIECE - 1) / PIECE ))
mkdir -p "$PIECES"
log "generating $TOTAL positions in $N pieces of up to $PIECE (${OPTS[*]})"
cd "$ENGINE_DIR" || fail "no engine dir"
for i in $(seq 1 "$N"); do
  f=$PIECES/piece$i.samples
  [ -s "$f" ] && continue
  size=$(( i < N ? PIECE : TOTAL - (N - 1) * PIECE ))
  rm -f "$f.tmp"
  log "piece $i/$N: $size positions"
  "$MOVEGEN" generate "$BOOK" "$f.tmp" "$WORKERS" "$size" "${OPTS[@]}" 2>&1 |
    grep -a "We got back\|stopping\|ERROR\|panick" || fail "piece $i: generate failed"
  [ -s "$f.tmp" ] || fail "piece $i: empty output"
  mv "$f.tmp" "$f"
done

# ---- 3. dedup across pieces
log "deduplicating $N pieces"
rm -f "$OUT.all.tmp" "$OUT.tmp"
cat "$PIECES"/piece*.samples > "$OUT.all.tmp" || fail "concatenation failed"
"$MOVEGEN" dedup "$OUT.all.tmp" "$OUT.tmp" 2>&1 | grep -a "dedup done" || fail "dedup failed"
rm -f "$OUT.all.tmp"
mv "$OUT.tmp" "$OUT"
[ "${KEEP_PIECES:-0}" = 1 ] || rm -rf "$PIECES"
log "done: $OUT ($(( $(stat -c %s "$OUT") / 18 )) positions)"
