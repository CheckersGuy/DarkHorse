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
#   3. deduplicates across all pieces, streaming them into movegen dedup and deleting
#      each piece once read (disk peak ~ pieces, not pieces + copy + output); if fewer
#      than TOTAL unique positions remain, it generates further pieces and deduplicates
#      them together with the positions kept so far, then writes exactly TOTAL.
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
#   OPENING_STATE=F file with the next opening index (default OUTPUT.pieces/next_opening;
#                   local_run.sh shares one file across the run and its extensions)
#   EXCLUDE_POL1="F1 F2 ..."  POL1 files (e.g. the training + validation file of an
#                   existing run): their positions are never produced, so a later run
#                   adds only NEW positions (local_run.sh extend)
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
# (endgames above all: 1.5B generated -> ~1.0B unique in the first cloud run). The
# pieces are streamed into the dedup (no concatenated copy on disk) and each piece is
# deleted as soon as it has been read, so the disk peak is about the size of the pieces
# plus a little, not pieces + copy + output. The unique positions so far are kept in
# OUTPUT.acc; if fewer than TOTAL, further pieces are generated and deduplicated
# together with OUTPUT.acc, until TOTAL is reached; then exactly TOTAL are kept.
# Resumable: a read piece leaves a piece<i>.used marker (it is not generated again);
# if a dedup is interrupted, the positions of its already deleted pieces are lost, and
# the next round simply generates more pieces for the deficit.
PIECE=${GEN_PIECE_POS:-$(( ${GEN_PIECE_M:-50} * 1000000 ))}
mkdir -p "$PIECES"
TF=$PIECES/pieces.target
OPENING_STATE=${OPENING_STATE:-$PIECES/next_opening}   # position in the opening book
ACC=$OUT.acc
# EXCLUDE_POL1: POL1 files whose positions must not be produced again (extensions of an
# existing data set: its training and validation files); counted once here
NEX=0
for x in ${EXCLUDE_POL1:-}; do
  [ -s "$x" ] || fail "exclude file $x missing"
  NEX=$(( NEX + $(od -An -t u8 -j 4 -N 8 "$x" | tr -d ' ') ))
done
EXCLUDE_POL1=${EXCLUDE_POL1:-}
[ -s "$TF" ] || echo $(( (TOTAL + PIECE - 1) / PIECE )) > "$TF"
cd "$ENGINE_DIR" || fail "no engine dir"
for round in $(seq 1 20); do
  N=$(cat "$TF")
  log "round $round: $N pieces of $PIECE positions (${OPTS[*]})"
  for i in $(seq 1 "$N"); do
    f=$PIECES/piece$i.samples
    { [ -s "$f" ] || [ -e "$PIECES/piece$i.used" ]; } && continue
    rm -f "$f.tmp"
    log "piece $i/$N: $PIECE positions"
    # continue through the opening book where the previous piece stopped (all pieces
    # starting at opening 0 replayed the same openings: many cross-piece duplicates)
    START=0; [ -s "$OPENING_STATE" ] && START=$(cat "$OPENING_STATE")
    "$MOVEGEN" generate "$BOOK" "$f.tmp" "$WORKERS" "$PIECE" "${OPTS[@]}" start=$START > "$f.log" 2>&1 ||
      fail "piece $i: generate failed (see $f.log)"
    grep -a "We got back\|stopping\|ERROR\|panick" "$f.log"
    [ -s "$f.tmp" ] || fail "piece $i: empty output"
    NEXT=$(grep -ao "next_opening [0-9]*" "$f.log" | tail -1 | grep -o "[0-9]*$")
    [ -n "$NEXT" ] && echo "$NEXT" > "$OPENING_STATE"
    rm -f "$f.log"
    mv "$f.tmp" "$f"
  done
  NEWP=$(ls "$PIECES"/piece*.samples 2>/dev/null | sort -V)
  if [ -n "$NEWP" ]; then
    ACCN=0; [ -s "$ACC" ] && ACCN=$(( $(stat -c %s "$ACC") / 18 ))
    EXP=$(( ACCN + NEX )); for f in $NEWP; do EXP=$(( EXP + $(stat -c %s "$f") / 18 )); done
    FP=1e-4; [ "$EXP" -gt 1500000000 ] && FP=1e-3      # Bloom filter RAM: ~2.4 vs ~1.8 bytes/entry
    log "deduplicating $ACCN kept + $(echo $NEWP | wc -w) new pieces$([ "$NEX" -gt 0 ] && echo " against $NEX excluded positions") ($EXP positions)"
    rm -f "$OUT.tmp"
    # stream: [excluded positions] [kept positions] [new pieces] -> dedup (keeps first
    # occurrences, so excluded positions win) -> drop the excluded prefix again
    {
      for x in $EXCLUDE_POL1; do "$MOVEGEN" pol2samples "$x" /dev/fd/3 3>&1 1>&2; done
      [ -s "$ACC" ] && cat "$ACC"
      for f in $NEWP; do
        cat "$f" && rm -f "$f" && touch "${f%.samples}.used"
      done
    } 2>/dev/null | "$MOVEGEN" dedup /dev/stdin /dev/fd/3 "$EXP" "$FP" 3>&1 1>"$OUT.dedup.log" |
      tail -c +$(( NEX * 18 + 1 )) > "$OUT.tmp" ||
      fail "dedup failed (rerun: read pieces are gone, the next round makes up the deficit)"
    grep -aq "dedup done" "$OUT.dedup.log" || fail "dedup failed (see $OUT.dedup.log)"
    [ -s "$OUT.tmp" ] || fail "dedup produced no new positions"
    mv "$OUT.tmp" "$ACC"
  fi
  [ -s "$ACC" ] || fail "no positions"
  U=$(( $(stat -c %s "$ACC") / 18 ))
  if [ "$U" -ge "$TOTAL" ]; then
    # keep exactly TOTAL (the file order is piece order; pieces are independent)
    truncate -s $(( TOTAL * 18 )) "$ACC"
    log "$U unique positions >= $TOTAL; keeping $TOTAL"
    break
  fi
  # deficit: further pieces, assuming new pieces add unique positions at 80 % of the
  # average rate so far (later pieces repeat more of what already exists)
  RATE_PPM=$(( U * 1000000 / (N * PIECE) ))
  [ "$RATE_PPM" -ge 1 ] || RATE_PPM=1
  EXTRA=$(( ( (TOTAL - U) * 1000000 * 10 / (RATE_PPM * 8) + PIECE - 1) / PIECE ))
  [ "$EXTRA" -ge 1 ] || EXTRA=1
  log "only $U unique positions of $TOTAL (unique rate $(( RATE_PPM / 10000 )) %); adding $EXTRA pieces"
  echo $(( N + EXTRA )) > "$TF"
done
U=$(( $(stat -c %s "$ACC") / 18 ))
[ "$U" -ge "$TOTAL" ] || fail "target not reached after 20 rounds"
mv "$ACC" "$OUT"
rm -rf "$PIECES"   # only markers and pieces.target are left
log "done: $OUT ($(( $(stat -c %s "$OUT") / 18 )) positions)"
