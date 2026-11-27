#!/bin/bash
# cloud_pipeline.sh WORKDIR WORKERS TOTAL
#
# The whole data run on a (spot) cloud machine, restartable at any point:
#   1. cloud_generate.sh  -> WORKDIR/gen.samples (TOTAL positions, resumable pieces,
#                            opening book built if missing, final dedup)
#   2. cloud_rescore.sh   -> WORKDIR/rescore/: new validation set (val.ready), random
#                            chunks of CHUNK_M million (default 10 for spot instances),
#                            early 500M block (block.ready), full training file
#                            (train.ready)
# After an interruption run the same command again (or let @reboot do it, below):
# every finished step/piece/chunk is skipped, at most one piece or chunk is redone.
# A lock file prevents two copies from running at the same time.
#
# Requirements on a spot instance:
#   - WORKDIR (and ideally the repo) on a PERSISTENT volume that survives termination
#     (instance-store/local NVMe disks are usually wiped). Data volume needed for 1.5B:
#     ~300 GB (3B: ~600 GB).
#   - current engine build in Training/Rust/MainEngine (or ENGINE_DIR), movegen built
#     (cargo build --release --bin movegen in Training/Rust), python3, optional zstd.
#   - automatic restart after the instance comes back, e.g. crontab -e:
#       @reboot cd /path/to/DarkHorse && Training/scripts/cloud_pipeline.sh /data/run 350 1500000000 >> /data/run/cron.out 2>&1
# Download when the marker files appear: rescore/val.children.samples (val.ready),
# rescore/block.children.samples (block.ready), rescore/train.children.samples
# (train.ready); with COMPRESS=9 also the .zst files.
#
# Environment: everything cloud_generate.sh / cloud_rescore.sh accept (MOVEGEN,
# ENGINE_DIR, BOOK, GEN_PIECE_M, NODES_PHASE, VAL_POS, COMPRESS, DELETE_INPUTS,
# DELETE_CHUNKS, ...) plus
#   CHUNK_M=N        rescoring chunk size in million positions (default 10)
#   BLOCK_CHUNKS=N   chunks in the early block (default 500 / CHUNK_M)
#   KEEP_GEN=1       keep gen.samples after the chunks were split off
set -u -o pipefail
W=$1; WORKERS=$2; TOTAL=$3
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"; W=$(readlink -f "$W")
exec 9> "$W/.pipeline.lock"
flock -n 9 || { echo "another cloud_pipeline.sh is running in $W"; exit 1; }
LOG=$W/pipeline.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

CHUNK_M=${CHUNK_M:-10}
BLOCK_CHUNKS=${BLOCK_CHUNKS:-$(( 500 / CHUNK_M ))}
log "pipeline start: $TOTAL positions, $WORKERS workers, chunks of ${CHUNK_M}M, block after $BLOCK_CHUNKS chunks"

if [ ! -f "$W/rescore/chunks.done" ]; then
  "$S/cloud_generate.sh" "$W/gen.samples" "$WORKERS" "$TOTAL" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { log "generation failed"; exit 1; }
  [ -s "$W/gen.samples" ] || { log "generation produced no positions ($W/gen.samples)"; exit 1; }
fi

"$S/cloud_rescore.sh" "$W/gen.samples" "$W/rescore" "$WORKERS" "$CHUNK_M" "$BLOCK_CHUNKS" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "rescoring failed"; exit 1; }
[ "${KEEP_GEN:-0}" = 1 ] || rm -f "$W/gen.samples"
log "pipeline finished"
