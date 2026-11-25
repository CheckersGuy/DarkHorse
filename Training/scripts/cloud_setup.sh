#!/bin/bash
# cloud_setup.sh - set up a fresh Linux (Debian/Ubuntu) cloud machine and run the whole
# data pipeline (generation -> rescoring -> training files), or resume it after a restart.
#
# Usage, either
#   a) as your normal user:   bash cloud_setup.sh
#      (installs an @reboot crontab entry, so the run resumes after every restart), or
#   b) pasted as the Google Cloud "startup-script" (VM metadata): runs as root at every
#      boot and switches to RUN_USER by itself (no crontab needed).
# Edit the settings below (or pass them as environment variables) before the first run.
#
# What it does:
#   1. installs missing packages (build tools, cmake, git, python3, zstd, cron) and a
#      nightly Rust toolchain
#   2. clones the repo (or fast-forwards it) and builds MainEngine + movegen
#   3. checks the engine (bench node count) and runs a small end-to-end pipeline test
#   4. freezes the binaries into RUN_DIR/bin and starts cloud_pipeline.sh detached
# On every later start (restart / spot interruption) it does NOT pull or rebuild once a
# run exists in RUN_DIR: it resumes the run with the frozen binaries, so all positions
# of a run are labelled by the same engine.
#
# Progress:  tail -f RUN_DIR/pipeline.log      Ready files: ls RUN_DIR/rescore/*.ready
# Setup log: RUN_DIR/setup.log
set -u -o pipefail

# ---------------- settings ----------------
RUN_USER=${RUN_USER:-}                    # user that owns the run (needed when run as root)
REPO_URL=${REPO_URL:-https://github.com/CheckersGuy/DarkHorse.git}
BRANCH=${BRANCH:-main}
TOTAL=${TOTAL:-1500000000}                # positions to generate
WORKERS=${WORKERS:-}                      # default: all hardware threads
COMPRESS=${COMPRESS-9}                    # zstd level for the ready files ("" = none)
EXPECT_BENCH=${EXPECT_BENCH-1281642}      # MainEngine --bench node count ("" = no check)
SELFTEST=${SELFTEST:-1}                   # 1 = small end-to-end test before the real run
BUILD_JOBS=${BUILD_JOBS:-}                # parallel build jobs (default: all threads)
NO_CRON=${NO_CRON:-}                      # 1 = do not install the @reboot crontab entry
# paths (relative to the run user's home unless absolute)
BASE_DIR=${BASE_DIR:-data}                # repo goes to BASE_DIR/DarkHorse
RUN_DIR=${RUN_DIR:-run}
# -------------------------------------------

# ---- started as root (cloud startup-script): continue as RUN_USER
if [ "$(id -u)" -eq 0 ]; then
  if [ -z "$RUN_USER" ]; then
    echo "cloud_setup.sh: run as root, set RUN_USER=<your user name> in the script" >&2
    exit 1
  fi
  id "$RUN_USER" >/dev/null 2>&1 || { echo "user $RUN_USER does not exist" >&2; exit 1; }
  SELF=$(mktemp /tmp/cloud_setup.XXXX.sh); cat "$0" > "$SELF"; chmod 755 "$SELF"
  exec sudo -u "$RUN_USER" -H env RUN_USER="$RUN_USER" REPO_URL="$REPO_URL" BRANCH="$BRANCH" \
    TOTAL="$TOTAL" WORKERS="$WORKERS" COMPRESS="$COMPRESS" EXPECT_BENCH="$EXPECT_BENCH" \
    SELFTEST="$SELFTEST" BASE_DIR="$BASE_DIR" RUN_DIR="$RUN_DIR" BUILD_JOBS="$BUILD_JOBS" \
    FROM_ROOT=1 bash "$SELF"
fi

cd "$HOME"
case "$BASE_DIR" in /*) ;; *) BASE_DIR=$HOME/$BASE_DIR;; esac
case "$RUN_DIR" in /*) ;; *) RUN_DIR=$HOME/$RUN_DIR;; esac
REPO=$BASE_DIR/DarkHorse
BIN=$RUN_DIR/bin
WORKERS=${WORKERS:-$(nproc)}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
mkdir -p "$RUN_DIR" "$BASE_DIR"
LOG=$RUN_DIR/setup.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() { log "ERROR: $*"; exit 1; }
log "==== cloud_setup.sh start (user $(id -un), $(nproc) threads, workers $WORKERS)"

# ---- 1. packages + Rust (only what is missing)
need=()
pkgs=(build-essential cmake git python3 util-linux)
[ -n "$COMPRESS" ] && pkgs+=(zstd)                       # only needed for .zst files
[ -z "$NO_CRON" ] && [ -z "${FROM_ROOT:-}" ] && pkgs+=(cron)  # @reboot entry
for p in "${pkgs[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || need+=("$p")
done
if [ ${#need[@]} -gt 0 ]; then
  log "installing ${need[*]}"
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -qq >> "$LOG" 2>&1 &&
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${need[@]}" >> "$LOG" 2>&1 ||
    fail "apt-get install failed (needs passwordless sudo; see $LOG)"
fi
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
if ! command -v cargo >/dev/null || ! rustc --version | grep -q nightly; then
  log "installing the nightly Rust toolchain"
  curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly >> "$LOG" 2>&1 ||
    fail "rustup failed"
  . "$HOME/.cargo/env"
fi
sudo -n systemctl enable --now cron >/dev/null 2>&1 || true

# ---- run already exists? then only resume (no pull, no rebuild)
RUN_EXISTS=0
[ -f "$RUN_DIR/pipeline.log" ] && RUN_EXISTS=1
if [ "$RUN_EXISTS" = 1 ] && grep -q "pipeline finished" "$RUN_DIR/pipeline.log" &&
   [ -f "$RUN_DIR/rescore/train.ready" ]; then
  log "run in $RUN_DIR is already finished; nothing to do"
  exit 0
fi

if [ "$RUN_EXISTS" = 0 ]; then
  # ---- 2. repo + build
  if [ -d "$REPO/.git" ]; then
    log "updating $REPO ($BRANCH)"
    git -C "$REPO" fetch -q origin && git -C "$REPO" checkout -q "$BRANCH" &&
      git -C "$REPO" pull -q --ff-only >> "$LOG" 2>&1 || fail "git update failed (local changes?)"
  else
    log "cloning $REPO_URL into $REPO"
    git clone -q -b "$BRANCH" "$REPO_URL" "$REPO" >> "$LOG" 2>&1 || fail "git clone failed"
  fi
  log "building MainEngine"
  ( cd "$REPO" && cmake . >> "$LOG" 2>&1 &&
    cmake --build . --target MainEngine -j "$BUILD_JOBS" >> "$LOG" 2>&1 ) || fail "engine build failed"
  bench=$("$REPO/cmake-build-debug/MainEngine" --bench | grep -o "bench nodes [0-9]*" | grep -o "[0-9]*$")
  log "engine bench: $bench nodes"
  [ -z "$EXPECT_BENCH" ] || [ "$bench" = "$EXPECT_BENCH" ] ||
    fail "bench $bench != expected $EXPECT_BENCH (set EXPECT_BENCH to the new value if the engine changed on purpose)"
  log "building movegen"
  ( cd "$REPO/Training/Rust" && cargo build --release -j "$BUILD_JOBS" --bin movegen >> "$LOG" 2>&1 ) ||
    fail "movegen build failed"

  # freeze the binaries for this run
  mkdir -p "$BIN"
  cp "$REPO/cmake-build-debug/MainEngine" "$BIN/MainEngine"
  cp "$REPO/Training/Rust/target/release/movegen" "$BIN/movegen"
  git -C "$REPO" rev-parse HEAD > "$BIN/commit"
  echo "$bench" > "$BIN/bench"

  # ---- resources
  free_gb=$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc 0-9)
  need_gb=$(( TOTAL / 5000000 ))             # ~200 bytes per position incl. temporary files
  mem_gb=$(free -g | awk '/^Mem:/ {print $2}')
  log "disk free ${free_gb} GB (needed ~${need_gb} GB), RAM ${mem_gb} GB (needed ~$(( WORKERS / 8 + 4 )) GB)"
  [ "$free_gb" -ge "$need_gb" ] || fail "not enough disk space in $RUN_DIR"
  [ "$mem_gb" -ge $(( WORKERS / 8 + 4 )) ] || fail "not enough RAM for $WORKERS workers"

  # ---- 3. end-to-end self-test
  if [ "$SELFTEST" = 1 ]; then
    T=$RUN_DIR.selftest; rm -rf "$T"
    log "self-test: 100k positions in $T (builds the opening book first if missing)"
    MOVEGEN=$BIN/movegen ENGINE_DIR=$BIN GEN_PIECE_POS=50000 CHUNK_POS=20000 VAL_POS=5000 \
      BLOCK_CHUNKS=2 "$REPO/Training/scripts/cloud_pipeline.sh" "$T" "$(( WORKERS < 16 ? WORKERS : 16 ))" 100000 \
      >> "$LOG" 2>&1 || fail "self-test failed (see $T/pipeline.log)"
    grep -q "pipeline finished" "$T/pipeline.log" && [ -f "$T/rescore/train.ready" ] &&
      [ -f "$T/rescore/val.ready" ] && [ -f "$T/rescore/block.ready" ] ||
      fail "self-test incomplete (see $T/pipeline.log)"
    log "self-test passed: $(cat "$T/rescore/train.ready")"
    rm -rf "$T"
  fi
else
  log "resuming the existing run in $RUN_DIR"
  if [ ! -x "$BIN/MainEngine" ] || [ ! -x "$BIN/movegen" ]; then
    # run started by hand before this script existed: freeze what it used
    mkdir -p "$BIN"
    cp "$REPO/Training/Rust/MainEngine" "$BIN/MainEngine" &&
      cp "$REPO/Training/Rust/target/release/movegen" "$BIN/movegen" ||
      fail "cannot find the binaries of the existing run"
    log "froze the binaries the existing run was started with"
  fi
fi

# ---- 4. (re)start the pipeline detached; its lock file prevents a second copy
RUN_CMD="cd $REPO && COMPRESS=$COMPRESS MOVEGEN=$BIN/movegen ENGINE_DIR=$BIN Training/scripts/cloud_pipeline.sh $RUN_DIR $WORKERS $TOTAL"
if pgrep -f "cloud_pipeline.sh $RUN_DIR " >/dev/null; then
  log "pipeline is already running"
else
  log "starting: $RUN_CMD"
  before=$(grep -c "pipeline start" "$RUN_DIR/pipeline.log" 2>/dev/null)
  setsid nohup bash -c "$RUN_CMD" >> "$RUN_DIR/start.out" 2>&1 < /dev/null &
  sleep 5
  # started = still running, or it already logged a new start (tiny runs finish fast)
  pgrep -f "cloud_pipeline.sh $RUN_DIR " >/dev/null ||
    [ "$(grep -c "pipeline start" "$RUN_DIR/pipeline.log" 2>/dev/null)" -gt "${before:-0}" ] ||
    fail "pipeline did not start (see $RUN_DIR/start.out)"
fi

# ---- restart after reboots: crontab entry (not needed when started as a cloud startup-script)
if [ -z "${FROM_ROOT:-}" ] && [ -z "$NO_CRON" ]; then
  SCRIPT=$REPO/Training/scripts/cloud_setup.sh
  ENTRY="@reboot TOTAL=$TOTAL WORKERS=$WORKERS COMPRESS=$COMPRESS BASE_DIR=$BASE_DIR RUN_DIR=$RUN_DIR bash $SCRIPT >> $RUN_DIR/cron.out 2>&1"
  ( crontab -l 2>/dev/null | grep -v "cloud_pipeline.sh\|cloud_setup.sh"; echo "$ENTRY" ) | crontab - &&
    log "crontab @reboot entry installed"
fi
log "==== setup done; progress: tail -f $RUN_DIR/pipeline.log"
