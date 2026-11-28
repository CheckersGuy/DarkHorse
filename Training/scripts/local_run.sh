#!/bin/bash
# local_run.sh start|pause|resume|stop|status [TOTAL] [WORKERS]
#
# Value-net data run on this PC (WSL) whenever it is available: generation + value-only
# rescoring (cloud_pipeline.sh with VALUE_ONLY=1) into RUN_DIR, restartable at any time.
#   start [TOTAL] [WORKERS]  start or resume the run (TOTAL default 2000000000, WORKERS
#                            default all threads; both are remembered for later starts)
#   pause                    freeze everything at once (CPU free immediately, nothing lost)
#   resume                   continue a paused run exactly where it stopped
#   stop                     end the processes (the current piece/chunk is redone later)
#   status                   progress (RUN_DIR/STATUS.txt, updated once a minute)
#   extend ADD               ADD more positions, additive (nothing is redone):
#                            - while the run is still generating: the target is raised
#                            - later: an extension segment RUN_DIR/seg<k> is queued; after
#                              the run it generates ADD new positions that exclude every
#                              existing one, rescores them and appends them to
#                              RUN_DIR/rescore/train.children.samples (same validation set)
#   daemon                   start if needed and stay in the foreground while the run is
#                            active (used by the Windows autostart: keeps WSL running)
#   autostart-install        start automatically: when WSL starts (crontab @reboot) and
#                            when you log in to Windows (hidden script in the Windows
#                            Startup folder that runs "local_run.sh daemon")
#   autostart-remove         remove both again
# Adaptive mode (default, ADAPTIVE=0 to disable): adaptive_throttle.py measures what
# Windows programs use and keeps only as many engines running as there are free CPUs;
# the rest are frozen and resumed when the load drops. All searches use node budgets,
# so freezing changes no result. The run also has the lowest priority inside WSL.
# After a reboot or a WSL shutdown simply run "local_run.sh start" again.
#
# Environment: RUN_DIR (default /mnt/c/TrainingData/vrun), ADAPTIVE, RESERVE (CPUs kept
# free while other programs are active, default 1), plus everything cloud_pipeline.sh
# accepts (CHUNK_M, GEN_PIECE_M, NODES_PHASE, VAL_POS, ...).
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)            # repo
S=$R/Training/scripts
RUN_DIR=${RUN_DIR:-/mnt/c/TrainingData/vrun}
BIN=$RUN_DIR/bin
CMD=${1:-status}
mkdir -p "$RUN_DIR"
PGF=$RUN_DIR/.pgid
pgid() { [ -s "$PGF" ] && cat "$PGF"; }
alive() { local g; g=$(pgid) && [ -n "$g" ] && kill -0 -- "-$g" 2>/dev/null; }
throttle_pid() { [ -s "$RUN_DIR/.throttle_pid" ] && cat "$RUN_DIR/.throttle_pid"; }

base_done() { [ -f "$RUN_DIR/rescore/train.ready" ]; }
next_segment() {   # first extension segment that is not merged yet
  local d; for d in $(ls -d "$RUN_DIR"/seg* 2>/dev/null | sort -V); do
    [ -e "$d/merged" ] || { echo "$d"; return; }
  done
}
finished() { base_done && [ -z "$(next_segment)" ]; }
pol1_n() { od -An -t u8 -j 4 -N 8 "$1" | tr -d ' '; }
merge_segment() {   # append a finished segment's training data to the main training file
  local seg=$1 B=$RUN_DIR/rescore/train.children.samples E=$1/rescore/train.children.samples
  [ -s "$seg/merge.state" ] || echo "$(stat -c %s "$B") $(pol1_n "$B")" > "$seg/merge.state"
  local orig nb ne; read -r orig nb < "$seg/merge.state"; ne=$(pol1_n "$E")
  truncate -s "$orig" "$B"                  # idempotent after an interrupted merge
  tail -c +13 "$E" >> "$B" || return 1
  python3 -c "import struct,sys; f=open(sys.argv[1],'r+b'); f.seek(4); f.write(struct.pack('<Q',int(sys.argv[2])))" "$B" $(( nb + ne ))
  [ "$(stat -c %s "$B")" = $(( orig + $(stat -c %s "$E") - 12 )) ] || return 1
  echo "$(( nb + ne )) positions, $(stat -c %s "$B") bytes (incl. $(basename "$seg"))" > "$RUN_DIR/rescore/train.ready"
  touch "$seg/merged"; rm -f "$E"
  echo "$(date '+%F %T') merged $(basename "$seg"): +$ne positions, now $(( nb + ne ))" >> "$RUN_DIR/pipeline.log"
}
status_pid() { [ -s "$RUN_DIR/.status_pid" ] && cat "$RUN_DIR/.status_pid"; }
start_status() {
  local p; p=$(status_pid)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && return
  setsid nohup python3 "$S/run_status.py" --run-dir "$RUN_DIR" >> "$RUN_DIR/status.out" 2>&1 < /dev/null &
  echo $! > "$RUN_DIR/.status_pid"
}
STARTUP_WIN="$(cmd.exe /c 'echo %APPDATA%' 2>/dev/null | tr -d '\r')\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\darkhorse_datagen.vbs"

case "$CMD" in
start)
  exec 8> "$RUN_DIR/.start.lock"; flock 8     # cron and the Windows autostart may fire together
  if finished; then echo "run in $RUN_DIR is finished ($(cat "$RUN_DIR/rescore/train.ready"))"; exit 0; fi
  if alive; then echo "already running (pgid $(pgid)); use pause/resume/stop"; start_status; exit 0; fi
  rm -f "$RUN_DIR/.paused"
  if [ ! -s "$RUN_DIR/.args" ]; then
    echo "${2:-2000000000} ${3:-$(nproc)}" > "$RUN_DIR/.args"
  fi
  read -r TOTAL WORKERS < "$RUN_DIR/.args"
  if [ ! -x "$BIN/MainEngine" ] || [ ! -x "$BIN/movegen" ]; then
    # freeze the binaries for the whole run (all positions labelled by the same engine)
    mkdir -p "$BIN"
    cp "$R/cmake-build-debug/MainEngine" "$BIN/MainEngine" &&
      cp "$R/Training/Rust/target/release/movegen" "$BIN/movegen" || { echo "binaries missing"; exit 1; }
    git -C "$R" rev-parse HEAD > "$BIN/commit"
    "$BIN/MainEngine" --bench | tail -1 > "$BIN/bench"
  fi
  setsid nohup nice -n 19 "$0" _chain >> "$RUN_DIR/start.out" 2>&1 < /dev/null &
  echo $! > "$PGF"
  sleep 2
  if ! alive; then echo "pipeline did not start, see $RUN_DIR/start.out"; exit 1; fi
  if [ "${ADAPTIVE:-1}" = 1 ]; then
    setsid nohup python3 "$S/adaptive_throttle.py" --pgid "$(pgid)" --run-dir "$RUN_DIR" \
      --reserve "${RESERVE:-1}" >> "$RUN_DIR/throttle.out" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/.throttle_pid"
  fi
  start_status
  echo "started: $TOTAL positions (value-only), $WORKERS workers, run dir $RUN_DIR$([ "${ADAPTIVE:-1}" = 1 ] && echo ", adaptive")"
  echo "status: $RUN_DIR/STATUS.txt (updated once a minute)"
  ;;
_chain)   # internal: the main run, then every queued extension segment (+ merge)
  read -r TOTAL WORKERS < "$RUN_DIR/.args"
  export VALUE_ONLY=1 MOVEGEN="$BIN/movegen" ENGINE_DIR="$BIN"
  export OPENING_STATE="$RUN_DIR/.next_opening"   # one book position for the run and its extensions
  if ! base_done; then
    echo "$RUN_DIR" > "$RUN_DIR/.active"
    "$S/cloud_pipeline.sh" "$RUN_DIR" "$WORKERS" "$TOTAL" || exit 1
  fi
  while seg=$(next_segment) && [ -n "$seg" ]; do
    read -r ADD W2 < "$seg/.args"
    if [ ! -f "$seg/rescore/train.ready" ]; then
      echo "$seg" > "$RUN_DIR/.active"
      VAL_POS=0 EXCLUDE_POL1="$RUN_DIR/rescore/train.children.samples $RUN_DIR/rescore/val.children.samples" \
        "$S/cloud_pipeline.sh" "$seg" "$W2" "$ADD" || exit 1
    fi
    merge_segment "$seg" || { echo "merge of $seg failed"; exit 1; }
  done
  echo "$RUN_DIR" > "$RUN_DIR/.active"
  ;;
extend)
  ADD=${2:?usage: local_run.sh extend ADD}
  read -r TOTAL WORKERS < "$RUN_DIR/.args"
  if ! base_done && [ ! -s "$RUN_DIR/gen.samples" ] && [ ! -e "$RUN_DIR/rescore/split.done" ] &&
     [ ! -e "$RUN_DIR/rescore/chunks.done" ]; then
    # still generating: raise the target of the main run (restart picks it up)
    was=0; alive && was=1
    [ $was = 1 ] && "$0" stop > /dev/null
    TOTAL=$(( TOTAL + ADD )); echo "$TOTAL $WORKERS" > "$RUN_DIR/.args"
    TF=$RUN_DIR/gen.samples.pieces/pieces.target
    if [ -s "$TF" ]; then
      P=$(grep -ao "pieces of [0-9]*" "$RUN_DIR/pipeline.log" | tail -1 | grep -o "[0-9]*$")
      NEWN=$(( (TOTAL + P - 1) / P ))
      [ "$NEWN" -gt "$(cat "$TF")" ] && echo "$NEWN" > "$TF"
    fi
    echo "target raised to $TOTAL positions (still in the generation phase)"
    [ $was = 1 ] && "$0" start
  else
    k=2; while [ -e "$RUN_DIR/seg$k" ]; do k=$(( k + 1 )); done
    mkdir -p "$RUN_DIR/seg$k"; echo "$ADD $WORKERS" > "$RUN_DIR/seg$k/.args"
    echo "extension seg$k queued: +$ADD new positions after the current work"
    alive || "$0" start
  fi
  ;;
daemon)
  "$0" start; exec 8>&-
  # stay in the foreground while the run exists (paused counts as active)
  while alive || { [ -e "$RUN_DIR/.paused" ] && alive; }; do sleep 60; done
  ;;
autostart-install)
  SELF=$(readlink -f "$0")
  ENTRY="@reboot RUN_DIR=$RUN_DIR $SELF start >> $RUN_DIR/autostart.out 2>&1"
  ( crontab -l 2>/dev/null | grep -v "local_run.sh"; echo "$ENTRY" ) | crontab -
  DISTRO=${WSL_DISTRO_NAME:-Ubuntu}
  WINF=$(wslpath "$STARTUP_WIN")
  printf 'Set sh = CreateObject("WScript.Shell")\r\nsh.Run "wsl.exe -d %s -u %s -- bash -lc ""RUN_DIR=%s %s daemon""", 0, False\r\n' \
    "$DISTRO" "$(id -un)" "$RUN_DIR" "$SELF" > "$WINF"
  echo "autostart installed:"
  echo "  WSL start:      crontab @reboot entry (crontab -l)"
  echo "  Windows logon:  $STARTUP_WIN"
  ;;
autostart-remove)
  crontab -l 2>/dev/null | grep -v "local_run.sh" | crontab -
  rm -f "$(wslpath "$STARTUP_WIN")"
  echo "autostart removed (crontab entry and Windows Startup script)"
  ;;
pause)
  alive || { echo "not running"; exit 0; }
  touch "$RUN_DIR/.paused"
  kill -STOP -- "-$(pgid)"
  echo "paused (CPU free); continue with: local_run.sh resume"
  ;;
resume)
  alive || { echo "not running; use start"; exit 0; }
  kill -CONT -- "-$(pgid)"
  rm -f "$RUN_DIR/.paused"
  echo "resumed$([ -n "$(throttle_pid)" ] && echo " (the throttle re-applies its engine count within ~15 s)")"
  ;;
stop)
  t=$(throttle_pid); [ -n "$t" ] && kill "$t" 2>/dev/null
  t=$(status_pid); [ -n "$t" ] && kill "$t" 2>/dev/null
  if alive; then
    g=$(pgid); kill -CONT -- "-$g" 2>/dev/null; kill -TERM -- "-$g" 2>/dev/null
    sleep 3; kill -KILL -- "-$g" 2>/dev/null
  fi
  rm -f "$RUN_DIR/.paused" "$RUN_DIR/.throttle_pid" "$RUN_DIR/.status_pid"
  python3 "$S/run_status.py" --run-dir "$RUN_DIR" --once 2>/dev/null
  echo "stopped; start again later with: local_run.sh start"
  ;;
status)
  [ -s "$RUN_DIR/STATUS.txt" ] && { python3 "$S/run_status.py" --run-dir "$RUN_DIR" --once 2>/dev/null; cat "$RUN_DIR/STATUS.txt"; exit 0; }
  if alive; then
    if [ -e "$RUN_DIR/.paused" ]; then echo "state: PAUSED"; else echo "state: running"; fi
  else
    echo "state: not running"
  fi
  [ -s "$RUN_DIR/.args" ] && { read -r T W < "$RUN_DIR/.args"; echo "target: $T positions, $W workers"; }
  [ -s "$RUN_DIR/throttle.state" ] && echo "throttle: $(cat "$RUN_DIR/throttle.state")"
  [ -s "$RUN_DIR/pipeline.log" ] && { echo "last steps:"; grep -a "piece\|chunk\|dedup\|ready\|pipeline\|ERROR" "$RUN_DIR/pipeline.log" | tail -4; }
  ;;
*)
  sed -n '2,20p' "$0"; exit 1;;
esac
