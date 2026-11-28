#!/usr/bin/env python3
"""Adapts a running data pipeline (generation / rescoring) to what else the PC is doing.

    adaptive_throttle.py --pgid PGID --run-dir DIR [--interval 15] [--reserve 1]

Runs under WSL next to local_run.sh. Every interval it measures
  - the Windows CPU load (all programs incl. WSL; PowerShell CIM, no admin rights), and
  - the WSL CPU load (/proc/stat),
so "other load" = Windows load - WSL load is what Windows programs (browser, video,
games, ...) use. It then keeps as many engine processes (MainEngine in the pipeline's
process group) running as there are free logical CPUs, minus a small reserve while
other programs are active, and freezes the rest with SIGSTOP (resumed with SIGCONT).

Freezing an engine is safe because all searches use node budgets, not time limits:
an engine that is paused in the middle of a search produces exactly the same result
later. Nothing is lost; the frozen workers just wait.

While DIR/.paused exists (local_run.sh pause) the throttle does nothing. It exits when
the process group is gone and resumes every engine it froze when it is terminated.
Log: DIR/throttle.log (one line per change), state: DIR/throttle.state.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time

PS_LOOP = ("while($true){ (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor "
           "-Filter \"Name='_Total'\").PercentProcessorTime; Start-Sleep -Seconds 5 }")


def cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v), idle


def engines(pgid):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                st = f.read()
            comm = st[st.index("(") + 1:st.rindex(")")]
            fields = st[st.rindex(")") + 2:].split()
            state, pg = fields[0], int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        # zombies (engines that already finished) do not count
        if pg == pgid and comm == "MainEngine" and state != "Z":
            out.append((int(d), state))
    return sorted(out)


def match_running():
    """the match tool (SPRT, SPSA, matches) runs in WSL: the data run must not disturb it"""
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/comm") as f:
                    if f.read().strip() == "Train":
                        return True
            except OSError:
                pass
    return False


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgid", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--reserve", type=int, default=1,
                    help="extra CPUs left free while other programs use the CPU")
    ap.add_argument("--min-active", type=int, default=1,
                    help="engines that always keep running (0 = may pause completely)")
    args = ap.parse_args()
    ncpu = os.cpu_count() or 1
    log = open(os.path.join(args.run_dir, "throttle.log"), "a", buffering=1)
    state_file = os.path.join(args.run_dir, "throttle.state")
    paused_flag = os.path.join(args.run_dir, ".paused")

    win = []                               # Windows load samples (percent)
    # full path: under cron/systemd the Windows directories are not in PATH
    pse = next((p for p in ("/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe",
                            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
                if os.path.exists(p)), "powershell.exe")
    try:
        ps = subprocess.Popen([pse, "-NoProfile", "-Command", PS_LOOP],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError as e:
        ps = None
        log.write(f"{time.strftime('%F %T')} cannot read the Windows CPU load ({e}); "
                  "all engines keep running (lowest priority only)\n")

    def reader():
        for line in ps.stdout:
            line = line.strip()
            if line.isdigit():
                win.append(float(line))
    if ps is not None:
        threading.Thread(target=reader, daemon=True).start()

    frozen = set()

    def release_all(*_):
        for pid in list(frozen):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        frozen.clear()

    def on_term(*_):
        release_all()
        if ps is not None:
            ps.kill()
        log.write(f"{time.strftime('%F %T')} throttle stopped, all engines running\n")
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    other_ema, last_target = None, None
    last_match = 0.0      # SPSA starts the match tool once per iteration: hold for 5 min
    t0, i0 = cpu_times()
    log.write(f"{time.strftime('%F %T')} throttle start: {ncpu} CPUs, interval {args.interval:g} s, "
              f"reserve {args.reserve}\n")
    while group_alive(args.pgid):
        time.sleep(args.interval)
        t1, i1 = cpu_times()
        wsl = 100.0 * (1 - (i1 - i0) / max(1, t1 - t0))
        t0, i0 = t1, i1
        if os.path.exists(paused_flag):
            continue                       # local_run.sh pause: everything is stopped
        samples, win[:] = win[:], []
        if match_running():
            last_match = time.time()
        match_hold = time.time() - last_match < 300
        if not samples and not match_hold:
            continue                       # no Windows reading yet
        other = max(0.0, sum(samples) / len(samples) - wsl) if samples else 0.0
        if os.environ.get("THROTTLE_FAKE_OTHER"):          # tests only
            other = float(os.environ["THROTTLE_FAKE_OTHER"])
        other_ema = other if other_ema is None else 0.5 * other_ema + 0.5 * other
        procs = engines(args.pgid)
        frozen &= {p for p, _ in procs}
        n = len(procs)
        if n == 0:
            continue                       # dedup / merge phase: no engines
        other_cpus = other_ema / 100.0 * ncpu
        busy = other_cpus + (args.reserve if other_ema > 5.0 else 0)
        target = max(min(args.min_active, n), min(n, int(round(ncpu - busy))))
        if match_hold:
            target = 0                     # SPRT / SPSA / match running: freeze everything
        running = [p for p, s in procs if p not in frozen]
        if len(running) > target:
            for p in running[target - len(running):] if target < len(running) else []:
                try:
                    os.kill(p, signal.SIGSTOP)
                    frozen.add(p)
                except ProcessLookupError:
                    pass
        elif len(running) < target:
            for p in sorted(frozen)[: target - len(running)]:
                try:
                    os.kill(p, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                frozen.discard(p)
        with open(state_file, "w") as f:
            wv = sum(samples) / len(samples) if samples else 0
            f.write(f"{time.strftime('%F %T')} windows {wv:.0f} % wsl {wsl:.0f} % other {other_ema:.0f} % "
                    f"-> {target}/{n} engines running{' (match tool running)' if match_hold else ''}\n")
        if target != last_target:
            why = "match tool (SPRT/SPSA) running" if match_hold else f"other programs {other_ema:.0f} % of the CPU"
            log.write(f"{time.strftime('%F %T')} {why} -> {target} of {n} engines running\n")
            last_target = target
    release_all()
    if ps is not None:
        ps.kill()
    log.write(f"{time.strftime('%F %T')} pipeline gone, throttle exits\n")


if __name__ == "__main__":
    main()
