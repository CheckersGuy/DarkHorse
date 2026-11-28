#!/usr/bin/env python3
"""Writes a human-readable status file for a local data run (local_run.sh), once a minute.

    run_status.py --run-dir DIR [--interval 60] [--once]

DIR/STATUS.txt (also readable from Windows, e.g. C:\\TrainingData\\vrun\\STATUS.txt):
state, phase, positions generated / rescored so far, speed over the last hour and an
estimated finish time. Exits when the pipeline is gone (after writing a final status).
"""
import argparse
import glob
import os
import re
import struct
import time


def fmt(n):
    return f"{int(n):,}"


def pol1_count(path):
    try:
        with open(path, "rb") as f:
            magic, n = struct.unpack("<IQ", f.read(12))
        return n if magic == 0x504F4C31 else 0
    except (OSError, struct.error):
        return 0


def size(path, rec):
    try:
        return os.path.getsize(path) // rec
    except OSError:
        return 0


def when(t):
    """absolute date, e.g. 'Mon 01.12. 14:30'"""
    return time.strftime("%a %d.%m. %H:%M", time.localtime(t))


def alive(run_dir):
    try:
        g = int(open(os.path.join(run_dir, ".pgid")).read())
        os.killpg(g, 0)
        return True
    except (OSError, ValueError, ProcessLookupError):
        return False


def snapshot(d):
    s = {}
    try:
        s["total"], s["workers"] = (int(x) for x in open(os.path.join(d, ".args")).read().split())
    except (OSError, ValueError):
        s["total"], s["workers"] = 0, 0
    pieces = os.path.join(d, "gen.samples.pieces")
    piece_size = 50_000_000
    m = re.search(r"pieces of (\d+) positions", open(os.path.join(d, "pipeline.log"), errors="ignore").read()
                  if os.path.exists(os.path.join(d, "pipeline.log")) else "")
    if m:
        piece_size = int(m.group(1))
    finished = sorted(glob.glob(os.path.join(pieces, "piece*.samples")))
    used = glob.glob(os.path.join(pieces, "piece*.used"))
    current = glob.glob(os.path.join(pieces, "piece*.samples.tmp"))
    try:
        target_pieces = int(open(os.path.join(pieces, "pieces.target")).read())
    except (OSError, ValueError):
        target_pieces = 0
    s["pieces_done"] = len(finished) + len(used)
    s["piece_size"] = piece_size
    s["pieces_target"] = target_pieces
    s["generated"] = sum(size(f, 18) for f in finished) + len(used) * piece_size + \
        sum(size(f, 18) for f in current)
    s["unique"] = size(os.path.join(d, "gen.samples.acc"), 18)
    s["gen_file"] = size(os.path.join(d, "gen.samples"), 18)
    r = os.path.join(d, "rescore")
    s["rescoring"] = os.path.exists(os.path.join(r, "chunks.done")) or os.path.exists(os.path.join(r, "split.done"))
    try:
        s["chunks"] = int(open(os.path.join(r, "chunks.done")).read())
    except (OSError, ValueError):
        s["chunks"] = 0
    done = sorted(glob.glob(os.path.join(r, "data.c*.children.samples")))
    s["chunks_done"] = len(done)
    rescored = sum(pol1_count(f) for f in done)
    rescored += pol1_count(os.path.join(r, "val.children.samples"))
    # chunk in progress: partition files of the value-only rescoring (19 bytes per sample)
    rescored += sum(size(f, 19) for f in glob.glob(os.path.join(r, "*.tmp[0-9]*")))
    s["rescored"] = rescored
    s["train_ready"] = os.path.exists(os.path.join(r, "train.ready"))
    s["train_count"] = pol1_count(os.path.join(r, "train.children.samples")) if s["train_ready"] else 0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    d = a.run_dir
    hist = []
    while True:
        now = time.time()
        try:
            active = open(os.path.join(d, ".active")).read().strip() or d
        except OSError:
            active = d
        segs = sorted(glob.glob(os.path.join(d, "seg*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
        s = snapshot(active)
        hist.append((now, s["generated"], s["rescored"]))
        hist = [h for h in hist if now - h[0] <= 3600]
        run = alive(d)
        paused = os.path.exists(os.path.join(d, ".paused"))
        state = "PAUSED" if run and paused else "running" if run else "not running"
        base_ready = os.path.exists(os.path.join(d, "rescore", "train.ready"))
        pending = [p for p in segs if not os.path.exists(os.path.join(p, "merged"))]
        if base_ready and not pending:
            state = "FINISHED"
        thr = ""
        try:
            t = open(os.path.join(d, "throttle.state")).read().strip()
            m = re.search(r"other (\d+) % -> (\d+)/(\d+)", t)
            if m and run and not paused:
                thr = f" (adaptive: {m.group(2)} of {m.group(3)} engines active, other programs {m.group(1)} % CPU)"
        except OSError:
            pass
        total = s["total"] or 1
        rate_g = rate_r = 0.0
        if len(hist) >= 2 and hist[-1][0] > hist[0][0]:
            dt = (hist[-1][0] - hist[0][0]) / 60
            rate_g = (hist[-1][1] - hist[0][1]) / dt
            rate_r = (hist[-1][2] - hist[0][2]) / dt
        lines = [f"DarkHorse value-data run      (updated {time.strftime('%F %T')})",
                 f"state:      {state}{thr}"]
        base_n = pol1_count(os.path.join(d, "rescore", "train.children.samples")) if base_ready else 0
        if segs:
            plan = []
            for p in segs:
                try:
                    add = int(open(os.path.join(p, ".args")).read().split()[0])
                except (OSError, ValueError, IndexError):
                    add = 0
                tag = "merged" if os.path.exists(os.path.join(p, "merged")) else \
                      "running" if os.path.abspath(p) == os.path.abspath(active) else "queued"
                plan.append(f"{os.path.basename(p)} +{fmt(add)} ({tag})")
            lines.append("extensions: " + ", ".join(plan))
        if active != d:
            lines.append(f"now:        extension {os.path.basename(active)}: "
                         f"{fmt(s['total'])} new positions (excluding the existing {fmt(base_n)})")
        lines.append(f"target:     {fmt(s['total'])} positions (value-only, {s['workers']} workers)")
        if state == "FINISHED":
            lines.append(f"result:     {fmt(base_n)} positions in rescore/train.children.samples")
        elif s["rescoring"]:
            lines.append(f"phase:      2/2 rescoring, chunk {s['chunks_done'] + 1} of {s['chunks']}")
            lines.append(f"rescored:   {fmt(s['rescored'])} of {fmt(total)} ({100 * s['rescored'] / total:.1f} %)")
            if rate_r > 0:
                left = (total - s["rescored"]) / rate_r
                lines.append(f"speed:      {rate_r / 1e3:,.0f}k positions/min (last hour)")
                lines.append(f"estimate:   rescoring done in ~{left / 60:.1f} h of running time")
                lines.append(f"finished:   {when(now + left * 60)}   (if it keeps running at this speed)")
        else:
            lines.append(f"phase:      1/2 generation, piece {min(s['pieces_done'] + 1, max(1, s['pieces_target']))} "
                         f"of {s['pieces_target']}")
            lines.append(f"generated:  {fmt(s['generated'])} positions so far"
                         + (f" ({fmt(s['unique'])} unique after the last dedup)" if s["unique"] else ""))
            if rate_g > 0:
                remaining = max(0, s["pieces_target"] * s["piece_size"] - s["generated"])
                lines.append(f"speed:      {rate_g / 1e6:,.2f}M positions/min (last hour)")
                if remaining:
                    gen_min = remaining / rate_g
                    # value rescoring at the measured local full speed (~850k/min), scaled by
                    # how fast the generation currently runs compared with full speed
                    # (~2.3M/min on 16 threads), plus ~1 h for dedup / merge / shuffle
                    slow = min(1.0, rate_g / 2.3e6 * 16 / max(1, s["workers"]))
                    res_min = total / (850_000 * max(0.05, slow)) + 60
                    lines.append(f"estimate:   generation done in ~{gen_min / 60:.1f} h, then dedup and value "
                                 f"rescoring ~{res_min / 60:.0f} h (running time)")
                    lines.append(f"generation: {when(now + gen_min * 60)}")
                    lines.append(f"finished:   {when(now + (gen_min + res_min) * 60)}   (if it keeps running at this speed)")
            lines.append(f"rescored:   0 (starts after the generation)")
        lines.append(f"log:        {os.path.join(active, 'pipeline.log')}")
        tmp = os.path.join(d, "STATUS.txt.tmp")
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, os.path.join(d, "STATUS.txt"))
        if a.once or not run:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
