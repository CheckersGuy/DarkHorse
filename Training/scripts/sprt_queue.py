#!/usr/bin/env python3
"""Runs search-feature SPRTs one after another with the project's procedure.

    sprt_queue.py ENGINE JOBS_FILE SUMMARY_FILE [--wait-log LOG --wait-opt KEY=VAL]

JOBS_FILE: one job per line "name type KEY=VAL [KEY=VAL ...]", type = sprt (stage 1
[0, 3]; H0 with Elo >= +1.0 -> stage 2 [0, 2], which decides) or nr (non-regression
[-3, 1]). Both sides use the same engine (Training/Engines/ENGINE); accepted options
(H1) are passed to both sides of all later jobs. --wait-log: first wait for an SPRT
that is already running and accept --wait-opt if its log ends with H1.
Logs: Training/<name>_sprt_20ms.log (stage 2: _sprt2_), summary in SUMMARY_FILE.
"""
import argparse
import os
import re
import subprocess
import time

ROOT = "/home/leaguerobin/DarkHorse"
BUILD = os.path.join(ROOT, "cmake-build-debug")


def train_running():
    return subprocess.run(["pgrep", "-x", "Train"], capture_output=True).returncode == 0


def result(log):
    text = open(log, errors="ignore").read()
    dec = "H1" if "H1 accepted" in text else "H0" if "H0 accepted" in text else "none"
    m = re.findall(r"Final: games (\d+).*?Elo ([+-][\d.]+) \+- ([\d.]+)", text)
    games, elo, err = (int(m[-1][0]), float(m[-1][1]), float(m[-1][2])) if m else (0, 0.0, 0.0)
    return dec, games, elo, err


def run(engine, log, bounds, opts1, opts2):
    cmd = ["./Train", "--match", "--engines", engine, engine, "--time", "20", "20",
           "--threads", "14", "--book", "../Training/Positions/unbalanced_70_150.book",
           "--sprt", str(bounds[0]), str(bounds[1])]
    if opts1:
        cmd += ["--opts1"] + opts1
    if opts2:
        cmd += ["--opts2"] + opts2
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=BUILD, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return result(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("jobs")
    ap.add_argument("summary")
    ap.add_argument("--wait-log")
    ap.add_argument("--wait-opt")
    ap.add_argument("--wait-kind", default="nr", help="sprt: stage-2 rule applies to the waited job")
    ap.add_argument("--accepted", nargs="*", default=[], help="options accepted earlier (both sides)")
    a = ap.parse_args()
    summary = open(a.summary, "a", buffering=1)
    accepted = list(a.accepted)
    if a.wait_log:
        while train_running():
            time.sleep(60)
        dec, games, elo, err = result(a.wait_log)
        text = f"{os.path.basename(a.wait_log)}: {dec} Elo {elo:+.1f} +- {err:.1f} ({games} games)"
        if a.wait_kind == "sprt" and dec == "H0" and elo >= 1.0 and a.wait_opt:
            log2 = a.wait_log.replace("_sprt_", "_sprt2_")
            dec, games, elo, err = run(a.engine, log2, (0, 2), accepted + [a.wait_opt], accepted)
            text += f"; stage 2 [0,2]: {dec} Elo {elo:+.1f} +- {err:.1f} ({games} games)"
        if dec == "H1" and a.wait_opt:
            accepted.append(a.wait_opt)
        summary.write(text + (" -> ACCEPTED" if dec == "H1" else " -> rejected") + "\n")
    for line in open(a.jobs):
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, kind, *opts = line.split()
        while train_running():  # never two matches at once
            time.sleep(60)
        log = os.path.join(ROOT, "Training", f"{name}_sprt_20ms.log")
        bounds = (-3, 1) if kind == "nr" else (0, 3)
        dec, games, elo, err = run(a.engine, log, bounds, accepted + opts, accepted)
        text = f"{name} {' '.join(opts)} [{bounds[0]},{bounds[1]}]: {dec} Elo {elo:+.1f} +- {err:.1f} ({games} games)"
        if kind == "sprt" and dec == "H0" and elo >= 1.0:
            log2 = os.path.join(ROOT, "Training", f"{name}_sprt2_20ms.log")
            dec, games, elo, err = run(a.engine, log2, (0, 2), accepted + opts, accepted)
            text += f"; stage 2 [0,2]: {dec} Elo {elo:+.1f} +- {err:.1f} ({games} games)"
        if dec == "H1":
            accepted += opts
            text += " -> ACCEPTED"
        else:
            text += " -> rejected"
        summary.write(text + "\n")
    summary.write(f"QUEUE DONE, accepted: {' '.join(accepted) or '(none)'}\n")


if __name__ == "__main__":
    main()
