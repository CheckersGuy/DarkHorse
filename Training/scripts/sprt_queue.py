#!/usr/bin/env python3
"""Runs a queue of SPRT tests one after another (two-stage scheme, no input needed).

    sprt_queue.py QUEUE.json ENGINE_NAME

QUEUE.json: {"tests": [{"idea": "B", "book": "...optional...",
                        "variants": [{"name": "B1", "opts": {"WIN_MODE": 1}}, ...]}, ...]}

Per idea the variants are tried in order until one passes:
  stage 1: --sprt 0 3; H1 -> accepted. H0 with Elo >= +1.0 -> stage 2 --sprt 0 2, its
  result decides. Otherwise the next variant of the idea is tried.
Accepted options are passed to BOTH engines in all later tests (--opts1/--opts2), so
every test measures its change on top of everything accepted before.
Both engines are the same binary Training/Engines/ENGINE_NAME (20 ms, 14 games).
Results: Training/sprt_queue_results.csv, logs Training/sq_<variant>_<stage>.log.
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = "/home/leaguerobin/DarkHorse"
RUNDIR = os.path.join(ROOT, "cmake-build-debug")
TRAIN = os.path.join(ROOT, "Training")
DEFAULT_BOOK = "../Training/Positions/unbalanced_70_150.book"
RESULTS = os.path.join(TRAIN, "sprt_queue_results.csv")


def opts_args(flag, opts):
    if not opts:
        return []
    return [flag] + [f"{k}={v}" for k, v in opts.items()]


def run_sprt(engine, label, test_opts, base_opts, bounds, book):
    log = os.path.join(TRAIN, f"sq_{label}.log")
    cmd = ["./Train", "--match", "--engines", engine, engine, "--time", "20", "20",
           "--threads", "14", "--book", book, "--sprt", str(bounds[0]), str(bounds[1])]
    cmd += opts_args("--opts1", {**base_opts, **test_opts}) + opts_args("--opts2", base_opts)
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n"); f.flush()
        subprocess.run(cmd, cwd=RUNDIR, stdout=f, stderr=subprocess.STDOUT)
    text = open(log).read()
    games = re.findall(r"^games\s+(\d+).*?Elo ([+-]?[\d.]+) \+- ([\d.]+)", text, re.M)
    n, elo, err = (int(games[-1][0]), float(games[-1][1]), float(games[-1][2])) if games else (0, 0.0, 0.0)
    verdict = "H1" if "H1 accepted" in text else "H0" if "H0 accepted" in text else "none"
    return verdict, elo, err, n


def main():
    queue = json.load(open(sys.argv[1]))["tests"]
    engine = sys.argv[2]
    accepted = {}
    if not os.path.exists(RESULTS):
        open(RESULTS, "w").write("time,idea,variant,stage,bounds,verdict,elo,err,games,opts,base\n")
    for test in queue:
        book = test.get("book", DEFAULT_BOOK)
        for var in test["variants"]:
            stages = [("s1", (0, 3))]
            verdict = None
            for stage, bounds in stages:
                label = f"{var['name']}_{stage}"
                print(f"{time.strftime('%F %T')} start {label} {var['opts']} base {accepted}", flush=True)
                verdict, elo, err, n = run_sprt(engine, label, var["opts"], accepted, bounds, book)
                print(f"{time.strftime('%F %T')} end {label}: {verdict} {elo:+.1f} +- {err:.1f} ({n} games)", flush=True)
                with open(RESULTS, "a") as f:
                    f.write(f"{time.strftime('%F %T')},{test['idea']},{var['name']},{stage},"
                            f"{bounds[0]}..{bounds[1]},{verdict},{elo:+.1f},{err:.1f},{n},"
                            f"\"{json.dumps(var['opts'])}\",\"{json.dumps(accepted)}\"\n")
                if stage == "s1" and verdict == "H0" and elo >= 1.0:
                    stages.append(("s2", (0, 2)))
            if verdict == "H1":
                accepted.update(var["opts"])
                print(f"{time.strftime('%F %T')} accepted {var['name']}; base now {accepted}", flush=True)
                break
    print(f"{time.strftime('%F %T')} queue finished; accepted {accepted}", flush=True)
    json.dump(accepted, open(os.path.join(TRAIN, "sprt_queue_accepted.json"), "w"))


if __name__ == "__main__":
    main()
