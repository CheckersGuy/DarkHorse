#!/usr/bin/env python3
"""SPSA tuning of the engine's search parameters (fishtest-style).

Every iteration plays a small match between theta+ = theta + c_k*delta and
theta- = theta - c_k*delta (delta in {-1,+1}^n) with the match tool and moves
theta towards the side that scored better:
    theta_i += a_k / c_k * (wins - losses of theta+) * delta_i

Parameters (name, default, min, max, c_end) come from the engine itself:
    Training/Engines/<engine> --list_params

Run from anywhere, e.g.
    python3 Training/spsa.py --engine params-dev --iterations 2000
    python3 Training/spsa.py --resume            # continue the last run

State (theta, iteration) is saved after every iteration to --state, every
iteration is appended to --log (CSV).

NOTE on other machines: --time (ms per move) is calibrated for the local
machine. Before tuning on a different machine (e.g. a cloud server), run
Training/bench_parallel.sh with the number of parallel games on both machines
and scale --time by nps_local / nps_other, so every move gets roughly the same
number of nodes as here.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "cmake-build-debug")   # the match tool runs here
ENGINE_DIR = os.path.join(ROOT, "Training", "Engines")
# tuned separately per net (spsa.py --only NAME), never in a normal SPSA run
SCALE_PARAMS = {"VALUE_SCALE_X100", "POLICY_SCALE_X100"}

FINAL_RE = re.compile(r"^Final: games (\d+)\s+W-L-D (\d+)-(\d+)-(\d+)")


def read_params(engine):
    out = subprocess.run([os.path.join(ENGINE_DIR, engine), "--list_params"],
                         capture_output=True, text=True, check=True).stdout
    params = []
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 5:
            continue
        name, value, lo, hi, c_end = parts
        params.append(dict(name=name, default=int(value), min=int(lo),
                           max=int(hi), c_end=float(c_end)))
    if not params:
        sys.exit(f"{engine} --list_params returned no parameters")
    return params


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def opts_string(params, values):
    return [f"{p['name']}={int(round(v))}" for p, v in zip(params, values)]


def play(args, plus, minus, seed):
    """Plays one mini match theta+ (engine 1) vs theta- (engine 2)."""
    # the match tool finishes about --threads games beyond --num_games
    num_games = max(1, args.games - args.threads)
    cmd = ["./Train", "--match", "--engines", args.engine, args.engine,
           "--time", str(args.time), str(args.time),
           "--threads", str(args.threads), "--num_games", str(num_games),
           "--book", args.book, "--book_seed", str(seed),
           "--opts1", *plus, *args.extra, "--opts2", *minus, *args.extra]
    # own process group: on abort the match tool *and* its engines are killed
    # (the engines do not notice when the match tool dies)
    proc = subprocess.Popen(cmd, cwd=BUILD_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        stdout, stderr = proc.communicate()
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise

    class _Out:
        pass
    out = _Out()
    out.stdout, out.stderr = stdout, stderr
    for line in out.stdout.replace("\r", "\n").splitlines():
        m = FINAL_RE.match(line.strip())
        if m:
            return int(m.group(2)), int(m.group(3)), int(m.group(4))
    raise RuntimeError("no result from match tool:\n" + out.stdout[-2000:]
                       + out.stderr[-2000:])


def _terminate(signum, frame):
    raise SystemExit(f"stopped by signal {signum}")


def main():
    signal.signal(signal.SIGTERM, _terminate)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="params-dev",
                    help="engine binary in Training/Engines")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--games", type=int, default=28,
                    help="games per iteration (both colours of each opening)")
    ap.add_argument("--time", type=int, default=5, help="ms per move")
    ap.add_argument("--threads", type=int, default=14,
                    help="games played in parallel")
    ap.add_argument("--book", default="../Training/Positions/unbalanced_70_150.book",
                    help="opening book, relative to cmake-build-debug")
    ap.add_argument("--r-end", type=float, default=0.002)
    ap.add_argument("--alpha", type=float, default=0.602)
    ap.add_argument("--gamma", type=float, default=0.101)
    ap.add_argument("--state", default=os.path.join(ROOT, "Training", "spsa_state.json"))
    ap.add_argument("--log", default=os.path.join(ROOT, "Training", "spsa_log.csv"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--start-from", default=None,
                    help="state JSON of an earlier run: start from its theta "
                         "(parameters missing there start at the engine default)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="fixed engine options for both sides, e.g. "
                         "policy_file=/abs/net.quant (key=value)")
    ap.add_argument("--only", default=None,
                    help="comma-separated parameter names to tune; all other "
                         "parameters stay fixed at their start value")
    args = ap.parse_args()

    if args.resume:
        with open(args.state) as f:
            st = json.load(f)
        for key in ("engine", "iterations", "games", "time", "threads", "book",
                    "r_end", "alpha", "gamma", "seed", "extra"):
            setattr(args, key, st["args"][key])
        params = st["params"]
        theta = st["theta"]
        k0 = st["k"]
        totals = st["totals"]
        print(f"resuming at iteration {k0 + 1}/{args.iterations}")
    else:
        params = read_params(args.engine)
        theta = [float(p["default"]) for p in params]
        if args.start_from:
            with open(args.start_from) as f:
                prev = json.load(f)
            prev_theta = {q["name"]: t for q, t in zip(prev["params"], prev["theta"])}
            theta = [prev_theta.get(p["name"], t) for p, t in zip(params, theta)]
        only = set(args.only.split(",")) if args.only else None
        if only:
            unknown = only - {p["name"] for p in params}
            if unknown:
                sys.exit(f"unknown parameters in --only: {sorted(unknown)}")
        # c_end 0 (engine) or not selected: listed but kept fixed
        for p in params:
            # net-specific scales are tuned only when named in --only (one 1-parameter
            # SPSA per new net); normal runs keep them fixed
            p["tune"] = p["c_end"] > 0 and (p["name"] in only if only else p["name"] not in SCALE_PARAMS)
        print("tuning: " + ", ".join(p["name"] for p in params if p["tune"]))
        k0 = 0
        totals = dict(w=0, l=0, d=0)
        with open(args.log, "w", newline="") as f:
            csv.writer(f).writerow(["k", "w", "l", "d", "c_k_scale", "a_k_scale"]
                                   + [p["name"] for p in params])

    n_iter = args.iterations
    big_a = 0.1 * n_iter
    rng = random.Random(args.seed)
    for _ in range(k0):          # keep the delta sequence identical on resume
        for _ in params:
            rng.random()

    t0 = time.time()
    for k in range(k0 + 1, n_iter + 1):
        c_scale = (n_iter ** args.gamma) / (k ** args.gamma)
        a_scale = ((big_a + n_iter) ** args.alpha) / ((big_a + k) ** args.alpha)
        delta, plus, minus, steps = [], [], [], []
        for p, th in zip(params, theta):
            d = 1 if rng.random() < 0.5 else -1
            if not p.get("tune", True):
                d = 0          # fixed parameter: same value on both sides
            c_k = p["c_end"] * c_scale
            delta.append(d)
            steps.append(c_k)
            plus.append(clamp(th + c_k * d, p["min"], p["max"]))
            minus.append(clamp(th - c_k * d, p["min"], p["max"]))

        w, l, dr = play(args, opts_string(params, plus), opts_string(params, minus),
                        seed=args.seed * 1000003 + k)
        totals["w"] += w
        totals["l"] += l
        totals["d"] += dr
        result = w - l

        for i, p in enumerate(params):
            if delta[i] == 0:
                continue
            c_k = steps[i]
            a_k = args.r_end * p["c_end"] ** 2 * a_scale
            theta[i] = clamp(theta[i] + a_k / c_k * result * delta[i],
                             p["min"], p["max"])

        with open(args.log, "a", newline="") as f:
            csv.writer(f).writerow([k, w, l, dr, f"{c_scale:.4f}", f"{a_scale:.4f}"]
                                   + [f"{t:.3f}" for t in theta])
        with open(args.state + ".tmp", "w") as f:
            json.dump(dict(args=vars(args), params=params, theta=theta, k=k,
                           totals=totals), f, indent=1)
        os.replace(args.state + ".tmp", args.state)

        if k % 50 == 0 or k == n_iter or k == k0 + 1:
            el = time.time() - t0
            done = k - k0
            eta = el / done * (n_iter - k)
            games = totals["w"] + totals["l"] + totals["d"]
            print(f"[{k}/{n_iter}] games {games} (W-L-D {totals['w']}-{totals['l']}-"
                  f"{totals['d']})  elapsed {el / 60:.1f} min  eta {eta / 60:.1f} min",
                  flush=True)
            print("   " + "  ".join(f"{p['name']}={t:.2f}" for p, t in zip(params, theta)),
                  flush=True)

    print("\nTuned values:")
    for p, t in zip(params, theta):
        print(f"  {p['name']:22s} {p['default']:5d} -> {int(round(t)):5d}   ({t:.2f})")
    print("\nMatch tool options:\n  --opts1 " + " ".join(opts_string(params, theta)))
    print("\nC++ defaults:")
    for p, t in zip(params, theta):
        print(f"  {p['name']} = {int(round(t))};")


if __name__ == "__main__":
    main()
