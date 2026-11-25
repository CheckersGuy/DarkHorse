#!/usr/bin/env python3
# Early-stop check for SPSA run 3 (agreed with the user, applied from iteration 2500):
#   converged if no tuned parameter drifted more than 0.4 c_end over the last 500
#   iterations AND no parameter moved in the same direction (> 0.05 c_end each)
#   in all of the last three 100-iteration windows.
import csv, json, sys
T = '/home/leaguerobin/DarkHorse/Training/'
r = list(csv.reader(open(T + 'spsa3_log.csv'))); h = r[0]; d = r[1:]
st = json.load(open(T + 'spsa3_state.json')); P = {p['name']: p for p in st['params']}
k = min(int(sys.argv[1]), len(d))
if k < 2500:
    sys.exit(0)
val = lambda i, j: float(d[i - 1][j])
drifters, steady = [], []
for j, n in enumerate(h[6:], 6):
    p = P[n]
    if not p.get('tune'):
        continue
    c = p['c_end']
    drift = (val(k, j) - val(k - 500, j)) / c
    if abs(drift) > 0.4:
        drifters.append(f"{n} {drift:+.2f}")
    steps = [(val(k - 100 * i, j) - val(k - 100 * (i + 1), j)) / c for i in range(3)]
    if all(s > 0.05 for s in steps) or all(s < -0.05 for s in steps):
        steady.append(f"{n} {sum(steps):+.2f}")
if not drifters and not steady:
    print(f"CONVERGED at iter {k}: no drift > 0.4 c_end over 500 iterations, no steady direction")
else:
    print(f"CHECK iter {k}: not converged | drift>0.4 over 500 iter: {len(drifters)} "
          f"({', '.join(drifters[:6])}) | steady direction: {len(steady)} ({', '.join(steady[:6])})")
