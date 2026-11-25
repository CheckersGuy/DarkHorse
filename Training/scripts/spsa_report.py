#!/usr/bin/env python3
# One-line report of the SPSA run 3 parameters that moved most (in c_end units),
# with the change since the previous report (arrow = direction of the last 100 iterations).
import csv, json, sys
T = '/home/leaguerobin/DarkHorse/Training/'
r = list(csv.reader(open(T + 'spsa3_log.csv'))); h = r[0]; d = r[1:]
st = json.load(open(T + 'spsa3_state.json')); P = {p['name']: p for p in st['params']}
k = int(sys.argv[1]) if len(sys.argv) > 1 else len(d)
k = min(k, len(d)); prev = max(1, k - 100)
rows = []
for j, n in enumerate(h[6:], 6):
    p = P[n]
    if not p.get('tune'):
        continue
    c = p['c_end']; s = p['default']
    v = float(d[k - 1][j]); v0 = float(d[prev - 1][j])
    rows.append((abs(v - s) / c, n, s, v, (v - s) / c, (v - v0) / c))
rows.sort(reverse=True)
parts = []
for _, n, s, v, rel, step in rows[:12]:
    arrow = '↑' if step > 0.05 else '↓' if step < -0.05 else '→'
    parts.append(f"{n} {s}→{v:.1f} ({rel:+.2f}{arrow})")
print(f"REPORT iter {k}: " + " | ".join(parts))
