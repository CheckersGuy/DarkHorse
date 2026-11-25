#!/usr/bin/env python3
"""Distribution profile of one or more 18-byte .samples files (side by side).

    ~/training/bin/python3 data_profile.py [--sample N] NAME=FILE [NAME=FILE ...]

Per file (on N evenly spread records, default 20M): piece count, number of
kings, value-net bucket usage (engine's 128 buckets: opponent occupancy of
squares 6,9,10,13,14,17,18 after flipping to white-to-move), |value| bands,
side to move. Columns are percentages, so files of different size compare.
"""
import argparse
import numpy as np

REC = np.dtype([('wp', '<u4'), ('bp', '<u4'), ('k', '<u4'), ('color', 'i1'),
                ('mlh', '<i2'), ('result', 'i1'), ('value', '<i2')])
BUCKET_SQ = [6, 9, 10, 13, 14, 17, 18]


def flip32(x):
    """Reverse the 32 bits (color flip = board rotation, as get_color_flip)."""
    x = x.astype(np.uint32)
    out = np.zeros_like(x)
    for i in range(32):
        out |= ((x >> np.uint32(i)) & np.uint32(1)) << np.uint32(31 - i)
    return out


def load(path, n):
    data = np.memmap(path, dtype=REC, mode='r')
    if len(data) <= n:
        return np.array(data), len(data)
    # evenly spread blocks of 100k records (cheap sequential reads)
    blocks = n // 100_000
    starts = np.linspace(0, len(data) - 100_000, blocks).astype(np.int64)
    return np.concatenate([np.array(data[s:s + 100_000]) for s in starts]), len(data)


def profile(d):
    wp, bp, k = d['wp'], d['bp'], d['k']
    black = d['color'] == -1
    pieces = np.bitwise_count(wp) + np.bitwise_count(bp)
    kings = np.bitwise_count(k)
    # opponent pieces after flipping to the side to move = white
    opp = np.where(black, flip32(wp), bp)
    bucket = np.zeros(len(d), dtype=np.int64)
    for j, sq in enumerate(BUCKET_SQ):
        bucket |= ((opp >> np.uint32(sq)) & np.uint32(1)).astype(np.int64) << j
    v = np.abs(d['value'].astype(np.int32))
    return dict(pieces=pieces, kings=kings, bucket=bucket, value=v, black=black)


def pct(mask):
    return 100.0 * np.count_nonzero(mask) / len(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=20_000_000)
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    names, profs = [], []
    for spec in a.files:
        name, path = spec.split('=', 1)
        d, total = load(path, a.sample)
        names.append(name)
        profs.append(profile(d))
        print(f"{name}: {path}  ({total:,} records, profiled {len(d):,})")
    head = lambda t: print(f"\n{t:<22}" + "".join(f"{n:>12}" for n in names))
    row = lambda lbl, f: print(f"{lbl:<22}" + "".join(f"{f(p):>11.2f}%" for p in profs))

    head("pieces on board")
    for lo, hi in [(21, 24), (17, 20), (13, 16), (9, 12), (6, 8), (2, 5)]:
        row(f"  {lo}-{hi}", lambda p: pct((p['pieces'] >= lo) & (p['pieces'] <= hi)))
    head("kings on board")
    for lo, hi, lbl in [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 32, ">=4")]:
        row(f"  {lbl}", lambda p: pct((p['kings'] >= lo) & (p['kings'] <= hi)))
    head("|value| (search eval)")
    for lo, hi in [(0, 50), (51, 150), (151, 400), (401, 1000), (1001, 99999)]:
        row(f"  {lo}-{hi}", lambda p: pct((p['value'] >= lo) & (p['value'] <= hi)))
    head("side to move")
    row("  black", lambda p: pct(p['black']))
    head("buckets (128)")
    for p, n in zip(profs, names):
        c = np.bincount(p['bucket'], minlength=128) / len(p['bucket'])
        srt = np.sort(c)[::-1]
        print(f"  {n}: used {np.count_nonzero(c)}/128, top-10 share {100 * srt[:10].sum():.1f}%, "
              f"smallest {100 * srt[-1]:.3f}%, empty-bucket share {100 * c[0]:.1f}%")


if __name__ == '__main__':
    main()
