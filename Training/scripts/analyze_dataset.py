#!/usr/bin/env python3
"""Pre-rescoring analysis of a binary .samples file (18-byte Sample records).

Reports: total positions, exact duplicates (wp, bp, k, color), positions the
rescorer skips (|value| >= 5000), TB-labelled positions, and how many positions
have a forced capture / no move / exactly one legal quiet move.

Run with the training venv:  ~/training/bin/python3 analyze_dataset.py FILE
"""
import sys
import time
import numpy as np

REC = np.dtype([('wp', '<u4'), ('bp', '<u4'), ('k', '<u4'), ('color', 'i1'),
                ('mlh', '<i2'), ('result', 'i1'), ('value', '<i2')])
assert REC.itemsize == 18

# masks from Training/Rust/src/Pos.rs
L3, L5, R3, R5 = np.uint32(14737632), np.uint32(117901063), np.uint32(117901056), np.uint32(3772834016)
U32 = np.uint32


def default_shift(black, m):  # black: bool array
    return np.where(black, m << U32(4), m >> U32(4))


def forward_mask(black, m):
    return np.where(black, ((m & L3) << U32(3)) | ((m & L5) << U32(5)),
                    ((m & R3) >> U32(3)) | ((m & R5) >> U32(5)))


def popcount(x):
    x = x.astype(np.uint32)
    return np.bitwise_count(x).astype(np.int32)


def jumpers(black, own, opp, kings, nocc):
    # Pos.rs get_jumpers with COLOR = side to move
    m = np.zeros_like(own)
    t = default_shift(~black, nocc) & opp
    m |= forward_mask(~black, t) & own
    t = forward_mask(~black, nocc) & opp
    m |= default_shift(~black, t) & own
    kk = own & kings
    t = default_shift(black, nocc) & opp
    m |= forward_mask(black, t) & kk
    t = forward_mask(black, nocc) & opp
    m |= default_shift(black, t) & kk
    return m


def quiet_moves(black, own, kings, nocc):
    pawns, kk = own & ~kings, own & kings
    n = popcount(default_shift(black, pawns) & nocc) + popcount(forward_mask(black, pawns) & nocc)
    n += popcount(default_shift(black, kk) & nocc) + popcount(forward_mask(black, kk) & nocc)
    n += popcount(default_shift(~black, kk) & nocc) + popcount(forward_mask(~black, kk) & nocc)
    return n


def main(path, chunk=20_000_000):
    data = np.memmap(path, dtype=REC, mode='r')
    n = len(data)
    print(f"{path}: {n:,} positions")
    keys = np.empty(n, dtype=np.uint64)
    stats = dict(decisive=0, tb=0, capture=0, nomove=0, one=0, two=0, cap_or_one=0)
    t0 = time.time()
    for s in range(0, n, chunk):
        d = np.array(data[s:s + chunk])
        wp, bp, k = d['wp'], d['bp'], d['k']
        black = d['color'] == -1
        own, opp = np.where(black, bp, wp), np.where(black, wp, bp)
        nocc = ~(wp | bp)
        h = (wp.astype(np.uint64) | (bp.astype(np.uint64) << np.uint64(32))) * np.uint64(0x9E3779B97F4A7C15)
        h ^= (k.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)) + black.astype(np.uint64)
        keys[s:s + len(d)] = h
        stats['decisive'] += int((np.abs(d['value'].astype(np.int32)) >= 5000).sum())
        stats['tb'] += int(np.isin(d['result'], [4, 5, 6]).sum())
        cap = jumpers(black, own, opp, k, nocc) != 0
        q = quiet_moves(black, own, k, nocc)
        stats['capture'] += int(cap.sum())
        stats['nomove'] += int((~cap & (q == 0)).sum())
        stats['one'] += int((~cap & (q == 1)).sum())
        stats['two'] += int((~cap & (q == 2)).sum())
        stats['cap_or_one'] += int((cap | (q <= 1)).sum())
        print(f"  {min(s + chunk, n):,}/{n:,}  {time.time() - t0:.0f}s", flush=True)
    keys.sort()
    uniq = int(1 + np.count_nonzero(keys[1:] != keys[:-1]))
    pct = lambda x: f"{x:,} ({100 * x / n:.1f} %)"
    print(f"unique positions:            {pct(uniq)}")
    print(f"duplicates:                  {pct(n - uniq)}")
    print(f"|value| >= 5000 (skipped):   {pct(stats['decisive'])}")
    print(f"TB-labelled results:         {pct(stats['tb'])}")
    print(f"forced capture:              {pct(stats['capture'])}")
    print(f"no legal move:               {pct(stats['nomove'])}")
    print(f"exactly 1 quiet move:        {pct(stats['one'])}")
    print(f"exactly 2 quiet moves:       {pct(stats['two'])}")
    print(f"capture or <= 1 quiet move:  {pct(stats['cap_or_one'])}")
    print(f"total time {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main(sys.argv[1])
