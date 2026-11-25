"""Neuron permutation for a quantised net (.quant), so that neurons that are often
active together share the 4-byte blocks the engine's SparseLayer skips when zero.

    ~/training/bin/python3 compute_permutation.py --net NET.quant --l1 2048 \
        --positions /mnt/c/TrainingData/evallog/search20ms.samples --out NET.perm
    ~/training/bin/python3 compute_permutation.py --net POLICY.quant --l1 1024 \
        --positions /mnt/c/TrainingData/reservoir-full.val.children.samples --out POLICY.perm

--l1: half width of the accumulator (value net 2048, policy net 1024, mlh net 128).
--positions: 18-byte sample files (e.g. a --log_evals search log; best for value nets,
    it is what the search evaluates) or a POL1 policy file (read via the Rust loader).
Output: raw size_t[2*L1] (perm, perm + L1) for Network::load_permutation(_from_array).
The permutation is fitted on half of the positions and reported on the other half.
Activations are computed exactly like the engine: a = clamp(acc, 0, 127) per half,
neuron active iff (a * b) >> 7 != 0.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def features_from_samples(path, limit):
    r = np.fromfile(path, dtype=np.uint8)
    r = r[: (len(r) // 18) * 18].reshape(-1, 18)[:limit]
    u = lambda o: r[:, o:o + 4].copy().view(np.uint32).ravel()
    wp, bp, k = u(0), u(4), u(8)
    if (r[:, 12].view(np.int8) != r[0, 12].view(np.int8)).any():
        sys.exit("mixed side to move in the sample file (expected all white to move)")
    X = np.zeros((len(r), 120), np.float32)
    # feature layout of AccumulatorState::apply: white pawns on squares 4..31 -> 0..27,
    # black pawns 0..27 -> 28..55, white kings -> 56..87, black kings -> 88..119
    for bits, off, lo, hi in ((wp & ~k, -4, 4, 32), (bp & ~k, 28, 0, 28),
                              (wp & k, 56, 0, 32), (bp & k, 88, 0, 32)):
        for sq in range(lo, hi):
            X[:, sq + off] = (bits >> sq) & 1
    return X


def features_from_pol1(path, limit):
    from Experimental import PolicyBatchDataSet
    ds = PolicyBatchDataSet(8192, 10**9, path, False, one_pass=True)
    out = []
    for batch in iter(ds):
        out.append(batch[2])
        if sum(len(x) for x in out) >= limit:
            break
    return np.concatenate(out)[:limit]


def load_accumulator(path, l1):
    out = 2 * l1
    raw = np.fromfile(path, dtype=np.uint8)
    W = np.frombuffer(raw[:120 * out * 2].tobytes(), dtype=np.int16).reshape(120, out)
    b = np.frombuffer(raw[120 * out * 2:120 * out * 2 + out * 2].tobytes(), dtype=np.int16)
    return W.astype(np.float32), b.astype(np.float32)


def activations(X, W, b, l1, dev):
    W, b = torch.tensor(W, device=dev), torch.tensor(b, device=dev)
    res = []
    for i in range(0, len(X), 65536):
        acc = torch.tensor(X[i:i + 65536], device=dev) @ W + b
        res.append(acc[:, :l1].clamp(0, 127) * acc[:, l1:].clamp(0, 127) >= 128)
    return torch.cat(res)


def nonzero_blocks(A, perm):
    return A[:, perm].view(A.shape[0], -1, 4).any(2).float().sum(1).mean().item()


def greedy_merge(G_act, groups):
    """pairs groups greedily by the smallest union of their activity"""
    F = G_act.float()
    n = F.sum(0)
    union = n[:, None] + n[None, :] - F.T @ F
    union.fill_diagonal_(float("inf"))
    G = len(groups)
    used = np.zeros(G, bool)
    new_g, new_a = [], []
    for idx in torch.argsort(union.flatten()).cpu().numpy():
        i, j = divmod(int(idx), G)
        if used[i] or used[j] or i == j:
            continue
        used[i] = used[j] = True
        new_g.append(groups[i] + groups[j])
        new_a.append(G_act[:, i] | G_act[:, j])
        if len(new_g) == G // 2:
            break
    return torch.stack(new_a, 1), new_g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True)
    ap.add_argument("--l1", type=int, required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1_000_000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.positions, "rb") as f:
        is_pol1 = int.from_bytes(f.read(4), "little") == 0x504F4C31
    X = (features_from_pol1 if is_pol1 else features_from_samples)(args.positions, args.limit)
    W, b = load_accumulator(args.net, args.l1)
    A = activations(X, W, b, args.l1, dev)
    A = A[torch.randperm(len(A), generator=torch.Generator().manual_seed(0)).to(dev)]
    fit, test = A[: len(A) // 2], A[len(A) // 2:]

    groups = [[i] for i in range(args.l1)]
    g_act = fit
    for _ in range(2):                       # singles -> pairs -> blocks of 4
        g_act, groups = greedy_merge(g_act, groups)
    perm = torch.tensor([i for g in groups for i in g], device=dev)
    assert sorted(perm.tolist()) == list(range(args.l1))

    ident = torch.arange(args.l1, device=dev)
    print(f"{len(A)} positions, active neurons {A.float().mean().item() * 100:.2f} % "
          f"({A.float().sum(1).mean().item():.0f} of {args.l1})")
    print(f"non-zero blocks of {args.l1 // 4} (held-out half): identity "
          f"{nonzero_blocks(test, ident):.1f} -> permuted {nonzero_blocks(test, perm):.1f}")
    p = perm.cpu().numpy().astype(np.uint64)
    np.concatenate([p, p + args.l1]).astype(np.uint64).tofile(args.out)
    print(f"wrote {args.out} ({2 * args.l1} x size_t)")


if __name__ == "__main__":
    main()
