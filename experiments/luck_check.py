import os, sys, glob
import numpy as np, pandas as pd

# Scores a planned queue on the simulated public splits that match every observed public score (luck_search stage 1 output).
#   luck_check.py S <prefix> <parts> <KEEP>
S, pre = sys.argv[1], sys.argv[2]
src = open(S + "harvest_kf.py").read(); exec(src[:src.index("names = list(O); ir")])
ob = list(obs); v = np.array(list(obs.values()))
split = lambda b: np.random.default_rng([3, b]).choice(len(y), 9000, replace=False)
parts, KEEP = map(int, sys.argv[3:5])
z = [np.load(S + f"luck_{p}.npz") for p in range(parts)]
bs = np.concatenate([q["b"] for q in z]); D = np.vstack([q["d"] for q in z])
lw = -((D - v) ** 2).sum(1) / (2 * sig ** 2); w = np.exp(lw - lw.max()); w /= w.sum()
keep = np.argsort(-w)[:KEEP]; wk = w[keep] / w[keep].sum()
q = [resolve(os.path.basename(f)[len(pre) + 4:-4]) for f in sorted(glob.glob(f"variants/{pre}_*.csv"))]
assert q and all(q), "a queue file is not a known candidate"
P = np.column_stack([O[n] for n in ["v4|top_mix"] + ob + q])
Dk = np.array([ks_cols(P[idx], y[idx]) for idx in map(split, bs[keep])]); Dk = Dk[:, 1:] - Dk[:, [0]]
print("matching splits reproduce the observed scores as:", dict(zip(ob, (wk @ Dk[:, :len(ob)]).round(5))))
goal = 0.565 - 0.55860; cur = np.maximum(0, Dk[:, :len(ob)].max(1))
for i, n in enumerate(q):
    d = Dk[:, len(ob) + i]; cur = np.maximum(cur, d)
    print(f"{i:02d} {n:55s} single {wk @ (d >= goal):.3f} mean {wk @ d:+.5f}  queue so far P(0.565) {wk @ (cur >= goal):.3f} E[best] {0.55860 + wk @ cur:.5f}")
