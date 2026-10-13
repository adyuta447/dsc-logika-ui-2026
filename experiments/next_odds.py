import glob, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
# Chance that each of the next queued files beats our best public, on the same conditioned splits as harvest_bayes.py (B 20000).
S = sys.argv[1]
y = pd.read_parquet("features.parquet", columns=["target"]).target.dropna().astype(int).values
rank = lambda v: rankdata(v) / len(v)
def variants(r):
    res = pd.read_pickle(f"{r}/models/predictions.pkl")
    pick = lambda pre: np.mean([rank(res[k][0]) for k in res if k.startswith(pre)], 0)
    c = {f"{m}_{t}": pick(f"{m}_{t}") for m in ("lgb", "cat", "xgb") for t in ("full", "lean", "top")}
    lin = rank(res["linear"][0]); tr = {t: (c[f"lgb_{t}"] + c[f"cat_{t}"] + c[f"xgb_{t}"]) / 3 for t in ("full", "lean", "top")}
    core = (tr["full"] + tr["lean"] + c["cat_full"] + c["cat_lean"]) / 4
    return {"top_mix": 0.8 * (core + tr["top"]) / 2 + 0.2 * lin, "trees_only": (core + tr["top"]) / 2, "linear30": 0.7 * (core + tr["top"]) / 2 + 0.3 * lin,
            "top_only": 0.8 * tr["top"] + 0.2 * lin, "full_only": tr["full"], "lgb_only": (c["lgb_full"] + c["lgb_lean"] + c["lgb_top"]) / 3}
O = {}
for g, rs in {"v4": ["run_v4_s42", "run_v4_s7", "run_v4_s13"], "fac": ["run_fac", "run_s7", "run_s13", "run_s21"]}.items():
    vo = [variants(r) for r in rs]
    for k in vo[0]: O[f"{g}|{k}"] = rank(np.mean([v[k] for v in vo], 0))
for name in ("lgb_fullset", "lgb_extra"):
    O[f"new|{name}"] = rank(np.mean([rank(np.load(p)["oof"]) for p in sorted(glob.glob(S + f"div/{name}_s*.npz"))], 0))
ext = lambda a, k: rank(O[f"v4|{k}"] + a * (O[f"v4|{k}"] - O[f"fac|{k}"]))
O["ext1.5|v4-fac|trees_only"], O["ext1.5|v4-fac|full_only"], O["ext0.5|v4-fac|top_only"] = ext(1.5, "trees_only"), ext(1.5, "full_only"), ext(0.5, "top_only")
names = ["v4|top_mix", "fac|top_mix", "fac|linear30", "ext1.5|v4-fac|trees_only", "new|lgb_fullset", "ext1.5|v4-fac|full_only", "ext0.5|v4-fac|top_only", "new|lgb_extra", "v4|lgb_only"]
P = np.column_stack([O[n] for n in names])
rng = np.random.default_rng(0); B = 20000; D = np.zeros((B, len(names)))
for b in range(B):
    idx = rng.choice(len(y), 9000, replace=False); o = np.argsort(-P[idx], axis=0); yy = y[idx][o]
    D[b] = np.max(np.cumsum(yy, 0) / yy[:, 0].sum() - np.cumsum(1 - yy, 0) / (9000 - yy[:, 0].sum()), 0)
diff = D - D[:, [0]]
obs = {"fac|top_mix": 0.00005, "fac|linear30": -0.00034, "ext1.5|v4-fac|trees_only": -0.00050, "new|lgb_fullset": 0.00071}
w = np.exp(-sum((diff[:, names.index(k)] - v) ** 2 for k, v in obs.items()) / (2 * 0.0006 ** 2)); w /= w.sum()
print("ESS", round(1 / (w ** 2).sum(), 1))
for n in names[5:]:
    d = diff[:, names.index(n)]; o = np.argsort(d); cw = np.cumsum(w[o])
    q = lambda p: 0.55860 + d[o][np.searchsorted(cw, p)]
    print(f"{n:26s} P(beats 0.55931) {(w * (d > 0.00071)).sum():.2f}  P(>=0.56275 rank 13) {(w * (d >= 0.00415)).sum():.2f}  expected {0.55860 + (w * d).sum():.5f}  10/50/90% {q(0.1):.5f} {q(0.5):.5f} {q(0.9):.5f}")
best = np.maximum(0.00071, diff[:, 5:8].max(1))
print(f"next 3 (18:01, 07:01, 12:01): P(new best) {(w * (best > 0.00071)).sum():.2f}  P(rank 13) {(w * (best >= 0.00415)).sum():.2f}  expected best {0.55860 + (w * best).sum():.5f}")
