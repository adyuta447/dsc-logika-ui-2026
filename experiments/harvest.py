import numpy as np, pandas as pd, sys
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

# How much public score can a set of submissions gain from the public split alone?
# Every candidate is scored on many random 9000 row subsets of the training OOF (the public split is 30% of 30000 rows).
S = sys.argv[1]
f = pd.read_parquet("features.parquet", columns=["target"]); y = f[f.target.notna()].target.astype(int).values
rank = lambda v: rankdata(v) / len(v)
runs = ["run_v4_s42", "run_v4_s7", "run_v4_s13", "run_v4_lr005", "run_fac", "run_s7", "run_s13", "run_s21", "run10", "run_seeds", "."]
cand = {}
for r in runs:
    res = pd.read_pickle(f"{r}/models/predictions.pkl")
    if not isinstance(res["linear"], tuple): continue
    pick = lambda pre: np.mean([rank(res[k][0]) for k in res if k.startswith(pre)], 0)
    c = {f"{m}_{t}": pick(f"{m}_{t}") for m in ("lgb", "cat", "xgb") for t in ("full", "lean", "top")}
    lin = rank(res["linear"][0])
    tr = {t: (c[f"lgb_{t}"] + c[f"cat_{t}"] + c[f"xgb_{t}"]) / 3 for t in ("full", "lean", "top")}
    core = (tr["full"] + tr["lean"] + c["cat_full"] + c["cat_lean"]) / 4
    name = "v1_5fold" if r == "." else r
    cand.update({
        f"{name}|top_mix": 0.8 * (core + tr["top"]) / 2 + 0.2 * lin, f"{name}|trees_only": (core + tr["top"]) / 2,
        f"{name}|linear30": 0.7 * (core + tr["top"]) / 2 + 0.3 * lin, f"{name}|main": 0.8 * core + 0.2 * lin,
        f"{name}|top_only": 0.8 * tr["top"] + 0.2 * lin, f"{name}|lgb_only": (c["lgb_full"] + c["lgb_lean"] + c["lgb_top"]) / 3,
        f"{name}|cat_only": (c["cat_full"] + c["cat_lean"] + c["cat_top"]) / 3, f"{name}|xgb_only": (c["xgb_full"] + c["xgb_lean"] + c["xgb_top"]) / 3,
        f"{name}|linear_only": lin})
v4 = [f"run_v4_s{s}|top_mix" for s in (42, 7, 13)]
cand["bag_v4|top_mix"] = np.mean([cand[k] for k in v4], 0)
ref = "bag_v4|top_mix"
names = list(cand); P = np.column_stack([cand[n] for n in names]); ir = names.index(ref)

def ks_cols(Ps, ys):
    o = np.argsort(-Ps, axis=0); yy = ys[o]
    tp = np.cumsum(yy, 0) / ys.sum(); fp = np.cumsum(1 - yy, 0) / (len(ys) - ys.sum())
    return np.max(tp - fp, 0)

rng = np.random.default_rng(0); B = 600
D = np.zeros((B, len(names)))
for b in range(B):
    idx = rng.choice(len(y), 9000, replace=False)
    D[b] = ks_cols(P[idx], y[idx])
diff = D - D[:, [ir]]
full_ks = ks_cols(P, y)
t = pd.DataFrame({"auc": [roc_auc_score(y, P[:, i]) for i in range(len(names))], "ks_full": full_ks,
                  "spearman_ref": [spearmanr(P[:, i], P[:, ir]).correlation for i in range(len(names))],
                  "mean_diff": diff.mean(0), "sd_diff": diff.std(0)}, index=names)
t["p_beats_ref"] = (diff > 0).mean(0)
print("public subset KS of the reference: mean", D[:, ir].mean().round(4), "sd", D[:, ir].std().round(4))
print(t.sort_values("auc", ascending=False).round(5).to_string())
# greedy family: keep adding the candidate that most raises the expected best public score over the reference
fam = [ir]; cur = np.zeros(B)
for step in range(15):
    gains = [(np.maximum(cur, diff[:, j]).mean(), j) for j in range(len(names)) if j not in fam]
    g, j = max(gains); fam.append(j); cur = np.maximum(cur, diff[:, j])
    print(f"family size {len(fam):2d}: + {names[j]:28s} expected best minus reference {g:+.5f}  p(best >= ref + 0.002) {(cur >= 0.002).mean():.2f}  p(>= +0.004) {(cur >= 0.004).mean():.2f}")
t.to_csv(S + "harvest.csv")
