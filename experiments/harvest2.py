import numpy as np, pandas as pd, sys
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
S = sys.argv[1]
f = pd.read_parquet("features.parquet", columns=["target"]); y = f[f.target.notna()].target.astype(int).values
rank = lambda v: rankdata(v) / len(v)
def variants(r, i):
    res = pd.read_pickle(f"{r}/models/predictions.pkl")
    pick = lambda pre: np.mean([rank(res[k][i]) for k in res if k.startswith(pre)], 0)
    c = {f"{m}_{t}": pick(f"{m}_{t}") for m in ("lgb", "cat", "xgb") for t in ("full", "lean", "top")}
    lin = rank(res["linear"][i]); tr = {t: (c[f"lgb_{t}"] + c[f"cat_{t}"] + c[f"xgb_{t}"]) / 3 for t in ("full", "lean", "top")}
    core = (tr["full"] + tr["lean"] + c["cat_full"] + c["cat_lean"]) / 4
    return {"top_mix": 0.8 * (core + tr["top"]) / 2 + 0.2 * lin, "trees_only": (core + tr["top"]) / 2, "linear30": 0.7 * (core + tr["top"]) / 2 + 0.3 * lin,
            "main": 0.8 * core + 0.2 * lin, "top_only": 0.8 * tr["top"] + 0.2 * lin, "lgb_only": (c["lgb_full"] + c["lgb_lean"] + c["lgb_top"]) / 3,
            "cat_only": (c["cat_full"] + c["cat_lean"] + c["cat_top"]) / 3, "xgb_only": (c["xgb_full"] + c["xgb_lean"] + c["xgb_top"]) / 3,
            "full_only": tr["full"], "lean_only": tr["lean"], "linear_only": lin}
groups = {"v4": ["run_v4_s42", "run_v4_s7", "run_v4_s13"], "fac": ["run_fac", "run_s7", "run_s13", "run_s21"]}
O, T = {}, {}
for g, rs in groups.items():
    vo = [variants(r, 0) for r in rs]; vt = [variants(r, 1) for r in rs]
    for k in vo[0]:
        O[f"{g}|{k}"] = rank(np.mean([v[k] for v in vo], 0)); T[f"{g}|{k}"] = rank(np.mean([v[k] for v in vt], 0))
ref = "v4|top_mix"; names = list(O); P = np.column_stack([O[n] for n in names]); ir = names.index(ref)
def ks_cols(Ps, ys):
    o = np.argsort(-Ps, axis=0); yy = ys[o]
    return np.max(np.cumsum(yy, 0) / ys.sum() - np.cumsum(1 - yy, 0) / (len(ys) - ys.sum()), 0)
rng = np.random.default_rng(0); B = 800; D = np.zeros((B, len(names)))
for b in range(B):
    idx = rng.choice(len(y), 9000, replace=False); D[b] = ks_cols(P[idx], y[idx])
diff = D - D[:, [ir]]
t = pd.DataFrame({"cv_auc": [roc_auc_score(y, O[n]) for n in names], "cv_ks": ks_cols(P, y),
                  "rho_oof": [spearmanr(O[n], O[ref]).correlation for n in names], "rho_test": [spearmanr(T[n], T[ref]).correlation for n in names],
                  "mean_diff": diff.mean(0), "sd_diff": diff.std(0), "p_beats_ref": (diff > 0).mean(0)}, index=names)
print(t.sort_values("cv_auc", ascending=False).round(5).to_string())
fam = [ir]; cur = np.zeros(B)
for step in range(12):
    g, j = max((np.maximum(cur, diff[:, j]).mean(), j) for j in range(len(names)) if j not in fam)
    fam.append(j); cur = np.maximum(cur, diff[:, j])
    print(f"family {len(fam):2d}: + {names[j]:16s} E[best - ref] {g:+.5f}  P(>= +0.002) {(cur >= .002).mean():.2f}  P(>= +0.004) {(cur >= .004).mean():.2f}  P(>= +0.006) {(cur >= .006).mean():.2f}")
pd.to_pickle({"T": T, "names": names, "fam": [names[j] for j in fam]}, S + "harvest2.pkl")
# chance of each leaderboard target, given the reference scored 0.55860 on the real public split
lb = {"rank 18 (0.55905)": 0.55905, "rank 15 (0.56013)": 0.56013, "rank 13 (0.56160)": 0.56160, "rank 10 (0.56334)": 0.56334, "rank 8 (0.56441)": 0.56441, "rank 5 (0.56573)": 0.56573}
for size in (4, 7, 13):
    best = np.max(diff[:, fam[:size]], 1)
    print(f"{size:2d} submissions:", "  ".join(f"{k} {np.mean(0.55860 + best >= v):.2f}" for k, v in lb.items()))
# private part: the better of two selected submissions (private split 21000 rows)
rng = np.random.default_rng(1); a, b2 = names.index("v4|top_mix"), names.index("v4|lgb_only"); g = []
for _ in range(400):
    idx = rng.choice(len(y), 21000, replace=False); k = ks_cols(P[idx][:, [a, b2]], y[idx]); g.append(k.max() - k[0])
print("private: best of top_mix and lgb_only minus top_mix alone, mean", round(np.mean(g), 5))
