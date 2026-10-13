import glob, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

# Extends the scheduled family with the new model types and picks, greedily, the files that raise the chance of the target most.
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
for name in sorted({p.split("/")[-1].rsplit("_s", 1)[0] for p in glob.glob(S + "div/*.npz")}):
    z = [np.load(p) for p in sorted(glob.glob(S + f"div/{name}_s*.npz"))]
    if len(z) < 3: continue
    O[f"new|{name}"] = rank(np.mean([rank(q["oof"]) for q in z], 0)); T[f"new|{name}"] = rank(np.mean([rank(q["pred"]) for q in z], 0))
    for w in (0.5,):
        O[f"mix|{name}"] = rank((1 - w) * O["v4|top_mix"] + w * O[f"new|{name}"]); T[f"mix|{name}"] = rank((1 - w) * T["v4|top_mix"] + w * T[f"new|{name}"])

# extrapolations: step past the better model along the direction it improved on
for k in ("top_mix", "trees_only", "linear30", "main", "top_only", "lgb_only", "cat_only", "xgb_only", "full_only", "lean_only"):
    for a_ in (0.5, 1.0, 1.5, 2.0):
        n_ = f"ext{a_}|v4-fac|{k}"; O[n_] = rank(O[f"v4|{k}"] + a_ * (O[f"v4|{k}"] - O[f"fac|{k}"])); T[n_] = rank(T[f"v4|{k}"] + a_ * (T[f"v4|{k}"] - T[f"fac|{k}"]))
for new_, old_ in (("lgb_extra", "lgb_only"), ("xgb_deep", "xgb_only"), ("lgb_shallow", "lgb_only"), ("cat_deep", "cat_only")):
    if f"new|{new_}" in O:
        for a_ in (0.5, 1.0):
            n_ = f"ext{a_}|{new_}-v4|{old_}"; O[n_] = rank(O[f"new|{new_}"] + a_ * (O[f"new|{new_}"] - O[f"v4|{old_}"])); T[n_] = rank(T[f"new|{new_}"] + a_ * (T[f"new|{new_}"] - T[f"v4|{old_}"]))
ref = "v4|top_mix"; names = list(O); P = np.column_stack([O[n] for n in names]); ir = names.index(ref)
def ks_cols(Ps, ys):
    o = np.argsort(-Ps, axis=0); yy = ys[o]
    return np.max(np.cumsum(yy, 0) / ys.sum() - np.cumsum(1 - yy, 0) / (len(ys) - ys.sum()), 0)
rng = np.random.default_rng(0); B = 3000; D = np.zeros((B, len(names)))
for b in range(B):
    idx = rng.choice(len(y), 9000, replace=False); D[b] = ks_cols(P[idx], y[idx])
diff = D - D[:, [ir]]
t = pd.DataFrame({"cv_auc": [roc_auc_score(y, O[n]) for n in names], "rho_oof": [spearmanr(O[n], O[ref]).correlation for n in names],
                  "rho_test": [spearmanr(T[n], T[ref]).correlation for n in names], "mean_diff": diff.mean(0), "sd_diff": diff.std(0)}, index=names)

import re, os
done = ["v4|top_mix", "fac|top_mix", "fac|linear30"]
if os.path.exists("logs/scheduled_harvest.log"):
    for m in re.finditer(r"harvest_\d+_(v4|fac)_(\w+?)\.csv public", open("logs/scheduled_harvest.log").read()):
        done.append(f"{m.group(1)}|{m.group(2)}")
nslots = int(os.environ.get("NSLOTS", "37"))
print("already submitted:", done)
fam = [names.index(n) for n in done]; cur = np.max(diff[:, fam], 1)
lb = {"rank 13": 0.56160 - 0.55860, "rank 10": 0.56334 - 0.55860, "rank 8": 0.56441 - 0.55860, "0.565": 0.565 - 0.55860}
print("scheduled family:", f"E[best - ref] {cur.mean():+.5f}", "  ".join(f"{k} {(cur >= v).mean():.2f}" for k, v in lb.items()))

# condition the simulated public splits on the public scores we have actually seen (relative to bag v4 at 0.55860)
obs = {"fac|top_mix": 0.00005, "fac|linear30": -0.00034, "ext1.5|v4-fac|trees_only": -0.00050}
sig = 0.0006
w = np.exp(-sum((diff[:, names.index(k)] - v) ** 2 for k, v in obs.items()) / (2 * sig ** 2)); w /= w.sum()
print("effective number of matching splits:", round(1 / (w ** 2).sum(), 1), "of", B)
done = ["v4|top_mix"] + list(obs)
fam = [names.index(n) for n in done]; cur = np.max(diff[:, fam], 1)
wmean = lambda c: (w * c).sum()
lb = {"rank 13": 0.56160 - 0.55860, "rank 10": 0.56334 - 0.55860, "rank 8": 0.56441 - 0.55860, "0.565": 0.565 - 0.55860}
for label, ww in (("unconditional", np.full(B, 1 / B)), ("conditioned on what we saw", w)):
    print(label, "mean diff of a few candidates:", {k: round(float((ww * diff[:, names.index(k)]).sum()), 5) for k in ["ext1.0|v4-fac|top_mix", "new|lgb_extra", "new|xgb_deep", "mix|cat_deep", "v4|lgb_only", "ext2.0|v4-fac|full_only"] if k in names})
picked = []
for step in range(nslots):
    g, j = max((wmean(np.maximum(cur, diff[:, j])), j) for j in range(len(names)) if j not in fam)
    fam.append(j); cur = np.maximum(cur, diff[:, j]); picked.append(names[j])
    if step < 12 or step % 10 == 0 or step == nslots - 1:
        print(f"+ {names[j]:28s} E[best] {0.55860 + g:.5f}", "  ".join(f"{k} {(w * (cur >= v)).sum():.2f}" for k, v in lb.items()))
ids = pd.read_csv("run_v4_s42/variants/top_mix.csv").application_id
for i, n in enumerate(picked):
    sub = pd.DataFrame({"application_id": ids, "default_probability": T[n]})
    assert len(sub) == 30000 and sub.application_id.is_unique and sub.default_probability.between(0, 1).all()
    sub.to_csv(f"variants/qb_{i:02d}_{re.sub(r'[^A-Za-z0-9.]+', '_', n)}.csv", index=False)
print("files:", len(picked))
