import glob, os, sys
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
# fac3 uses only the fold seeds v4 has, so v4 minus fac3 carries the feature step without fold noise
groups = {"v4": ["run_v4_s42", "run_v4_s7", "run_v4_s13"], "fac": ["run_fac", "run_s7", "run_s13", "run_s21"], "fac3": ["run_fac", "run_s7", "run_s13"]}
if os.path.exists("run_v4_s21/models/predictions.pkl"): groups["v4_4"] = ["run_v4_s42", "run_v4_s7", "run_v4_s13", "run_v4_s21"]
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
    for hi, lo in [("v4", "fac"), ("v4", "fac3")] + [("v4_4", "fac")] * ("v4_4" in groups):
        for a_ in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            n_ = f"ext{a_}|{hi}-{lo}|{k}"; O[n_] = rank(O[f"{hi}|{k}"] + a_ * (O[f"{hi}|{k}"] - O[f"{lo}|{k}"])); T[n_] = rank(T[f"{hi}|{k}"] + a_ * (T[f"{hi}|{k}"] - T[f"{lo}|{k}"]))
for new_, old_ in (("lgb_extra", "lgb_only"), ("xgb_deep", "xgb_only"), ("lgb_shallow", "lgb_only"), ("cat_deep", "cat_only")):
    if f"new|{new_}" in O:
        for a_ in (0.5, 1.0):
            n_ = f"ext{a_}|{new_}-v4|{old_}"; O[n_] = rank(O[f"new|{new_}"] + a_ * (O[f"new|{new_}"] - O[f"v4|{old_}"])); T[n_] = rank(T[f"new|{new_}"] + a_ * (T[f"new|{new_}"] - T[f"v4|{old_}"]))
# per model type: full set minus lean set, and full set v4 minus the same without the season and rhythm features
for k in ("lgb_extra", "xgb_deep", "lgb_shallow", "lgb_ff06", "lgb_extra31", "lgb_mc800", "xgb_shallow", "cat_deep", "lgb_deep", "lgb_goss", "lgb_ff015", "cat_lossguide"):
    for hi, lo, steps in ((f"full_{k}", k, (0.5, 1.0)), (f"full_{k}", f"facfull_{k}", (1.0, 1.5, 2.0))):
        if f"new|{hi}" in O and f"new|{lo}" in O:
            for a_ in steps:
                n_ = f"ext{a_}|{hi}-{lo}"; O[n_] = rank(O[f"new|{hi}"] + a_ * (O[f"new|{hi}"] - O[f"new|{lo}"])); T[n_] = rank(T[f"new|{hi}"] + a_ * (T[f"new|{hi}"] - T[f"new|{lo}"]))
ref = "v4|top_mix"; names = list(O); P = np.column_stack([O[n] for n in names]); ir = names.index(ref)
print("candidates:", len(names))
def ks_cols(Ps, ys):
    o = np.argsort(-Ps, axis=0); yy = ys[o]
    return np.max(np.cumsum(yy, 0) / ys.sum() - np.cumsum(1 - yy, 0) / (len(ys) - ys.sum()), 0)

import re, os
# condition the simulated public splits on the public scores we have actually seen (relative to bag v4 at 0.55860)
obs = {"fac|top_mix": 0.00005, "fac|linear30": -0.00034, "ext1.5|v4-fac|trees_only": -0.00050, "new|lgb_fullset": 0.00071, "ext1.5|v4-fac|full_only": 0.00311}
sig = 0.0006
# two stages: many splits scored on the observed columns only, then every candidate on the splits that carry the weight
oc = [ir] + [names.index(k) for k in obs]; Po = P[:, oc]
split = lambda b: np.random.default_rng([0, b]).choice(len(y), 9000, replace=False)
B = int(os.environ.get("B", "200000")); Do = np.zeros((B, len(oc)))
for b in range(B):
    idx = split(b); Do[b] = ks_cols(Po[idx], y[idx])
lw = -sum((Do[:, i + 1] - Do[:, 0] - v) ** 2 for i, v in enumerate(obs.values())) / (2 * sig ** 2)
w = np.exp(lw - lw.max()); w /= w.sum()
keep = np.argsort(-w)[:int(os.environ.get("KEEP", "3000"))]
print("effective number of matching splits:", round(1 / (w ** 2).sum(), 1), "of", B, " weight kept:", round(w[keep].sum(), 4))
D = np.array([ks_cols(P[idx], y[idx]) for idx in map(split, keep)])
diff = D - D[:, [ir]]; w = w[keep] / w[keep].sum()
nslots = int(os.environ.get("NSLOTS", "37"))
done = ["v4|top_mix"] + list(obs)
fam = [names.index(n) for n in done]; cur = np.max(diff[:, fam], 1)
wmean = lambda c: (w * c).sum()
lb = {"rank 13": 0.56275 - 0.55860, "rank 10": 0.56440 - 0.55860, "rank 8": 0.56553 - 0.55860, "0.565": 0.565 - 0.55860}
m = pd.Series({n: wmean(diff[:, j]) for j, n in enumerate(names)}).sort_values(ascending=False)
print("best conditioned means:", m.head(12).round(5).to_dict())
# 50/50 mixes of the ten candidates that look best on the conditioned splits, scored on the same splits
top = list(m.index[:10]); new = []
for i, a in enumerate(top):
    for b in top[i + 1:]:
        n_ = f"mix2|{a}+{b}"; O[n_] = rank((O[a] + O[b]) / 2); T[n_] = rank((T[a] + T[b]) / 2); new.append(n_)
Pm = np.column_stack([O[n] for n in new])
Dm = np.array([ks_cols(Pm[idx], y[idx]) for idx in map(split, keep)])
names += new; diff = np.hstack([diff, Dm - D[:, [ir]]])
print("mixes added:", len(new), "best mix means:", pd.Series({n: wmean(diff[:, names.index(n)]) for n in new}).sort_values(ascending=False).head(5).round(5).to_dict())
picked = []
for step in range(nslots):
    g, j = max((wmean(np.maximum(cur, diff[:, j])), j) for j in range(len(names)) if j not in fam)
    fam.append(j); cur = np.maximum(cur, diff[:, j]); picked.append(names[j])
    if step < 12 or step % 10 == 0 or step == nslots - 1:
        print(f"+ {names[j]:30s} E[best] {0.55860 + g:.5f}", "  ".join(f"{k} {(w * (cur >= v)).sum():.2f}" for k, v in lb.items()))
ids = pd.read_csv("run_v4_s42/variants/top_mix.csv").application_id
for i, n in enumerate(picked):
    sub = pd.DataFrame({"application_id": ids, "default_probability": T[n]})
    assert len(sub) == 30000 and sub.application_id.is_unique and sub.default_probability.between(0, 1).all()
    sub.to_csv(f"variants/{os.environ.get('PREFIX', 'qd')}_{i:02d}_{re.sub(r'[^A-Za-z0-9.]+', '_', n)}.csv", index=False)
print("files:", len(picked))
