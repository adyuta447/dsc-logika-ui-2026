import numpy as np, pandas as pd, sys
from scipy.optimize import minimize
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R = sys.argv[1]
res = pd.read_pickle(R + "/models/predictions.pkl")
y = pd.read_csv("data/train_applications.csv").pipe(lambda d: d)  # placeholder, replaced below
f = pd.read_parquet("features.parquet", columns=["application_id", "target"]); y = f[f.target.notna()].target.astype(int).values
rank = lambda v: rankdata(v) / len(v)
cat = lambda t: [k for k in res if k.startswith(f"cat_{t}")]
xg = lambda t: [k for k in res if k.startswith(f"xgb_{t}")]
comp = {}
for t in ["full", "lean", "top"]:
    comp[f"lgb_{t}"] = np.mean([rank(res[f"lgb_{t}_{s}"][0]) for s in range(3)], 0), np.mean([rank(res[f"lgb_{t}_{s}"][1]) for s in range(3)], 0)
    comp[f"cat_{t}"] = np.mean([rank(res[k][0]) for k in cat(t)], 0), np.mean([rank(res[k][1]) for k in cat(t)], 0)
    comp[f"xgb_{t}"] = np.mean([rank(res[k][0]) for k in xg(t)], 0), np.mean([rank(res[k][1]) for k in xg(t)], 0)
comp["linear"] = rank(res["linear"][0]), rank(res["linear"][1])
names = list(comp); O = np.column_stack([comp[n][0] for n in names]); T = np.column_stack([comp[n][1] for n in names])
ks = lambda yy, p: ks_2samp(p[yy == 1], p[yy == 0]).statistic
# current top_mix weights expressed on the same components
tm = dict(lgb_full=1/30, cat_full=1/30+1/20, xgb_full=1/30, lgb_lean=1/30, cat_lean=1/30+1/20, xgb_lean=1/30, lgb_top=1/15, cat_top=1/15, xgb_top=1/15, linear=0.2)
w_tm = np.array([tm[n] * (0.8 if n != "linear" else 1) for n in names]); w_tm[names.index("linear")] = 0.2
def fit(idx):
    obj = lambda w: -roc_auc_score(y[idx], O[idx] @ (np.abs(w) / np.abs(w).sum()))
    return np.abs(minimize(obj, w_tm, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-4, fatol=1e-7)).x)
# nested check: weights fitted on half the rows are scored on the other half
sk = StratifiedKFold(2, shuffle=True, random_state=7)
for a, b in sk.split(O, y):
    w = fit(a); w /= w.sum()
    print("half: fitted", round(roc_auc_score(y[b], O[b] @ w), 5), round(ks(y[b], O[b] @ w), 5), " top_mix", round(roc_auc_score(y[b], O[b] @ w_tm), 5), round(ks(y[b], O[b] @ w_tm), 5))
w = fit(np.arange(len(y))); w /= w.sum()
print(pd.Series(w, index=names).round(3).to_string())
print("full: fitted", round(roc_auc_score(y, O @ w), 5), round(ks(y, O @ w), 5), " top_mix", round(roc_auc_score(y, O @ w_tm), 5), round(ks(y, O @ w_tm), 5))
pd.to_pickle((names, w, T @ w), R + "/models/blendopt.pkl")
