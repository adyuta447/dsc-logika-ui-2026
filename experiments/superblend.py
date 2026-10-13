import glob, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
# Does a weighted blend of every model beat bag v4 top_mix? Weights are fitted on one half of the rows and scored on the other.
S = sys.argv[1]
y = pd.read_parquet("features.parquet", columns=["target"]).target.dropna().astype(int).values
rank = lambda v: rankdata(v) / len(v)
ks = lambda p, t: ks_2samp(p[t == 1], p[t == 0]).statistic
def top_mix(r):
    res = pd.read_pickle(f"{r}/models/predictions.pkl")
    pick = lambda pre: np.mean([rank(res[k][0]) for k in res if k.startswith(pre)], 0)
    c = {f"{m}_{t}": pick(f"{m}_{t}") for m in ("lgb", "cat", "xgb") for t in ("full", "lean", "top")}
    tr = {t: (c[f"lgb_{t}"] + c[f"cat_{t}"] + c[f"xgb_{t}"]) / 3 for t in ("full", "lean", "top")}
    core = (tr["full"] + tr["lean"] + c["cat_full"] + c["cat_lean"]) / 4
    return 0.8 * (core + tr["top"]) / 2 + 0.2 * rank(res["linear"][0])
C = {"top_mix": rank(np.mean([top_mix(r) for r in ("run_v4_s42", "run_v4_s7", "run_v4_s13")], 0))}
for name in sorted({p.split("/")[-1].rsplit("_s", 1)[0] for p in glob.glob(S + "div/*.npz")}):
    z = sorted(glob.glob(S + f"div/{name}_s*.npz"))
    if len(z) == 3: C[name] = rank(np.mean([rank(np.load(p)["oof"]) for p in z], 0))
names = list(C); X = np.column_stack([C[n] for n in names])
print({n: round(roc_auc_score(y, C[n]), 5) for n in names})
strong = [n for n in names if n != "top_mix" and roc_auc_score(y, C[n]) > 0.8465]
fixed = {"top_mix": C["top_mix"], "half lgb_extra": (C["top_mix"] + C["lgb_extra"]) / 2,
         "half strong new": (C["top_mix"] + np.mean([C[n] for n in strong], 0)) / 2,
         "third lgb_extra xgb_deep": (C["top_mix"] + C["lgb_extra"] + C["xgb_deep"]) / 3}
rows = []
for rep in range(10):
    for a, b in StratifiedKFold(2, shuffle=True, random_state=rep).split(X, y):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(np.log(X[a] / (1 - X[a] + 1e-9) + 1e-9), y[a])
        p = m.decision_function(np.log(X[b] / (1 - X[b] + 1e-9) + 1e-9))
        r = {"stack": (roc_auc_score(y[b], p), ks(p, y[b]))}
        r.update({k: (roc_auc_score(y[b], v[b]), ks(v[b], y[b])) for k, v in fixed.items()})
        rows.append({f"{k} {i}": v[i] for k, v in r.items() for i in (0, 1)})
t = pd.DataFrame(rows)
for k in ["stack"] + list(fixed)[1:]:
    da, dk = t[f"{k} 0"] - t["top_mix 0"], t[f"{k} 1"] - t["top_mix 1"]
    print(f"{k:26s} auc gain {da.mean():+.5f} (wins {(da > 0).mean():.2f})  ks gain {dk.mean():+.5f} (wins {(dk > 0).mean():.2f})")
print("strong new types:", strong)
