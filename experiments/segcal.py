import numpy as np, pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
res = pd.read_pickle("run_fac/models/predictions.pkl")
f = pd.read_parquet("features.parquet"); tr = f[f.target.notna()].reset_index(drop=True); y = tr.target.astype(int).values
rank = lambda v: rankdata(v) / len(v)
def trees(t): return (np.mean([rank(res[f"lgb_{t}_{s}"][0]) for s in range(3)], 0) + rank(res[f"cat_{t}"][0]) + rank(res[f"xgb_{t}"][0])) / 3
core = (trees("full") + trees("lean") + rank(res["cat_full"][0]) + rank(res["cat_lean"][0])) / 4
p = 0.8 * (core + trees("top")) / 2 + 0.2 * rank(res["linear"][0])
ks = lambda q: ks_2samp(q[y == 1], q[y == 0]).statistic
print("top_mix oof auc", round(roc_auc_score(y, p), 5), "ks", round(ks(p), 5))
cal = IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(p)
# KS threshold in calibrated probability
thr = np.sort(np.unique(p)); best = max(((np.mean(p[y == 1] > t) - np.mean(p[y == 0] > t)), t) for t in np.quantile(p, np.linspace(.3, .95, 300)))
t = best[1]; print("KS threshold at rank", round(t, 3), "calibrated p", round(cal[np.argmin(abs(p - t))], 3), "share above", round((p > t).mean(), 3))
seg = {"new customer": tr.rh_n.isna(), "rate grade": tr.interest_rate.round(2), "no account data": tr.ab_months.isna(),
       "few ec items": pd.qcut(tr.ec_orders, 4, labels=False, duplicates="drop"), "bureau reports": pd.qcut(tr.bu_reports.fillna(0), 3, labels=False, duplicates="drop")}
for name, s in seg.items():
    d = pd.DataFrame({"s": s.values, "y": y, "cal": cal, "p": p})
    g = d.groupby("s").agg(n=("y", "size"), actual=("y", "mean"), predicted=("cal", "mean"))
    g["gap"] = g.predicted - g.actual
    g["auc_within"] = d.groupby("s").apply(lambda x: roc_auc_score(x.y, x.p) if x.y.nunique() > 1 else np.nan)
    g["near_thr"] = d.assign(nt=(abs(d.p - t) < 0.08)).groupby("s").nt.mean()
    print("\n" + name); print(g.round(4).to_string())
# CV test: recalibrate each segment (new vs existing x rate grade) with isotonic fitted on other folds
key = tr.rh_n.isna().astype(str) + "_" + tr.interest_rate.round(2).astype(str)
out = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=11).split(p, y):
    glob = IsotonicRegression(out_of_bounds="clip").fit(p[a], y[a])
    out[b] = glob.predict(p[b])
    for k in key.unique():
        ia, ib = a[key.values[a] == k], b[key.values[b] == k]
        if len(ib) and y[ia].sum() > 30:
            out[ib] = IsotonicRegression(out_of_bounds="clip").fit(p[ia], y[ia]).predict(p[ib])
out = out + 1e-9 * p  # keep ties in the original order
print("\nsegment recalibrated auc", round(roc_auc_score(y, out), 5), "ks", round(ks(out), 5), "| original", round(roc_auc_score(y, p), 5), round(ks(p), 5))
