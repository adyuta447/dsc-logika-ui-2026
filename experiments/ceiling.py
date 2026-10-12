import numpy as np, pandas as pd
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score
from features import load_apps, factor_inputs, factor_features
res = pd.read_pickle("run_fac/models/predictions.pkl")
f = pd.read_parquet("features.parquet", columns=["application_id", "target"]); tr = f[f.target.notna()].reset_index(drop=True); y = tr.target.astype(int).values
rank = lambda v: rankdata(v) / len(v)
def trees(t): return (np.mean([rank(res[f"lgb_{t}_{s}"][0]) for s in range(3)], 0) + rank(res[f"cat_{t}"][0]) + rank(res[f"xgb_{t}"][0])) / 3
p = 0.8 * ((trees("full") + trees("lean") + rank(res["cat_full"][0]) + rank(res["cat_lean"][0])) / 4 + trees("top")) / 2 + 0.2 * rank(res["linear"][0])
app = load_apps(); M, E = factor_inputs(app)
pos = pd.Series(np.arange(len(app)), index=app.application_id)[tr.application_id].values
F = factor_features(M, E, app.target.values, pos).iloc[pos].reset_index(drop=True)
d = pd.DataFrame({"y": y, "p": p, "sd": F.factor_sd.values})
d["dec"] = pd.qcut(d.sd, 10, labels=False)
ks = lambda x: ks_2samp(x.p[x.y == 1], x.p[x.y == 0]).statistic
g = d.groupby("dec").apply(lambda x: pd.Series({"posterior_sd": x.sd.mean(), "n": len(x), "default": x.y.mean(), "auc": roc_auc_score(x.y, x.p), "ks": ks(x)}))
print(g.round(4).to_string())
# extrapolate auc to posterior sd 0 with a straight line over the deciles
c = np.polyfit(g.posterior_sd ** 2, g.auc, 1); print("auc extrapolated to perfectly measured risk", round(np.polyval(c, 0), 4), "| now", round(roc_auc_score(y, p), 4))
