import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
# Does a plain rank average of bag v4 top_mix with the best of the many new models (passes 3 to 10) beat bag v4 top_mix on CV?
# K and the weight are picked on one half of the train rows and scored on the other half, both ways.
S = sys.argv[1]
src = open(S + "harvest_kf.py").read(); exec(src[:src.index("# lean set plus one group")])
def ks_cols(Ps, ys):
    o = np.argsort(-Ps, axis=0); yy = ys[o]
    return np.max(np.cumsum(yy, 0) / ys.sum() - np.cumsum(1 - yy, 0) / (len(ys) - ys.sum()), 0)
new = [n for n in O if n.startswith("new|")]
auc = pd.Series({n: roc_auc_score(y, O[n]) for n in new}).sort_values(ascending=False)
v4 = O["v4|top_mix"]; print("bag v4 top_mix AUC", round(roc_auc_score(y, v4), 5), "KS", round(ks_cols(v4[:, None], y)[0], 5))
print("best new models:", auc.head(8).round(5).to_dict())
half = np.random.default_rng(0).random(len(y)) < 0.5
def blend(K, w):
    return w * v4 + (1 - w) * np.mean([O[n] for n in auc.index[:K]], 0)
grid = [(K, w) for K in (3, 5, 10, 20, 40, 80) for w in (0.0, 0.3, 0.5, 0.7)]
for a, b in ((half, ~half), (~half, half)):
    best = max(grid, key=lambda g: roc_auc_score(y[a], blend(*g)[a]))
    p = blend(*best)
    print(f"picked K {best[0]} w {best[1]} on one half; other half AUC {roc_auc_score(y[b], p[b]):.5f} vs bag v4 {roc_auc_score(y[b], v4[b]):.5f},",
          f"KS {ks_cols(p[b][:, None], y[b])[0]:.5f} vs {ks_cols(v4[b][:, None], y[b])[0]:.5f}")
for K, w in ((10, 0.5), (20, 0.5), (40, 0.3)):
    p = blend(K, w); print(f"K {K} w {w}: all rows AUC {roc_auc_score(y, p):.5f} KS {ks_cols(p[:, None], y)[0]:.5f}")
