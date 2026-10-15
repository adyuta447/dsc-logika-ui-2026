import sys
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import build, load_apps, factor_inputs, factor_features

# KS is decided by the order of the rows near its threshold. A second model trained only on the rows of bag v4 top_mix that sit in a
# band around that threshold (with the v4 score as one input) re-ranks them; rows outside the band keep their order. Honest 10 fold.
S = sys.argv[1]
src = open(S + "harvest_kf.py").read(); exec(src[src.index("rank = lambda"):src.index("# fac3 uses")])
v4 = rank(np.mean([variants(r, 0)["top_mix"] for r in ["run_v4_s42", "run_v4_s7", "run_v4_s13"]], 0))
f = build(); tr = f[f.target.notna()].reset_index(drop=True); y = tr.target.astype(int).values
assert len(v4) == len(y) and roc_auc_score(y, v4) > 0.847, "row order differs from the bag v4 OOF"
noise = ["region", "employment_type", "housing_status", "age", "employment_tenure_months", "dependents",
         "tenure_to_age", "income_per_dependent", "loan_purpose", "decision_lag_h", "app_hour", "app_dow"]
noise += [c for c in tr.columns if c.startswith(("ec_share_", "ec_catqty_", "cs_shift", "buh_"))]
cols = [c for c in tr.columns if c not in noise + ["application_id", "customer_id", "target"]]
app = load_apps(); M, E = factor_inputs(app)
p_tr = pd.Series(np.arange(len(app)), index=app.application_id)[tr.application_id].values

def ks(p, yy):
    o = np.argsort(-p); t = yy[o]
    return np.max(np.cumsum(t) / t.sum() - np.cumsum(1 - t) / (len(t) - t.sum()))
o = np.argsort(-v4); t = y[o]; d = np.cumsum(t) / t.sum() - np.cumsum(1 - t) / (len(t) - t.sum())
thr = 1 - (np.argmax(d) + 1) / len(y)  # v4 rank at the KS threshold
print(f"bag v4 top_mix AUC {roc_auc_score(y, v4):.5f} KS {ks(v4, y):.5f}, threshold at rank {thr:.3f}")
p = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=20, max_bin=127, seed=0, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(tr, y))
for width in (0.10, 0.20, 0.30):
    band = np.abs(v4 - thr) < width; s2 = np.zeros(len(y))
    for a, b in folds:
        F = factor_features(M, E, app.target.values, p_tr[a]).iloc[p_tr].reset_index(drop=True)
        X = pd.concat([tr[cols], F], axis=1).assign(v4=v4)
        a, b = a[band[a]], b[band[b]]
        av, af = a[::10], np.setdiff1d(a, a[::10])  # early stopping on held back training rows, never on the scored fold
        m = lgb.train(p, lgb.Dataset(X.iloc[af], y[af]), 3000, valid_sets=[lgb.Dataset(X.iloc[av], y[av])], callbacks=[lgb.early_stopping(200, verbose=False)])
        s2[b] = m.predict(X.iloc[b])
    for mixw in (1.0, 0.5):
        new = v4.copy(); inner = rankdata(mixw * rankdata(s2[band]) + (1 - mixw) * rankdata(v4[band]))
        new[band] = np.sort(v4[band])[(inner - 1).astype(int)]
        print(f"band {width:.2f} ({band.mean():.0%} of rows) weight {mixw}: AUC {roc_auc_score(y, new):.5f} KS {ks(new, y):.5f}"
              f"  | halves KS {ks(new[::2], y[::2]):.5f} vs {ks(v4[::2], y[::2]):.5f}, {ks(new[1::2], y[1::2]):.5f} vs {ks(v4[1::2], y[1::2]):.5f}", flush=True)
