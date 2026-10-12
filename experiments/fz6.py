import pandas as pd, numpy as np, lightgbm as lgb, sys
sys.path.insert(0, sys.argv[1])
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, rankdata
from features import load_apps
from features import factor_inputs, factor_features
app = load_apps(); M, E = factor_inputs(app); ya = app.target.values
f = pd.read_parquet('features.parquet')
tr = f[f.target.notna()].reset_index(drop=True); y = tr.target.astype(int).values
pos = pd.Series(np.arange(len(app)), index=app.application_id)[tr.application_id].values
assert (ya[pos] == y).all()
noise = ["region","employment_type","housing_status","age","employment_tenure_months","dependents","tenure_to_age","income_per_dependent","loan_purpose","decision_lag_h","app_hour","app_dow"]
base = [c for c in tr.columns if c not in noise+['application_id','customer_id','target'] and not c.startswith(('ec_share_','ec_catqty_','cs_shift','buh_'))]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
FF = [factor_features(M, E, ya, pos[a]).iloc[pos].reset_index(drop=True) for a, b in folds]
print('fold scale check', [round(roc_auc_score(y[b], FF[k].factor_z.values[b]), 4) for k, (a, b) in enumerate(folds)], flush=True)
def run(seed, extra, mono=None):
    p = dict(objective="binary",learning_rate=0.01,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,max_bin=127,seed=seed,deterministic=True,force_row_wise=True,n_jobs=-1,verbose=-1)
    p.update(extra)
    oof = np.zeros(len(tr))
    for k, (a, b) in enumerate(folds):
        X = pd.concat([tr[base], FF[k]], axis=1)
        if mono: p['monotone_constraints'] = [mono.get(c, 0) for c in X.columns]; p['monotone_constraints_method'] = 'advanced'
        m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), 10000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[b] = m.predict(X.iloc[b])
    return oof
ks = lambda p: ks_2samp(p[y==1], p[y==0]).statistic
mono = {"interest_rate": 1, "factor_z": 1, "fs_price": 1, "fs_late": 1, "fs_ldpd": 1, "fs_bu_z": -1, "ec_price_rel": 1,
        "bua_score": -1, "rh_rate1_smooth": 1, "rh_rate15_smooth": 1, "rh_rate30_smooth": 1, "bu_score_mean": -1}
for name, extra, mo in [("monotone", {}, mono), ("leaves31", dict(num_leaves=31), None), ("minchild100", dict(min_child_samples=100), None),
                        ("ff0.5", dict(feature_fraction=0.5), None)]:
    o = [run(s, extra, mo) for s in (0, 1)]
    m = np.mean([rankdata(v) for v in o], 0)
    print(name, "seeds", [round(roc_auc_score(y, v), 5) for v in o], "| 2 seed auc", round(roc_auc_score(y, m), 5), "ks", round(ks(m), 5), "(factor v1 base 0.84524 / 0.52203)", flush=True)
