import os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import build, load_apps, factor_inputs, factor_features

# Extra model types on the v4 features, each over the three fold splits, kept for the public split harvest.
OUT = sys.argv[1] + "div/"; os.makedirs(OUT, exist_ok=True)
f = build(); tr = f[f.target.notna()].reset_index(drop=True); te = f[f.target.isna()].reset_index(drop=True)
y = tr.target.astype(int).values
noise = ["region", "employment_type", "housing_status", "age", "employment_tenure_months", "dependents",
         "tenure_to_age", "income_per_dependent", "loan_purpose", "decision_lag_h", "app_hour", "app_dow"]
noise += [c for c in tr.columns if c.startswith(("ec_share_", "ec_catqty_", "cs_shift", "buh_"))]
cols = [c for c in tr.columns if c not in noise + ["application_id", "customer_id", "target"]]
app = load_apps(); M, E = factor_inputs(app)
pos = pd.Series(np.arange(len(app)), index=app.application_id)
p_tr, p_te = pos[tr.application_id].values, pos[te.application_id].values

def lgb_fit(extra):
    def fit(X, Xt, a, b):
        p = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=200, feature_fraction=0.3, bagging_fraction=0.8,
                 bagging_freq=1, lambda_l2=20, max_bin=127, seed=0, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
        p.update(extra)
        m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), 10000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict(X.iloc[b]), m.predict(Xt)
    return fit

def xgb_fit(X, Xt, a, b):
    m = xgb.XGBClassifier(n_estimators=10000, learning_rate=0.02, max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.3,
                          reg_lambda=20, tree_method="hist", max_bin=128, enable_categorical=True, eval_metric="auc", early_stopping_rounds=200,
                          random_state=0, n_jobs=-1)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
    return m.predict_proba(X.iloc[b])[:, 1], m.predict_proba(Xt)[:, 1]

def xgb_shallow(X, Xt, a, b):
    m = xgb.XGBClassifier(n_estimators=10000, learning_rate=0.02, max_depth=3, min_child_weight=20, subsample=0.8, colsample_bytree=0.3,
                          reg_lambda=20, tree_method="hist", max_bin=128, enable_categorical=True, eval_metric="auc", early_stopping_rounds=200,
                          random_state=0, n_jobs=-1)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
    return m.predict_proba(X.iloc[b])[:, 1], m.predict_proba(Xt)[:, 1]

def cat_lossguide(X, Xt, a, b):
    X, Xt = X.copy(), Xt.copy(); cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    X[cats], Xt[cats] = X[cats].astype(str), Xt[cats].astype(str)
    m = CatBoostClassifier(iterations=4000, learning_rate=0.06, grow_policy="Lossguide", max_leaves=31, min_data_in_leaf=100, l2_leaf_reg=10,
                           eval_metric="AUC", od_type="Iter", od_wait=200, random_seed=0, cat_features=cats, verbose=0, thread_count=-1)
    m.fit(X.iloc[a], y[a], eval_set=(X.iloc[b], y[b]))
    return m.predict_proba(X.iloc[b])[:, 1], m.predict_proba(Xt)[:, 1]

def full_set(fit):
    # same LightGBM, but the profile and timing fields that the lean set drops are put back
    def g(X, Xt, a, b):
        extra = [c for c in noise if c in tr.columns]
        return fit(pd.concat([X, tr[extra]], axis=1), pd.concat([Xt, te[extra]], axis=1), a, b)
    return g

def cat_fit(X, Xt, a, b):
    X, Xt = X.copy(), Xt.copy(); cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    X[cats], Xt[cats] = X[cats].astype(str), Xt[cats].astype(str)
    m = CatBoostClassifier(iterations=4000, learning_rate=0.06, depth=8, l2_leaf_reg=10, eval_metric="AUC", od_type="Iter", od_wait=200,
                           random_seed=0, cat_features=cats, verbose=0, thread_count=-1)
    m.fit(X.iloc[a], y[a], eval_set=(X.iloc[b], y[b]))
    return m.predict_proba(X.iloc[b])[:, 1], m.predict_proba(Xt)[:, 1]

configs = {"lgb_deep": lgb_fit(dict(num_leaves=63, min_child_samples=50)), "lgb_shallow": lgb_fit(dict(num_leaves=7, min_child_samples=300)),
           "lgb_extra": lgb_fit(dict(extra_trees=True)), "lgb_ff06": lgb_fit(dict(feature_fraction=0.6)), "xgb_deep": xgb_fit, "cat_deep": cat_fit,
           "lgb_extra31": lgb_fit(dict(extra_trees=True, num_leaves=31, min_child_samples=100)), "lgb_lintree": lgb_fit(dict(linear_tree=True, linear_lambda=1.0)),
           "lgb_ff015": lgb_fit(dict(feature_fraction=0.15)), "lgb_mc800": lgb_fit(dict(min_child_samples=800)),
           "lgb_fullset": full_set(lgb_fit({})), "lgb_goss": lgb_fit(dict(data_sample_strategy="goss", bagging_fraction=1.0, bagging_freq=0)),
           "xgb_shallow": xgb_shallow, "cat_lossguide": cat_lossguide}
# third pass: the full feature set did well on the public split, so the strongest model types are repeated on it
configs.update({f"full_{k}": full_set(configs[k]) for k in ("lgb_extra", "xgb_deep", "lgb_shallow", "lgb_ff06", "lgb_extra31", "lgb_mc800", "xgb_shallow", "cat_deep")})
for seed in (42, 7, 13):
    folds = list(StratifiedKFold(10, shuffle=True, random_state=seed).split(tr, y))
    frames = []
    for a, b in folds:
        F = factor_features(M, E, app.target.values, p_tr[a])
        frames.append((pd.concat([tr[cols], F.iloc[p_tr].reset_index(drop=True)], axis=1), pd.concat([te[cols], F.iloc[p_te].reset_index(drop=True)], axis=1)))
    for name, fit in configs.items():
        fn = f"{OUT}{name}_s{seed}.npz"
        if os.path.exists(fn): continue
        t0 = time.time(); oof, pred = np.zeros(len(tr)), np.zeros(len(te))
        try:
            for k, (a, b) in enumerate(folds):
                X, Xt = frames[k]; o, p = fit(X, Xt, a, b); oof[b] = o; pred += p / len(folds)
        except Exception as e:
            print(f"{name} seed {seed} failed: {e}", flush=True); continue
        np.savez(fn, oof=oof, pred=pred)
        print(f"{name} seed {seed} auc {roc_auc_score(y, oof):.5f} minutes {(time.time() - t0) / 60:.1f}", flush=True)
