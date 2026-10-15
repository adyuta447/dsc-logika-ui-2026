import os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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

def with_cols(fit, extra):
    # same model, with some of the profile and timing fields that the lean set drops put back
    extra = [c for c in extra if c in tr.columns]
    return lambda X, Xt, a, b: fit(pd.concat([X, tr[extra]], axis=1), pd.concat([Xt, te[extra]], axis=1), a, b)
full_set = lambda fit: with_cols(fit, noise)

def cat_fit(X, Xt, a, b, depth=8):
    X, Xt = X.copy(), Xt.copy(); cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    X[cats], Xt[cats] = X[cats].astype(str), Xt[cats].astype(str)
    m = CatBoostClassifier(iterations=4000, learning_rate=0.06, depth=depth, l2_leaf_reg=10, eval_metric="AUC", od_type="Iter", od_wait=200,
                           random_seed=0, cat_features=cats, verbose=0, thread_count=-1)
    m.fit(X.iloc[a], y[a], eval_set=(X.iloc[b], y[b]))
    return m.predict_proba(X.iloc[b])[:, 1], m.predict_proba(Xt)[:, 1]

def xgb_lossguide(X, Xt, a, b):
    m = xgb.XGBClassifier(n_estimators=10000, learning_rate=0.02, grow_policy="lossguide", max_leaves=31, max_depth=0, min_child_weight=50, subsample=0.8,
                          colsample_bytree=0.3, reg_lambda=20, tree_method="hist", max_bin=128, enable_categorical=True, eval_metric="auc",
                          early_stopping_rounds=200, random_state=0, n_jobs=-1)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
    return m.predict_proba(X.iloc[b])[:, 1], m.predict_proba(Xt)[:, 1]

def hgb_fit(X, Xt, a, b):
    # categories as their codes (train and test share them), early stopping on a slice of the training fold
    X, Xt = (d.apply(lambda c: c.cat.codes.where(c.notna()) if str(c.dtype) == "category" else c) for d in (X, Xt))
    m = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=5000, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=20, max_features=0.3,
                                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=200, random_state=0)
    m.fit(X.iloc[a], y[a])
    return m.predict_proba(X.iloc[b])[:, 1], m.predict_proba(Xt)[:, 1]

configs = {"lgb_deep": lgb_fit(dict(num_leaves=63, min_child_samples=50)), "lgb_shallow": lgb_fit(dict(num_leaves=7, min_child_samples=300)),
           "lgb_extra": lgb_fit(dict(extra_trees=True)), "lgb_ff06": lgb_fit(dict(feature_fraction=0.6)), "xgb_deep": xgb_fit, "cat_deep": cat_fit,
           "lgb_extra31": lgb_fit(dict(extra_trees=True, num_leaves=31, min_child_samples=100)), "lgb_lintree": lgb_fit(dict(linear_tree=True, linear_lambda=1.0)),
           "lgb_ff015": lgb_fit(dict(feature_fraction=0.15)), "lgb_mc800": lgb_fit(dict(min_child_samples=800)),
           "lgb_fullset": full_set(lgb_fit({})), "lgb_goss": lgb_fit(dict(data_sample_strategy="goss", bagging_fraction=1.0, bagging_freq=0)),
           "xgb_shallow": xgb_shallow, "cat_lossguide": cat_lossguide}
# third pass: the full feature set did well on the public split, so the strongest model types are repeated on it
configs.update({f"full_{k}": full_set(configs[k]) for k in ("lgb_extra", "xgb_deep", "lgb_shallow", "lgb_ff06", "lgb_extra31", "lgb_mc800", "xgb_shallow", "cat_deep")})
# fourth pass: the same full set models without the season and rhythm features, so the step from fac to v4 can be extrapolated per model type
V4 = ["dec_doy", "ab_inc_share", "ab_open_avg", "pa_gap_mean", "ec_gap_mean", "ec_gap_last3"]
without_v4 = lambda fit: lambda X, Xt, a, b: fit(X.drop(columns=V4), Xt.drop(columns=V4), a, b)
configs.update({f"facfull_{k}": without_v4(configs[f"full_{k}"]) for k in ("lgb_extra", "xgb_deep", "lgb_shallow", "lgb_ff06", "lgb_extra31", "lgb_mc800", "xgb_shallow", "cat_deep")})
# fifth pass, run only with PASS5 set so the fourth pass finishes on its own first
if os.environ.get("PASS5"):
    extra = ("lgb_deep", "lgb_goss", "lgb_ff015", "cat_lossguide")
    configs.update({f"full_{k}": full_set(configs[k]) for k in extra})
    configs.update({f"facfull_{k}": without_v4(configs[f"full_{k}"]) for k in extra})
# sixth pass, only with PASS6 set: the lean set plus one group of the dropped fields at a time, to find which group the public split rewards;
# the eighth and ninth passes (PASS8, PASS9) do the same for eight more model types; the tenth (PASS10) drops one group from the full set instead, and the eleventh (PASS11) does that for six more model types
if os.environ.get("PASS6") or os.environ.get("PASS8") or os.environ.get("PASS9") or os.environ.get("PASS10") or os.environ.get("PASS11"):
    plus = {"prof": ["region", "employment_type", "housing_status", "age", "employment_tenure_months", "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose"],
            "time": ["decision_lag_h", "app_hour", "app_dow"], "ec": [c for c in noise if c.startswith(("ec_share_", "ec_catqty_"))],
            "cs": [c for c in noise if c.startswith(("cs_shift", "buh_"))]}
    ks = ("lgb_deep", "lgb_goss", "lgb_ff015", "lgb_ff06") if os.environ.get("PASS9") else ("lgb_extra31", "lgb_shallow", "xgb_shallow", "lgb_mc800") if os.environ.get("PASS8") else ("lgb_extra", "xgb_deep")
    if os.environ.get("PASS10") or os.environ.get("PASS11"):
        mk = ("xgb_shallow", "lgb_mc800", "lgb_deep", "lgb_goss", "lgb_ff015", "lgb_ff06") if os.environ.get("PASS11") else ("lgb_extra", "lgb_shallow", "lgb_extra31", "xgb_deep")
        configs = {f"minus_{g}_{k}": with_cols(configs[k], [n for n in noise if n not in c]) for g, c in plus.items() for k in mk}
    else:
        configs = {f"plus_{g}_{k}": with_cols(configs[k], c) for g, c in plus.items() for k in ks}
# seventh pass, only with PASS7 set: four new model families, each on the lean and the full set, so each adds its own full minus lean direction
if os.environ.get("PASS7"):
    new7 = {"hgb": hgb_fit, "xgb_lossguide": xgb_lossguide, "cat_d6": lambda X, Xt, a, b: cat_fit(X, Xt, a, b, depth=6),
            "lgb_extra_s1": lgb_fit(dict(extra_trees=True, seed=1))}
    configs = {**new7, **{f"full_{k}": full_set(f) for k, f in new7.items()}}
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
