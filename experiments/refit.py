import os, joblib, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer
from features import build

SEED, K, M = 42, 10, "run10/models/"
OUT = "run_full/"
os.makedirs(OUT + "models", exist_ok=True); os.makedirs(OUT + "variants", exist_ok=True)
f = build()
tr = f[f.target.notna()].reset_index(drop=True); te = f[f.target.isna()].reset_index(drop=True)
ids = ["application_id", "customer_id", "target"]
X_all, X_test_all = tr.drop(columns=ids), te.drop(columns=ids)
y = tr.target.astype(int).values
noise = ["region", "employment_type", "housing_status", "age", "employment_tenure_months", "dependents",
         "tenure_to_age", "income_per_dependent", "loan_purpose", "decision_lag_h", "app_hour", "app_dow"]
noise += [c for c in X_all.columns if c.startswith(("ec_share_", "ec_catqty_", "cs_shift", "buh_"))]
sets = {"full": list(X_all.columns), "lean": [c for c in X_all.columns if c not in noise]}
rank = lambda v: rankdata(v) / len(v)
# 10 fold models saw 90 percent of rows, so the full fit gets 10 percent more rounds
grow = lambda its: int(round(np.mean(its) * 1.1))


def fit_lgb(cols, tag, seed):
    n = grow([lgb.Booster(model_file=f"{M}lgb_{tag}_s{seed}_f{k}.txt").num_trees() for k in range(K)])
    params = dict(objective="binary", learning_rate=0.01, num_leaves=15, min_child_samples=200,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, max_bin=127,
                  seed=seed, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
    m = lgb.train(params, lgb.Dataset(X_all[cols], y), n)
    m.save_model(f"{OUT}models/lgb_{tag}_s{seed}.txt")
    return m.predict(X_test_all[cols])


def fit_cat(cols, tag):
    n = grow([CatBoostClassifier().load_model(f"{M}cat_{tag}_f{k}.cbm").tree_count_ for k in range(K)])
    X, Xt = X_all[cols].copy(), X_test_all[cols].copy()
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    X[cats], Xt[cats] = X[cats].astype(str), Xt[cats].astype(str)
    m = CatBoostClassifier(iterations=n, learning_rate=0.03, depth=6, l2_leaf_reg=5, random_seed=0, cat_features=cats, verbose=0)
    m.fit(X, y)
    m.save_model(f"{OUT}models/cat_{tag}.cbm")
    return m.predict_proba(Xt)[:, 1]


def fit_xgb(cols, tag):
    its = []
    for k in range(K):
        b = xgb.XGBClassifier(); b.load_model(f"{M}xgb_{tag}_f{k}.json"); its.append(b.best_iteration + 1)
    m = xgb.XGBClassifier(n_estimators=grow(its), learning_rate=0.01, max_depth=4, min_child_weight=20,
                          subsample=0.8, colsample_bytree=0.3, reg_lambda=20, tree_method="hist", max_bin=128,
                          enable_categorical=True, random_state=0, n_jobs=-1)
    m.fit(X_all[cols], y)
    m.save_model(f"{OUT}models/xgb_{tag}.json")
    return m.predict_proba(X_test_all[cols])[:, 1]


res = {}
def fit_set(cols, tag):
    for s in range(3):
        res[f"lgb_{tag}_{s}"] = fit_lgb(cols, tag, s); print(tag, "lgb", s, flush=True)
    res[f"cat_{tag}"] = fit_cat(cols, tag); print(tag, "cat", flush=True)
    res[f"xgb_{tag}"] = fit_xgb(cols, tag); print(tag, "xgb", flush=True)


for tag, cols in sets.items():
    fit_set(cols, tag)
num = [c for c in sets["lean"] if str(X_all[c].dtype) != "category"]
flag = [c for c in num if X_all[c].isna().mean() > 0.01]
med = X_all[num].median()
lm = lambda X: pd.concat([X[num].astype(float).fillna(med), X[flag].isna().astype(float).add_suffix("_na")], axis=1)
m = make_pipeline(QuantileTransformer(n_quantiles=200, subsample=10**6, random_state=SEED),
                  SplineTransformer(n_knots=4, degree=2), LogisticRegression(C=0.03, max_iter=3000))
m.fit(lm(X_all), y); joblib.dump((m, med), f"{OUT}models/linear.pkl")
res["linear"] = m.predict_proba(lm(X_test_all))[:, 1]
gain = sum(lgb.Booster(model_file=f"{OUT}models/lgb_lean_s{s}.txt").feature_importance("gain") for s in range(3))
top = list(pd.Series(gain, index=sets["lean"]).sort_values(ascending=False).index[:80])
# the top set iteration counts come from the 10 fold top models, which used per fold lists of the same size
fit_set(top, "top")


def trees(tag):
    lg = np.mean([rank(res[f"lgb_{tag}_{s}"]) for s in range(3)], axis=0)
    return (lg + rank(res[f"cat_{tag}"]) + rank(res[f"xgb_{tag}"])) / 3


core = (trees("full") + trees("lean") + rank(res["cat_full"]) + rank(res["cat_lean"])) / 4
lin, tp = rank(res["linear"]), trees("top")
variants = {"main": 0.8 * core + 0.2 * lin, "top_mix": 0.8 * (core + tp) / 2 + 0.2 * lin,
            "top_heavy": 0.8 * (core + 2 * tp) / 3 + 0.2 * lin, "top_only": 0.8 * tp + 0.2 * lin,
            "trees_only": (core + tp) / 2, "linear30": 0.7 * (core + tp) / 2 + 0.3 * lin}
for name, p in variants.items():
    sub = pd.DataFrame({"application_id": te.application_id, "default_probability": p})
    assert len(sub) == 30000 and sub.application_id.is_unique and sub.default_probability.between(0, 1).all()
    sub.to_csv(f"{OUT}variants/{name}.csv", index=False)
    ref = pd.read_csv(f"run10/variants/{name}.csv")
    assert (ref.application_id.values == sub.application_id.values).all()
    print(name, "spearman vs 10 fold", round(spearmanr(p, ref.default_probability).correlation, 5))
pd.to_pickle(res, f"{OUT}models/predictions.pkl")
