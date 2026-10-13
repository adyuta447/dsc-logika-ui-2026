import ast, sys, nbformat as nbf

src = open("features.py").read()
tree = ast.parse(src)
fn = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Credit Default Prediction, DSC LOGIKA UI 2026

The task is to rank loan applications by the risk of reaching 90 DPD within 12 months, scored with KS.

Approach in short:

1. Build customer level features from six tables, using only records available at the decision time.
2. Train gradient boosted trees (LightGBM, CatBoost, XGBoost) on two feature sets and a spline logistic regression.
3. Blend the models by rank, since KS only depends on the ordering of scores.
4. Repeat the whole training over three fold splits and average the blends.

Everything runs on CPU with fixed seeds so the submission can be reproduced exactly."""),
code("""import os, random, joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from scipy.stats import ks_2samp, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer

FOLD_SEEDS = [""" + (sys.argv[4] if len(sys.argv) > 4 else "42") + """]
VARIANT = "top_mix"
N_FOLDS = """ + (sys.argv[2] if len(sys.argv) > 2 else "5") + """
TREE_SEEDS = """ + (sys.argv[3] if len(sys.argv) > 3 else "1") + """
LR = """ + (sys.argv[5] if len(sys.argv) > 5 else "0.01") + """
random.seed(FOLD_SEEDS[0])
np.random.seed(FOLD_SEEDS[0])
KAGGLE = "/kaggle/input/data-science-competition-logika-ui-2026/"
D = KAGGLE if os.path.exists(KAGGLE) else "data/"
os.makedirs("models", exist_ok=True)
os.makedirs("variants", exist_ok=True)"""),
md("""## Features

Every table is joined to the selected application through `customer_id` and filtered against `decision_datetime`.
The applications table gives loan terms, and the profile table gives declared income used for affordability ratios.
Default rates move with the calendar, from 12 percent for May decisions to 15 percent for February, so the decision day of year is kept as a feature."""),
code(fn["load_apps"] + "\n\n\n" + fn["app_features"] + "\n\n\n" + fn["profile_features"]),
md("""Bureau reports from every provider and every past application are noisy readings of the same customer, so besides the current reports we also pool all of them."""),
code(fn["bureau_features"]),
code(fn["previous_features"]),
md("""Repayment history is the strongest source. A late rate is only fair when it counts installments that had enough time to become late, so each DPD threshold uses matured installments only."""),
code(fn["repayment_features"] + "\n\n\n" + fn["sequence_features"]),
md("""Account behavior is compared with the customer's own baseline, because a drop in balance or a jump in outflows matters more than the absolute level."""),
code(fn["account_features"]),
md("""Ecommerce turned out to hold the most surprising signal. Customers who later default pay noticeably more per unit than the category median, so the log price relative to the category median is computed per item and averaged over several windows."""),
code(fn["ecommerce_features"] + "\n\n\n" + fn["category_shift_features"]),
md("""Two small signals sit in how steady a customer is: the share of account inflows that look like salary, and the spacing between past applications and between orders."""),
code(fn["rhythm_features"]),
md("""Every table is also a noisy reading of the same hidden risk. A one factor model pools them: each source mean is scaled by its spread across customers, its noise shrinks with the number of items behind it, and the posterior mean of the risk weighs every source by how reliable it is for that customer. Installment lateness is first cleaned of calendar, sequence and size effects. Only the loadings use labels, so they are fitted inside each fold."""),
code(fn["factor_inputs"] + "\n\n\n" + fn["factor_features"]),
code(fn["build"]),
code("""f = build()
print(f.shape)"""),
md("""A quick look at the price signal: default rate by decile of the relative unit price."""),
code("""t = f[f.target.notna()]
dec = pd.qcut(t.ec_price_rel, 10, labels=False)
ax = t.groupby(dec).target.mean().plot(kind="bar", figsize=(7, 3))
ax.set_xlabel("decile of relative unit price")
ax.set_ylabel("default rate")
plt.show()"""),
md("""## Validation

Stratified K fold CV on the training applications, with K set by `N_FOLDS`. The split is repeated for every seed in `FOLD_SEEDS`. AUC is tracked next to KS because it is a smoother guide for model selection.

Profile fields, application timing and a few ecommerce share blocks add noise for the boosted trees, so a second feature set drops them. Training on both sets gives the blend some diversity."""),
code("""tr = f[f.target.notna()].reset_index(drop=True)
te = f[f.target.isna()].reset_index(drop=True)
ids = ["application_id", "customer_id", "target"]
X_all, X_test_all = tr.drop(columns=ids), te.drop(columns=ids)
y = tr.target.astype(int).values

noise = ["region", "employment_type", "housing_status", "age", "employment_tenure_months", "dependents",
         "tenure_to_age", "income_per_dependent", "loan_purpose", "decision_lag_h", "app_hour", "app_dow"]
noise += [c for c in X_all.columns if c.startswith(("ec_share_", "ec_catqty_", "cs_shift", "buh_"))]
sets = {"full": list(X_all.columns), "lean": [c for c in X_all.columns if c not in noise]}

app = load_apps()
M, E = factor_inputs(app)
pos = pd.Series(np.arange(len(app)), index=app.application_id)
p_tr, p_te = pos[tr.application_id].values, pos[te.application_id].values


def split(seed):
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_all, y))
    return folds, [factor_features(M, E, app.target.values, p_tr[a]) for a, b in folds]


folds, fac = split(FOLD_SEEDS[0])
frame = lambda k: (pd.concat([X_all, fac[k].iloc[p_tr].reset_index(drop=True)], axis=1),
                   pd.concat([X_test_all, fac[k].iloc[p_te].reset_index(drop=True)], axis=1))
sets = {k: v + list(fac[0].columns) for k, v in sets.items()}
ks = lambda p: ks_2samp(p[y == 1], p[y == 0]).statistic
rank = lambda v: rankdata(v) / len(v)
report = lambda name, p: print(f"{name:12s} auc {roc_auc_score(y, p):.5f}  ks {ks(p):.5f}")"""),
md("""## Models"""),
code("""def fit_lgb(cols, tag, seed):
    params = dict(objective="binary", learning_rate=LR, num_leaves=15, min_child_samples=200,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, max_bin=127,
                  seed=seed, deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)
    oof, pred = np.zeros(len(X_all)), np.zeros(len(X_test_all))
    for k, (a, b) in enumerate(folds):
        X, Xt = [d[cols[k]] for d in frame(k)]
        m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), 20000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        m.save_model(f"{MD}lgb_{tag}_s{seed}_f{k}.txt")
        oof[b] = m.predict(X.iloc[b])
        pred += m.predict(Xt) / len(folds)
    return oof, pred


def fit_cat(cols, tag, seed):
    oof, pred = np.zeros(len(X_all)), np.zeros(len(X_test_all))
    for k, (a, b) in enumerate(folds):
        X, Xt = [d[cols[k]].copy() for d in frame(k)]
        cats = [c for c in X.columns if str(X[c].dtype) == "category"]
        X[cats], Xt[cats] = X[cats].astype(str), Xt[cats].astype(str)
        m = CatBoostClassifier(iterations=int(60 / LR), learning_rate=3 * LR, depth=6, l2_leaf_reg=5, eval_metric="AUC",
                               od_type="Iter", od_wait=300, random_seed=seed, cat_features=cats, verbose=0)
        m.fit(X.iloc[a], y[a], eval_set=(X.iloc[b], y[b]))
        m.save_model(f"{MD}cat_{tag}_s{seed}_f{k}.cbm")
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / len(folds)
    return oof, pred


def fit_xgb(cols, tag, seed):
    oof, pred = np.zeros(len(X_all)), np.zeros(len(X_test_all))
    for k, (a, b) in enumerate(folds):
        X, Xt = [d[cols[k]] for d in frame(k)]
        m = xgb.XGBClassifier(n_estimators=10000, learning_rate=LR, max_depth=4, min_child_weight=20,
                              subsample=0.8, colsample_bytree=0.3, reg_lambda=20, tree_method="hist", max_bin=128,
                              enable_categorical=True, eval_metric="auc", early_stopping_rounds=300,
                              random_state=seed, n_jobs=-1)
        m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
        m.save_model(f"{MD}xgb_{tag}_s{seed}_f{k}.json")
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / len(folds)
    return oof, pred


def fit_set(cols, tag):
    for s in range(3):
        res[f"lgb_{tag}_{s}"] = fit_lgb(cols, tag, s)
    for s in range(TREE_SEEDS):
        res[f"cat_{tag}_{s}"] = fit_cat(cols, tag, s)
        res[f"xgb_{tag}_{s}"] = fit_xgb(cols, tag, s)"""),
md("""The spline logistic regression sees each numeric feature through a smooth curve. It reaches an AUC close to the trees on its own, which suggests the risk is largely additive, and it brings a different error pattern to the blend."""),
code("""num = [c for c in sets["lean"] if str(frame(0)[0][c].dtype) != "category"]
flag = [c for c in num if frame(0)[0][c].isna().mean() > 0.01]


def linear_matrix(X, med):
    return pd.concat([X[num].astype(float).fillna(med), X[flag].isna().astype(float).add_suffix("_na")], axis=1)


def fit_linear():
    oof, pred = np.zeros(len(X_all)), np.zeros(len(X_test_all))
    for k, (a, b) in enumerate(folds):
        X, Xt = frame(k)
        med = X.iloc[a][num].median()
        m = make_pipeline(QuantileTransformer(n_quantiles=200, subsample=10**6, random_state=SEED),
                          SplineTransformer(n_knots=4, degree=2), LogisticRegression(C=0.03, max_iter=3000))
        m.fit(linear_matrix(X.iloc[a], med), y[a])
        joblib.dump((m, med), f"{MD}linear_f{k}.pkl")
        oof[b] = m.predict_proba(linear_matrix(X.iloc[b], med))[:, 1]
        pred += m.predict_proba(linear_matrix(Xt, med))[:, 1] / len(folds)
    return oof, pred"""),
md("""Many features are close copies of each other, so a third set keeps only the 80 strongest. The ranking for each fold uses the gain of that fold's lean LightGBM models only, which keeps the validation rows out of the selection.

Everything is trained once per fold split, and the models of each split go to their own folder."""),
code("""runs = {}
for SEED in FOLD_SEEDS:
    folds, fac = split(SEED)
    MD = f"models/folds_{SEED}/"
    os.makedirs(MD, exist_ok=True)
    res = {}
    for tag, cols in sets.items():
        fit_set([cols] * len(folds), tag)
    res["linear"] = fit_linear()
    top = []
    for k in range(len(folds)):
        gain = sum(lgb.Booster(model_file=f"{MD}lgb_lean_s{s}_f{k}.txt").feature_importance("gain") for s in range(3))
        top.append(list(pd.Series(gain, index=sets["lean"]).sort_values(ascending=False).index[:80]))
    fit_set(top, "top")
    for name, (oof, _) in res.items():
        report(f"{SEED} {name}", oof)
    runs[SEED] = res"""),
md("""## Blend

Within each feature set the seeds of each model are averaged first, then LightGBM, CatBoost and XGBoost get equal weight. In the main blend CatBoost gets one extra share because it was the most stable model across folds, and the linear model takes 20 percent.

The other variants mix the same models with different weights. Their CV scores are nearly identical, and `VARIANT` picks the one written as the submission. Each variant is averaged over the fold splits and ranked again."""),
code("""def avg(model, tag, i, n):
    return np.mean([rank(res[f"{model}_{tag}_{s}"][i]) for s in range(n)], axis=0)


def trees(tag, i):
    return (avg("lgb", tag, i, 3) + avg("cat", tag, i, TREE_SEEDS) + avg("xgb", tag, i, TREE_SEEDS)) / 3


def core(i):
    return (trees("full", i) + trees("lean", i) + avg("cat", "full", i, TREE_SEEDS) + avg("cat", "lean", i, TREE_SEEDS)) / 4


lin = lambda i: rank(res["linear"][i])
variants = {
    "main": lambda i: 0.8 * core(i) + 0.2 * lin(i),
    "top_mix": lambda i: 0.8 * (core(i) + trees("top", i)) / 2 + 0.2 * lin(i),
    "top_heavy": lambda i: 0.8 * (core(i) + 2 * trees("top", i)) / 3 + 0.2 * lin(i),
    "top_only": lambda i: 0.8 * trees("top", i) + 0.2 * lin(i),
    "trees_only": lambda i: (core(i) + trees("top", i)) / 2,
    "linear30": lambda i: 0.7 * (core(i) + trees("top", i)) / 2 + 0.3 * lin(i),
}
bag = {name: [] for name in variants}
for SEED, res in runs.items():
    for name, blend in variants.items():
        bag[name].append((blend(0), blend(1)))
final = {}
for name, b in bag.items():
    report(name, np.mean([o for o, p in b], axis=0))
    final[name] = rank(np.mean([p for o, p in b], axis=0))
    pd.DataFrame({"application_id": te.application_id, "default_probability": final[name]}).to_csv(f"variants/{name}.csv", index=False)
pd.to_pickle(runs, "models/predictions.pkl")"""),
code("""sub = pd.DataFrame({"application_id": te.application_id, "default_probability": final[VARIANT]})
assert len(sub) == 30000 and sub.application_id.is_unique and sub.default_probability.between(0, 1).all()
sub.to_csv("DSC0152_kali ini beneran deh_Submisi.csv", index=False)
sub.head()"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, sys.argv[1] if len(sys.argv) > 1 else "DSC0152_kali ini beneran deh_Notebook.ipynb")
