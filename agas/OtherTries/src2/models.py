"""Ensemble training for the 90+ DPD model, scored by KS."""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from scipy.stats import ks_2samp, rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer

SEED = 42
N_FOLDS = 5

LGB_BASE = dict(objective="binary", metric="auc", learning_rate=0.01, num_leaves=15,
                min_data_in_leaf=200, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=20, max_bin=127, verbose=-1, deterministic=True, force_row_wise=True)
LGB_ET = dict(LGB_BASE, extra_trees=True, seed=1)
LGB_RND = dict(LGB_BASE, seed=0)

XGB_BASE = dict(max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.3,
                reg_lambda=20, tree_method="hist", max_bin=128, eval_metric="auc",
                n_jobs=-1, random_state=0, enable_categorical=True)

CAT_BASE = dict(iterations=6000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                eval_metric="AUC", od_type="Iter", od_wait=300, verbose=0)


def ks_score(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def rankify(v):
    return rankdata(v) / len(v)


def make_folds(X, y, n_folds=N_FOLDS, seed=SEED):
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y))


def fit_lgb(Xtr, ytr, Xva, yva, Xte, params, cat_cols=()):
    ds = lgb.Dataset(Xtr, ytr, categorical_feature=list(cat_cols))
    dv = lgb.Dataset(Xva, yva, reference=ds)
    m = lgb.train(params, ds, 12000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(400, verbose=False)])
    return (m.predict(Xva, num_iteration=m.best_iteration),
            m.predict(Xte, num_iteration=m.best_iteration), m)


def fit_xgb(Xtr, ytr, Xva, yva, Xte, params):
    p = dict(objective="binary:logistic", **params)
    dtr, dva, dte = xgb.DMatrix(Xtr, ytr), xgb.DMatrix(Xva, yva), xgb.DMatrix(Xte)
    m = xgb.train(p, dtr, 12000, evals=[(dva, "va")], early_stopping_rounds=400, verbose_eval=False)
    best = m.best_iteration + 1
    return m.predict(dva, iteration_range=(0, best)), m.predict(dte, iteration_range=(0, best)), m


def fit_cat(Xtr, ytr, Xva, yva, Xte, params, cat_cols=()):
    cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in cats:
        Xtr[c], Xva[c], Xte[c] = Xtr[c].astype(str), Xva[c].astype(str), Xte[c].astype(str)
    m = CatBoostClassifier(**dict(params, random_seed=0), cat_features=cats)
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m


def linear_matrix(X, med, num, flags):
    vals = X[num].astype(float)
    return pd.concat([vals.fillna(med), vals.isna().astype(float).add_suffix("_na")], axis=1)


def fit_linear(Xtr, ytr, Xva, Xte, num, flags):
    med = Xtr[num].median()
    m = make_pipeline(QuantileTransformer(n_quantiles=200, subsample=10 ** 6, random_state=SEED),
                      SimpleImputer(strategy="median"),
                      SplineTransformer(n_knots=4, degree=2),
                      LogisticRegression(C=0.03, max_iter=3000))
    m.fit(linear_matrix(Xtr, med, num, flags), ytr)
    return m.predict_proba(linear_matrix(Xva, med, num, flags))[:, 1], \
        m.predict_proba(linear_matrix(Xte, med, num, flags))[:, 1], m