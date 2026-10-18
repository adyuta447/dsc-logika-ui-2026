"""Does self-training on the test set actually help, measured without cheating?

Pseudo-labelling is the standard transductive move: train once, predict the
unlabelled test rows, attach hard labels to the rows the model is most confident
about, then retrain with those rows appended. It uses only data the committee
provided, so it is legal, and it often helps because it sharpens the decision
boundary in regions where the labelled set is thin.

Measuring it honestly is the hard part. The naive version scores the second
model on rows it just trained on, which always looks good. So here the
pseudo-labels are drawn from the test set only, and the score is computed on the
held-out validation fold, which never enters training in either pass:

    fold-train  ->  model A  ->  pseudo-label the test rows
    fold-train + pseudo-test  ->  model B  ->  score fold-val

Model B is only allowed to be compared against model A on the same rows, so the
comparison is clean. Both models use the same columns, chosen on fold-train.

Confidence is taken as a quantile cut on the predicted probability rather than
an absolute threshold, so the fraction of pseudo-labelled rows is fixed even
though the probability scale differs between folds.
"""

import json
import os
import statistics
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv_harness as H  # noqa: E402

LR, L2, LEAVES, FF, MCS, BAG, MAXBIN = 0.01, 5.0, 15, 0.3, 200, 0.8, 127
FOLDS, FOLD_SEEDS, TOP_N, SEED = 5, [42, 7, 13], 250, 0


def params():
    return dict(objective="binary", learning_rate=LR, num_leaves=LEAVES,
                min_child_samples=MCS, feature_fraction=FF, bagging_fraction=BAG,
                bagging_freq=1, lambda_l2=L2, max_bin=MAXBIN, seed=SEED,
                deterministic=True, force_row_wise=True, n_jobs=-1, verbose=-1)


def fit(Xa, ya, Xb, yb, rounds=8000):
    m = lgb.train(params(), lgb.Dataset(Xa, ya), num_boost_round=rounds,
                  valid_sets=[lgb.Dataset(Xb, yb)],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m


def main():
    if os.environ.get("V65_NO_STUDY"):
        print("V65_NO_STUDY set, skipping the pseudo-labelling study")
        return
    f, tr, te = H.load_frames(True)
    y = tr.target.astype(int).values
    X = tr.drop(columns=["application_id", "customer_id", "target"])
    Xte = te.drop(columns=["application_id", "customer_id", "target"])
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
            Xte[c] = Xte[c].astype("category")

    import features as ft

    app = ft.load_apps()
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    res = {"frac": frac, "weight": weight, "runs": []}
    for fs in FOLD_SEEDS:
        folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=fs).split(X, y))
        fac, p_tr, p_te, fac_cols = H.factor_block(tr, te, folds, app, 1)
        base_tr = X[[c for c in X.columns if c not in fac_cols]]
        base_te = Xte[[c for c in Xte.columns if c not in fac_cols]]
        Xa = pd.concat([base_tr, fac[0].iloc[p_tr].reset_index(drop=True)], axis=1)
        Xt = pd.concat([base_te, fac[0].iloc[p_te].reset_index(drop=True)], axis=1)

        oof_a = np.zeros(len(y))
        oof_b = np.zeros(len(y))
        for a, b in folds:
            probe = lgb.train({**params(), "feature_fraction": 1.0, "num_boost_round": 300,
                               "learning_rate": 0.1},
                              lgb.Dataset(Xa.iloc[a], y[a]), num_boost_round=300)
            g = pd.Series(probe.feature_importance(importance_type="gain"), index=Xa.columns)
            cols = list(g.sort_values(ascending=False).index[:TOP_N])

            m1 = fit(Xa.iloc[a][cols], y[a], Xa.iloc[b][cols], y[b])
            oof_a[b] = m1.predict(Xa.iloc[b][cols])
            pt = m1.predict(Xt[cols])

            lo, hi = np.quantile(pt, frac), np.quantile(pt, 1 - frac)
            keep = (pt <= lo) | (pt >= hi)
            pseudo = pt[keep] >= hi
            # the test rows are appended as extra rows; their weights are scaled
            # down because a hard label on a prediction is weaker evidence than
            # a real observation
            Xp = pd.concat([Xa.iloc[a][cols], Xt[cols].iloc[keep]], ignore_index=True)
            yp = np.concatenate([y[a], pseudo.astype(int)])
            w = np.concatenate([np.ones(len(a)), np.full(int(keep.sum()), weight)])

            dtr = lgb.Dataset(Xp, yp, weight=w)
            dva = lgb.Dataset(Xa.iloc[b][cols], y[b])
            m2 = lgb.train(params(), dtr, num_boost_round=m1.best_iteration,
                           valid_sets=[dva], callbacks=[lgb.early_stopping(300, verbose=False)])
            oof_b[b] = m2.predict(Xa.iloc[b][cols])

        run = {
            "fold_seed": fs,
            "auc_a": float(roc_auc_score(y, oof_a)),
            "auc_b": float(roc_auc_score(y, oof_b)),
            "ks_a": float(ks_2samp(oof_a[y == 1], oof_a[y == 0]).statistic),
            "ks_b": float(ks_2samp(oof_b[y == 1], oof_b[y == 0]).statistic),
            "n_pseudo": int(keep.sum()),
        }
        res["runs"].append(run)
        print(json.dumps(run), flush=True)

    for m in ("auc", "ks"):
        res[f"{m}_a"] = statistics.mean(r[f"{m}_a"] for r in res["runs"])
        res[f"{m}_b"] = statistics.mean(r[f"{m}_b"] for r in res["runs"])
        res[f"{m}_sd_b"] = statistics.pstdev([r[f"{m}_b"] for r in res["runs"]])
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2), flush=True)

    out = ROOT / "agas/v65/output"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"pseudo_frac{frac}_w{weight}.json").write_text(json.dumps(res, indent=2))
    print(f"auc {res['auc_a']:.6f} -> {res['auc_b']:.6f}   "
          f"ks {res['ks_a']:.6f} -> {res['ks_b']:.6f}", flush=True)


if __name__ == "__main__":
    main()