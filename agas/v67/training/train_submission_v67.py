"""v67: stacking of proven pipelines.

v66's best public result is 0.56424 (`submission_v66_04_ext0.5_tm30_lgb_lean.csv`),
built from one pipeline trained harder. The remaining gap to the leaderboard top is
large, and a single pipeline is unlikely to close it alone. This run stacks the
out-of-fold predictions of every proven pipeline (v61..v66) into one meta model.

Leakage control: base predictions are already out-of-fold, the meta model is
trained only on those OOF vectors, and the test vectors are produced by base
models that never saw the test labels. No external data, no AutoML, no AutoFE.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "agas/shared"))
sys.path.insert(0, str(HERE.parents[2]))
os.chdir(HERE.parents[2])

import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE.parent / "output"

LEG_NAMES = {
    "v61": ["lgb_full", "lgb_lean", "lgbx_full", "cat_full", "xgb_full", "linear", "fac_full"],
    "v62": ["lgb_full", "lgb_fac", "lgb_lean", "lgb_faclean", "lgb_top", "lgb_topwide", "lgbx_full", "cat_full", "xgb_full", "linear"],
    "v63": ["sel100", "sel200", "sel200b", "sel300", "sel400", "lgb_full", "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"],
    "v64": ["sel200", "sel400", "lgb_full", "lgb_faclean", "lgbx_full", "cat_full", "xgb_full", "linear"],
    "v65": ["sel", "lean", "xgb", "cat"],
    "v66": ["fac_full", "lgb_full", "lgb_lean", "lgbx_full", "cat_full", "xgb_full", "linear"],
}


def r01(v):
    return rankdata(v) / len(v)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z, list(z["train_ids"]), list(z["test_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="v67")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log_file = OUT / f"train_log_{args.out_prefix}.txt"
    if log_file.exists():
        log_file.unlink()
    t0 = time.time()

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(log_file, "a") as fh:
            fh.write(line + "\n")

    # labels
    tr = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"])
    te = pd.read_csv(ROOT / "data/test_applications.csv", usecols=["application_id"])
    y = tr.set_index("application_id")

    # load all stored predictions
    preds = {}
    for v in ["v61", "v62", "v63", "v64", "v65", "v66"]:
        p = ROOT / f"agas/{v}/output/preds_{v}.npz"
        z, tr_ids, te_ids = load_npz(p)
        preds[v] = (z, tr_ids, te_ids)
        log(f"loaded {v}: {len(z.files)} arrays")

    # reference ids from v66
    ref_tr = list(preds["v66"][1])
    ref_te = list(preds["v66"][2])
    n_tr = len(ref_tr)
    n_te = len(ref_te)

    # build OOF/test matrices
    oof_cols = []
    test_cols = []
    names = []

    def add(name, oof, test):
        oof_cols.append(np.asarray(oof, dtype=float).ravel())
        test_cols.append(np.asarray(test, dtype=float).ravel())
        names.append(name)

    for v in ["v61", "v62", "v63", "v64", "v65", "v66"]:
        z, tr_ids, te_ids = preds[v]
        assert list(tr_ids) == ref_tr and list(te_ids) == ref_te, v
        prefix = "oof_" if v != "v65" else "oof_"
        # v65 uses oof_sel etc, others use oof_<leg>
        if v == "v65":
            for leg in ["sel", "lean", "xgb", "cat"]:
                add(f"{v}_{leg}", z[f"oof_{leg}"], z[f"test_{leg}"])
        else:
            for leg in LEG_NAMES[v]:
                key = f"oof_{leg}"
                tkey = f"test_{leg}"
                if key in z.files:
                    add(f"{v}_{leg}", z[key], z[tkey])

    # derived v66 blends as extra meta features
    z66 = preds["v66"][0]
    oof_map = {n: oof_cols[names.index(n)] for n in names}
    test_map = {n: test_cols[names.index(n)] for n in names}
    # v66 legs
    lgb_full = oof_map["v66_lgb_full"]; lgb_lean = oof_map["v66_lgb_lean"]
    cat_full = oof_map["v66_cat_full"]; xgb_full = oof_map["v66_xgb_full"]
    lgbx_full = oof_map["v66_lgbx_full"]; linear = oof_map["v66_linear"]
    r_trees4 = r01(lgb_full) + r01(lgb_lean) + r01(cat_full) + r01(xgb_full)
    r_trees4 = r_trees4 / 4.0
    r_lgbx = r01(lgbx_full); r_linear = r01(linear)
    r_core = (2 * r_trees4 + r_lgbx) / 3.0
    r_tm20 = 0.8 * r_core + 0.2 * r_linear
    r_tm30 = 0.7 * r_core + 0.3 * r_linear
    add("v66_tm20", r_tm20, None)
    # test derived
    t_lgb_full = test_map["v66_lgb_full"]; t_lgb_lean = test_map["v66_lgb_lean"]
    t_cat_full = test_map["v66_cat_full"]; t_xgb_full = test_map["v66_xgb_full"]
    t_lgbx = test_map["v66_lgbx_full"]; t_linear = test_map["v66_linear"]
    t_trees4 = (r01(t_lgb_full)+r01(t_lgb_lean)+r01(t_cat_full)+r01(t_xgb_full))/4.0
    t_core = (2*t_trees4 + r01(t_lgbx))/3.0
    t_tm20 = 0.8*t_core + 0.2*r01(t_linear)
    t_tm30 = 0.7*t_core + 0.3*r01(t_linear)
    # fix last added test placeholder
    test_cols[-1] = t_tm20
    # add tm30
    add("v66_tm30", r_tm30, t_tm30)

    X = np.column_stack(oof_cols)
    Xte = np.column_stack(test_cols)
    log(f"meta feature matrix: {X.shape}, test {Xte.shape}, names={len(names)}")

    ytr = y.loc[ref_tr, "target"].values.astype(int)

    # rank-transform features for the meta model
    Xr = np.column_stack([r01(c) for c in oof_cols])
    Xter = np.column_stack([r01(c) for c in test_cols])

    # 5-fold CV for honest estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_logit = np.zeros(n_tr)
    oof_lgb = np.zeros(n_tr)
    for fold, (a, b) in enumerate(skf.split(Xr, ytr)):
        m = LogisticRegression(C=0.5, max_iter=5000, solver="lbfgs")
        m.fit(Xr[a], ytr[a])
        oof_logit[b] = m.predict_proba(Xr[b])[:, 1]

        ml = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                 min_child_samples=300, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=10, random_state=2026, n_jobs=-1, verbose=-1)
        ml.fit(Xr[a], ytr[a])
        oof_lgb[b] = ml.predict_proba(Xr[b])[:, 1]
        log(f"fold {fold} logit auc={roc_auc_score(ytr[b], oof_logit[b]):.6f}")

    log(f"CV logit AUC={roc_auc_score(ytr, oof_logit):.6f} KS={np.max([0])}")
    from scipy.stats import ks_2samp
    log(f"CV logit KS={ks_2samp(oof_logit[ytr==1], oof_logit[ytr==0]).statistic:.6f}")
    log(f"CV lgb   AUC={roc_auc_score(ytr, oof_lgb):.6f} KS={ks_2samp(oof_lgb[ytr==1], oof_lgb[ytr==0]).statistic:.6f}")

    # final models on all OOF
    final_logit = LogisticRegression(C=0.5, max_iter=5000, solver="lbfgs")
    final_logit.fit(Xr, ytr)
    test_logit = final_logit.predict_proba(Xter)[:, 1]

    final_lgb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                    min_child_samples=300, subsample=0.8, colsample_bytree=0.8,
                                    reg_lambda=10, random_state=2026, n_jobs=-1, verbose=-1)
    final_lgb.fit(Xr, ytr)
    test_lgb = final_lgb.predict_proba(Xter)[:, 1]

    # anchor = v66_04 vector (ext0.5 tm30 lean), already a proven public vector
    anchor_oof = r01(lgb_full)  # placeholder replaced below
    # build ext0.5 tm30 lean oof/test from v66 legs
    anchor_oof = r01(r_tm30 + 0.5 * (r_tm30 - r01(lgb_lean)))  # note: r_tm30 already rank, lgb_lean rank
    # Actually ext formula uses rank(tm30 + 0.5*(tm30 - rank(lgb_lean))) in raw prob space.
    # Rebuild from raw legs to stay consistent with make_submission_v66.
    # (use stored v66 raw legs)
    raw_tm30 = r_tm30  # already rank scale; ext on rank scale is fine
    anchor_oof = r01(raw_tm30 + 0.5 * (raw_tm30 - r01(lgb_lean)))
    anchor_test = r01(t_tm30 + 0.5 * (t_tm30 - r01(t_lgb_lean)))

    # save
    np.savez_compressed(OUT / f"preds_{args.out_prefix}.npz",
                        train_ids=np.array(ref_tr), test_ids=np.array(ref_te),
                        y=ytr,
                        oof_meta_logit=oof_logit,
                        test_meta_logit=test_logit,
                        oof_meta_lgb=oof_lgb,
                        test_meta_lgb=test_lgb,
                        oof_anchor=anchor_oof,
                        test_anchor=anchor_test,
                        **{f"oof_{n}": oof_cols[i] for i, n in enumerate(names)},
                        **{f"test_{n}": test_cols[i] for i, n in enumerate(names)})

    summary = {
        "n_features": X.shape[1],
        "feature_names": names,
        "cv_logit_auc": float(roc_auc_score(ytr, oof_logit)),
        "cv_logit_ks": float(ks_2samp(oof_logit[ytr == 1], oof_logit[ytr == 0]).statistic),
        "cv_lgb_auc": float(roc_auc_score(ytr, oof_lgb)),
        "cv_lgb_ks": float(ks_2samp(oof_lgb[ytr == 1], oof_lgb[ytr == 0]).statistic),
        "minutes": (time.time() - t0) / 60.0,
    }
    (OUT / f"summary_{args.out_prefix}.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {(time.time()-t0)/60:.1f} min; features={X.shape[1]}")


if __name__ == "__main__":
    main()
