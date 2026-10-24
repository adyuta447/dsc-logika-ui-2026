"""Assemble the v71 feature cache.

Three additions on top of the v70 cache:

1. `panel_score` — cross-fitted historical panel prediction.
2. A compact, domain-selected subset of the 375 event-time panel features.
3. `rf_*` — target-independent auxiliary rate-tier model.  The model predicts the
   observed underwriting grade from every feature except the current rate and
   the features directly derived from it.  The residual is the part of the
   lender's grade not explained by the observable risk profile.

No current target is used by the panel score or the rate model.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = HERE.parent / "output"

PANEL_SELECTED = [
    # underwriting trajectory
    "pn_pa_n", "pn_pa_appr_rate", "pn_pa_reject_rate", "pn_pa_rate_mean",
    "pn_pa_rate_max", "pn_pa_rate_trend", "pn_pa_last_rate_minus_cur",
    "pn_pa_max_rate_minus_cur", "pn_pa_appr_over_req",
    "pn_pa_last_appr_ratio_minus_cur", "pn_pa_rej90_rate", "pn_pa_rej365_rate",
    "pn_pa_active_inst", "pn_pa_req_to_income", "pn_dti_total",
    # repayment and prior-loan outcomes
    "pn_rh_late_share", "pn_rh_late90_share", "pn_rh_dpd_max",
    "pn_rh_unpaid_share", "pn_rh_delay_mean", "pn_rh_delay_trend",
    "pn_rh_late_trend", "pn_rh_unpaid_trend", "pn_rh_pay_rate",
    "pn_lg_bad_share", "pn_lg_bad_due_share", "pn_lg_worst_dpd",
    "pn_lg_last_late", "pn_lg_last_dpd", "pn_lg_active_late",
    "pn_lg_active_inst", "pn_lg_fully_paid_share",
    # bureau disagreement and trend
    "pn_bu_score_mean", "pn_bu_score_std", "pn_bu_provider_spread",
    "pn_bu_util_mean", "pn_bu_inq_max", "pn_bu_delinq_max",
    "pn_bu_delinq_sum", "pn_buh_score_trend", "pn_bu_unscored_rate",
    # cash-flow shape
    "pn_ab_income_trend_3_12", "pn_ab_net_trend_3_12",
    "pn_ab_bal_trend_3_12", "pn_ab_low_trend_3_12", "pn_ab_low_share",
    "pn_ab_neg_share", "pn_ab_income_to_declared", "pn_ab_net_to_income",
    # e-commerce shape
    "pn_ec_spend_trend", "pn_ec_orders_trend", "pn_ec_basket_trend",
    "pn_ec_cancel_trend", "pn_ec_spend_to_income", "pn_ec_cancel_rate",
    "pn_ec_cod_rate", "pn_ec_cc_rate", "pn_ec_need_share",
    "pn_ec_basket_cv", "pn_ec_cc_rate_minus_cod",
    # interactions
    "pn_rate_x_dpd", "pn_rate_x_late90", "pn_rate_x_pa_rej",
    "pn_loan_to_income_x_dpd", "pn_late_x_net", "pn_late_x_low",
    "pn_bad_due_x_rate", "pn_bu_delinq_x_util",
]

RATE_VALUES = np.array([0.16, 0.20, 0.24, 0.28, 0.32], dtype=float)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def add_panel_features(base):
    log("adding panel features")
    panel = pd.read_pickle(OUT / "panel_dataset.pkl")
    cur = panel[panel.is_hist == 0].copy()
    cur = cur.rename(columns={"own_application_id": "application_id"})
    z = np.load(OUT / "preds_panel_v71.npz", allow_pickle=True)
    ps = pd.DataFrame({
        "application_id": z["current_application_id"],
        "panel_score": z["current_score"],
        "panel_fold": z["current_cust_fold"],
    })
    cur = cur.merge(ps, on="application_id", how="left", validate="one_to_one")
    assert cur.application_id.is_unique
    assert len(cur) == 100000
    assert cur.panel_score.notna().all()

    cols = ["application_id", "panel_score"] + [c for c in PANEL_SELECTED if c in cur.columns]
    missing = [c for c in PANEL_SELECTED if c not in cur.columns]
    if missing:
        log(f"  WARNING missing panel columns: {missing}")
    add = cur[cols].copy()
    f = base.merge(add, on="application_id", how="left", validate="one_to_one")
    assert len(f) == len(base)
    assert f.application_id.is_unique
    assert f.panel_score.notna().all()
    log(f"  added {len(cols) - 1} panel columns")
    return f, set(cols[1:])


def build_rate_features(base):
    log("building auxiliary rate model")
    ids = ["application_id", "customer_id", "target"]
    tr = base[base.target.notna()].reset_index(drop=True)
    te = base[base.target.isna()].reset_index(drop=True)
    exclude = set(ids) | {
        "interest_rate", "rate_vs_history",
        # the current installment is a deterministic function of amount, rate and
        # term; leaving it in makes the auxiliary rate task trivially perfect and
        # the residual exactly zero
        "installment", "inst_to_income", "inst_to_net",
        "cx_inst_vs_hist_max", "cx_inst_vs_hist_mean", "cx_remaining_vs_inst",
        "af_inst_to_declared", "af_inst_to_observed", "af_inst_to_best_estimate",
        "af_total_inst_to_income", "ix_inst_x_late", "ix_inst_x_ec_spend",
        "gn_int_inst_to_income", "gn_emp_inst_to_income", "gn_reg_inst_to_income",
    }
    # never feed the current rate or its direct derivative to the auxiliary model
    predictors = [c for c in base.columns if c not in exclude
                  and not c.startswith("rf_")]
    # panel score/features are target-independent and allowed, but the rate model
    # should be a pure feature-map model; keep the v70 columns only
    predictors = [c for c in predictors if not c.startswith("panel_")
                  and not c.startswith("pn_")]
    for c in predictors:
        if str(base[c].dtype) == "object":
            tr[c] = tr[c].astype("category")
            te[c] = te[c].astype("category")

    y_rate = tr.interest_rate.round(2).values
    classes = np.array(sorted(np.unique(y_rate)))
    assert np.allclose(classes, RATE_VALUES), classes
    y_cls = np.searchsorted(classes, y_rate)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(tr, y_cls))
    oof_prob = np.zeros((len(tr), len(classes)))
    test_prob = np.zeros((len(te), len(classes)))
    for k, (a, b) in enumerate(folds):
        params = dict(objective="multiclass", num_class=len(classes),
                      learning_rate=0.03, num_leaves=15, min_child_samples=200,
                      feature_fraction=0.45, bagging_fraction=0.80, bagging_freq=1,
                      lambda_l2=20.0, max_bin=127, extra_trees=True,
                      deterministic=True, force_row_wise=True,
                      n_jobs=-1, verbose=-1, seed=k)
        m = lgb.train(params, lgb.Dataset(tr.iloc[a][predictors], y_cls[a]),
                      num_boost_round=1200,
                      valid_sets=[lgb.Dataset(tr.iloc[b][predictors], y_cls[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_prob[b] = m.predict(tr.iloc[b][predictors])
        test_prob += m.predict(te[predictors]) / len(folds)
        log(f"  rate fold {k} best_iter={m.best_iteration}")

    def frame(prob, rate):
        pred_rate = prob @ classes
        actual = np.searchsorted(classes, rate.round(2).values)
        p_actual = prob[np.arange(len(prob)), actual]
        out = pd.DataFrame({
            "rf_pred_rate": pred_rate,
            "rf_rate_residual": rate.values - pred_rate,
            "rf_rate_abs_residual": np.abs(rate.values - pred_rate),
            "rf_rate_surprise": -np.log(np.clip(p_actual, 1e-9, 1.0)),
            "rf_rate_entropy": -(prob * np.log(np.clip(prob, 1e-9, 1.0))).sum(1),
            "rf_pred_grade": prob.argmax(1).astype(float),
            "rf_grade_mismatch": actual - prob.argmax(1),
        })
        for i, v in enumerate(classes):
            out[f"rf_p{int(round(v * 100))}"] = prob[:, i]
        return out

    oof = frame(oof_prob, tr.interest_rate)
    tst = frame(test_prob, te.interest_rate)
    oof["application_id"] = tr.application_id.values
    tst["application_id"] = te.application_id.values
    add = pd.concat([oof, tst], ignore_index=True)
    f = base.merge(add, on="application_id", how="left", validate="one_to_one")
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score((tr.interest_rate >= 0.24).astype(int),
                                  oof.rf_pred_rate))
    except Exception:
        pass
    meta = {"predictors": len(predictors), "classes": classes.tolist(),
            "oof_auc_rate_ge_024": auc}
    log(f"  rate model done: predictors={len(predictors)} auc(rate>=.24)={auc}")
    return f, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "agas/v70/output/features_v70.pkl"))
    ap.add_argument("--out", default=str(OUT / "features_v71.pkl"))
    args = ap.parse_args()
    t0 = time.time()
    base = pd.read_pickle(args.base)
    base, panel_cols = add_panel_features(base)
    base, rate_meta = build_rate_features(base)
    base.to_pickle(args.out)
    meta = {
        "base": args.base,
        "rows": int(len(base)),
        "total_columns": int(base.shape[1]),
        "panel_columns": sorted(panel_cols),
        "rate_model": rate_meta,
        "minutes": (time.time() - t0) / 60.0,
        "note": "panel_score is cross-fitted by customer; rf_* is target-independent.",
    }
    (OUT / "summary_build_features_v71.json").write_text(json.dumps(meta, indent=2))
    log(f"saved {args.out} shape={base.shape} in {meta['minutes']:.1f} min")


if __name__ == "__main__":
    main()
