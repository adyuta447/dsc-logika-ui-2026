"""Build the v71 historical panel and event-time features.

The panel is the second training set that v66-v70 never used: one row per
approved historical loan whose 12-month outcome window is fully observed before
the customer's current decision.  Every predictor is computed at that historical
loan's own ``decision_datetime``, so no future information enters.

The same builder also computes the same event-time feature block for the current
train/test applications.  Those rows have no label; they are the rows the panel
model scores for the v71 stack.

Output
------
``agas/v71/output/panel_dataset.pkl``
    columns: row_id, customer_id, cut_id, is_hist, label, the loan terms, the
    profile snapshot, and every ``pn_`` feature.

No external data, no target of the current application, no AutoML.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

D = ROOT / "data"
OUT = HERE.parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

PROFILE_CAT = ["region", "employment_type", "housing_status"]


def safe_div(a, b):
    if isinstance(b, pd.Series):
        return a / b.replace(0, np.nan)
    return a / b if b else np.nan


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_cuts():
    """Current applications plus fully observed historical loans."""
    tr = pd.read_csv(D / "train_applications.csv",
                     parse_dates=["application_datetime", "decision_datetime"])
    te = pd.read_csv(D / "test_applications.csv",
                     parse_dates=["application_datetime", "decision_datetime"])
    app = pd.concat([tr, te], ignore_index=True)
    prof = pd.read_csv(D / "customer_profiles.csv")
    for c in PROFILE_CAT:
        prof[c] = prof[c].astype("category")
    app = app.merge(prof, on="customer_id", how="left")
    cur_cut = app[["customer_id", "decision_datetime"]].drop_duplicates("customer_id")
    cur_cut = cur_cut.rename(columns={"decision_datetime": "cur_cut"})

    prev = pd.read_csv(
        D / "previous_applications.csv",
        parse_dates=["application_datetime", "decision_datetime"],
    )
    prev = prev[prev.decision.eq("approved") & prev.loan_id.notna()].copy()

    rh = pd.read_csv(
        D / "repayment_history.csv",
        parse_dates=["due_date"],
        usecols=["loan_id", "customer_id", "due_date", "max_days_past_due"],
    )
    rh = rh.sort_values(["loan_id", "due_date"])
    rh["seq"] = rh.groupby("loan_id", sort=False).cumcount()
    first12 = rh[rh.seq < 12]
    lab = first12.groupby("loan_id", sort=False).agg(
        first_due=("due_date", "min"),
        last_due_12=("due_date", "max"),
        n12=("due_date", "size"),
        maxdpd12=("max_days_past_due", "max"),
    )
    del rh, first12
    gc.collect()
    lab["window_end"] = lab.first_due + pd.DateOffset(months=12)
    lab["label"] = (lab.maxdpd12 >= 90).astype(int)
    prev = prev.merge(lab.reset_index(), on="loan_id", how="inner")
    prev = prev.merge(cur_cut, on="customer_id", how="left")
    prev = prev[prev.window_end <= prev.cur_cut].copy()
    log(f"historical panel rows: {len(prev)}  positive rate={prev.label.mean():.4f}")

    hist = pd.DataFrame({
        "row_id": "H_" + prev.loan_id.astype(str),
        "customer_id": prev.customer_id,
        "cut": prev.decision_datetime,
        "cur_cut": prev.cur_cut,
        "is_hist": 1,
        "label": prev.label.astype(float),
        "own_application_id": prev.application_id,
        "loan_id": prev.loan_id,
        "loan_amount": prev.approved_amount.astype(float),
        "requested_amount": prev.requested_amount.astype(float),
        "approved_amount": prev.approved_amount.astype(float),
        "rate": prev.annual_interest_rate.astype(float),
        "installment": prev.scheduled_monthly_installment.astype(float),
        "term_months": prev.term_months.astype(float),
        "application_datetime": prev.application_datetime,
        "decision_datetime": prev.decision_datetime,
        "loan_purpose": np.nan,
        "hist_first_due": prev.first_due,
        "hist_window_end": prev.window_end,
        "hist_maxdpd12": prev.maxdpd12.astype(float),
    })
    hist = hist.merge(prof, on="customer_id", how="left")
    months_back = (hist.cur_cut - hist.decision_datetime).dt.days / 30.44
    hist["age"] = hist.age - np.floor(months_back / 12)
    hist["employment_tenure_months"] = (hist.employment_tenure_months - months_back).clip(lower=0)

    cur = pd.DataFrame({
        "row_id": "C_" + app.application_id.astype(str),
        "customer_id": app.customer_id,
        "cut": app.decision_datetime,
        "cur_cut": app.decision_datetime,
        "is_hist": 0,
        "label": np.nan,
        "own_application_id": app.application_id,
        "loan_id": np.nan,
        "loan_amount": app.loan_amount.astype(float),
        "requested_amount": app.loan_amount.astype(float),
        "approved_amount": app.loan_amount.astype(float),
        "rate": app.annual_interest_rate.astype(float),
        "installment": app.proposed_monthly_installment.astype(float),
        "term_months": app.term_months.astype(float),
        "application_datetime": app.application_datetime,
        "decision_datetime": app.decision_datetime,
        "loan_purpose": app.loan_purpose,
        "hist_first_due": pd.NaT,
        "hist_window_end": pd.NaT,
        "hist_maxdpd12": np.nan,
    })
    for c in PROFILE_CAT + ["age", "employment_tenure_months", "declared_monthly_income", "dependents"]:
        cur[c] = app[c]
    cur["label"] = app.target.astype(float)

    cuts = pd.concat([hist, cur], ignore_index=True)
    cuts["cut_id"] = np.arange(len(cuts), dtype=np.int64)
    cuts["decision_lag_h"] = (cuts.decision_datetime - cuts.application_datetime).dt.total_seconds() / 3600
    cuts["app_hour"] = cuts.application_datetime.dt.hour
    cuts["app_dow"] = cuts.application_datetime.dt.dayofweek
    cuts["dec_doy"] = cuts.decision_datetime.dt.dayofyear
    cuts["month"] = cuts.decision_datetime.dt.month
    cuts["pn_age"] = cuts.age.astype(float)
    cuts["pn_tenure"] = cuts.employment_tenure_months.astype(float)
    cuts["pn_income"] = cuts.declared_monthly_income.astype(float)
    cuts["pn_dependents"] = cuts.dependents.astype(float)
    cuts["pn_inst_to_income"] = safe_div(cuts.installment, cuts.pn_income)
    cuts["pn_loan_to_income"] = safe_div(cuts.loan_amount, cuts.pn_income)
    cuts["pn_req_to_income"] = safe_div(cuts.requested_amount, cuts.pn_income)
    cuts["pn_inst_to_req"] = safe_div(cuts.installment, cuts.requested_amount)
    cuts["pn_appr_ratio"] = safe_div(cuts.approved_amount, cuts.requested_amount)
    cuts["pn_rate"] = cuts.rate.astype(float)
    cuts["pn_loan_amount"] = cuts.loan_amount.astype(float)
    cuts["pn_requested_amount"] = cuts.requested_amount.astype(float)
    cuts["pn_installment"] = cuts.installment.astype(float)
    cuts["pn_term"] = cuts.term_months.astype(float)
    cuts["pn_decision_lag_h"] = cuts.decision_lag_h.astype(float)
    cuts["pn_app_hour"] = cuts.app_hour.astype(float)
    cuts["pn_app_dow"] = cuts.app_dow.astype(float)
    cuts["pn_dec_doy"] = cuts.dec_doy.astype(float)
    cuts["pn_month"] = cuts.month.astype(float)

    keep = ["row_id", "customer_id", "cut_id", "cut", "cur_cut", "is_hist", "label",
            "own_application_id", "loan_id", "loan_purpose", "hist_first_due",
            "hist_window_end", "hist_maxdpd12"] + [c for c in cuts.columns if c.startswith("pn_")] + PROFILE_CAT
    cuts = cuts[keep].copy()
    log(f"cuts total={len(cuts)} current={(cuts.is_hist == 0).sum()} historical={(cuts.is_hist == 1).sum()}")
    return cuts


def previous_block(cuts):
    log("previous applications block")
    pa = pd.read_csv(
        D / "previous_applications.csv",
        parse_dates=["application_datetime", "decision_datetime"],
    )
    pa = pa.merge(cuts[["cut_id", "customer_id", "cut"]], on="customer_id", how="inner")
    pa["days_ago"] = (pa.cut - pa.decision_datetime).dt.total_seconds() / 86400
    pa = pa[pa.days_ago > 0].copy()
    pa["approved"] = pa.decision.eq("approved").astype(float)
    pa["rejected"] = pa.decision.eq("rejected").astype(float)
    pa["cancelled"] = pa.decision.eq("cancelled").astype(float)
    pa["appr_ratio"] = safe_div(pa.approved_amount, pa.requested_amount)
    pa["month_elapsed"] = pa.days_ago / 30.44
    pa["active"] = ((pa.approved == 1) & (pa.month_elapsed < pa.term_months)).astype(float)
    pa["active_inst"] = pa.scheduled_monthly_installment.where(pa.active == 1, 0.0)
    pa = pa.sort_values(["cut_id", "decision_datetime"])

    g = pa.groupby("cut_id", sort=False)
    out = pd.DataFrame({
        "pn_pa_n": g.size(),
        "pn_pa_approved": g.approved.sum(),
        "pn_pa_rejected": g.rejected.sum(),
        "pn_pa_cancelled": g.cancelled.sum(),
        "pn_pa_req_sum": g.requested_amount.sum(),
        "pn_pa_req_mean": g.requested_amount.mean(),
        "pn_pa_req_max": g.requested_amount.max(),
        "pn_pa_appr_sum": g.approved_amount.sum(),
        "pn_pa_rate_mean": g.annual_interest_rate.mean(),
        "pn_pa_rate_max": g.annual_interest_rate.max(),
        "pn_pa_rate_min": g.annual_interest_rate.min(),
        "pn_pa_term_mean": g.term_months.mean(),
        "pn_pa_inst_mean": g.scheduled_monthly_installment.mean(),
        "pn_pa_days_since_last": g.days_ago.min(),
        "pn_pa_days_since_first": g.days_ago.max(),
        "pn_pa_active_n": g.active.sum(),
        "pn_pa_active_inst": g.active_inst.sum(),
        "pn_pa_appr_ratio_mean": g.appr_ratio.mean(),
        "pn_pa_appr_ratio_min": g.appr_ratio.min(),
        "pn_pa_in90": g.apply(lambda d: (d.days_ago <= 90).sum(), include_groups=False),
        "pn_pa_in180": g.apply(lambda d: (d.days_ago <= 180).sum(), include_groups=False),
        "pn_pa_in365": g.apply(lambda d: (d.days_ago <= 365).sum(), include_groups=False),
        "pn_pa_rej90": g.apply(lambda d: ((d.days_ago <= 90) & (d.rejected == 1)).sum(), include_groups=False),
        "pn_pa_rej365": g.apply(lambda d: ((d.days_ago <= 365) & (d.rejected == 1)).sum(), include_groups=False),
        "pn_pa_last_rate": g.annual_interest_rate.last(),
        "pn_pa_first_rate": g.annual_interest_rate.first(),
        "pn_pa_last_req": g.requested_amount.last(),
        "pn_pa_last_appr_ratio": g.appr_ratio.last(),
        "pn_pa_last_status": g.decision.last(),
    })
    out["pn_pa_reject_rate"] = safe_div(out.pn_pa_rejected, out.pn_pa_n)
    out["pn_pa_appr_rate"] = safe_div(out.pn_pa_approved, out.pn_pa_n)
    out["pn_pa_appr_over_req"] = safe_div(out.pn_pa_appr_sum, out.pn_pa_req_sum)
    out["pn_pa_rate_trend"] = out.pn_pa_last_rate - out.pn_pa_first_rate
    out["pn_pa_last_rate_minus_cur"] = out.pn_pa_last_rate - cuts.set_index("cut_id").pn_rate
    out["pn_pa_max_rate_minus_cur"] = out.pn_pa_rate_max - cuts.set_index("cut_id").pn_rate
    out["pn_pa_last_appr_ratio_minus_cur"] = out.pn_pa_last_appr_ratio - cuts.set_index("cut_id").pn_appr_ratio
    out["pn_pa_rej90_rate"] = safe_div(out.pn_pa_rej90, out.pn_pa_in90)
    out["pn_pa_rej365_rate"] = safe_div(out.pn_pa_rej365, out.pn_pa_in365)
    out["pn_pa_last_status"] = out.pn_pa_last_status.map({"approved": 1.0, "rejected": -1.0, "cancelled": 0.0})
    out = out.reset_index()
    log(f"  previous block features={out.shape[1] - 1}")
    return out


def repayment_block(cuts):
    log("repayment block")
    rh = pd.read_csv(
        D / "repayment_history.csv",
        parse_dates=["due_date", "last_payment_datetime"],
        usecols=["loan_id", "customer_id", "due_date", "amount_due", "amount_paid",
                 "last_payment_datetime", "days_past_due", "max_days_past_due"],
    )
    rh = rh.merge(cuts[["cut_id", "customer_id", "cut"]], on="customer_id", how="inner")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.total_seconds() / 86400
    rh = rh[rh.days_ago >= 0].copy()
    rh["late"] = (rh.max_days_past_due > 0).astype(float)
    rh["late30"] = (rh.max_days_past_due >= 30).astype(float)
    rh["late90"] = (rh.max_days_past_due >= 90).astype(float)
    rh["unpaid"] = (rh.amount_due - rh.amount_paid).clip(lower=0)
    rh["paid_ratio"] = safe_div(rh.amount_paid, rh.amount_due)
    rh["delay"] = (rh.last_payment_datetime - rh.due_date).dt.total_seconds() / 86400
    rh["early"] = (rh.delay <= 0).astype(float)
    rh["delay_gt5"] = (rh.delay > 5).astype(float)

    g = rh.groupby("cut_id", sort=False)
    out = pd.DataFrame({
        "pn_rh_n": g.size(),
        "pn_rh_loans": g.loan_id.nunique(),
        "pn_rh_due_sum": g.amount_due.sum(),
        "pn_rh_paid_sum": g.amount_paid.sum(),
        "pn_rh_unpaid_sum": g.unpaid.sum(),
        "pn_rh_paid_ratio_mean": g.paid_ratio.mean(),
        "pn_rh_paid_ratio_min": g.paid_ratio.min(),
        "pn_rh_late_n": g.late.sum(),
        "pn_rh_late30_n": g.late30.sum(),
        "pn_rh_late90_n": g.late90.sum(),
        "pn_rh_late_mean": g.late.mean(),
        "pn_rh_late30_mean": g.late30.mean(),
        "pn_rh_late90_mean": g.late90.mean(),
        "pn_rh_dpd_max": g.max_days_past_due.max(),
        "pn_rh_dpd_mean": g.max_days_past_due.mean(),
        "pn_rh_dpd_now_max": g.days_past_due.max(),
        "pn_rh_delay_mean": g.delay.mean(),
        "pn_rh_delay_max": g.delay.max(),
        "pn_rh_early_share": g.early.mean(),
        "pn_rh_delay_gt5_share": g.delay_gt5.mean(),
        "pn_rh_last_due_days": g.days_ago.min(),
        "pn_rh_first_due_days": g.days_ago.max(),
    })
    out["pn_rh_late_share"] = safe_div(out.pn_rh_late_n, out.pn_rh_n)
    out["pn_rh_late90_share"] = safe_div(out.pn_rh_late90_n, out.pn_rh_n)
    out["pn_rh_unpaid_share"] = safe_div(out.pn_rh_unpaid_sum, out.pn_rh_due_sum)
    out["pn_rh_pay_rate"] = safe_div(out.pn_rh_paid_sum, out.pn_rh_due_sum)

    # windows
    for n in (30, 90, 180, 365):
        t = rh[rh.days_ago <= n].groupby("cut_id", sort=False)
        p = pd.DataFrame({
            f"pn_rh_n_{n}": t.size(),
            f"pn_rh_late_{n}": t.late.sum(),
            f"pn_rh_late30_{n}": t.late30.sum(),
            f"pn_rh_late90_{n}": t.late90.sum(),
            f"pn_rh_unpaid_{n}": t.unpaid.sum(),
            f"pn_rh_due_{n}": t.amount_due.sum(),
            f"pn_rh_paid_ratio_{n}": t.paid_ratio.mean(),
            f"pn_rh_delay_{n}": t.delay.mean(),
        })
        p[f"pn_rh_late_share_{n}"] = safe_div(p[f"pn_rh_late_{n}"], p[f"pn_rh_n_{n}"])
        p[f"pn_rh_late90_share_{n}"] = safe_div(p[f"pn_rh_late90_{n}"], p[f"pn_rh_n_{n}"])
        p[f"pn_rh_unpaid_share_{n}"] = safe_div(p[f"pn_rh_unpaid_{n}"], p[f"pn_rh_due_{n}"])
        out = out.join(p)
    out["pn_rh_delay_trend"] = out.pn_rh_delay_90 - out.pn_rh_delay_365
    out["pn_rh_late_trend"] = out.pn_rh_late_share_90 - out.pn_rh_late_share_365
    out["pn_rh_unpaid_trend"] = out.pn_rh_unpaid_share_90 - out.pn_rh_unpaid_share_365

    # per-loan view at this cut
    lg = rh.groupby(["cut_id", "loan_id"], sort=False).agg(
        l_n=("late", "size"),
        l_late=("late", "mean"),
        l_late30=("late30", "mean"),
        l_late90=("late90", "max"),
        l_dpd=("max_days_past_due", "max"),
        l_dpd_now=("days_past_due", "max"),
        l_unpaid=("unpaid", "sum"),
        l_due=("amount_due", "sum"),
        l_paid_ratio=("paid_ratio", "mean"),
        l_last_days=("days_ago", "min"),
        l_first_days=("days_ago", "max"),
        l_inst=("amount_due", "mean"),
    ).reset_index()
    lg["l_active"] = (lg.l_last_days <= 45).astype(float)
    lg["l_active_inst"] = lg.l_inst.where(lg.l_active == 1, 0.0)
    lg["l_active_unpaid"] = lg.l_unpaid.where(lg.l_active == 1, 0.0)
    lg["l_bad"] = (lg.l_dpd >= 90).astype(float)
    lg["l_bad_due"] = lg.l_due.where(lg.l_bad == 1, 0.0)
    lg = lg.sort_values(["cut_id", "l_last_days"])
    gl = lg.groupby("cut_id", sort=False)
    lo = pd.DataFrame({
        "pn_lg_loans": gl.size(),
        "pn_lg_bad_n": gl.l_bad.sum(),
        "pn_lg_bad_share": gl.l_bad.mean(),
        "pn_lg_bad_due_sum": gl.l_bad_due.sum(),
        "pn_lg_due_sum": gl.l_due.sum(),
        "pn_lg_worst_late": gl.l_late.max(),
        "pn_lg_worst_dpd": gl.l_dpd.max(),
        "pn_lg_last_late": gl.l_late.last(),
        "pn_lg_last_dpd": gl.l_dpd.last(),
        "pn_lg_last_paid_ratio": gl.l_paid_ratio.last(),
        "pn_lg_active_n": gl.l_active.sum(),
        "pn_lg_active_inst": gl.l_active_inst.sum(),
        "pn_lg_active_unpaid": gl.l_active_unpaid.sum(),
        "pn_lg_active_late": gl.apply(lambda d: d.l_late[d.l_active == 1].mean(), include_groups=False),
        "pn_lg_newest_age": gl.l_last_days.min(),
        "pn_lg_oldest_age": gl.l_first_days.max(),
        "pn_lg_span_mean": gl.apply(lambda d: (d.l_first_days - d.l_last_days).mean(), include_groups=False),
        "pn_lg_paid_ratio_min": gl.l_paid_ratio.min(),
        "pn_lg_fully_paid_share": gl.apply(lambda d: ((d.l_unpaid <= 1) & (d.l_paid_ratio >= 0.99)).mean(),
                                           include_groups=False),
    })
    lo["pn_lg_bad_due_share"] = safe_div(lo.pn_lg_bad_due_sum, lo.pn_lg_due_sum)
    out = out.join(lo)
    out = out.reset_index()
    log(f"  repayment block features={out.shape[1] - 1}")
    return out


def bureau_block(cuts):
    log("bureau block")
    bu = pd.read_csv(
        D / "bureau_observations.csv",
        parse_dates=["report_datetime", "available_datetime"],
    )
    bu["unscored"] = bu.score.isna().astype(float)
    bu["free_limit"] = bu.revolving_credit_limit - bu.revolving_outstanding_balance

    # current application report
    own = bu.merge(cuts[["cut_id", "cut", "own_application_id"]], left_on="application_id",
                   right_on="own_application_id", how="inner")
    own = own[own.available_datetime <= own.cut]
    own["age_days"] = (own.cut - own.available_datetime).dt.total_seconds() / 86400
    g = own.groupby("cut_id", sort=False)
    out = pd.DataFrame({
        "pn_bu_n": g.size(),
        "pn_bu_score_mean": g.score.mean(),
        "pn_bu_score_min": g.score.min(),
        "pn_bu_score_max": g.score.max(),
        "pn_bu_score_std": g.score.std(),
        "pn_bu_score_median": g.score.median(),
        "pn_bu_unscored": g.unscored.sum(),
        "pn_bu_active_mean": g.active_accounts.mean(),
        "pn_bu_active_max": g.active_accounts.max(),
        "pn_bu_outstanding_mean": g.outstanding_balance.mean(),
        "pn_bu_outstanding_max": g.outstanding_balance.max(),
        "pn_bu_rev_limit_max": g.revolving_credit_limit.max(),
        "pn_bu_rev_bal_max": g.revolving_outstanding_balance.max(),
        "pn_bu_util_mean": g.utilization.mean(),
        "pn_bu_util_max": g.utilization.max(),
        "pn_bu_util_std": g.utilization.std(),
        "pn_bu_inq_mean": g.inquiries_last_6m.mean(),
        "pn_bu_inq_max": g.inquiries_last_6m.max(),
        "pn_bu_delinq_mean": g.delinquent_accounts_last_12m.mean(),
        "pn_bu_delinq_max": g.delinquent_accounts_last_12m.max(),
        "pn_bu_delinq_sum": g.delinquent_accounts_last_12m.sum(),
        "pn_bu_age_min": g.age_days.min(),
        "pn_bu_age_mean": g.age_days.mean(),
        "pn_bu_free_limit_max": g.free_limit.max(),
    })
    pv = own.pivot_table(index="cut_id", columns="bureau_provider", values="score", aggfunc="mean")
    pv.columns = ["pn_bu_score_" + str(c).replace("Bureau_", "prov") for c in pv.columns]
    out = out.join(pv)
    prov_cols = [c for c in out.columns if c.startswith("pn_bu_score_prov")]
    if len(prov_cols) >= 2:
        out["pn_bu_provider_spread"] = out[prov_cols].max(axis=1) - out[prov_cols].min(axis=1)
    else:
        out["pn_bu_provider_spread"] = np.nan
    out["pn_bu_unscored_rate"] = safe_div(out.pn_bu_unscored, out.pn_bu_n)

    # prior reports before this cut, excluding this application's own reports
    hist = bu.merge(cuts[["cut_id", "cut", "customer_id", "own_application_id"]], on="customer_id", how="inner")
    hist = hist[(hist.available_datetime <= hist.cut) & (hist.application_id != hist.own_application_id)]
    hist = hist.sort_values(["cut_id", "available_datetime"])
    gh = hist.groupby("cut_id", sort=False)
    h = pd.DataFrame({
        "pn_buh_n": gh.size(),
        "pn_buh_score_mean": gh.score.mean(),
        "pn_buh_score_min": gh.score.min(),
        "pn_buh_util_mean": gh.utilization.mean(),
        "pn_buh_inq_max": gh.inquiries_last_6m.max(),
        "pn_buh_delinq_max": gh.delinquent_accounts_last_12m.max(),
        "pn_buh_outstanding_max": gh.outstanding_balance.max(),
        "pn_buh_last_score": gh.score.last(),
        "pn_buh_last_util": gh.utilization.last(),
    })
    out = out.join(h)
    out["pn_buh_score_trend"] = out.pn_bu_score_mean - out.pn_buh_last_score
    out = out.reset_index()
    log(f"  bureau block features={out.shape[1] - 1}")
    return out


def account_block(cuts):
    log("account block")
    ab = pd.read_csv(
        D / "account_behavior.csv",
        parse_dates=["month"],
        usecols=["customer_id", "month", "opening_balance", "closing_balance",
                 "inflows", "outflows", "income_like_credits", "average_balance",
                 "minimum_balance", "low_balance_days"],
    )
    ab = ab.merge(cuts[["cut_id", "customer_id", "cut"]], on="customer_id", how="inner")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut].copy()
    ab["days_ago"] = (ab.cut - ab.month).dt.total_seconds() / 86400
    ab["net"] = ab.inflows - ab.outflows
    ab["income_share"] = safe_div(ab.income_like_credits, ab.inflows)
    ab["out_in"] = safe_div(ab.outflows, ab.inflows)
    ab["net_to_income"] = safe_div(ab.net, ab.income_like_credits)
    ab["bal_to_income"] = safe_div(ab.average_balance, ab.income_like_credits)
    ab["minbal_to_income"] = safe_div(ab.minimum_balance, ab.income_like_credits)
    ab["low"] = (ab.low_balance_days > 0).astype(float)
    ab["neg"] = (ab.net < 0).astype(float)
    ab["deficit"] = (ab.net < 0).astype(float)
    ab = ab.sort_values(["cut_id", "month"])

    g = ab.groupby("cut_id", sort=False)
    out = pd.DataFrame({
        "pn_ab_n": g.size(),
        "pn_ab_inflows_mean": g.inflows.mean(),
        "pn_ab_outflows_mean": g.outflows.mean(),
        "pn_ab_income_mean": g.income_like_credits.mean(),
        "pn_ab_income_sum": g.income_like_credits.sum(),
        "pn_ab_net_mean": g.net.mean(),
        "pn_ab_net_min": g.net.min(),
        "pn_ab_net_std": g.net.std(),
        "pn_ab_net_cv": g.net.std() / g.net.mean().replace(0, np.nan),
        "pn_ab_income_std": g.income_like_credits.std(),
        "pn_ab_income_cv": g.income_like_credits.std() / g.income_like_credits.mean().replace(0, np.nan),
        "pn_ab_avg_bal_mean": g.average_balance.mean(),
        "pn_ab_avg_bal_min": g.average_balance.min(),
        "pn_ab_avg_bal_std": g.average_balance.std(),
        "pn_ab_min_bal_mean": g.minimum_balance.mean(),
        "pn_ab_min_bal_min": g.minimum_balance.min(),
        "pn_ab_close_bal_mean": g.closing_balance.mean(),
        "pn_ab_open_bal_mean": g.opening_balance.mean(),
        "pn_ab_low_days_mean": g.low_balance_days.mean(),
        "pn_ab_low_days_max": g.low_balance_days.max(),
        "pn_ab_low_days_sum": g.low_balance_days.sum(),
        "pn_ab_income_share": g.income_share.mean(),
        "pn_ab_out_in": g.out_in.mean(),
        "pn_ab_net_to_income": g.net_to_income.mean(),
        "pn_ab_net_to_income_min": g.net_to_income.min(),
        "pn_ab_bal_to_income": g.bal_to_income.mean(),
        "pn_ab_minbal_to_income": g.minbal_to_income.mean(),
        "pn_ab_low_months": g.low.sum(),
        "pn_ab_neg_months": g.neg.sum(),
        "pn_ab_deficit_months": g.deficit.sum(),
        "pn_ab_last_income": g.income_like_credits.last(),
        "pn_ab_last_net": g.net.last(),
        "pn_ab_last_avg_bal": g.average_balance.last(),
        "pn_ab_last_low": g.low_balance_days.last(),
        "pn_ab_last_close": g.closing_balance.last(),
        "pn_ab_first_income": g.income_like_credits.first(),
        "pn_ab_first_avg_bal": g.average_balance.first(),
    })
    out["pn_ab_low_share"] = safe_div(out.pn_ab_low_months, out.pn_ab_n)
    out["pn_ab_neg_share"] = safe_div(out.pn_ab_neg_months, out.pn_ab_n)
    out["pn_ab_deficit_share"] = safe_div(out.pn_ab_deficit_months, out.pn_ab_n)
    out["pn_ab_bal_cv"] = safe_div(out.pn_ab_avg_bal_std, out.pn_ab_avg_bal_mean)
    out["pn_ab_income_trend"] = safe_div(out.pn_ab_last_income, out.pn_ab_first_income) - 1
    out["pn_ab_bal_trend"] = safe_div(out.pn_ab_last_avg_bal, out.pn_ab_first_avg_bal) - 1

    for n in (3, 6, 12):
        t = ab[ab.days_ago <= n * 30.44 + 1].groupby("cut_id", sort=False)
        p = pd.DataFrame({
            f"pn_ab_income_{n}": t.income_like_credits.mean(),
            f"pn_ab_net_{n}": t.net.mean(),
            f"pn_ab_avg_bal_{n}": t.average_balance.mean(),
            f"pn_ab_min_bal_{n}": t.minimum_balance.mean(),
            f"pn_ab_low_{n}": t.low_balance_days.mean(),
            f"pn_ab_outflows_{n}": t.outflows.mean(),
            f"pn_ab_inflows_{n}": t.inflows.mean(),
            f"pn_ab_neg_share_{n}": t.neg.mean(),
            f"pn_ab_low_share_{n}": t.low.mean(),
            f"pn_ab_net_to_income_{n}": t.net_to_income.mean(),
            f"pn_ab_bal_to_income_{n}": t.bal_to_income.mean(),
        })
        out = out.join(p)
    out["pn_ab_income_trend_3_12"] = safe_div(out.pn_ab_income_3, out.pn_ab_income_12) - 1
    out["pn_ab_net_trend_3_12"] = out.pn_ab_net_3 - out.pn_ab_net_12
    out["pn_ab_bal_trend_3_12"] = safe_div(out.pn_ab_avg_bal_3, out.pn_ab_avg_bal_12) - 1
    out["pn_ab_low_trend_3_12"] = out.pn_ab_low_3 - out.pn_ab_low_12
    out = out.reset_index()
    log(f"  account block features={out.shape[1] - 1}")
    return out


def ecommerce_block(cuts):
    log("ecommerce block")
    ec = pd.read_csv(
        D / "ecommerce_transactions.csv",
        parse_dates=["transaction_datetime"],
        usecols=["order_id", "customer_id", "transaction_datetime", "product_category",
                 "quantity", "item_total", "order_total", "payment_method", "order_status"],
    )
    ec = ec[ec.quantity > 0].copy()
    ec["date"] = ec.transaction_datetime.dt.floor("D")
    ec["done"] = ec.order_status.eq("completed").astype(float)
    ec["spend"] = ec.item_total.where(ec.done == 1, 0.0)
    ec["need"] = ec.product_category.isin(["groceries", "health", "education"]).astype(float)
    ec["want"] = ec.product_category.isin(["fashion", "electronics", "personal_care"]).astype(float)
    ec["home"] = ec.product_category.eq("home").astype(float)
    ec["night"] = ec.transaction_datetime.dt.hour.isin([0, 1, 2, 3, 4, 22, 23]).astype(float)
    ec["weekend"] = (ec.transaction_datetime.dt.dayofweek >= 5).astype(float)
    item = ec.groupby(["customer_id", "date"], sort=False).agg(
        i_items=("quantity", "sum"),
        i_spend=("spend", "sum"),
        i_need=("need", "sum"),
        i_want=("want", "sum"),
        i_home=("home", "sum"),
        i_night=("night", "sum"),
        i_weekend=("weekend", "sum"),
        i_unit=("item_total", "sum"),
    ).reset_index()

    od = ec.drop_duplicates("order_id").copy()
    od["cod"] = od.payment_method.eq("COD").astype(float)
    od["cc"] = od.payment_method.eq("credit_card").astype(float)
    od["ewallet"] = od.payment_method.eq("e_wallet").astype(float)
    od["bank"] = od.payment_method.eq("bank_transfer").astype(float)
    od["debit"] = od.payment_method.eq("debit_card").astype(float)
    od["cancel"] = (1 - od.done).astype(float)
    order = od.groupby(["customer_id", "date"], sort=False).agg(
        o_orders=("order_id", "nunique"),
        o_basket=("order_total", "sum"),
        o_cancel=("cancel", "sum"),
        o_cod=("cod", "sum"),
        o_cc=("cc", "sum"),
        o_ewallet=("ewallet", "sum"),
        o_bank=("bank", "sum"),
        o_debit=("debit", "sum"),
        o_night=("night", "sum"),
        o_weekend=("weekend", "sum"),
    ).reset_index()
    day = item.merge(order, on=["customer_id", "date"], how="outer")
    for c in [c for c in day.columns if c not in ("customer_id", "date")]:
        day[c] = day[c].fillna(0.0)
    del ec, item, od, order
    gc.collect()

    # compute order gap once on the unique customer-day panel, before copying it
    # across the multiple cuts each customer can have
    day = day.sort_values(["customer_id", "date"])
    day["gap"] = day.groupby("customer_id", sort=False).date.diff().dt.total_seconds() / 86400

    day = day.merge(cuts[["cut_id", "customer_id", "cut"]], on="customer_id", how="inner")
    day["days_ago"] = (day.cut - day.date).dt.total_seconds() / 86400
    day = day[day.days_ago >= 0].copy()
    day = day.sort_values(["cut_id", "date"])

    g = day.groupby("cut_id", sort=False)
    out = pd.DataFrame({
        "pn_ec_days": g.size(),
        "pn_ec_orders": g.o_orders.sum(),
        "pn_ec_items": g.i_items.sum(),
        "pn_ec_spend": g.i_spend.sum(),
        "pn_ec_span": g.days_ago.max(),
        "pn_ec_last_days": g.days_ago.min(),
        "pn_ec_basket_mean": g.o_basket.mean(),
        "pn_ec_basket_max": g.o_basket.max(),
        "pn_ec_basket_std": g.o_basket.std(),
        "pn_ec_gap_mean": g.gap.mean(),
        "pn_ec_gap_last": g.gap.last(),
        "pn_ec_gap_std": g.gap.std(),
        "pn_ec_cancel": g.o_cancel.sum(),
        "pn_ec_cod": g.o_cod.sum(),
        "pn_ec_cc": g.o_cc.sum(),
        "pn_ec_ewallet": g.o_ewallet.sum(),
        "pn_ec_bank": g.o_bank.sum(),
        "pn_ec_debit": g.o_debit.sum(),
        "pn_ec_night": g.i_night.sum(),
        "pn_ec_weekend": g.i_weekend.sum(),
        "pn_ec_need": g.i_need.sum(),
        "pn_ec_want": g.i_want.sum(),
        "pn_ec_home": g.i_home.sum(),
        "pn_ec_unit_mean": g.i_unit.mean(),
        "pn_ec_spend_std": g.i_spend.std(),
        "pn_ec_last_spend": g.i_spend.last(),
        "pn_ec_last_orders": g.o_orders.last(),
    })
    out["pn_ec_cancel_rate"] = safe_div(out.pn_ec_cancel, out.pn_ec_orders)
    out["pn_ec_cod_rate"] = safe_div(out.pn_ec_cod, out.pn_ec_orders)
    out["pn_ec_cc_rate"] = safe_div(out.pn_ec_cc, out.pn_ec_orders)
    out["pn_ec_ewallet_rate"] = safe_div(out.pn_ec_ewallet, out.pn_ec_orders)
    out["pn_ec_bank_rate"] = safe_div(out.pn_ec_bank, out.pn_ec_orders)
    out["pn_ec_debit_rate"] = safe_div(out.pn_ec_debit, out.pn_ec_orders)
    out["pn_ec_night_rate"] = safe_div(out.pn_ec_night, out.pn_ec_items)
    out["pn_ec_weekend_rate"] = safe_div(out.pn_ec_weekend, out.pn_ec_items)
    out["pn_ec_need_share"] = safe_div(out.pn_ec_need, out.pn_ec_items)
    out["pn_ec_want_share"] = safe_div(out.pn_ec_want, out.pn_ec_items)
    out["pn_ec_home_share"] = safe_div(out.pn_ec_home, out.pn_ec_items)
    out["pn_ec_basket_cv"] = safe_div(out.pn_ec_basket_std, out.pn_ec_basket_mean)
    out["pn_ec_gap_cv"] = safe_div(out.pn_ec_gap_std, out.pn_ec_gap_mean)
    out["pn_ec_spend_per_order"] = safe_div(out.pn_ec_spend, out.pn_ec_orders)
    out["pn_ec_spend_per_month"] = safe_div(out.pn_ec_spend, out.pn_ec_span / 30.44 + 1)
    out["pn_ec_orders_per_month"] = safe_div(out.pn_ec_orders, out.pn_ec_span / 30.44 + 1)

    for n in (30, 90, 180, 365):
        t = day[day.days_ago <= n].groupby("cut_id", sort=False)
        p = pd.DataFrame({
            f"pn_ec_spend_{n}": t.i_spend.sum(),
            f"pn_ec_orders_{n}": t.o_orders.sum(),
            f"pn_ec_items_{n}": t.i_items.sum(),
            f"pn_ec_basket_{n}": t.o_basket.mean(),
            f"pn_ec_cancel_{n}": t.o_cancel.sum(),
            f"pn_ec_cod_{n}": t.o_cod.sum(),
            f"pn_ec_need_{n}": t.i_need.sum(),
            f"pn_ec_want_{n}": t.i_want.sum(),
        })
        p[f"pn_ec_cancel_rate_{n}"] = safe_div(p[f"pn_ec_cancel_{n}"], p[f"pn_ec_orders_{n}"])
        p[f"pn_ec_cod_rate_{n}"] = safe_div(p[f"pn_ec_cod_{n}"], p[f"pn_ec_orders_{n}"])
        p[f"pn_ec_need_share_{n}"] = safe_div(p[f"pn_ec_need_{n}"], p[f"pn_ec_items_{n}"])
        out = out.join(p)
    out["pn_ec_spend_trend"] = safe_div(out.pn_ec_spend_90, out.pn_ec_spend_365 / 4 + 1) - 1
    out["pn_ec_orders_trend"] = safe_div(out.pn_ec_orders_90, out.pn_ec_orders_365 / 4 + 1) - 1
    out["pn_ec_basket_trend"] = out.pn_ec_basket_90 - out.pn_ec_basket_365
    out["pn_ec_cancel_trend"] = out.pn_ec_cancel_rate_90 - out.pn_ec_cancel_rate_365
    out = out.reset_index()
    log(f"  ecommerce block features={out.shape[1] - 1}")
    return out


def assemble(cuts, blocks):
    log("assemble")
    f = cuts.copy()
    for b in blocks:
        f = f.merge(b, on="cut_id", how="left")
    profile_income = f.pn_income.replace(0, np.nan)
    f["pn_ab_income_to_declared"] = safe_div(f.pn_ab_income_mean, profile_income)
    f["pn_ab_l3_income_to_declared"] = safe_div(f.pn_ab_income_3, profile_income)
    if "pn_ec_spend_365" in f:
        f["pn_ec_spend_to_income"] = safe_div(f.pn_ec_spend_365, 12 * profile_income)
        f["pn_ec_basket_to_income"] = safe_div(f.pn_ec_basket_mean, profile_income)
    f["pn_dti_total"] = safe_div(f.pn_pa_active_inst.fillna(0) + f.pn_installment, profile_income)
    f["pn_pa_req_to_income"] = safe_div(f.pn_pa_req_mean, profile_income)
    f["pn_rate_x_dpd"] = f.pn_rate * f.pn_rh_dpd_max
    f["pn_rate_x_late90"] = f.pn_rate * f.pn_rh_late90_n
    f["pn_rate_x_pa_rej"] = f.pn_rate * f.pn_pa_reject_rate
    f["pn_loan_to_income_x_dpd"] = f.pn_loan_to_income * f.pn_rh_dpd_max
    f["pn_late_x_net"] = f.pn_rh_late_share * f.pn_ab_net_to_income
    f["pn_late_x_low"] = f.pn_rh_late_share * f.pn_ab_low_share
    f["pn_bad_due_x_rate"] = f.pn_lg_bad_due_share * f.pn_rate
    f["pn_cc_rate_minus_cod"] = f.pn_ec_cc_rate - f.pn_ec_cod_rate
    f["pn_bu_delinq_x_util"] = f.pn_bu_delinq_max * f.pn_bu_util_mean
    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "panel_dataset.pkl"))
    ap.add_argument("--skip-ec", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cuts = load_cuts()
    blocks = [previous_block(cuts), repayment_block(cuts), bureau_block(cuts), account_block(cuts)]
    if not args.skip_ec:
        blocks.append(ecommerce_block(cuts))
    f = assemble(cuts, blocks)
    f.to_pickle(args.out)
    meta = {
        "rows": int(len(f)),
        "historical_rows": int((f.is_hist == 1).sum()),
        "current_rows": int((f.is_hist == 0).sum()),
        "historical_positive_rate": float(f.loc[f.is_hist == 1, "label"].mean()),
        "features": int(sum(c.startswith("pn_") for c in f.columns)),
        "minutes": (time.time() - t0) / 60.0,
        "note": "All pn_ features use each row's own decision_datetime cutoff. "
                "Historical rows are approved loans with a fully observed 12-month window.",
    }
    (OUT / "summary_panel_build_v71.json").write_text(json.dumps(meta, indent=2))
    log(f"saved {args.out} shape={f.shape} in {meta['minutes']:.1f} min")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
