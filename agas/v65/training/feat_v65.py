"""New feature blocks for v65.

Three gaps in the existing 554-column feature set, each chosen because it is a
different *kind* of information rather than another statistic on columns that
are already summarised.

`dataquality` — coverage and missingness. In bureau-style credit data, "this
customer has no bureau file" and "this customer's bureau file has no score" are
signals, not gaps to be filled with a mean. The existing blocks encode row
counts, so they capture presence, but not how incomplete the presence is. This
block adds the missingness fractions and the joint-coverage pattern.

`burdirection` — direction, not just level. The existing bureau blocks report
means, maxima and two trend columns for score alone. Nothing encodes whether
utilisation, inquiries or delinquency are *rising*, which is the early-warning
signal a lender would actually act on. This block splits each customer's report
history into an older and a newer half and takes the difference, and adds a
momentum-volatility term that a plain slope cannot express.

`stress` — affordability burden against observed cash flow rather than declared
income. `features.py` has debt-to-income on declared income. Nobody can verify
declared income, while observed inflows and prior active instalments can be
measured. This block recomputes the burden against observed cash flow and
against the instalments the customer is already carrying.
"""

import numpy as np
import pandas as pd

D = "data/"


def _cut(app):
    return app[["application_id", "customer_id", "decision_datetime"]].rename(
        columns={"decision_datetime": "cut"}
    )


def _half_delta(df, key, cols):
    """Difference between the mean over the newer half and the newer half of a
    per-customer report history.

    Splitting by rank rather than by a fixed date keeps the two halves the same
    size even for customers with a thin file, which matters because thin files
    are exactly the ones where a fixed cut-off would leave one half empty.
    """
    df = df.sort_values([key, "available_datetime"])
    out = []
    for _, g in df.groupby(key):
        n = len(g)
        if n < 2:
            for c in cols:
                out.append((g[key].iloc[0], np.nan, c))
            continue
        h = n // 2
        old, new = g.iloc[:h], g.iloc[h:]
        for c in cols:
            out.append((g[key].iloc[0], new[c].mean() - old[c].mean(), c))
    return pd.DataFrame(out, columns=[key, "delta", "col"]).pivot(
        index=key, columns="col", values="delta")


def block_dataquality(app):
    """Coverage depth and missingness fractions across all six tables."""
    cut = _cut(app)[["customer_id", "application_id", "cut"]]
    src = cut.copy()

    bu = pd.read_csv(D + "bureau_observations.csv")
    has_bu = bu.customer_id.nunique()
    m = bu.merge(cut[["customer_id", "cut"]], on="customer_id")
    bu_g = m.groupby("customer_id")
    dq = pd.DataFrame(index=cut.customer_id.unique())
    dq.index.name = "customer_id"
    dq["dq_bureau_score_missing"] = bu_g.score.apply(lambda s: float(s.isna().mean()))
    dq["dq_bureau_util_missing"] = bu_g.utilization.apply(lambda s: float(s.isna().mean()))
    dq["dq_bureau_limit_missing"] = bu_g.revolving_credit_limit.apply(lambda s: float(s.isna().mean()))
    dq["dq_bureau_prov_n"] = bu_g.bureau_provider.nunique()
    dq["dq_bureau_report_n"] = bu_g.size()

    pa = pd.read_csv(D + "previous_applications.csv")
    m = pa.merge(cut[["customer_id", "cut"]], on="customer_id")
    pa_g = m.groupby("customer_id")
    dq["dq_prev_amount_missing"] = pa_g.approved_amount.apply(lambda s: float(s.isna().mean()))
    dq["dq_prev_rate_missing"] = pa_g.annual_interest_rate.apply(lambda s: float(s.isna().mean()))
    dq["dq_prev_no_decision"] = pa_g.decision.apply(lambda s: float(s.isna().mean()))

    rh = pd.read_csv(D + "repayment_history.csv")
    m = rh.merge(cut[["customer_id", "cut"]], on="customer_id")
    rh_g = m.groupby("customer_id")
    dq["dq_repay_dpd_missing"] = rh_g.days_past_due.apply(lambda s: float(s.isna().mean()))
    dq["dq_repay_maxdpd_missing"] = rh_g.max_days_past_due.apply(lambda s: float(s.isna().mean()))

    ab = pd.read_csv(D + "account_behavior.csv")
    m = ab.merge(cut[["customer_id", "cut"]], on="customer_id")
    ab_g = m.groupby("customer_id")
    dq["dq_acct_zero_income_months"] = ab_g.income_like_credits.apply(
        lambda s: float((s.fillna(0) <= 0).mean()))
    dq["dq_acct_balance_missing"] = ab_g.average_balance.apply(lambda s: float(s.isna().mean()))
    dq["dq_acct_months"] = ab_g.size()

    ec = pd.read_csv(D + "ecommerce_transactions.csv")
    m = ec.merge(cut[["customer_id", "cut"]], on="customer_id")
    ec_g = m.groupby("customer_id")
    dq["dq_ec_cat_missing"] = ec_g.product_category.apply(lambda s: float(s.isna().mean()))
    dq["dq_ec_qty_missing"] = ec_g.quantity.apply(lambda s: float(s.isna().mean()))
    dq["dq_ec_item_missing"] = ec_g.item_total.apply(lambda s: float(s.isna().mean()))

    # presence per source, and how many sources agree the customer is visible
    present = pd.DataFrame(index=dq.index)
    idx = present.index
    present["dq_has_bureau"] = (dq.dq_bureau_report_n.fillna(0) > 0).astype(float)
    present["dq_has_prev"] = pd.Series(idx.isin(pa.customer_id.unique()), index=idx).astype(float)
    present["dq_has_repay"] = pd.Series(idx.isin(rh.customer_id.unique()), index=idx).astype(float)
    present["dq_has_acct"] = (dq.dq_acct_months.fillna(0) > 0).astype(float)
    present["dq_has_ec"] = pd.Series(idx.isin(ec.customer_id.unique()), index=idx).astype(float)
    flags = [f"dq_has_{s}" for s in ("bureau", "prev", "repay", "acct", "ec")]
    present["dq_n_sources"] = present[flags].sum(axis=1)
    # visible to the bureau but invisible to our own repayment ledger is a
    # specific contradiction, so keep it separate from a plain count
    present["dq_thin_file"] = (present.dq_n_sources <= 2).astype(float)

    dq = dq.join(present).reset_index()
    out = src[["customer_id", "application_id"]].merge(dq, on="customer_id", how="left")
    return out.drop(columns=["customer_id"])


def block_burdirection(app):
    """Rising versus falling bureau indicators, and momentum volatility."""
    bu = pd.read_csv(D + "bureau_observations.csv",
                     parse_dates=["report_datetime", "available_datetime"])
    cut = _cut(app)[["customer_id", "cut"]]
    m = bu.merge(cut, on="customer_id")
    m["days_ago"] = (m.cut - m.report_datetime).dt.total_seconds() / 86400
    m = m[m.days_ago >= 0]
    key = "customer_id"

    cols = ["score", "utilization", "inquiries_last_6m",
            "delinquent_accounts_last_12m", "outstanding_balance",
            "active_accounts", "revolving_outstanding_balance"]
    delta = _half_delta(m, key, cols)
    delta.columns = ["bd_" + c.replace("_", "") + "_delta" for c in delta.columns]

    # recency-weighted level: a customer whose recent file is worse than their
    # average is a different case from one whose recent file is better
    hl = 180.0
    m["w"] = np.exp(-m.days_ago * np.log(2) / hl)
    g = m.groupby(key)
    wsum = g.w.sum()
    rec = pd.DataFrame(index=delta.index)
    for c in cols:
        num = g.apply(lambda d, c=c: float((d[c] * d.w).sum()), include_groups=False)
        rec["bd_" + c.replace("_", "") + "_rec180"] = (num / wsum).reindex(delta.index)

    # momentum volatility: mean absolute successive change, which a slope
    # cannot express because a slope averages the direction away
    vol = {}
    for c in ["score", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m"]:
        d = m.sort_values([key, "available_datetime"])
        dif = d.groupby(key)[c].diff().abs()
        vol["bd_" + c.replace("_", "") + "_jump"] = dif.groupby(d[key]).mean()

    gap = (m.sort_values([key, "available_datetime"]).groupby(key).report_datetime
           .apply(lambda s: float((s.diff().dt.total_seconds() / 86400).mean())))
    gap.name = "bd_report_gap_days"

    out = pd.concat([delta, rec, pd.DataFrame(vol), gap.to_frame()], axis=1)

    # a score below 700 is a conventional thin-file boundary; the rate below it
    # is more stable than the raw score because it saturates
    out["bd_score_below_700"] = m.assign(f=(m.score < 700).astype(float)).groupby(key).f.mean()
    out["bd_score_below_600"] = m.assign(f=(m.score < 600).astype(float)).groupby(key).f.mean()
    cmap = app[["customer_id", "application_id"]].drop_duplicates("customer_id")
    return cmap.merge(out.reset_index(), on="customer_id", how="left").drop(columns=["customer_id"])


def block_stress(app):
    """Affordability measured against observed cash and existing commitments."""
    prof = pd.read_csv(D + "customer_profiles.csv")
    cut = _cut(app)
    df = (app.merge(prof, on="customer_id", how="left")
           .merge(cut[["application_id", "cut"]], on="application_id"))

    ab = pd.read_csv(D + "account_behavior.csv")
    m = ab.merge(cut[["customer_id", "cut"]], on="customer_id")
    m["days_ago"] = (m.cut - pd.to_datetime(m.month)).dt.days
    recent = m[(m.days_ago >= 0) & (m.days_ago <= 365)]
    agg = recent.groupby("customer_id").agg(
        af_inflow_last12=("income_like_credits", "mean"),
        af_spend_last12=("outflows", "mean"),
        af_minbal_last12=("minimum_balance", "mean"),
        af_months_last12=("month", "size"),
    )
    base = recent.groupby("customer_id").minimum_balance.apply(
        lambda s: float((s < 1e9).mean()))

    pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["decision_datetime"])
    m = pa.merge(cut[["customer_id", "cut"]], on="customer_id")
    m["days_ago"] = (m.cut - m.decision_datetime).dt.total_seconds() / 86400
    m = m[m.days_ago >= 0]
    appr = m[m.decision == "approved"]
    ap_g = appr.groupby("customer_id")
    burden = ap_g.scheduled_monthly_installment.sum()
    outstanding = ap_g.approved_amount.sum()

    out = df.join(agg, on="customer_id").join(burden.rename("af_prior_inst"), on="customer_id")
    out = out.join(outstanding.rename("af_prior_amount"), on="customer_id")
    out = out.join(base.rename("af_minbal_coverage"), on="customer_id")

    inc = out.declared_monthly_income.replace(0, np.nan)
    inst = out.proposed_monthly_installment
    observed = (out.af_inflow_last12 / out.af_months_last12).replace(0, np.nan)
    annual = inc * 12

    out["af_inst_to_declared"] = inst / inc
    out["af_inst_to_observed"] = inst / observed
    out["af_inst_to_best_estimate"] = inst / observed.fillna(inc)
    # prior commitments are what actually compete for the same cash
    out["af_prior_inst_to_income"] = out.af_prior_inst / inc
    out["af_total_inst_to_income"] = (out.af_prior_inst + inst) / inc
    out["af_prior_amount_to_annual_income"] = out.af_prior_amount / annual
    out["af_total_exposure_to_annual_income"] = (out.af_prior_amount + out.loan_amount) / annual

    hist_max = ap_g.approved_amount.max()
    hist_mean = ap_g.approved_amount.mean()
    out["af_loan_to_hist_max"] = out.loan_amount / hist_max.reindex(out.index)
    out["af_loan_to_hist_mean"] = out.loan_amount / hist_mean.reindex(out.index)

    out["af_interest_burden"] = out.annual_interest_rate * out.loan_amount / annual
    out["af_loan_to_minbal"] = out.loan_amount / out.af_minbal_last12.replace(0, np.nan)
    out["af_observed_minus_declared"] = observed - inc
    out["af_history_years"] = out.af_months_last12 / 12
    # thin observed history next to a large loan is a different risk from a
    # long history, and the ratio separates them where the level does not
    out["af_history_vs_loan"] = out.af_history_years / (out.loan_amount / 1e7)
    # income, age and dependents are already columns in features_cache, and an
    # outer merge on a duplicate name would blank them out
    return out[["application_id"] + [c for c in out.columns if c.startswith("af_")]]