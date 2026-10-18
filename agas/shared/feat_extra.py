"""Additional feature blocks that are not part of `features.py`.

Every block is a pure function of the competition tables and the application
decision timestamp, so no label information is used and no fold-specific state
leaks. Nothing here modifies `features.py`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = "data/"
CAT_DISC = ["electronics", "home", "personal_care", "fashion"]
CAT_ESS = ["groceries", "health"]


def _cut(app):
    return app[["application_id", "customer_id", "decision_datetime"]].rename(
        columns={"decision_datetime": "cut"}
    )


def _expw(days_ago, hl):
    return np.exp(-days_ago * np.log(2) / hl)


# ---------------------------------------------------------------- bureau ----
def block_bureau(app):
    bu = pd.read_csv(D + "bureau_observations.csv", parse_dates=["report_datetime", "available_datetime"])
    bu = bu.merge(_cut(app)[["customer_id", "application_id", "cut"]].rename(
        columns={"application_id": "application_id_sel"}), on="customer_id")
    bu["days_ago"] = (bu.cut - bu.report_datetime).dt.total_seconds() / 86400
    bu["avail_ago"] = (bu.cut - bu.available_datetime).dt.total_seconds() / 86400
    bu["prov_lag"] = (bu.available_datetime - bu.report_datetime).dt.total_seconds() / 86400
    bu["delinq_per_acct"] = bu.delinquent_accounts_last_12m / (bu.active_accounts + 1)
    bu["inq_per_acct"] = bu.inquiries_last_6m / (bu.active_accounts + 1)
    bu["out_per_acct"] = bu.outstanding_balance / (bu.active_accounts + 1)
    bu["rev_free"] = bu.revolving_credit_limit - bu.revolving_outstanding_balance

    g = bu.groupby("application_id_sel")
    out = g.agg(
        bx_prov_n=("bureau_provider", "nunique"),
        bx_report_n=("report_datetime", "nunique"),
        bx_avail_ago_min=("avail_ago", "min"),
        bx_avail_ago_mean=("avail_ago", "mean"),
        bx_prov_lag_mean=("prov_lag", "mean"),
        bx_delinq_per_acct_max=("delinq_per_acct", "max"),
        bx_inq_per_acct_max=("inq_per_acct", "max"),
        bx_out_per_acct_max=("out_per_acct", "max"),
        bx_rev_free_min=("rev_free", "min"),
        bx_active_min=("active_accounts", "min"),
        bx_zero_util=("utilization", lambda s: float((s.fillna(0) <= 0).mean())),
    )
    out["bx_score_range"] = g.score.max() - g.score.min()
    out["bx_util_range"] = g.utilization.max() - g.utilization.min()
    out["bx_delinq_range"] = g.delinquent_accounts_last_12m.max() - g.delinquent_accounts_last_12m.min()
    out["bx_inq_range"] = g.inquiries_last_6m.max() - g.inquiries_last_6m.min()

    # provider disagreement: each bureau reads the same customer differently
    z = bu.groupby(["application_id_sel", "bureau_provider"]).score.mean().unstack()
    out["bx_prov_score_std"] = z.std(axis=1)
    out["bx_prov_score_min"] = z.min(axis=1)
    zl = bu.groupby(["application_id_sel", "bureau_provider"]).delinquent_accounts_last_12m.mean().unstack()
    out["bx_prov_delinq_std"] = zl.std(axis=1)

    # short horizon trend: recent reports against everything older
    for col, name in [("score", "bx_score_trend"), ("utilization", "bx_util_trend"),
                      ("delinquent_accounts_last_12m", "bx_delinq_trend"),
                      ("inquiries_last_6m", "bx_inq_trend")]:
        rec = bu[bu.days_ago <= 180].groupby("application_id_sel")[col].mean()
        old = bu[bu.days_ago > 180].groupby("application_id_sel")[col].mean()
        out[name] = (rec - old).reindex(out.index)
        out[name + "_rec"] = rec.reindex(out.index)
    return out.reset_index().rename(columns={"application_id_sel": "application_id"})


# ------------------------------------------------------------- repayment ----
def block_repayment(app):
    rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date", "last_payment_datetime"])
    rh = rh.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.total_seconds() / 86400
    rh = rh[rh.days_ago >= 0]
    rh["late"] = (rh.max_days_past_due > 0).astype(int)
    rh["late90"] = (rh.max_days_past_due >= 90).astype(int)
    rh["paid_ratio"] = rh.amount_due.replace(0, np.nan)
    rh["paid_ratio"] = rh.amount_paid / rh.paid_ratio
    rh["pay_delay"] = (rh.last_payment_datetime - rh.due_date).dt.total_seconds() / 86400
    rh["partial"] = ((rh.amount_paid > 0) & (rh.amount_paid < rh.amount_due)).astype(int)
    rh["matured"] = (rh.days_ago >= 30).astype(int)
    # a late instalment that was eventually cleared is a different signal from one that was not
    rh["cured"] = ((rh.max_days_past_due > 0) & (rh.paid_ratio >= 0.95)).astype(int)
    rh["seq"] = rh.groupby(["customer_id", "loan_id"]).cumcount()
    first_late = rh[rh.late == 1].groupby(["customer_id", "loan_id"]).seq.min()
    rh = rh.sort_values(["customer_id", "loan_id", "due_date"])

    g = rh.groupby("customer_id")
    out = g.agg(
        rx_late_recent_days=("days_ago", lambda s: float(s[rh.loc[s.index, "late"].eq(1)].min()) if rh.loc[s.index, "late"].eq(1).any() else np.nan),
        rx_late90_recent_days=("days_ago", lambda s: float(s[rh.loc[s.index, "late90"].eq(1)].min()) if rh.loc[s.index, "late90"].eq(1).any() else np.nan),
        rx_partial_rate=("partial", "mean"),
        rx_matured_rate=("matured", "mean"),
        rx_delay_ge5=("pay_delay", lambda s: float((s > 5).mean())),
        rx_delay_ge15=("pay_delay", lambda s: float((s > 15).mean())),
        rx_delay_ge30=("pay_delay", lambda s: float((s > 30).mean())),
        rx_dpd_mean_matured=("max_days_past_due", "mean"),
    )
    out["rx_cured_of_late"] = rh[rh.late == 1].groupby("customer_id").cured.mean()
    out["rx_late_n"] = g.late.sum()
    out["rx_late90_n"] = g.late90.sum()

    for hl in (60, 180, 365):
        w = _expw(rh.days_ago, hl)
        out[f"rx_late_w{hl}"] = (w * rh.late).groupby(rh.customer_id).sum()
        out[f"rx_dpd_w{hl}"] = (w * rh.max_days_past_due).groupby(rh.customer_id).sum()
        out[f"rx_late_n_w{hl}"] = w.groupby(rh.customer_id).sum()

    per = rh.groupby(["customer_id", "loan_id"]).agg(
        n=("late", "size"), nl=("late", "sum"), mx=("max_days_past_due", "max"),
        end=("due_date", "max"), dpd=("days_past_due", "last"),
    )
    per["late_rate"] = per.nl / per.n
    pg = per.groupby(level=0)
    out["rx_loan_late_rate_mean"] = pg.late_rate.mean()
    out["rx_loan_late_rate_max"] = pg.late_rate.max()
    out["rx_n_loans_late"] = pg.late_rate.apply(lambda s: float((s > 0).mean()))
    out["rx_loan_dpd_max"] = pg.mx.max()
    out["rx_first_late_idx_mean"] = first_late.groupby(level=0).mean()
    out["rx_first_late_idx_min"] = first_late.groupby(level=0).min()
    out["rx_loan_n"] = pg.n.mean()
    out["rx_dpd_open_max"] = pg.dpd.max()
    return out.reset_index()


# ----------------------------------------------------- previous apps --------
def block_previous(app):
    pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
    pa = pa.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    pa["days_ago"] = (pa.cut - pa.application_datetime).dt.total_seconds() / 86400
    pa = pa[pa.days_ago >= 0].sort_values(["customer_id", "application_datetime"])
    pa["rej"] = (pa.decision == "rejected").astype(int)
    pa["can"] = (pa.decision == "cancelled").astype(int)
    pa["app"] = (pa.decision == "approved").astype(int)
    pa["months_elapsed"] = pa.days_ago / 30.44
    pa["active"] = ((pa.decision == "approved") & (pa.months_elapsed < pa.term_months)).astype(int)
    pa["gap"] = pa.groupby("customer_id").application_datetime.diff().dt.total_seconds() / 86400

    g = pa.groupby("customer_id")
    out = pd.DataFrame({
        "px_n30": g.apply(lambda d: float((d.days_ago <= 30).sum()), include_groups=False),
        "px_n90": g.apply(lambda d: float((d.days_ago <= 90).sum()), include_groups=False),
        "px_gap_std": g.gap.std(),
        "px_gap_min": g.gap.min(),
        "px_gap_max": g.gap.max(),
    })
    for flag, name in [("rej", "px_rej"), ("can", "px_can"), ("app", "px_app")]:
        out[f"{name}_recent_days"] = g.apply(
            lambda d: float(d.days_ago[d[flag].eq(1)].min()) if d[flag].eq(1).any() else np.nan,
            include_groups=False)
        out[f"{name}_n90"] = g.apply(
            lambda d: float(((d[flag].eq(1)) & (d.days_ago <= 90)).sum()), include_groups=False)
        out[f"{name}_amt"] = g.apply(
            lambda d: float(d.requested_amount[d[flag].eq(1)].sum()), include_groups=False)
    out["px_active_inst_max"] = g.apply(
        lambda d: float(d.scheduled_monthly_installment[d.active.eq(1)].max()) if d.active.any() else np.nan,
        include_groups=False)
    out["px_active_recent_n"] = g.apply(
        lambda d: float(((d.active.eq(1)) & (d.days_ago <= 365)).sum()), include_groups=False)
    out["px_peak_active_90"] = g.apply(
        lambda d: float(((d.application_datetime > d.application_datetime.max() - pd.Timedelta(days=90)) & d.active.eq(1)).sum()),
        include_groups=False)
    # 90-day rejection bursts: repeated rejections in a short window signal stress
    out["px_rej_burst"] = g.apply(
        lambda d: float(max(
            [((d.rej == 1) & (d.application_datetime >= t) & (d.application_datetime < t + pd.Timedelta(days=90))).sum()
             for t in d.application_datetime], default=0)), include_groups=False)
    return out.reset_index()


# -------------------------------------------------------------- account -----
def block_account(app):
    ab = pd.read_csv(D + "account_behavior.csv", parse_dates=["month"])
    ab = ab.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut].sort_values(["customer_id", "month"])
    ab["net"] = ab.inflows - ab.outflows
    ab["inst"] = ab.customer_id.map(app.set_index("customer_id").proposed_monthly_installment)
    ab["stress"] = (ab.minimum_balance < ab.inst).astype(int)
    ab["cash_stress"] = (ab.income_like_credits < ab.inst).astype(int)
    ab["neg"] = (ab.net < 0).astype(int)

    g = ab.groupby("customer_id")

    def cv(s):
        return float(s.std() / s.abs().mean()) if s.abs().mean() > 0 else np.nan

    out = pd.DataFrame({
        "ax_inc_cv": g.income_like_credits.apply(cv),
        "ax_inflow_cv": g.inflows.apply(cv),
        "ax_outflow_cv": g.outflows.apply(cv),
        "ax_bal_cv": g.average_balance.apply(cv),
        "ax_net_cv": g.net.apply(cv),
        "ax_stress_months": g.stress.sum(),
        "ax_cash_stress_months": g.cash_stress.sum(),
        "ax_neg_months": g.neg.sum(),
        "ax_low_days_mean": g.low_balance_days.mean(),
        "ax_low_days_std": g.low_balance_days.std(),
        "ax_minbal_mean": g.minimum_balance.mean(),
        "ax_close_min": g.closing_balance.min(),
        "ax_max_withdraw_months": g.apply(
            lambda d: float(((d.closing_balance - d.opening_balance) < 0).sum()), include_groups=False),
    })

    def longest_zero(d):
        z = (d.income_like_credits <= 0).to_numpy()
        best = cur = 0
        for v in z:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return float(best)

    def inc_gap(d):
        v = d.income_like_credits.to_numpy() > 0
        best = 0
        for i in range(1, len(v)):
            if v[i] and v[i - 1]:
                best = max(best, i)
        return float(best)

    def slope(d):
        y = d.income_like_credits.to_numpy(dtype=float)
        n = len(y)
        if n < 3 or y.mean() == 0:
            return np.nan
        x = np.arange(n, dtype=float)
        return float(np.polyfit(x, y, 1)[0] / (abs(y.mean()) + 1e-9) * 12)

    agg = {
        "ax_inc_zero_streak": g.apply(longest_zero, include_groups=False),
        "ax_inc_gap_max": g.apply(inc_gap, include_groups=False),
        "ax_inc_slope": g.apply(slope, include_groups=False),
    }
    out = out.join(pd.DataFrame(agg))

    last3 = ab.groupby("customer_id").tail(3)
    l3 = last3.groupby("customer_id")
    out["ax_inc_last3_min"] = l3.income_like_credits.min()
    out["ax_inc_last3_mean_over_all"] = out.ax_inc_last3_min / g.income_like_credits.mean().replace(0, np.nan)
    out["ax_stress_last3"] = last3.groupby("customer_id").stress.mean()
    out["ax_lowbal_slope"] = out.ax_low_days_mean - g.low_balance_days.mean()
    out["ax_stress_share"] = out.ax_stress_months / g.size()
    return out.reset_index()


# ------------------------------------------------------------ ecommerce -----
def block_ecommerce(app):
    ec = pd.read_csv(
        D + "ecommerce_transactions.csv",
        parse_dates=["transaction_datetime"],
        usecols=["customer_id", "order_id", "transaction_datetime", "product_category",
                 "quantity", "item_total", "order_total", "order_status", "payment_method"],
    )
    ec = ec.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    ec = ec[ec.transaction_datetime <= ec.cut]
    ec["days_ago"] = (ec.cut - ec.transaction_datetime).dt.total_seconds() / 86400
    done = ec[ec.order_status == "completed"].copy()
    od = done.drop_duplicates("order_id")

    od["disc"] = od.product_category.isin(CAT_DISC).astype(float)
    g = od.groupby("customer_id")
    out = pd.DataFrame({
        "ex_disc_share": g.disc.mean(),
        "ex_items_per_order": g.quantity.mean(),
        "ex_distinct_items": od.groupby("customer_id").quantity.nunique(),
    })
    out["ex_ess_share"] = g.apply(
        lambda d: float(d.product_category.isin(CAT_ESS).mean()), include_groups=False)
    out["ex_order_max_over_mean"] = g.order_total.max() / g.order_total.mean().replace(0, np.nan)
    out["ex_order_mean_cv"] = g.order_total.apply(
        lambda s: float(s.std() / s.mean()) if s.mean() > 0 else np.nan)
    out["ex_cat_n"] = g.product_category.nunique()
    catshare = od.pivot_table(index="customer_id", columns="product_category",
                              values="order_total", aggfunc="size", fill_value=0)
    share = catshare.div(catshare.sum(axis=1), axis=0)
    out["ex_cat_entropy"] = -(share * np.log(share.replace(0, np.nan))).sum(axis=1)
    out["ex_top_cat_share"] = share.max(axis=1)
    out["ex_electronics_share"] = share.get("electronics", 0.0)

    it = done[done.quantity > 0].copy()
    unit = it.item_total / it.quantity
    it["lrel"] = np.log(unit / unit.groupby(it.product_category).transform("median"))
    gi = it.groupby("customer_id")
    out["ex_lrel_n"] = gi.lrel.size()
    out["ex_lrel_abs_mean"] = gi.lrel.apply(lambda s: float(s.abs().mean()))
    out["ex_lrel_abs_max"] = gi.lrel.apply(lambda s: float(s.abs().max()))
    for hl in (30, 90, 240):
        w = _expw(it.days_ago, hl)
        out[f"ex_lrel_w{hl}"] = (w * it.lrel).groupby(it.customer_id).sum() / w.groupby(it.customer_id).sum()
    out["ex_spend_per_item"] = it.groupby("customer_id").item_total.sum() / it.groupby("customer_id").size()
    return out.reset_index()


# ----------------------------------------------------------- interactions ---
def block_interactions(extra: pd.DataFrame):
    f = extra
    out = pd.DataFrame({"application_id": f.application_id})
    inc = f.declared_monthly_income
    out["ix_rate_x_dti"] = f.interest_rate * f.dti_total
    out["ix_rate_x_income"] = f.interest_rate * np.log1p(inc)
    out["ix_amt_x_age"] = f.loan_amount / (f.age * 1e6 + 1)
    out["ix_inst_x_late"] = f.installment * f.rh_rate1_smooth
    out["ix_rate_x_rent"] = f.interest_rate * (f.housing_status.astype(str) == "rented")
    out["ix_rate_x_self"] = f.interest_rate * (f.employment_type.astype(str) == "self_employed")
    out["ix_rate_x_var"] = f.interest_rate * (f.employment_type.astype(str) == "variable_income")
    out["ix_inst_x_ec_spend"] = f.installment / (f.ec_spend90 / 3 + 1)
    out["ix_inc_vs_spend"] = np.log1p(inc) / np.log1p(f.ec_spend_base / 9 + 1)
    out["ix_rate_x_late"] = f.interest_rate * f.rh_late
    out["ix_rate_x_price"] = f.interest_rate * f.ec_price_rel
    out["ix_dti_x_price"] = f.dti_total * f.ec_price_rel
    out["ix_stress"] = f.installment / (f.ab_l3_income_like_credits + 1)
    return out


BLOCKS = {
    "bureau": block_bureau,
    "repayment": block_repayment,
    "previous": block_previous,
    "account": block_account,
    "ecommerce": block_ecommerce,
    "interactions": block_interactions,
}


# ---------------------------------------------------------------- round 2 ---
GN_KEYS = [
    "ec_price_rel", "ec_price_rel_trend", "ec_spend90_vs_base", "dti_total",
    "loan_to_income", "ab_l3_income_like_credits", "inst_to_income", "bu_score_mean",
    "bu_score_min", "bu_util_max", "bu_delinq_max", "rh_late", "rh_rate1_smooth",
    "rh_maxdpd_max", "pa_reject_rate", "pa_active_n", "ec_orders", "ab_months",
    "declared_monthly_income", "age", "dependents", "employment_tenure_months",
]


def block_gradenorm(app, base=None):
    """Within-grade z-scores.

    The interest rate is a coarse 5 level risk grade. A customer who buys
    expensive electronics is unremarkable inside a high grade bucket but
    remarkable inside a low one, so the same raw feature carries different
    information per grade. Trees cannot express that without a very deep
    interaction, so it is written out explicitly.
    """
    out = pd.DataFrame({"application_id": base.application_id.values})
    for grp in ("interest_rate", "employment_type", "region"):
        g = base[grp].astype(str)
        for c in GN_KEYS:
            if c not in base.columns:
                continue
            v = pd.to_numeric(base[c], errors="coerce")
            m = v.groupby(g).transform("mean")
            s = v.groupby(g).transform("std")
            key = f"gn_{grp[:3]}_{c}"
            out[key] = (v - m) / s.replace(0, np.nan)
    return out


def block_rxtra(app):
    """Affordability history and on-time payment discipline."""
    rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date", "last_payment_datetime"])
    rh = rh.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.total_seconds() / 86400
    rh = rh[rh.days_ago >= 0]
    rh["ontime"] = ((rh.max_days_past_due == 0) & (rh.amount_paid >= rh.amount_due)).astype(int)
    rh["matured30"] = (rh.days_ago >= 30).astype(int)
    rh["untouched"] = (rh.max_days_past_due == 0).astype(int)
    per = rh.groupby(["customer_id", "loan_id"]).agg(
        amt=("amount_due", "mean"), mx=("max_days_past_due", "max"),
        n=("amount_due", "size"), end=("due_date", "max"), seq=("due_date", "size"),
    )
    g = per.groupby(level=0)
    out = pd.DataFrame({
        "r2_amt_mean": g.amt.mean(),
        "r2_amt_max": g.amt.max(),
        "r2_amt_min": g.amt.min(),
        "r2_amt_cv": g.amt.apply(lambda s: float(s.std() / s.mean()) if s.mean() > 0 else np.nan),
        "r2_loan_dpd_max": g.mx.max(),
        "r2_loan_dpd_mean": g.mx.mean(),
        "r2_last_amt": per.sort_values("end").groupby(level=0).amt.last(),
    })
    gi = rh.groupby("customer_id")
    out["r2_ontime_rate"] = gi.ontime.sum() / gi.matured30.sum().replace(0, np.nan)
    out["r2_untouched_rate"] = gi.untouched.mean()
    out["r2_amt_slope"] = g.amt.apply(
        lambda s: float(np.polyfit(np.arange(len(s), dtype=float), s.to_numpy(), 1)[0])
        if len(s) > 2 else np.nan)
    out["r2_amt_per_loan_trend"] = out.r2_amt_slope / out.r2_amt_mean.replace(0, np.nan)
    return out.reset_index()


def block_bxtra(app):
    """Depth, span and direction of the bureau file."""
    bu = pd.read_csv(D + "bureau_observations.csv", parse_dates=["report_datetime"])
    bu = bu.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    bu = bu[bu.report_datetime <= bu.cut]
    bu["span"] = (bu.cut - bu.report_datetime).dt.total_seconds() / 86400

    def slope(d):
        v = d.score.to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() < 3:
            return np.nan
        x = d.span.to_numpy(dtype=float)[ok]
        return float(np.polyfit(x, v[ok], 1)[0])

    g = bu.groupby("customer_id")
    out = pd.DataFrame({
        "b2_span_max": g.span.max(),
        "b2_months_covered": g.report_datetime.apply(lambda s: s.dt.to_period("M").nunique()),
        "b2_report_gap": g.report_datetime.apply(
            lambda s: float(s.sort_values().diff().dt.total_seconds().median() / 86400)),
        "b2_util_max": g.utilization.max(),
        "b2_util_min": g.utilization.min(),
        "b2_delinq_ever": g.delinquent_accounts_last_12m.max(),
        "b2_active_mean": g.active_accounts.mean(),
        "b2_out_max": g.outstanding_balance.max(),
        "b2_score_slope": g.apply(slope, include_groups=False),
    })
    out["b2_score_span"] = g.span.max() - g.span.min()
    out["b2_thickness"] = g.size() / out.b2_months_covered.replace(0, np.nan)
    first = bu.sort_values("report_datetime").groupby("customer_id").score.first()
    last = bu.sort_values("report_datetime").groupby("customer_id").score.last()
    out["b2_score_first"] = first
    out["b2_score_improve"] = last - first
    return out.reset_index()


def block_extra2(app):
    """Ecommerce tenure, repeat buying and basket composition drift."""
    ec = pd.read_csv(
        D + "ecommerce_transactions.csv",
        parse_dates=["transaction_datetime"],
        usecols=["customer_id", "order_id", "transaction_datetime", "product_category",
                 "quantity", "order_total", "order_status"],
    )
    ec = ec.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    ec = ec[ec.transaction_datetime <= ec.cut]
    done = ec[ec.order_status == "completed"].drop_duplicates("order_id").sort_values(
        ["customer_id", "transaction_datetime"])
    done["days_ago"] = (done.cut - done.transaction_datetime).dt.total_seconds() / 86400
    seen = done.groupby("customer_id").product_category.transform("first").notna()
    prior = done.groupby(["customer_id", "product_category"]).cumcount()
    done["repeat_cat"] = (prior > 0).astype(int)

    g = done.groupby("customer_id")
    out = pd.DataFrame({
        "e2_repeat_cat_rate": done.groupby("customer_id").repeat_cat.mean(),
        "e2_qty_std": g.quantity.std(),
        "e2_qty_max": g.quantity.max(),
        "e2_basket_mean": g.order_total.mean(),
        "e2_basket_std": g.order_total.std(),
        "e2_recent_repeat": done[done.days_ago <= 180].groupby("customer_id").repeat_cat.mean(),
    })
    span = (g.transaction_datetime.max() - g.transaction_datetime.min()).dt.total_seconds() / 86400
    out["e2_span_days"] = span
    out["e2_orders_per_month"] = g.size() / (span / 30.44 + 1)
    out["e2_basket_cv"] = out.e2_basket_std / out.e2_basket_mean.replace(0, np.nan)
    firstq = done.groupby("customer_id").quantity.first()
    lastq = done.groupby("customer_id").quantity.last()
    out["e2_qty_trend"] = lastq - firstq
    return out.reset_index()


def block_axtra(app):
    """Monthly cash-flow discipline rather than plain levels."""
    ab = pd.read_csv(D + "account_behavior.csv", parse_dates=["month"])
    ab = ab.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut].sort_values(["customer_id", "month"])
    ab["net"] = ab.inflows - ab.outflows
    ab["disposable"] = ab.income_like_credits - ab.outflows

    g = ab.groupby("customer_id")

    def cv(s):
        return float(s.std() / abs(s.mean())) if abs(s.mean()) > 0 else np.nan

    out = pd.DataFrame({
        "a2_disposable_cv": g.disposable.apply(cv),
        "a2_net_cv": g.net.apply(cv),
        "a2_bal_slope": g.average_balance.apply(
            lambda s: float(np.polyfit(np.arange(len(s), dtype=float), s.to_numpy(), 1)[0])
            if len(s) > 2 else np.nan),
        "a2_min_slope": g.minimum_balance.apply(
            lambda s: float(np.polyfit(np.arange(len(s), dtype=float), s.to_numpy(), 1)[0])
            if len(s) > 2 else np.nan),
        "a2_inc_slope": g.income_like_credits.apply(
            lambda s: float(np.polyfit(np.arange(len(s), dtype=float), s.to_numpy(), 1)[0])
            if len(s) > 2 else np.nan),
    })
    out["a2_net_mean"] = g.net.mean()
    out["a2_net_negative_share"] = ab.groupby("customer_id").net.apply(lambda s: float((s < 0).mean()))
    out["a2_out_gt_inc"] = ab.groupby("customer_id").apply(
        lambda d: float((d.outflows > d.income_like_credits).mean()), include_groups=False)
    out["a2_low_days_worst"] = ab.groupby("customer_id").low_balance_days.max()
    out["a2_low_recent"] = ab.groupby("customer_id").tail(3).groupby("customer_id").low_balance_days.mean()
    out["a2_low_delta"] = out.a2_low_recent - out.a2_low_days_worst
    return out.reset_index()




# ---------------------------------------------------------------- round 3 ---
def block_consistency(app):
    """Cross-source mismatches.

    A customer whose bureau file looks clean while the internal repayment file
    says otherwise is behaving differently from what the bureau reports, and
    the gap itself carries risk information that neither source carries alone.
    The same idea is applied to declared income versus observed cash flow,
    bureau-reported debt versus the instalments we can see, and the size of the
    new instalment versus the largest instalment the customer already carries.
    """
    cp = pd.read_csv(D + "customer_profiles.csv", usecols=["customer_id", "declared_monthly_income"])
    prof = app[["customer_id"]].merge(cp, on="customer_id", how="left")
    inc = prof.set_index("customer_id").declared_monthly_income
    inst = app.set_index("customer_id").proposed_monthly_installment

    pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
    pa = pa.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    pa["days_ago"] = (pa.cut - pa.application_datetime).dt.total_seconds() / 86400
    pa = pa[pa.days_ago >= 0]
    pa["months_elapsed"] = pa.days_ago / 30.44
    pa["active"] = ((pa.decision == "approved") & (pa.months_elapsed < pa.term_months)).astype(float)
    pa["active_inst"] = np.where(pa.active > 0, pa.scheduled_monthly_installment, 0.0)
    pa["remaining"] = np.where(pa.active > 0,
                                pa.approved_amount * (1 - pa.months_elapsed / pa.term_months), 0.0)
    g = pa.groupby("customer_id")
    pa_n_approved = g.apply(lambda d: float((d.decision == "approved").sum()), include_groups=False)
    pa_max_inst = g.active_inst.max()
    pa_inst_sum = g.active_inst.sum()
    pa_recent = pa[pa.days_ago <= 180]
    pa_first_days = g.days_ago.max()

    bu = pd.read_csv(D + "bureau_observations.csv", parse_dates=["report_datetime"])
    bu = bu.merge(_cut(app)[["customer_id", "cut"]], on="customer_id")
    bu = bu[bu.report_datetime <= bu.cut]
    gb = bu.groupby("customer_id")
    bu_score_min = gb.score.min()
    bu_delinq_max = gb.delinquent_accounts_last_12m.max()
    bu_inq_max = gb.inquiries_last_6m.max()
    bu_out_max = gb.outstanding_balance.max()
    bu_active_max = gb.active_accounts.max()
    bu_first_days = gb.report_datetime.apply(lambda s: float((bu.cut.loc[s.index] - s).dt.days.max())
                                             if len(s) else np.nan)

    out = pd.DataFrame(index=app.customer_id)
    yr_income = inc * 12
    out["cx_bureau_vs_declared_debt"] = bu_out_max / (yr_income * 0.35).replace(0, np.nan)
    out["cx_bureau_debt_vs_visible_inst"] = bu_out_max / ((pa_inst_sum + inst) * 12).replace(0, np.nan)
    out["cx_bureau_accounts_vs_our_loans"] = bu_active_max / (pa_n_approved + 1)
    out["cx_inquiries_vs_our_apps"] = bu_inq_max / (pa_n_approved + 1)
    rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date"]).merge(
        _cut(app)[["customer_id", "cut"]], on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.days
    rh = rh[rh.days_ago >= 1]
    rh["bad"] = (rh.max_days_past_due >= 30).astype(float)
    rh_bad_loans = rh.groupby(["customer_id", "loan_id"]).bad.max().groupby("customer_id").sum()
    rh_late_loans = rh[rh.bad > 0].groupby("loan_id").size()
    out["cx_delinq_gap"] = np.log1p(bu_delinq_max.reindex(out.index).fillna(0)) - np.log1p(rh_bad_loans.reindex(out.index).fillna(0))
    out["cx_bureau_delinq_per_internal_late"] = bu_delinq_max / (rh_bad_loans + 1)
    out["cx_file_age_gap_days"] = bu_first_days - pa_first_days
    out["cx_inst_vs_hist_max"] = inst / pa_max_inst.replace(0, np.nan)
    out["cx_inst_vs_hist_mean"] = inst / (pa_max_inst / 2 + 1).replace(0, np.nan)
    out["cx_remaining_vs_inst"] = g.remaining.sum() / (inst * 12).replace(0, np.nan)
    out["cx_recent_app_pressure"] = pa_recent.groupby("customer_id").size()
    out["cx_bureau_known_but_no_apps"] = (bu_score_min.notna()).astype(float) - (pa_n_approved > 0).astype(float)
    return out.reset_index().rename(columns={"index": "customer_id"})


def _load_v65_blocks():
    """Register the v65 blocks from agas/v65/training/feat_v65.py.

    They live in the version folder rather than in this shared module because
    they were written for one run and have not earned a place in the shared
    vocabulary yet. Registering them here keeps `build_extra.py` and the CV
    harness working without either knowing where the code sits.
    """
    import importlib.util as _il
    here = Path(__file__).resolve().parent
    for p in here.parents:
        cand = p / "v65" / "training" / "feat_v65.py"
        if cand.exists():
            spec = _il.spec_from_file_location("feat_v65", cand)
            mod = _il.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


_v65 = _load_v65_blocks()
if _v65 is not None:
    BLOCKS.update({
        "dataquality": _v65.block_dataquality,
        "burdirection": _v65.block_burdirection,
        "stress": _v65.block_stress,
    })


BLOCKS.update({
    "gradenorm": block_gradenorm,
    "rxtra": block_rxtra,
    "bxtra": block_bxtra,
    "ecommerce2": block_extra2,
    "axtra": block_axtra,
    "consistency": block_consistency,
})



BASE_COLS = [
    "declared_monthly_income", "interest_rate", "loan_amount", "installment",
    "housing_status", "employment_type", "dti_total", "rh_rate1_smooth",
    "rh_late", "ec_price_rel", "ec_spend90", "ec_spend_base", "ab_l3_income_like_credits",
    "region", "age", "dependents", "loan_purpose",
]


def build(names=None, app=None, base=None):
    names = list(names) if names else list(BLOCKS)
    if app is None:
        app = pd.read_csv(
            D + "train_applications.csv",
            parse_dates=["application_datetime", "decision_datetime"],
        )
        te = pd.read_csv(D + "test_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
        app = pd.concat([app, te], ignore_index=True)

    cmap = app[["customer_id", "application_id"]].drop_duplicates("customer_id")
    parts = []
    for n in names:
        if n == "interactions":
            continue
        p = BLOCKS[n](app, base=base) if n == "gradenorm" else BLOCKS[n](app)
        if "application_id" not in p.columns:
            p = p.merge(cmap, on="customer_id", how="left").drop(columns=["customer_id"])
        parts.append(p[["application_id"] + [c for c in p.columns if c != "application_id"]])
    if not parts:
        extra = pd.DataFrame({"application_id": app.application_id})
    else:
        extra = parts[0]
    for p in parts[1:]:
        extra = extra.merge(p, on="application_id", how="outer")

    return extra


def with_interactions(extra, base):
    if base is None:
        return extra
    inter = block_interactions(base.merge(extra, on="application_id", how="left"))
    return inter


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths

    e = build()
    print(e.shape)
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    e.to_pickle(paths.EXTRA_CACHE)