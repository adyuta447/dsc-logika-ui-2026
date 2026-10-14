"""Feature engineering v2: union of the temporal-window framework and the
signal ideas that proved strongest on the leaderboard.

Two DPD views are produced for the repayment table:

* ``asof``  - DPD recomputed at the application cutoff (strictly causal).
* ``raw``   - the DPD columns exactly as the dataset publishes them.

Both are kept so their contribution can be measured; the notebook documents
which one is used for the submitted model.
"""

import numpy as np
import pandas as pd

from fe_core import asof_point, window_stats

W_LONG = {"m3": 91, "m6": 182, "m12": 365, "all": None}
W_SHORT = {"d30": 30, "d90": 90, "d180": 180, "d365": 365, "all": None}


def _queries(apps):
    return apps[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})


# --------------------------------------------------------------------------- #
# 1. application + profile
# --------------------------------------------------------------------------- #
def app_features(apps, profiles):
    df = apps.merge(profiles, on="customer_id", how="left", validate="many_to_one")
    out = pd.DataFrame(index=df.index)
    out["loan_amount"] = df.loan_amount
    out["term_months"] = df.term_months
    out["interest_rate"] = df.annual_interest_rate
    out["installment"] = df.proposed_monthly_installment
    out["app_hour"] = df.application_datetime.dt.hour
    out["app_dow"] = df.application_datetime.dt.dayofweek
    out["app_month"] = df.application_datetime.dt.month
    out["decision_hour"] = df.decision_datetime.dt.hour
    out["decision_dow"] = df.decision_datetime.dt.dayofweek
    out["decision_month"] = df.decision_datetime.dt.month
    out["decision_weekofyear"] = df.decision_datetime.dt.isocalendar().week.astype(int)
    out["decision_lag_h"] = (df.decision_datetime - df.application_datetime).dt.total_seconds() / 3600
    out["decision_tenure_days"] = (df.decision_datetime - pd.Timestamp("2023-01-01")).dt.days
    out["age"] = df.age
    out["employment_tenure_months"] = df.employment_tenure_months
    out["declared_monthly_income"] = df.declared_monthly_income
    out["dependents"] = df.dependents
    for c in ["region", "employment_type", "housing_status", "loan_purpose"]:
        out[c] = df[c].astype("category")

    inc = df.declared_monthly_income.replace(0, np.nan)
    out["inst_to_income"] = df.proposed_monthly_installment / inc
    out["loan_to_income"] = df.loan_amount / (inc * 12)
    out["loan_to_installment"] = df.loan_amount / df.proposed_monthly_installment.replace(0, np.nan)
    out["installment_x_term"] = df.proposed_monthly_installment * df.term_months
    out["total_repay"] = df.proposed_monthly_installment * df.term_months
    out["dep_per_income_mio"] = df.dependents / inc * 1e6
    out["age_minus_tenure"] = df.age * 12 - df.employment_tenure_months
    out["tenure_to_age"] = df.employment_tenure_months / (df.age * 12)
    out["income_per_dependent"] = inc / (1 + df.dependents)
    out["installment_per_dependent"] = df.proposed_monthly_installment / (1 + df.dependents)
    out["rate_x_loan"] = df.annual_interest_rate * df.loan_amount / 1e6
    out["rate_per_term"] = df.annual_interest_rate / df.term_months.replace(0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# 2. credit bureau
# --------------------------------------------------------------------------- #
def bureau_features(apps, bureau):
    ev = bureau.copy()
    ev["report_lag_days"] = (ev.available_datetime - ev.report_datetime).dt.total_seconds() / 86400
    ev = ev.rename(columns={"available_datetime": "event_time"})
    ev["score_f"] = ev.score.fillna(ev.score.mean())
    ev["score_missing"] = ev.score.isna().astype(float)
    ev["utilization_f"] = ev.utilization.fillna(0.0)
    ev["revolving_free"] = ev.revolving_credit_limit - ev.revolving_outstanding_balance
    num = ["score_f", "score_missing", "active_accounts", "outstanding_balance", "revolving_credit_limit",
           "revolving_outstanding_balance", "utilization_f", "revolving_free", "inquiries_last_6m",
           "delinquent_accounts_last_12m", "report_lag_days"]
    q = _queries(apps)
    stats, _ = window_stats(ev, q, value_cols=num, windows=W_LONG,
                            ops=("sum", "mean", "std", "min", "max", "last"))
    latest, _ = asof_point(ev, q, num)
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["bu_" + c for c in stats.columns] + ["bu_last_" + c for c in latest.columns]

    # laporan yang terpaut langsung pada pengajuan terpilih
    cur = ev.merge(apps[["application_id", "customer_id", "decision_datetime"]]
                   .rename(columns={"decision_datetime": "cut"}),
                   on=["application_id", "customer_id"], how="inner")
    g = cur.groupby("application_id")
    agg = g.agg(
        bu_cur_score_mean=("score", "mean"), bu_cur_score_min=("score", "min"),
        bu_cur_score_max=("score", "max"), bu_cur_score_std=("score", "std"),
        bu_cur_n=("score", "size"), bu_cur_unscored=("score_missing", "sum"),
        bu_cur_out_max=("outstanding_balance", "max"), bu_cur_out_mean=("outstanding_balance", "mean"),
        bu_cur_limit_max=("revolving_credit_limit", "max"),
        bu_cur_revbal_max=("revolving_outstanding_balance", "max"),
        bu_cur_util_max=("utilization_f", "max"), bu_cur_util_mean=("utilization_f", "mean"),
        bu_cur_inq_max=("inquiries_last_6m", "max"), bu_cur_delinq_max=("delinquent_accounts_last_12m", "max"),
        bu_cur_delinq_sum=("delinquent_accounts_last_12m", "sum"),
        bu_cur_active_max=("active_accounts", "max"), bu_cur_age_min=("report_lag_days", "min"),
    )
    last_cur = cur.sort_values("event_time").groupby("application_id").last()
    for c in ["score", "outstanding_balance", "utilization_f", "inquiries_last_6m",
              "delinquent_accounts_last_12m", "active_accounts"]:
        agg[f"bu_cur_latest_{c}"] = last_cur[c]
    piv = cur.pivot_table(index="application_id", columns="bureau_provider", values="score")
    agg = agg.join(piv.add_prefix("bu_score_"))
    out = out.reset_index(drop=True)
    apps_idx = apps.reset_index(drop=True)
    pos = pd.Series(np.arange(len(apps_idx)), index=apps_idx.application_id)
    for c in agg.columns:
        out[c] = agg[c].reindex(apps_idx.application_id).to_numpy()
    out["bu_score_trend"] = out["bu_last_score_f"] - out["bu_mean_score_f__m12"]
    out["bu_score_trend_6m"] = out["bu_last_score_f"] - out["bu_mean_score_f__m6"]
    out["bu_score_trend_cur"] = out["bu_cur_latest_score"] - out["bu_cur_score_mean"]
    out["bu_provider_spread"] = agg[[f"bu_score_{p}" for p in ["Bureau_A", "Bureau_B", "Bureau_C"]]].max(axis=1).to_numpy() - \
                                agg[[f"bu_score_{p}" for p in ["Bureau_A", "Bureau_B", "Bureau_C"]]].min(axis=1).to_numpy()
    out["bu_score_min_history"] = out[["bu_cur_score_min", "bu_min_score_f__all"]].min(axis=1)
    return out


# --------------------------------------------------------------------------- #
# 3. previous applications
# --------------------------------------------------------------------------- #
def previous_features(apps, prev):
    ev = prev.rename(columns={"decision_datetime": "event_time"}).copy()
    ev["approved"] = (ev.decision == "approved").astype(float)
    ev["rejected"] = (ev.decision == "rejected").astype(float)
    ev["cancelled"] = (ev.decision == "cancelled").astype(float)
    ev["was_originated"] = ev.loan_id.notna().astype(float)
    ev["approved_amount_f"] = ev.approved_amount.fillna(0.0)
    ev["requested_amount"] = ev.requested_amount.astype(float)
    ev["rate_f"] = ev.annual_interest_rate.fillna(0.0)
    ev["sched_inst"] = ev.scheduled_monthly_installment.fillna(0.0)

    q = _queries(apps)
    # posisi pinjaman sebelumnya terhadap pengajuan terpilih (aktif atau sudah jatuh tempo)
    days_ago = np.zeros(len(ev))
    cut_map = apps.set_index("customer_id")["decision_datetime"]
    ev["cut"] = ev.customer_id.map(cut_map)
    days_ago = (ev["cut"] - ev["event_time"]).dt.total_seconds() / 86400
    ev["months_elapsed"] = days_ago / 30.44
    active = (ev.approved == 1) & (ev.months_elapsed < ev.term_months)
    ev["active"] = active.astype(float)
    ev["active_inst"] = np.where(active, ev.scheduled_monthly_installment, 0.0)
    ev["active_amount"] = np.where(active, ev.approved_amount.fillna(0.0), 0.0)
    ev["remaining_principal"] = np.where(active, ev.approved_amount.fillna(0.0) *
                                         (1 - ev.months_elapsed / ev.term_months.replace(0, np.nan)), 0.0)
    ev = ev.sort_values(["customer_id", "event_time"], kind="mergesort")
    ev["days_since_prev_app"] = (ev.groupby("customer_id")["event_time"].diff()
                                 .dt.total_seconds() / 86400).fillna(-1.0)

    stats, _ = window_stats(
        ev, q,
        value_cols=["requested_amount", "approved_amount_f", "approved", "rejected", "cancelled",
                    "was_originated", "days_since_prev_app", "active", "active_inst", "active_amount",
                    "remaining_principal", "rate_f", "sched_inst"],
        windows=W_LONG, ops=("sum", "mean", "std", "max", "last"))
    latest, _ = asof_point(ev, q, ["approved", "rejected", "cancelled", "active", "active_inst",
                                   "approved_amount_f", "requested_amount", "rate_f"])
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["pa_" + c for c in stats.columns] + ["pa_last_" + c for c in latest.columns]
    out["pa_approval_rate"] = out["pa_sum_approved__all"] / out["pa_n_events__all"].replace(0, np.nan)
    out["pa_reject_rate"] = out["pa_sum_rejected__all"] / out["pa_n_events__all"].replace(0, np.nan)
    out["pa_cancel_rate"] = out["pa_sum_cancelled__all"] / out["pa_n_events__all"].replace(0, np.nan)
    out["pa_amount_ratio"] = out["pa_sum_approved_amount_f__all"] / out["pa_sum_requested_amount__all"].replace(0, np.nan)
    out["pa_n_active"] = out["pa_sum_active__all"]
    out["pa_active_inst"] = out["pa_sum_active_inst__all"]
    out["pa_remaining"] = out["pa_sum_remaining_principal__all"]
    out["pa_rate_mean"] = out["pa_mean_rate_f__all"]
    out["pa_recent_rejection"] = (out["pa_sum_rejected__m3"] > 0).astype(float)
    out["pa_days_since_last_event"] = out["pa_last_days_since_ts"]
    out["pa_active_amount"] = out["pa_sum_active_amount__all"]
    return out


# --------------------------------------------------------------------------- #
# 4. repayment history
# --------------------------------------------------------------------------- #
def _repayment_events(repayments, apps):
    ev = repayments.rename(columns={"due_date": "event_time"}).copy()
    ev["cut"] = ev.customer_id.map(apps.set_index("customer_id")["decision_datetime"])
    if ev["cut"].isna().any():
        raise ValueError("repayment_history memuat customer di luar tabel aplikasi")
    ev = ev[ev.event_time <= ev.cut].copy()
    ev = ev.dropna(subset=["last_payment_datetime"], how="all")

    settled = (ev.amount_paid >= ev.amount_due) & (ev.last_payment_datetime <= ev.cut)
    days_ago = (ev.cut - ev.event_time).dt.total_seconds() / 86400
    ev["days_ago"] = days_ago
    ev["late_asof"] = ((~settled) & (days_ago >= 1)).astype(float)
    ev["dpd_asof"] = np.where(settled, 0.0, days_ago)
    ev["outstanding_asof"] = np.where(settled, 0.0, ev.amount_due.astype(float))
    ev["pay_ratio"] = np.where(settled, 1.0, 0.0)
    ev["pay_delay"] = (ev.last_payment_datetime - ev.event_time).dt.total_seconds() / 86400
    ev["amount_due"] = ev.amount_due.astype(float)
    ev["unpaid"] = (ev.amount_due - ev.amount_paid).astype(float)
    ev["dpd_raw"] = ev.days_past_due.astype(float)
    ev["maxdpd_raw"] = ev.max_days_past_due.astype(float)
    ev["late_raw"] = (ev.max_days_past_due > 0).astype(float)
    ev["late30_raw"] = (ev.max_days_past_due >= 30).astype(float)
    ev["late90_raw"] = (ev.max_days_past_due >= 90).astype(float)
    return ev.sort_values(["customer_id", "loan_id", "event_time"], kind="mergesort").reset_index(drop=True)


def _matured_rates(ev, tag, dpd_col, prefix):
    """Tingkat keterlambatan hanya atas cicilan yang sempat punya waktu untuk terlambat."""
    out = {}
    for t in [1, 15, 30, 60, 90]:
        matured = (ev.days_ago >= t).astype(float)
        hit = ((ev[dpd_col] >= t) & (ev.days_ago >= t)).astype(float)
        out[f"{prefix}mat{t}"] = matured.groupby(ev.customer_id).sum()
        out[f"{prefix}hit{t}"] = hit.groupby(ev.customer_id).sum()
    cols = {k: v for k, v in out.items()}
    frame = pd.DataFrame(cols)
    for t in [1, 15, 30, 60, 90]:
        frame[f"{prefix}rate{t}"] = frame[f"{prefix}hit{t}"] / frame[f"{prefix}mat{t}"].replace(0, np.nan)
        frame[f"{prefix}rate{t}_smooth"] = (frame[f"{prefix}hit{t}"] + 0.3) / (frame[f"{prefix}mat{t}"] + 1)
    return frame.drop(columns=[f"{prefix}mat{t}" for t in [1, 15, 30, 60, 90]])


def repayment_features(apps, repayments):
    ev = _repayment_events(repayments, apps)
    q = _queries(apps)
    ev["dpd_ge_1"] = (ev.dpd_asof >= 1).astype(float)
    ev["dpd_ge_30"] = (ev.dpd_asof >= 30).astype(float)
    ev["dpd_ge_60"] = (ev.dpd_asof >= 60).astype(float)
    ev["dpd_ge_89"] = (ev.dpd_asof >= 89).astype(float)
    asof_cols = ["amount_due", "outstanding_asof", "pay_ratio", "dpd_asof", "late_asof", "dpd_raw",
                 "dpd_ge_1", "dpd_ge_30", "dpd_ge_60", "dpd_ge_89"]
    raw_cols = ["unpaid", "maxdpd_raw", "late_raw", "late30_raw", "late90_raw", "pay_delay"]

    stats, _ = window_stats(ev, q, value_cols=asof_cols + raw_cols, windows=W_LONG,
                            ops=("sum", "mean", "std", "max", "last"))
    latest, _ = asof_point(ev, q, ["dpd_asof", "maxdpd_raw", "outstanding_asof", "pay_ratio", "late_raw"])
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["rh_" + c for c in stats.columns] + ["rh_last_" + c for c in latest.columns]

    rates = pd.concat([_matured_rates(ev, "asof", "dpd_asof", "rha_"),
                       _matured_rates(ev, "raw", "maxdpd_raw", "rhr_")], axis=1)
    out = pd.concat([out.reset_index(drop=True), rates.reset_index(drop=True)], axis=1)

    # level pinjaman
    per_loan = ev.groupby(["customer_id", "loan_id"]).agg(
        l_late=("late_raw", "mean"), l_late_asof=("late_asof", "mean"),
        l_dpd=("maxdpd_raw", "max"), l_dpd_mean=("maxdpd_raw", "mean"),
        l_n=("event_time", "size"), l_out=("outstanding_asof", "sum"),
        l_start=("event_time", "min"), l_end=("event_time", "max"),
        l_unpaid=("unpaid", "sum"), l_due=("amount_due", "sum"))
    g = per_loan.groupby("customer_id")
    pl = pd.DataFrame({
        "ln_n_loans": g.size(),
        "ln_n_inst": g.l_n.sum(),
        "ln_worst_loan_late": g.l_late.max(),
        "ln_best_loan_late": g.l_late.min(),
        "ln_max_dpd": g.l_dpd.max(),
        "ln_mean_worst_dpd": g.l_dpd.mean(),
        "ln_worst_late_asof": g.l_late_asof.max(),
        "ln_outstanding": g.l_out.sum(),
        "ln_unpaid": g.l_unpaid.sum(),
        "ln_first_start": g.l_start.min().astype("int64"),
        "ln_last_end": g.l_end.max().astype("int64"),
    })
    pl["ln_last_loan_late"] = per_loan.sort_values("l_end").groupby("customer_id").l_late.last()
    pl["ln_last_loan_dpd"] = per_loan.sort_values("l_end").groupby("customer_id").l_dpd.last()
    pl["ln_unpaid_ratio"] = pl.ln_unpaid / g.l_due.sum().replace(0, np.nan)
    apps_idx = apps.reset_index(drop=True)
    decision = pd.Series(pd.to_datetime(apps_idx.decision_datetime.values),
                         index=pd.RangeIndex(len(apps_idx)))
    last_end = pd.to_datetime(pl.ln_last_end)
    pl["ln_days_since_last_loan"] = (decision - last_end).dt.total_seconds() / 86400
    pl["ln_loan_per_year"] = pl.ln_n_loans / (pl.ln_days_since_last_loan / 365.25).replace(0, np.nan)
    out = pd.concat([out.reset_index(drop=True), pl.reset_index(drop=True)], axis=1)

    # urutan cicilan: roll rate, cure rate, streak, slope
    sq = ev.copy()
    sq["seq"] = sq.groupby(["customer_id", "loan_id"]).cumcount()
    prev_late = sq.groupby(["customer_id", "loan_id"]).late_raw.shift()
    sq["roll"] = sq.late_raw.where(prev_late == 1)
    sq["cure"] = (1 - sq.late_raw).where(prev_late == 1)
    run = sq.late_raw.groupby((sq.late_raw != prev_late).cumsum()).cumsum()
    sq["streak"] = run.where(sq.late_raw == 1, 0)
    gs = sq.groupby("customer_id")
    seqf = pd.DataFrame({
        "sq_roll_rate": gs.roll.mean(),
        "sq_cure_rate": gs.cure.mean(),
        "sq_max_streak": gs.streak.max(),
        "sq_mean_streak": gs.streak.mean(),
        "sq_early_late": sq[sq.seq <= 3].groupby("customer_id").late_raw.mean(),
        "sq_early_late_asof": sq[sq.seq <= 3].groupby("customer_id").late_asof.mean(),
    })
    slope = sq.groupby(["customer_id", "loan_id"]).apply(
        lambda d: np.polyfit(d.seq, d.maxdpd_raw, 1)[0] if len(d) > 2 else np.nan, include_groups=False)
    seqf["sq_dpd_slope_mean"] = slope.groupby(level=0).mean()
    seqf["sq_dpd_slope_max"] = slope.groupby(level=0).max()
    tail3 = sq.groupby("customer_id").tail(3).groupby("customer_id")
    seqf["sq_last3_late"] = tail3.late_raw.mean()
    seqf["sq_last3_dpd"] = tail3.maxdpd_raw.mean()
    seqf["sq_last_inst_dpd"] = sq.groupby("customer_id").maxdpd_raw.last()
    return pd.concat([out.reset_index(drop=True), seqf.reset_index(drop=True)], axis=1)


# --------------------------------------------------------------------------- #
# 5. account behaviour (dibandingkan dengan baseline customer sendiri)
# --------------------------------------------------------------------------- #
def account_features(apps, acct):
    ev = acct.rename(columns={"month": "event_time"}).copy()
    q = _queries(apps)
    ev["net"] = ev.inflows - ev.outflows
    ev["out_in"] = ev.outflows / ev.inflows.replace(0, np.nan)
    ev["income_share"] = ev.income_like_credits / ev.inflows.replace(0, np.nan)
    ev["balance_change"] = ev.closing_balance - ev.opening_balance
    ev["buffer_months"] = ev.average_balance / (ev.outflows / 12).replace(0, np.nan)
    ev["low_balance_flag"] = (ev.low_balance_days > 0).astype(float)
    ev["zero_income"] = (ev.income_like_credits == 0).astype(float)
    cols = ["inflows", "outflows", "income_like_credits", "average_balance", "minimum_balance",
            "low_balance_days", "net", "out_in", "income_share", "balance_change",
            "buffer_months", "closing_balance", "opening_balance"]

    stats, _ = window_stats(ev, q, value_cols=cols, windows=W_LONG, ops=("sum", "mean", "std", "min", "max", "last"))
    latest, _ = asof_point(ev, q, cols)
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["ab_" + c for c in stats.columns] + ["ab_last_" + c for c in latest.columns]

    # bulan k = jarak bulan dari keputusan; k=1 adalah bulan lengkap terakhir
    dec_month = apps.decision_datetime.dt.to_period("M").dt.to_timestamp().reset_index(drop=True)
    ev = ev.reset_index(drop=True)
    ev["_cust"] = ev.customer_id
    ev["k"] = ((dec_month.dt.year * 12 + dec_month.dt.month) -
               (ev.event_time.dt.year * 12 + ev.event_time.dt.month)).to_numpy()
    kmap = pd.Series(np.arange(len(apps)), index=apps.customer_id.reset_index(drop=True))
    ev["_app_idx"] = ev.customer_id.map(kmap).to_numpy()
    base = ev[(ev.k >= 4) & (ev.k <= 12)].groupby("_app_idx")[cols].mean()
    l3 = ev[ev.k <= 3].groupby("_app_idx")[cols].mean()
    k1 = ev[ev.k == 1].groupby("_app_idx")[cols].mean()
    n_idx = len(apps)
    base = base.reindex(range(n_idx))
    l3 = l3.reindex(range(n_idx))
    k1 = k1.reindex(range(n_idx))
    for c in ["inflows", "outflows", "income_like_credits", "average_balance", "net", "minimum_balance"]:
        out[f"ab_k1_vs_base_{c}"] = k1[c].to_numpy() / base[c].abs().replace(0, np.nan).to_numpy()
        out[f"ab_l3_vs_base_{c}"] = l3[c].to_numpy() / base[c].abs().replace(0, np.nan).to_numpy()
    for c in ["low_balance_days", "buffer_months", "out_in", "income_share"]:
        out[f"ab_l3_{c}"] = l3[c].to_numpy()
    out["ab_last_k"] = ev.groupby("_app_idx").k.min().reindex(range(n_idx)).to_numpy()
    out["ab_months"] = ev.groupby("_app_idx").size().reindex(range(n_idx)).to_numpy()
    out["ab_low_months"] = ev.groupby("_app_idx").low_balance_flag.sum().reindex(range(n_idx)).to_numpy()
    out["ab_zero_income_months"] = ev.groupby("_app_idx").zero_income.sum().reindex(range(n_idx)).to_numpy()
    out["ab_balance_growth"] = out["ab_last_closing_balance"] - out["ab_first_opening_balance"] \
        if "ab_first_opening_balance" in out else np.nan
    out["ab_balance_growth"] = (ev.sort_values("event_time").groupby("_app_idx").closing_balance.last() -
                                ev.sort_values("event_time").groupby("_app_idx").opening_balance.first()
                                ).reindex(range(n_idx)).to_numpy()
    out["ab_inc_trend"] = l3.income_like_credits.to_numpy() - out["ab_mean_income_like_credits__all"].to_numpy()
    out["ab_bal_trend"] = l3.average_balance.to_numpy() - out["ab_mean_average_balance__all"].to_numpy()
    out["ab_low_balance_month_frac"] = out["ab_sum_low_balance_days__m12"] / (30.0 * out["ab_n_events__m12"])
    f6 = out["ab_mean_average_balance__m6"].to_numpy()
    f12 = out["ab_mean_average_balance__m12"].to_numpy()
    out["ab_balance_trend_6v6"] = f6 - (f12 * 12 - f6 * 6) / 6.0
    out["ab_income_volatility"] = out["ab_std_income_like_credits__m12"] / \
        out["ab_mean_income_like_credits__m12"].replace(0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# 6. e-commerce
# --------------------------------------------------------------------------- #
def ecommerce_features(apps, ecom):
    XINC = apps.set_index("customer_id")["declared_monthly_income"] \
        if "declared_monthly_income" in apps.columns else pd.Series(np.nan, index=apps.customer_id)
    ev = ecom.sort_values(["order_id", "transaction_datetime"]).drop_duplicates("order_id", keep="first")
    ev = ev.merge(apps[["customer_id", "decision_datetime"]], on="customer_id", how="inner")
    ev = ev[ev.transaction_datetime <= ev.decision_datetime].copy()
    ev["days_ago"] = (ev.decision_datetime - ev.transaction_datetime).dt.total_seconds() / 86400
    ev = ev.rename(columns={"transaction_datetime": "event_time"})
    ev["done"] = ev.order_status == "completed"
    ev["spend"] = ev.order_total.where(ev.done, 0.0)
    ev["cancel"] = (~ev.done).astype(float)
    ev["is_completed"] = ev.done.astype(float)
    ev["quantity_f"] = ev.quantity.astype(float)
    for m in ["COD", "credit_card", "e_wallet", "bank_transfer", "debit_card"]:
        ev[f"pm_{m}"] = (ev.payment_method == m).astype(float)
    categories = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
    for c in categories:
        ev[f"cat_{c}"] = np.where(ev.product_category == c, ev.spend, 0.0)

    q = _queries(apps)
    value_cols = (["spend", "order_total", "is_completed", "cancel", "quantity_f"]
                  + [f"pm_{m}" for m in ["COD", "credit_card", "e_wallet", "bank_transfer", "debit_card"]]
                  + [f"cat_{c}" for c in categories])
    stats, _ = window_stats(ev, q, value_cols=value_cols, windows=W_LONG, ops=("sum", "mean", "max", "last"))
    latest, _ = asof_point(ev, q, ["spend", "order_total", "quantity_f"])
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["ec_" + c for c in stats.columns] + ["ec_last_" + c for c in latest.columns]

    # customer_id -> baris matriks fitur (semua customer unik, satu pengajuan per customer)
    row_of = pd.Series(np.arange(len(apps)), index=apps.customer_id.to_numpy())
    idx = apps.customer_id.map(row_of).to_numpy()

    base_mask = (ev.days_ago > 90) & (ev.days_ago <= 365)
    base_spend = ev.spend.where(base_mask, 0).groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
    base_n = base_mask.groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
    for c in categories:
        out[f"ec_share_{c}"] = out[f"ec_sum_cat_{c}__m12"] / out["ec_sum_spend__m12"].replace(0, np.nan)
    out["ec_orders_per_month"] = out["ec_n_events__all"] / 12.0
    out["ec_cancel_rate"] = out["ec_sum_cancel__all"] / out["ec_n_events__all"].replace(0, np.nan)
    out["ec_avg_basket"] = out["ec_sum_order_total__all"] / out["ec_n_events__all"].replace(0, np.nan)

    # jendela pendek + perbandingan dengan baseline 90-365 hari milik customer sendiri
    base_spend_m = base_spend / 9.0 + 1.0
    base_n_m = base_n / 9.0 + 0.1
    for w in [7, 14, 30, 60, 90]:
        m = ev.days_ago <= w
        sp = ev.spend.where(m, 0).groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
        cnt = m.groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
        out[f"ec_spend{w}"] = sp
        out[f"ec_n{w}"] = cnt
        out[f"ec_spend{w}_vs_base"] = sp * 30.0 / w / base_spend_m
        out[f"ec_n{w}_vs_base"] = cnt * 30.0 / w / base_n_m
    for hl in [15, 45]:
        wgt = np.exp(-ev.days_ago * np.log(2) / hl)
        out[f"ec_decay_n{hl}"] = wgt.groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
        out[f"ec_decay_spend{hl}"] = (wgt * ev.spend).groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()

    # harga satuan relatif terhadap median kategori -> sinyal terkuat di e-commerce
    items = ecom[ecom.order_status == "completed"].merge(
        apps[["customer_id", "decision_datetime"]], on="customer_id", how="inner")
    items = items[items.transaction_datetime <= items.decision_datetime]
    items["days_ago"] = (items.decision_datetime - items.transaction_datetime).dt.total_seconds() / 86400
    unit = items.item_total / items.quantity.replace(0, np.nan)
    items["lrel"] = np.log(unit / unit.groupby(items.product_category).transform("median"))
    gi = items.groupby("customer_id")
    out["ec_price_rel"] = gi.lrel.mean().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_std"] = gi.lrel.std().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_max"] = gi.lrel.max().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_min"] = gi.lrel.min().reindex(apps.customer_id).to_numpy()
    for lo, hi in [(0, 30), (0, 90), (90, 365), (365, 800)]:
        m = (items.days_ago > lo) & (items.days_ago <= hi)
        out[f"ec_price_rel_{lo}_{hi}"] = items[m].groupby("customer_id").lrel.mean().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_trend"] = out["ec_price_rel_0_90"] - out["ec_price_rel_365_800"]
    out["ec_price_rel_trend2"] = out["ec_price_rel_0_90"] - out["ec_price_rel_90_365"]
    slope = gi.apply(lambda d: np.polyfit(-d.days_ago, d.lrel, 1)[0] if len(d) > 2 else np.nan,
                     include_groups=False)
    out["ec_price_rel_slope"] = (slope * 365).reindex(apps.customer_id).to_numpy()
    bycat = items.pivot_table(index="customer_id", columns="product_category", values="lrel", aggfunc="mean")
    bycat = bycat.reindex(apps.customer_id)
    bycat.columns = [f"ec_price_rel_cat_{c}" for c in bycat.columns]
    out = pd.concat([out, bycat.reset_index(drop=True)], axis=1)
    catqty = items.pivot_table(index="customer_id", columns="product_category",
                               values="quantity", aggfunc="sum", fill_value=0).reindex(apps.customer_id)
    catqty.columns = [f"ec_catqty_{c}" for c in catqty.columns]
    out = pd.concat([out, catqty.reset_index(drop=True)], axis=1)

    # pergeseran komposisi kategori: 90 hari terakhir vs 90-365 hari
    idx_of = pd.Series(np.arange(len(apps)), index=apps.customer_id)
    frames = {}
    for name, m in [("r", items.days_ago <= 90), ("b", (items.days_ago > 90) & (items.days_ago <= 365))]:
        p = items[m].pivot_table(index="customer_id", columns="product_category",
                                 values="item_total", aggfunc="sum", fill_value=0)
        p = p.reindex(apps.customer_id).fillna(0.0)
        frames[name] = p.div(p.sum(axis=1).replace(0, np.nan), axis=0)
    shift = frames["r"].sub(frames["b"], fill_value=0.0)
    shift.columns = [f"cs_shift_{c}" for c in shift.columns]
    shift["cs_shift_l1"] = shift.abs().sum(axis=1)
    out = pd.concat([out, shift.reset_index(drop=True)], axis=1)

    # --- turunan tambahan dari sinyal terkuat (harga relatif & komposisi belanja) ---
    # harga relatif terhadap median customer sendiri, bukan hanya median kategori global
    own_med = unit.groupby([items.customer_id, items.product_category]).transform("median")
    items["lrel_own"] = np.log(unit / own_med.replace(0, np.nan))
    go = items.groupby("customer_id")
    out["ec_price_rel_own"] = go.lrel_own.mean().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_own_recent"] = (items[items.days_ago <= 90].groupby("customer_id").lrel_own.mean()
                                      .reindex(apps.customer_id).to_numpy())
    out["ec_price_rel_own_gap"] = out["ec_price_rel_own_recent"] - out["ec_price_rel_own"]
    out["ec_price_rel_to_income"] = (items.groupby("customer_id").item_total.mean()
                                     .reindex(apps.customer_id).to_numpy()
                                     / XINC.reindex(apps.customer_id).to_numpy())
    # belanja diskresioner vs esensial -> proxy tekanan keuangan
    disc = ["fashion", "electronics", "personal_care"]
    ess = ["groceries", "health"]
    disc_spend = sum(out[f"ec_sum_cat_{c}__m12"].fillna(0) for c in disc)
    ess_spend = sum(out[f"ec_sum_cat_{c}__m12"].fillna(0) for c in ess)
    out["ec_disc_share_m12"] = disc_spend / (disc_spend + ess_spend).replace(0, np.nan)
    out["ec_disc_vs_base"] = disc_spend / (disc_spend / 4 + ess_spend / 4 + 1).replace(0, np.nan)
    # struktur keranjang
    out["ec_items_per_order"] = (items.groupby(["customer_id", "transaction_datetime"]).size()
                                 .groupby("customer_id").mean().reindex(apps.customer_id).to_numpy())
    out["ec_distinct_categories"] = items.groupby("customer_id").product_category.nunique().reindex(apps.customer_id).to_numpy()
    out["ec_qty_per_item"] = items.quantity.groupby(items.customer_id).mean().reindex(apps.customer_id).to_numpy()
    out["ec_disp_price"] = go.lrel.std().reindex(apps.customer_id).to_numpy()
    month_spend = items.assign(m=items.days_ago // 30).groupby(["customer_id", "m"]).item_total.sum()
    out["ec_month_spend_std"] = month_spend.groupby("customer_id").std().reindex(apps.customer_id).to_numpy()
    month_mean = month_spend.groupby("customer_id").mean().reindex(apps.customer_id).to_numpy()
    out["ec_month_spend_cv"] = out["ec_month_spend_std"] / pd.Series(month_mean).replace(0, np.nan).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# 7. cross features
# --------------------------------------------------------------------------- #
def cross_features(apps, f):
    out = pd.DataFrame(index=f.index)
    inc = f["app__declared_monthly_income"].replace(0, np.nan)
    active_inst = f.get("pa__pa_active_inst")
    inst = f["app__installment"]
    out["x_dti_total"] = (active_inst.fillna(0) + inst) / inc
    cash = f.get("ab__ab_mean_income_like_credits__m6")
    out["x_dti_cashflow"] = (active_inst.fillna(0) + inst) / cash
    out["x_inst_to_net"] = inst / f["ab__ab_mean_net__m6"]
    out["x_income_verified_ratio"] = cash / inc
    out["x_bureau_debt_to_income"] = f["bu__bu_last_outstanding_balance"] / (inc * 12)
    out["x_loan_to_balance"] = f["app__loan_amount"] / f["ab__ab_mean_average_balance__m6"]
    out["x_rate_vs_history"] = f["app__interest_rate"] - f["pa__pa_rate_mean"]
    out["x_loan_vs_history"] = f["app__loan_amount"] / f["pa__pa_mean_requested_amount__all"].replace(0, np.nan)
    out["x_spend30_to_balance"] = f["ec__ec_spend30"] / f["ab__ab_mean_average_balance__m3"]
    out["x_spend90_to_inflow"] = f["ec__ec_sum_spend__m3"] / (3 * f["ab__ab_mean_inflows__m3"])
    out["x_surge_and_drain"] = f["ec__ec_n30_vs_base"] * f["ab__ab_k1_vs_base_outflows"]
    out["x_price_and_late"] = f["ec__ec_price_rel"] * f["rh__rha_rate1_smooth"]
    out["x_delinq_and_income"] = f["bu__bu_last_delinquent_accounts_last_12m"] / inc
    out["x_price_and_balance"] = f["ec__ec_price_rel"] * f["ab__ab_k1_vs_base_average_balance"]
    out["x_late_and_bureau"] = f["rh__ln_max_dpd"] + f["bu__bu_last_score_f"] / 100
    return out