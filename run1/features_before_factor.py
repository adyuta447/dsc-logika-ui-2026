import numpy as np
import pandas as pd

D = "data/"


def load_apps():
    dt = ["application_datetime", "decision_datetime"]
    tr = pd.read_csv(D + "train_applications.csv", parse_dates=dt)
    te = pd.read_csv(D + "test_applications.csv", parse_dates=dt)
    return pd.concat([tr, te], ignore_index=True)


def app_features(app):
    f = app[["application_id", "customer_id", "decision_datetime"]].copy()
    f["loan_amount"] = app.loan_amount
    f["interest_rate"] = app.annual_interest_rate
    f["installment"] = app.proposed_monthly_installment
    f["loan_purpose"] = app.loan_purpose.astype("category")
    f["decision_lag_h"] = (app.decision_datetime - app.application_datetime).dt.total_seconds() / 3600
    f["app_hour"] = app.application_datetime.dt.hour
    f["app_dow"] = app.application_datetime.dt.dayofweek
    return f


def profile_features(f):
    cp = pd.read_csv(D + "customer_profiles.csv")
    for c in ["region", "employment_type", "housing_status"]:
        cp[c] = cp[c].astype("category")
    f = f.merge(cp, on="customer_id", how="left")
    inc = f.declared_monthly_income
    f["inst_to_income"] = f.installment / inc
    f["loan_to_income"] = f.loan_amount / inc
    f["income_per_dependent"] = inc / (f.dependents + 1)
    f["tenure_to_age"] = f.employment_tenure_months / (f.age * 12)
    return f


def bureau_features(app):
    bu = pd.read_csv(D + "bureau_observations.csv", parse_dates=["report_datetime", "available_datetime"])
    bu = bu.merge(app[["customer_id", "application_id", "decision_datetime"]], on="customer_id", suffixes=("", "_sel"))
    bu["age_days"] = (bu.decision_datetime - bu.report_datetime).dt.total_seconds() / 86400
    bu["unscored"] = bu.score.isna().astype(int)
    bu["revolving_free"] = bu.revolving_credit_limit - bu.revolving_outstanding_balance

    cur = bu[bu.application_id == bu.application_id_sel]
    g = cur.groupby("application_id_sel")
    out = g.agg(
        bu_score_mean=("score", "mean"), bu_score_min=("score", "min"), bu_score_max=("score", "max"),
        bu_score_std=("score", "std"), bu_unscored=("unscored", "sum"), bu_reports=("score", "size"),
        bu_active_max=("active_accounts", "max"), bu_active_mean=("active_accounts", "mean"),
        bu_outstanding_max=("outstanding_balance", "max"), bu_outstanding_mean=("outstanding_balance", "mean"),
        bu_rev_limit_max=("revolving_credit_limit", "max"), bu_rev_bal_max=("revolving_outstanding_balance", "max"),
        bu_util_max=("utilization", "max"), bu_util_mean=("utilization", "mean"),
        bu_inq_max=("inquiries_last_6m", "max"), bu_inq_mean=("inquiries_last_6m", "mean"),
        bu_delinq_max=("delinquent_accounts_last_12m", "max"), bu_delinq_sum=("delinquent_accounts_last_12m", "sum"),
        bu_age_min=("age_days", "min"), bu_age_max=("age_days", "max"),
    )
    # the freshest report is the best snapshot of the customer at decision time
    latest = cur.sort_values("report_datetime").groupby("application_id_sel").last()
    for c in ["score", "outstanding_balance", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m", "active_accounts"]:
        out["bu_latest_" + c] = latest[c]
    piv = cur.pivot_table(index="application_id_sel", columns="bureau_provider", values="score")
    out = out.join(piv.add_prefix("bu_score_"))

    hist = bu[bu.application_id != bu.application_id_sel]
    h = hist.groupby("application_id_sel").agg(
        buh_score_mean=("score", "mean"), buh_score_min=("score", "min"),
        buh_delinq_max=("delinquent_accounts_last_12m", "max"), buh_outstanding_max=("outstanding_balance", "max"),
        buh_inq_max=("inquiries_last_6m", "max"), buh_n=("score", "size"),
    )
    hl = hist.sort_values("report_datetime").groupby("application_id_sel").last()
    h["buh_last_score"] = hl.score
    h["buh_last_age_days"] = hl.age_days
    out = out.join(h)
    out["bu_score_trend"] = out.bu_score_mean - out.buh_score_mean
    out["bu_score_trend_last"] = out.bu_latest_score - out.buh_last_score
    # every report is a noisy reading of the same customer, so pooling all of them reduces noise
    pool = ["score", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m", "outstanding_balance",
            "revolving_outstanding_balance", "active_accounts"]
    out = out.join(bu.groupby("application_id_sel")[pool].mean().add_prefix("bua_"))
    out["bua_score_median"] = bu.groupby("application_id_sel").score.median()
    out["bua_unscored_rate"] = bu.groupby("application_id_sel").unscored.mean()
    return out.rename_axis("application_id").reset_index()


def previous_features(app):
    pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
    pa = pa.merge(app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"}), on="customer_id")
    pa["days_ago"] = (pa.cut - pa.decision_datetime).dt.total_seconds() / 86400
    pa["approved"] = (pa.decision == "approved").astype(int)
    pa["rejected"] = (pa.decision == "rejected").astype(int)
    pa["cancelled"] = (pa.decision == "cancelled").astype(int)
    # months elapsed since origination tells whether the loan is still being repaid
    pa["months_elapsed"] = pa.days_ago / 30.44
    pa["active"] = (pa.approved == 1) & (pa.months_elapsed < pa.term_months)
    pa["active_inst"] = np.where(pa.active, pa.scheduled_monthly_installment, 0)
    pa["remaining_principal"] = np.where(
        pa.active, pa.approved_amount * (1 - pa.months_elapsed / pa.term_months), 0)
    for w in [90, 180, 365]:
        pa[f"in{w}"] = (pa.days_ago <= w).astype(int)
        pa[f"rej{w}"] = pa.rejected * pa[f"in{w}"]

    g = pa.groupby("customer_id")
    out = g.agg(
        pa_n=("decision", "size"), pa_approved=("approved", "sum"), pa_rejected=("rejected", "sum"),
        pa_cancelled=("cancelled", "sum"), pa_req_mean=("requested_amount", "mean"),
        pa_req_max=("requested_amount", "max"), pa_appr_sum=("approved_amount", "sum"),
        pa_rate_mean=("annual_interest_rate", "mean"), pa_rate_max=("annual_interest_rate", "max"),
        pa_term_mean=("term_months", "mean"), pa_last_days=("days_ago", "min"), pa_first_days=("days_ago", "max"),
        pa_active_n=("active", "sum"), pa_active_inst=("active_inst", "sum"),
        pa_remaining=("remaining_principal", "sum"),
        pa_in90=("in90", "sum"), pa_in180=("in180", "sum"), pa_in365=("in365", "sum"),
        pa_rej90=("rej90", "sum"), pa_rej365=("rej365", "sum"),
    )
    out["pa_reject_rate"] = out.pa_rejected / out.pa_n
    return out.reset_index()


def repayment_features(app):
    rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date", "last_payment_datetime"])
    rh = rh.merge(app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"}), on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.total_seconds() / 86400
    rh["unpaid"] = rh.amount_due - rh.amount_paid
    rh["paid_ratio"] = rh.amount_paid / rh.amount_due
    rh["pay_delay"] = (rh.last_payment_datetime - rh.due_date).dt.total_seconds() / 86400
    rh["late"] = (rh.max_days_past_due > 0).astype(int)
    rh["late30"] = (rh.max_days_past_due >= 30).astype(int)
    rh["late90"] = (rh.max_days_past_due >= 90).astype(int)
    rh["open_dpd"] = (rh.days_past_due > 0).astype(int)
    rh["recent"] = rh.days_ago <= 180
    for w in [90, 365]:
        rh[f"late_{w}"] = rh.late.where(rh.days_ago <= w)
        rh[f"dpd_{w}"] = rh.max_days_past_due.where(rh.days_ago <= w)
    for c in ["max_days_past_due", "late", "late30", "paid_ratio", "pay_delay"]:
        rh["r_" + c] = rh[c].where(rh.recent)
    # an installment due t days ago cannot show more than t days past due, so each rate only counts matured ones
    thresholds = [1, 15, 30, 60, 90]
    for t in thresholds:
        rh[f"mat{t}"] = (rh.days_ago >= t).astype(int)
        rh[f"hit{t}"] = ((rh.max_days_past_due >= t) & (rh.days_ago >= t)).astype(int)
    rh["sev"] = rh.max_days_past_due.where(rh.late == 1)
    rh["loan_seq"] = rh.groupby(["customer_id", "loan_id"]).due_date.rank()

    g = rh.groupby("customer_id")
    out = g.agg(
        rh_n=("amount_due", "size"), rh_loans=("loan_id", "nunique"),
        rh_dpd_now_max=("days_past_due", "max"), rh_dpd_now_sum=("days_past_due", "sum"),
        rh_open_dpd=("open_dpd", "sum"), rh_maxdpd_max=("max_days_past_due", "max"),
        rh_maxdpd_mean=("max_days_past_due", "mean"), rh_late=("late", "mean"), rh_late_n=("late", "sum"),
        rh_late30=("late30", "sum"), rh_late90=("late90", "sum"), rh_unpaid=("unpaid", "sum"),
        rh_paid_ratio=("paid_ratio", "mean"), rh_paid_ratio_min=("paid_ratio", "min"),
        rh_delay_mean=("pay_delay", "mean"), rh_delay_max=("pay_delay", "max"),
        rh_r_maxdpd=("r_max_days_past_due", "max"), rh_r_late=("r_late", "mean"),
        rh_r_late30=("r_late30", "sum"), rh_r_paid_ratio=("r_paid_ratio", "mean"),
        rh_r_delay=("r_pay_delay", "mean"), rh_last_due_days=("days_ago", "min"),
        rh_late_90=("late_90", "mean"), rh_late_365=("late_365", "mean"),
        rh_dpd_90=("dpd_90", "max"), rh_dpd_365=("dpd_365", "max"), rh_dpd_365_mean=("dpd_365", "mean"),
        rh_sev_mean=("sev", "mean"),
        **{f"rh_hit{t}": (f"hit{t}", "sum") for t in thresholds},
        **{f"rh_mat{t}": (f"mat{t}", "sum") for t in thresholds},
    )
    for t in thresholds:
        out[f"rh_rate{t}"] = out[f"rh_hit{t}"] / out[f"rh_mat{t}"].replace(0, np.nan)
        out[f"rh_rate{t}_smooth"] = (out[f"rh_hit{t}"] + 0.3) / (out[f"rh_mat{t}"] + 1)
    out = out.drop(columns=[f"rh_mat{t}" for t in thresholds])
    # worst loan and first installments of each loan, early delinquency is a strong warning sign
    per_loan = rh.groupby(["customer_id", "loan_id"]).agg(l_late=("late", "mean"), l_dpd=("max_days_past_due", "mean"), l_end=("due_date", "max"))
    out["rh_worst_loan_late"] = per_loan.groupby("customer_id").l_late.max()
    out["rh_best_loan_late"] = per_loan.groupby("customer_id").l_late.min()
    out["rh_last_loan_late"] = per_loan.sort_values("l_end").groupby("customer_id").l_late.last()
    out["rh_last_loan_dpd"] = per_loan.sort_values("l_end").groupby("customer_id").l_dpd.last()
    out["rh_early_late"] = rh[rh.loan_seq <= 3].groupby("customer_id").late.mean()
    last = rh.sort_values("due_date").groupby("customer_id").last()
    out["rh_last_dpd"] = last.days_past_due
    out["rh_last_paid_ratio"] = last.paid_ratio
    return out.reset_index()


def account_features(app):
    ab = pd.read_csv(D + "account_behavior.csv", parse_dates=["month"])
    ab = ab.merge(app[["customer_id", "decision_datetime"]], on="customer_id")
    dm = ab.decision_datetime.dt.year * 12 + ab.decision_datetime.dt.month
    ab["k"] = dm - (ab.month.dt.year * 12 + ab.month.dt.month)
    ab = ab.sort_values(["customer_id", "month"])
    ab["net"] = ab.inflows - ab.outflows
    ab["out_in"] = ab.outflows / ab.inflows.replace(0, np.nan)
    ab["rank"] = ab.groupby("customer_id").cumcount(ascending=False)
    cols = ["inflows", "outflows", "income_like_credits", "average_balance", "minimum_balance", "low_balance_days", "net", "out_in"]

    g = ab.groupby("customer_id")
    out = g[cols].mean().add_prefix("ab_mean_")
    out = out.join(ab[ab["rank"] < 3].groupby("customer_id")[cols].mean().add_prefix("ab_l3_"))
    out = out.join(ab[ab.k == 1].groupby("customer_id")[cols].mean().add_prefix("ab_k1_"))
    base = ab[(ab.k >= 4) & (ab.k <= 12)].groupby("customer_id")[cols].mean()
    for c in ["inflows", "outflows", "income_like_credits", "average_balance", "net"]:
        out[f"ab_k1_vs_base_{c}"] = out[f"ab_k1_{c}"] / base[c].abs().replace(0, np.nan)
        out[f"ab_l3_vs_base_{c}"] = out[f"ab_l3_{c}"] / base[c].abs().replace(0, np.nan)
    out["ab_last_k"] = g.k.min()
    out = out.join(g[["income_like_credits", "inflows", "average_balance"]].std().add_prefix("ab_std_"))
    out["ab_months"] = g.size()
    out["ab_low_days_max"] = g.low_balance_days.max()
    out["ab_low_months"] = g.low_balance_days.apply(lambda s: (s > 0).sum())
    out["ab_min_balance_min"] = g.minimum_balance.min()
    out["ab_last_close"] = g.closing_balance.last()
    out["ab_first_open"] = g.opening_balance.first()
    out["ab_balance_growth"] = out.ab_last_close - out.ab_first_open
    out["ab_inc_trend"] = out.ab_l3_income_like_credits - out.ab_mean_income_like_credits
    out["ab_bal_trend"] = out.ab_l3_average_balance - out.ab_mean_average_balance
    out["ab_zero_income_months"] = g.income_like_credits.apply(lambda s: (s == 0).sum())
    return out.reset_index()


def ecommerce_features(app):
    ec = pd.read_csv(D + "ecommerce_transactions.csv", parse_dates=["transaction_datetime"])
    ec = ec.merge(app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"}), on="customer_id")
    ec["done"] = ec.order_status == "completed"
    ec["spend"] = ec.item_total.where(ec.done, 0)

    od = ec.drop_duplicates("order_id").copy()
    od["days_ago"] = (od.cut - od.transaction_datetime).dt.total_seconds() / 86400
    od["spend"] = od.order_total.where(od.done, 0)
    od["cancel"] = (~od.done).astype(int)
    windows = [7, 14, 30, 60, 90, 180, 365]
    for w in windows:
        od[f"spend{w}"] = od.spend.where(od.days_ago <= w, 0)
        od[f"n{w}"] = (od.days_ago <= w).astype(int)
    od["spend_base"] = od.spend.where((od.days_ago > 90) & (od.days_ago <= 365), 0)
    od["n_base"] = ((od.days_ago > 90) & (od.days_ago <= 365)).astype(int)
    for m in ["COD", "credit_card", "e_wallet", "bank_transfer", "debit_card"]:
        od["pm_" + m] = (od.payment_method == m).astype(int)

    g = od.groupby("customer_id")
    out = g.agg(
        ec_orders=("order_id", "size"), ec_spend=("spend", "sum"), ec_spend_mean=("spend", "mean"),
        ec_spend_max=("spend", "max"), ec_spend_base=("spend_base", "sum"), ec_n_base=("n_base", "sum"),
        **{f"ec_spend{w}": (f"spend{w}", "sum") for w in windows},
        **{f"ec_n{w}": (f"n{w}", "sum") for w in windows},
        ec_cancel_rate=("cancel", "mean"), ec_last_days=("days_ago", "min"), ec_first_days=("days_ago", "max"),
        ec_cod=("pm_COD", "mean"), ec_cc=("pm_credit_card", "mean"), ec_ewallet=("pm_e_wallet", "mean"),
        ec_bank=("pm_bank_transfer", "mean"), ec_debit=("pm_debit_card", "mean"),
    )
    out["ec_spend90_share"] = out.ec_spend90 / out.ec_spend.replace(0, np.nan)
    # recent monthly spend against the customer's own baseline month (days 91 to 365)
    base_month = out.ec_spend_base / 9 + 1
    base_n = out.ec_n_base / 9 + 0.1
    for w in [7, 14, 30, 60, 90]:
        out[f"ec_spend{w}_vs_base"] = out[f"ec_spend{w}"] * 30 / w / base_month
        out[f"ec_n{w}_vs_base"] = out[f"ec_n{w}"] * 30 / w / base_n
    recent = od[od.days_ago <= 90]
    out["ec_r_cancel_rate"] = recent.groupby("customer_id").cancel.mean()
    out["ec_r_cc"] = recent.groupby("customer_id").pm_credit_card.mean()
    out["ec_r_cod"] = recent.groupby("customer_id").pm_COD.mean()
    out["ec_r_order_max"] = recent.groupby("customer_id").spend.max()
    out["ec_r_item_mean"] = ec[ec.done & (ec.cut - ec.transaction_datetime).dt.days.le(90)].groupby("customer_id").item_total.mean()
    out["ec_item_mean"] = ec[ec.done].groupby("customer_id").item_total.mean()
    out["ec_item_mean_ratio"] = out.ec_r_item_mean / out.ec_item_mean
    # exponentially decayed activity, the half life controls how much weight the newest orders get
    for hl in [15, 45]:
        w = np.exp(-od.days_ago * np.log(2) / hl)
        out[f"ec_decay_n{hl}"] = w.groupby(od.customer_id).sum()
        out[f"ec_decay_spend{hl}"] = (w * od.spend).groupby(od.customer_id).sum()
        out[f"ec_decay_n{hl}_vs_base"] = out[f"ec_decay_n{hl}"] / base_n
    cat = ec[ec.done].pivot_table(index="customer_id", columns="product_category", values="spend", aggfunc="sum", fill_value=0)
    out = out.join(cat.div(cat.sum(axis=1), axis=0).add_prefix("ec_share_"))
    out = out.join(cat.add_prefix("ec_catspend_"))
    # unit price against the category median shows whether a customer is trading up beyond their usual basket
    it = ec[ec.quantity > 0].copy()
    unit = it.item_total / it.quantity
    it["lrel"] = np.log(unit / unit.groupby(it.product_category).transform("median"))
    it["days_ago"] = (it.cut - it.transaction_datetime).dt.days
    gi = it.groupby("customer_id")
    out["ec_price_rel"] = gi.lrel.mean()
    out["ec_price_rel_std"] = gi.lrel.std()
    out["ec_price_rel_max"] = gi.lrel.max()
    for lo, hi in [(0, 30), (0, 90), (90, 365), (365, 800)]:
        m = (it.days_ago > lo) & (it.days_ago <= hi)
        out[f"ec_price_rel_{lo}_{hi}"] = it[m].groupby("customer_id").lrel.mean()
    out["ec_price_rel_trend"] = out.ec_price_rel_0_90 - out.ec_price_rel_365_800
    out["ec_price_rel_trend2"] = out.ec_price_rel_0_90 - out.ec_price_rel_90_365
    slope = gi.apply(lambda d: np.polyfit(-d.days_ago, d.lrel, 1)[0] if len(d) > 2 else np.nan)
    out["ec_price_rel_slope"] = slope * 365
    out = out.join(it.pivot_table(index="customer_id", columns="product_category", values="lrel", aggfunc="mean").add_prefix("ec_price_rel_"))
    cnt = ec[ec.done].pivot_table(index="customer_id", columns="product_category", values="quantity", aggfunc="sum", fill_value=0)
    out = out.join(cnt.add_prefix("ec_catqty_"))
    return out.reset_index()


def sequence_features(app):
    rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date"])
    rh = rh.sort_values(["customer_id", "loan_id", "due_date"])
    rh["late"] = (rh.max_days_past_due > 0).astype(int)
    rh["seq"] = rh.groupby(["customer_id", "loan_id"]).cumcount()
    prev = rh.groupby(["customer_id", "loan_id"]).late.shift()
    # roll rate: how often a late installment is followed by another late one
    rh["roll"] = rh.late.where(prev == 1)
    rh["cure"] = (1 - rh.late).where(prev == 1)
    run = rh.late.groupby((rh.late != prev).cumsum()).cumsum()
    rh["streak"] = run.where(rh.late == 1, 0)
    g = rh.groupby("customer_id")
    out = pd.DataFrame({"sq_roll_rate": g.roll.mean(), "sq_cure_rate": g.cure.mean(), "sq_max_streak": g.streak.max()})
    # slope of lateness along each loan tells whether the customer is getting worse
    per = rh.groupby(["customer_id", "loan_id"]).apply(
        lambda d: np.polyfit(d.seq, d.max_days_past_due, 1)[0] if len(d) > 2 else np.nan)
    out["sq_dpd_slope_mean"] = per.groupby(level=0).mean()
    out["sq_dpd_slope_max"] = per.groupby(level=0).max()
    last = rh.groupby("customer_id").tail(3).groupby("customer_id")
    out["sq_last3_late"] = last.late.mean()
    out["sq_last3_dpd"] = last.max_days_past_due.mean()
    return out.reset_index()


def category_shift_features(app):
    ec = pd.read_csv(D + "ecommerce_transactions.csv", parse_dates=["transaction_datetime"],
                     usecols=["customer_id", "transaction_datetime", "product_category", "item_total", "order_status"])
    ec = ec[ec.order_status == "completed"].merge(app[["customer_id", "decision_datetime"]], on="customer_id")
    days = (ec.decision_datetime - ec.transaction_datetime).dt.days
    out = []
    for name, m in [("r", days <= 90), ("b", (days > 90) & (days <= 365))]:
        p = ec[m].pivot_table(index="customer_id", columns="product_category", values="item_total", aggfunc="sum", fill_value=0)
        out.append(p.div(p.sum(axis=1), axis=0).add_prefix(f"cs_{name}_"))
    r, b = out
    shift = r.sub(b.rename(columns=lambda c: c.replace("cs_b_", "cs_r_")), fill_value=0)
    res = shift.rename(columns=lambda c: c.replace("cs_r_", "cs_shift_"))
    res["cs_shift_l1"] = shift.abs().sum(axis=1)
    return res.reset_index()


def build():
    app = load_apps()
    f = profile_features(app_features(app))
    f = f.merge(bureau_features(app), on="application_id", how="left")
    for fn in [previous_features, repayment_features, account_features, ecommerce_features,
               sequence_features, category_shift_features]:
        f = f.merge(fn(app), on="customer_id", how="left")

    f = f.copy()
    inc = f.declared_monthly_income
    f["dti_total"] = (f.pa_active_inst.fillna(0) + f.installment) / inc
    f["dti_cashflow"] = (f.pa_active_inst.fillna(0) + f.installment) / f.ab_l3_income_like_credits
    f["inst_to_net"] = f.installment / f.ab_l3_net
    f["income_verified_ratio"] = f.ab_l3_income_like_credits / inc
    f["bureau_debt_to_income"] = f.bu_outstanding_max / (inc * 12)
    f["loan_to_balance"] = f.loan_amount / f.ab_mean_average_balance
    f["rate_vs_history"] = f.interest_rate - f.pa_rate_mean
    f["loan_vs_history"] = f.loan_amount / f.pa_req_mean
    f["x_spend30_to_balance"] = f.ec_spend30 / f.ab_k1_average_balance
    f["x_spend90_to_inflow"] = f.ec_spend90 / (3 * f.ab_l3_inflows)
    f["x_surge_and_drain"] = f.ec_n30_vs_base * f.ab_k1_vs_base_outflows
    f["x_price_and_late"] = f.ec_price_rel * f.rh_rate1_smooth
    f = f.replace([np.inf, -np.inf], np.nan)
    f["target"] = app.target
    return f.drop(columns=["decision_datetime"])


if __name__ == "__main__":
    build().to_parquet("features.parquet")
