"""Feature blocks for v70.

Earlier rounds built aggregates per source. The blocks here are the ones that
cross sources or describe a state that no existing block expresses:

`monthstress` (`ms_`)
    Repayment lateness aligned to the *same calendar month* as account
    cash flow. Existing repayment features count late instalments and existing
    account features summarise balances, but neither asks whether a customer
    was late exactly in the months their account was negative or thin. Also
    contains the month-level budget: amount due divided by the income-like
    credits of that month, and how many months the observed balance could
    cover the instalment.

`loanstatus` (`ls_`)
    The current state of each loan, reconstructed from repayment history
    instead of assumed from `term_months`: which loans are still active, what
    they require per month, how much is unpaid, and how delinquent the active
    loans are right now.

`ecomdepth` (`ed_`)
    Month-level e-commerce panel: spend volatility, zero months, unit price
    level and slope, category entropy, need-versus-want share, payment-method
    drift, and purchase timing. Existing features are windows and totals; this
    block is the shape of the time series.

`ratefit` (`rf_`)
    `annual_interest_rate` is an underwriting grade with five values and
    default rates from 5.5% to 50.3%. These features are the residual of
    bureau score, utilization, delinquency and internal lateness *within*
    grade, plus the interactions between grade and the strongest risk
    indicators.

All blocks cut at `decision_datetime`. No target is used anywhere, and the
grade residuals are unsupervised (they use features, not labels), so they are
fold-safe.
"""

import numpy as np
import pandas as pd

D = "data/"


def _cut(app):
    return app[["application_id", "customer_id", "decision_datetime"]].rename(
        columns={"decision_datetime": "cut"}
    )


def _corr(num, den, cnt):
    """Vectorised per-group Pearson correlation from running sums."""
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (cnt * num - den[0] * den[1]) ** 2
        r = np.where(
            (cnt > 2) & (v > 0),
            (cnt * num - den[0] * den[1])
            / np.sqrt((cnt * den[2] - den[0] ** 2) * (cnt * den[3] - den[1] ** 2)),
            np.nan,
        )
    return pd.Series(r, index=cnt.index)


def _safe_div(a, b):
    return a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / b


def block_monthstress(app):
    rh = pd.read_csv(
        D + "repayment_history.csv",
        parse_dates=["due_date"],
        usecols=["customer_id", "due_date", "amount_due", "amount_paid",
                 "max_days_past_due", "days_past_due"],
    )
    cut = _cut(app)[["customer_id", "cut"]]
    rh = rh.merge(cut, on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.total_seconds() / 86400
    rh = rh[rh.days_ago >= 0].copy()
    rh["month"] = rh.due_date.dt.to_period("M").dt.to_timestamp()
    rh["late"] = (rh.max_days_past_due > 0).astype(float)
    rh["late30"] = (rh.max_days_past_due >= 30).astype(float)
    rh["late90"] = (rh.max_days_past_due >= 90).astype(float)
    rh["paid_ratio"] = rh.amount_paid / rh.amount_due.replace(0, np.nan)
    rh["unpaid"] = (rh.amount_due - rh.amount_paid).clip(lower=0)

    r = rh.groupby(["customer_id", "month"]).agg(
        r_due_n=("late", "size"), r_late_n=("late", "sum"),
        r_late30_n=("late30", "sum"), r_late90_n=("late90", "sum"),
        r_due=("amount_due", "sum"), r_paid=("amount_paid", "sum"),
        r_unpaid=("unpaid", "sum"), r_dpd_max=("max_days_past_due", "max"),
        r_dpd_now=("days_past_due", "max"), r_paid_ratio=("paid_ratio", "mean"),
    ).reset_index()
    r["r_late_share"] = r.r_late_n / r.r_due_n.replace(0, np.nan)

    ab = pd.read_csv(
        D + "account_behavior.csv",
        parse_dates=["month"],
        usecols=["customer_id", "month", "inflows", "outflows",
                 "income_like_credits", "average_balance", "minimum_balance",
                 "closing_balance", "low_balance_days"],
    )
    ab = ab.merge(cut, on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut].copy()
    ab["k"] = ab.groupby("customer_id").month.rank(ascending=False)
    ab["net"] = ab.inflows - ab.outflows

    m = ab.merge(r, on=["customer_id", "month"], how="left")
    for c in ["r_due_n", "r_late_n", "r_late30_n", "r_late90_n", "r_due",
              "r_paid", "r_unpaid"]:
        m[c] = m[c].fillna(0.0)
    m["has_due"] = (m.r_due_n > 0).astype(float)
    m["inc"] = m.income_like_credits.replace(0, np.nan)
    m["due_to_inc"] = m.r_due / m.inc
    m["paid_to_inc"] = m.r_paid / m.inc
    m["unpaid_to_inc"] = m.r_unpaid / m.inc
    m["bal_over_due"] = m.average_balance / m.r_due.replace(0, np.nan)
    m["minbal_over_due"] = m.minimum_balance / m.r_due.replace(0, np.nan)
    m["net_after_due"] = (m.net - m.r_due) / m.inc
    m["neg"] = (m.net < 0).astype(float)
    m["low"] = (m.low_balance_days > 0).astype(float)
    m["late_neg"] = m.r_late_n * m.neg
    m["late_low"] = m.r_late_n * m.low
    m["due_neg"] = m.r_due * m.neg
    m["unpaid_neg"] = m.r_unpaid * m.neg
    m["paid_ratio_neg"] = m.r_paid_ratio.where(m.neg == 1)
    m["paid_ratio_pos"] = m.r_paid_ratio.where(m.neg == 0)
    m["late_share_neg"] = m.r_late_share.where(m.neg == 1)
    m["late_share_pos"] = m.r_late_share.where(m.neg == 0)
    # income shock: month income below the customer's own median month
    med = m.groupby("customer_id").income_like_credits.transform("median")
    m["inc_low"] = (m.income_like_credits < 0.75 * med).astype(float)
    m["late_inc_low"] = m.r_late_n * m.inc_low

    m = m.sort_values(["customer_id", "month"])
    m["rk"] = m.groupby("customer_id").cumcount(ascending=False)

    g = m.groupby("customer_id")

    def wsum(col, n):
        return m[m.rk < n].groupby("customer_id")[col].sum()

    def wmean(col, n):
        return m[m.rk < n].groupby("customer_id")[col].mean()

    out = pd.DataFrame({
        "ms_months": g.size(),
        "ms_months_due": g.has_due.sum(),
        "ms_months_late": g.apply(lambda d: float((d.r_late_n > 0).sum()), include_groups=False),
        "ms_months_neg": g.neg.sum(),
        "ms_months_low": g.low.sum(),
        "ms_late_n": g.r_late_n.sum(),
        "ms_late30_n": g.r_late30_n.sum(),
        "ms_late90_n": g.r_late90_n.sum(),
        "ms_due_sum": g.r_due.sum(),
        "ms_unpaid_sum": g.r_unpaid.sum(),
        "ms_late_share": g.r_late_share.mean(),
        "ms_due_to_inc": g.due_to_inc.mean(),
        "ms_due_to_inc_max": g.due_to_inc.max(),
        "ms_unpaid_to_inc": g.unpaid_to_inc.mean(),
        "ms_bal_over_due": g.bal_over_due.mean(),
        "ms_bal_over_due_min": g.bal_over_due.min(),
        "ms_minbal_over_due": g.minbal_over_due.mean(),
        "ms_net_after_due": g.net_after_due.mean(),
        "ms_net_after_due_min": g.net_after_due.min(),
        "ms_late_neg": g.late_neg.sum(),
        "ms_late_low": g.late_low.sum(),
        "ms_due_neg": g.due_neg.sum(),
        "ms_unpaid_neg": g.unpaid_neg.sum(),
        "ms_paid_ratio_neg": g.paid_ratio_neg.mean(),
        "ms_paid_ratio_pos": g.paid_ratio_pos.mean(),
        "ms_late_share_neg": g.late_share_neg.mean(),
        "ms_late_share_pos": g.late_share_pos.mean(),
        "ms_late_inc_low": g.late_inc_low.sum(),
    })

    # share of lateness that happened in stressed months
    out["ms_late_neg_share"] = out.ms_late_neg / out.ms_late_n.replace(0, np.nan)
    out["ms_late_low_share"] = out.ms_late_low / out.ms_late_n.replace(0, np.nan)
    out["ms_late_inc_low_share"] = out.ms_late_inc_low / out.ms_late_n.replace(0, np.nan)
    out["ms_due_neg_share"] = out.ms_due_neg / out.ms_due_sum.replace(0, np.nan)
    out["ms_unpaid_neg_share"] = out.ms_unpaid_neg / out.ms_unpaid_sum.replace(0, np.nan)
    out["ms_paid_ratio_gap"] = out.ms_paid_ratio_neg - out.ms_paid_ratio_pos
    out["ms_late_share_gap"] = out.ms_late_share_neg - out.ms_late_share_pos
    out["ms_late_month_share"] = out.ms_months_late / out.ms_months_due.replace(0, np.nan)

    for n in (3, 6, 12):
        out[f"ms_due_to_inc_{n}"] = wmean("due_to_inc", n)
        out[f"ms_late_share_{n}"] = wmean("r_late_share", n)
        out[f"ms_unpaid_to_inc_{n}"] = wmean("unpaid_to_inc", n)
        out[f"ms_bal_over_due_{n}"] = wmean("bal_over_due", n)
        out[f"ms_net_after_due_{n}"] = wmean("net_after_due", n)
        out[f"ms_late_n_{n}"] = wsum("r_late_n", n)
        out[f"ms_due_n_{n}"] = wsum("r_due_n", n)
        out[f"ms_unpaid_{n}"] = wsum("r_unpaid", n)
    out["ms_late_share_delta"] = out.ms_late_share_3 - out.ms_late_share_12
    out["ms_due_to_inc_delta"] = out.ms_due_to_inc_3 - out.ms_due_to_inc_12
    out["ms_net_after_due_delta"] = out.ms_net_after_due_3 - out.ms_net_after_due_12

    # per-customer correlation between monthly stress and monthly lateness
    tmp = m.dropna(subset=["r_late_share", "net"]).copy()
    grp = tmp.groupby("customer_id")
    cnt = grp.size().rename("n")
    s = pd.DataFrame({
        "n": cnt,
        "sx": grp.net.sum(), "sy": grp.r_late_share.sum(),
        "sxx": grp.net.apply(lambda v: float((v ** 2).sum())),
        "syy": grp.r_late_share.apply(lambda v: float((v ** 2).sum())),
        "sxy": grp.apply(lambda d: float((d.net * d.r_late_share).sum()), include_groups=False),
        "sl": grp.low_balance_days.sum(),
        "sll": grp.low_balance_days.apply(lambda v: float((v ** 2).sum())),
        "syl": grp.apply(lambda d: float((d.r_late_share * d.low_balance_days).sum()), include_groups=False),
    })
    out["ms_corr_late_net"] = _corr(s.sxy, (s.sx, s.sy, s.sxx, s.syy), s.n)
    out["ms_corr_late_low"] = _corr(s.syl, (s.sl, s.sy, s.sll, s.syy), s.n)

    # recency of the last month that had any lateness
    last_late = m[m.r_late_n > 0].groupby("customer_id").rk.min()
    out["ms_months_since_late"] = last_late
    out["ms_last_late_share"] = m[m.r_late_n > 0].sort_values("month").groupby(
        "customer_id").r_late_share.last()

    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out


def block_spendstress(app):
    """Three-way month panel: repayment due/late, cash flow, e-commerce spend.

    The point is the joint month. A customer who is late in a month where the
    account went negative *and* e-commerce spending exceeded income-like
    credits is in a different state from one who is late while the account is
    healthy. Existing blocks measure each source separately, and `monthstress`
    pairs repayment with cash flow only.
    """
    cut = _cut(app)[["customer_id", "cut"]]
    rh = pd.read_csv(
        D + "repayment_history.csv",
        parse_dates=["due_date"],
        usecols=["customer_id", "due_date", "amount_due", "amount_paid",
                 "max_days_past_due"],
    )
    rh = rh.merge(cut, on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.total_seconds() / 86400
    rh = rh[rh.days_ago >= 0].copy()
    rh["month"] = rh.due_date.dt.to_period("M").dt.to_timestamp()
    rh["late"] = (rh.max_days_past_due > 0).astype(float)
    rh["unpaid"] = (rh.amount_due - rh.amount_paid).clip(lower=0)
    r = rh.groupby(["customer_id", "month"]).agg(
        due=("amount_due", "sum"), late=("late", "sum"),
        due_n=("late", "size"), unpaid=("unpaid", "sum"),
    ).reset_index()
    r["late_share"] = r.late / r.due_n.replace(0, np.nan)

    ab = pd.read_csv(
        D + "account_behavior.csv",
        parse_dates=["month"],
        usecols=["customer_id", "month", "inflows", "outflows",
                 "income_like_credits", "average_balance"],
    )
    ab = ab.merge(cut, on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut].copy()
    ab["net"] = ab.inflows - ab.outflows

    ec = pd.read_csv(
        D + "ecommerce_transactions.csv",
        parse_dates=["transaction_datetime"],
        usecols=["customer_id", "transaction_datetime", "item_total",
                 "order_status", "quantity"],
    )
    ec = ec.merge(cut, on="customer_id")
    ec["days_ago"] = (ec.cut - ec.transaction_datetime).dt.total_seconds() / 86400
    ec = ec[(ec.days_ago >= 0) & (ec.order_status == "completed")
            & (ec.quantity > 0)].copy()
    ec["month"] = ec.transaction_datetime.dt.to_period("M").dt.to_timestamp()
    e = ec.groupby(["customer_id", "month"]).item_total.sum().rename("spend").reset_index()

    keys = pd.concat([ab[["customer_id", "month"]], r[["customer_id", "month"]],
                      e[["customer_id", "month"]]]).drop_duplicates()
    m = keys.merge(ab[["customer_id", "month", "inflows", "outflows",
                       "income_like_credits", "average_balance", "net"]],
                   on=["customer_id", "month"], how="left")
    m = m.merge(r, on=["customer_id", "month"], how="left")
    m = m.merge(e, on=["customer_id", "month"], how="left")
    m["spend"] = m.spend.fillna(0.0)
    for c in ["due", "late", "due_n", "unpaid"]:
        m[c] = m[c].fillna(0.0)
    m["inc"] = m.income_like_credits.replace(0, np.nan)
    m["spend_to_inc"] = m.spend / m.inc
    m["spend_plus_out_to_inc"] = (m.spend + m.outflows) / m.inc
    m["spend_minus_net_to_inc"] = (m.spend - m.net) / m.inc
    m["due_to_inc"] = m.due / m.inc
    m["unpaid_to_inc"] = m.unpaid / m.inc
    m["neg"] = (m.net < 0).astype(float)
    m["spend_gt_inc"] = (m.spend_to_inc > 1).astype(float)
    m["spend_gt_2inc"] = (m.spend_to_inc > 2).astype(float)
    m["late_spend"] = m.late * m.spend_gt_inc
    m["late_neg"] = m.late * m.neg
    m["late_spend_neg"] = m.late * m.spend_gt_inc * m.neg
    m["spend_neg"] = m.spend * m.neg
    m["unpaid_spend"] = m.unpaid * m.spend_gt_inc
    m = m.sort_values(["customer_id", "month"])
    m["rk"] = m.groupby("customer_id").cumcount(ascending=False)

    g = m.groupby("customer_id")
    out = pd.DataFrame({
        "ss_months": g.size(),
        "ss_spend": g.spend.sum(),
        "ss_spend_to_inc": g.spend_to_inc.mean(),
        "ss_spend_to_inc_max": g.spend_to_inc.max(),
        "ss_spend_plus_out_to_inc": g.spend_plus_out_to_inc.mean(),
        "ss_spend_minus_net_to_inc": g.spend_minus_net_to_inc.mean(),
        "ss_due_to_inc": g.due_to_inc.mean(),
        "ss_unpaid_to_inc": g.unpaid_to_inc.mean(),
        "ss_months_spend_gt_inc": g.spend_gt_inc.sum(),
        "ss_months_spend_gt_2inc": g.spend_gt_2inc.sum(),
        "ss_late": g.late.sum(),
        "ss_late_spend": g.late_spend.sum(),
        "ss_late_neg": g.late_neg.sum(),
        "ss_late_spend_neg": g.late_spend_neg.sum(),
        "ss_spend_neg": g.spend_neg.sum(),
        "ss_unpaid_spend": g.unpaid_spend.sum(),
        "ss_spend_gt_inc_share": g.spend_gt_inc.mean(),
    })
    out["ss_spend_neg_share"] = out.ss_spend_neg / out.ss_spend.replace(0, np.nan)
    out["ss_late_spend_share"] = out.ss_late_spend / out.ss_late.replace(0, np.nan)
    out["ss_late_neg_share"] = out.ss_late_neg / out.ss_late.replace(0, np.nan)
    out["ss_late_spend_neg_share"] = out.ss_late_spend_neg / out.ss_late.replace(0, np.nan)
    out["ss_unpaid_spend_share"] = out.ss_unpaid_spend / out.ss_spend.replace(0, np.nan)

    for n in (3, 6, 12):
        t = m[m.rk < n].groupby("customer_id")
        out[f"ss_spend_to_inc_{n}"] = t.spend_to_inc.mean()
        out[f"ss_late_share_{n}"] = t.late_share.mean()
        out[f"ss_due_to_inc_{n}"] = t.due_to_inc.mean()
        out[f"ss_spend_{n}"] = t.spend.mean()
        out[f"ss_late_{n}"] = t.late.sum()
        out[f"ss_late_spend_{n}"] = t.late_spend.sum()

    # correlation between monthly lateness and monthly spending pressure
    tmp = m.dropna(subset=["late_share", "spend_to_inc"]).copy()
    grp = tmp.groupby("customer_id")
    s = pd.DataFrame({
        "n": grp.size(),
        "sx": grp.spend_to_inc.sum(), "sy": grp.late_share.sum(),
        "sxx": grp.spend_to_inc.apply(lambda v: float((v ** 2).sum())),
        "syy": grp.late_share.apply(lambda v: float((v ** 2).sum())),
        "sxy": grp.apply(lambda d: float((d.spend_to_inc * d.late_share).sum()),
                         include_groups=False),
        "sp": grp.spend.sum(),
        "spp": grp.spend.apply(lambda v: float((v ** 2).sum())),
        "syp": grp.apply(lambda d: float((d.spend * d.late_share).sum()),
                         include_groups=False),
    })
    out["ss_corr_late_spendinc"] = _corr(s.sxy, (s.sx, s.sy, s.sxx, s.syy), s.n)
    out["ss_corr_late_spend"] = _corr(s.syp, (s.sp, s.sy, s.spp, s.syy), s.n)

    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out


def block_loanstatus(app):
    rh = pd.read_csv(
        D + "repayment_history.csv",
        parse_dates=["due_date"],
        usecols=["loan_id", "customer_id", "due_date", "amount_due",
                 "amount_paid", "days_past_due", "max_days_past_due"],
    )
    cut = _cut(app)[["customer_id", "cut"]]
    rh = rh.merge(cut, on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.total_seconds() / 86400
    rh = rh[rh.days_ago >= 0].copy()
    rh["late"] = (rh.max_days_past_due > 0).astype(float)
    rh["late30"] = (rh.max_days_past_due >= 30).astype(float)
    rh["late90"] = (rh.max_days_past_due >= 90).astype(float)
    rh["unpaid"] = (rh.amount_due - rh.amount_paid).clip(lower=0)
    rh["paid_ratio"] = rh.amount_paid / rh.amount_due.replace(0, np.nan)

    lg = rh.groupby(["customer_id", "loan_id"]).agg(
        l_n=("late", "size"), l_first=("days_ago", "max"), l_last=("days_ago", "min"),
        l_late=("late", "mean"), l_late30=("late30", "mean"),
        l_late90=("late90", "mean"), l_dpd_max=("max_days_past_due", "max"),
        l_dpd_last=("max_days_past_due", "last"), l_dpd_now=("days_past_due", "last"),
        l_due=("amount_due", "sum"), l_unpaid=("unpaid", "sum"),
        l_paid_ratio=("paid_ratio", "mean"), l_inst=("amount_due", "mean"),
    ).reset_index()
    lg["l_span"] = lg.l_first - lg.l_last
    lg["l_active"] = (lg.l_last <= 45).astype(float)
    lg["l_finished"] = (lg.l_last > 45).astype(float)
    lg["l_active_inst"] = lg.l_inst.where(lg.l_active == 1, 0.0)
    lg["l_active_unpaid"] = lg.l_unpaid.where(lg.l_active == 1, 0.0)
    lg["l_active_late"] = lg.l_late.where(lg.l_active == 1)
    lg["l_active_dpd"] = lg.l_dpd_now.where(lg.l_active == 1)
    lg["l_fully_paid"] = ((lg.l_unpaid <= 1) & (lg.l_paid_ratio >= 0.99)).astype(float)

    g = lg.groupby("customer_id")
    out = pd.DataFrame({
        "ls_loans": g.size(),
        "ls_active_n": g.l_active.sum(),
        "ls_finished_n": g.l_finished.sum(),
        "ls_active_inst": g.l_active_inst.sum(),
        "ls_active_unpaid": g.l_active_unpaid.sum(),
        "ls_active_late": g.l_active_late.mean(),
        "ls_active_late_max": g.l_active_late.max(),
        "ls_active_dpd_max": g.l_active_dpd.max(),
        "ls_active_dpd_sum": g.l_active_dpd.sum(),
        "ls_newest_age": g.l_last.min(),
        "ls_oldest_age": g.l_first.max(),
        "ls_span_mean": g.l_span.mean(),
        "ls_span_max": g.l_span.max(),
        "ls_late_mean": g.l_late.mean(),
        "ls_late_max": g.l_late.max(),
        "ls_late90_max": g.l_late90.max(),
        "ls_dpd_max": g.l_dpd_max.max(),
        "ls_dpd_last_max": g.l_dpd_last.max(),
        "ls_unpaid_sum": g.l_unpaid.sum(),
        "ls_paid_ratio_mean": g.l_paid_ratio.mean(),
        "ls_paid_ratio_min": g.l_paid_ratio.min(),
        "ls_fully_paid_share": g.l_fully_paid.mean(),
        "ls_inst_mean": g.l_inst.mean(),
        "ls_inst_max": g.l_inst.max(),
    })
    out["ls_active_share"] = out.ls_active_n / out.ls_loans.replace(0, np.nan)
    out["ls_active_unpaid_share"] = out.ls_active_unpaid / out.ls_unpaid_sum.replace(0, np.nan)
    out["ls_active_inst_over_all"] = out.ls_active_inst / out.ls_inst_mean.replace(0, np.nan)
    out["ls_unpaid_over_due"] = out.ls_unpaid_sum / (
        out.ls_inst_mean * out.ls_loans).replace(0, np.nan)

    # last two loans only, because the newest loan is the one still being repaid
    lg2 = lg.sort_values("l_last")
    for n in (1, 2):
        t = lg2.groupby("customer_id").tail(n).groupby("customer_id")
        out[f"ls_last{n}_late"] = t.l_late.mean()
        out[f"ls_last{n}_dpd"] = t.l_dpd_max.max()
        out[f"ls_last{n}_paid_ratio"] = t.l_paid_ratio.mean()
        out[f"ls_last{n}_inst"] = t.l_inst.mean()
        out[f"ls_last{n}_unpaid"] = t.l_unpaid.sum()
        out[f"ls_last{n}_active"] = t.l_active.mean()

    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out


def block_ecomdepth(app):
    ec = pd.read_csv(
        D + "ecommerce_transactions.csv",
        parse_dates=["transaction_datetime"],
        usecols=["order_id", "customer_id", "transaction_datetime",
                 "product_category", "quantity", "item_total", "order_total",
                 "payment_method", "order_status"],
    )
    cut = _cut(app)[["customer_id", "cut"]]
    ec = ec.merge(cut, on="customer_id")
    ec["days_ago"] = (ec.cut - ec.transaction_datetime).dt.total_seconds() / 86400
    ec = ec[(ec.days_ago >= 0) & (ec.quantity > 0)].copy()
    ec["done"] = (ec.order_status == "completed").astype(float)
    ec["spend"] = ec.item_total.where(ec.done == 1, 0.0)
    ec["unit"] = ec.item_total / ec.quantity.replace(0, np.nan)
    ec["lunit"] = np.log(ec.unit.clip(lower=1))
    ec["month"] = ec.transaction_datetime.dt.to_period("M").dt.to_timestamp()
    ec["is_need"] = ec.product_category.isin(["groceries", "health", "education"]).astype(float)
    ec["is_want"] = ec.product_category.isin(["fashion", "electronics", "personal_care"]).astype(float)
    ec["night"] = ec.transaction_datetime.dt.hour.isin([0, 1, 2, 3, 4, 22, 23]).astype(float)
    ec["weekend"] = (ec.transaction_datetime.dt.dayofweek >= 5).astype(float)

    # order level
    od = ec.drop_duplicates("order_id").copy()
    od["n_items"] = od.order_id.map(ec.groupby("order_id").size())
    od["cod"] = (od.payment_method == "COD").astype(float)
    od["cc"] = (od.payment_method == "credit_card").astype(float)
    od["cancel"] = (1 - od.done)
    od = od.sort_values(["customer_id", "transaction_datetime"])
    od["gap"] = od.groupby("customer_id").transaction_datetime.diff().dt.total_seconds() / 86400
    od["rk"] = od.groupby("customer_id").cumcount(ascending=False)

    g = od.groupby("customer_id")
    out = pd.DataFrame({
        "ed_orders": g.size(),
        "ed_span": g.days_ago.max(),
        "ed_first": g.days_ago.max(),
        "ed_last": g.days_ago.min(),
        "ed_per_month": g.size() / (g.days_ago.max() / 30.44 + 1),
        "ed_gap_mean": g.gap.mean(),
        "ed_gap_last": g.gap.last(),
        "ed_items_mean": g.n_items.mean(),
        "ed_items_max": g.n_items.max(),
        "ed_cancel": g.cancel.mean(),
        "ed_cod": g.cod.mean(),
        "ed_cc": g.cc.mean(),
        "ed_night": g.night.mean(),
        "ed_weekend": g.weekend.mean(),
        "ed_basket_mean": g.order_total.mean(),
        "ed_basket_max": g.order_total.max(),
        "ed_basket_cv": g.order_total.std() / g.order_total.mean().replace(0, np.nan),
    })
    out["ed_gap_cv"] = od.groupby("customer_id").gap.std() / out.ed_gap_mean
    out["ed_cancel_last10"] = od[od.rk < 10].groupby("customer_id").cancel.mean()
    out["ed_cod_last10"] = od[od.rk < 10].groupby("customer_id").cod.mean()
    out["ed_cc_last10"] = od[od.rk < 10].groupby("customer_id").cc.mean()

    # month-level spend panel
    mo = ec.groupby(["customer_id", "month"]).agg(
        m_spend=("spend", "sum"), m_orders=("order_id", "nunique"),
        m_items=("quantity", "sum"), m_lunit=("lunit", "mean"),
        m_need=("is_need", "mean"), m_want=("is_want", "mean"),
    ).reset_index()
    mg = mo.groupby("customer_id")
    out["ed_months"] = mg.size()
    out["ed_spend_cv"] = mg.m_spend.std() / mg.m_spend.mean().replace(0, np.nan)
    out["ed_spend_zero_share"] = mg.m_spend.apply(lambda s: float((s <= 0).mean()))
    out["ed_orders_cv"] = mg.m_orders.std() / mg.m_orders.mean().replace(0, np.nan)
    out["ed_lunit_cv"] = mg.m_lunit.std() / mg.m_lunit.mean().replace(0, np.nan)
    out["ed_need_mean"] = mg.m_need.mean()
    out["ed_want_mean"] = mg.m_want.mean()

    mo = mo.sort_values(["customer_id", "month"])
    mo["rk"] = mo.groupby("customer_id").cumcount(ascending=False)
    for n in (3, 6):
        t = mo[mo.rk < n].groupby("customer_id")
        out[f"ed_spend_{n}"] = t.m_spend.mean()
        out[f"ed_orders_{n}"] = t.m_orders.mean()
        out[f"ed_lunit_{n}"] = t.m_lunit.mean()
        out[f"ed_need_{n}"] = t.m_need.mean()
        out[f"ed_want_{n}"] = t.m_want.mean()
    out["ed_spend_trend"] = out.ed_spend_3 - out.ed_spend_6
    out["ed_lunit_trend"] = out.ed_lunit_3 - out.ed_lunit_6
    out["ed_need_trend"] = out.ed_need_3 - out.ed_need_6

    # linear slope of monthly spend and unit price, normalised by the mean
    for col, name in [("m_spend", "ed_spend_slope"), ("m_lunit", "ed_lunit_slope")]:
        def sl(s):
            v = s.to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 3:
                return np.nan
            x = np.arange(len(v))[ok]
            yy = v[ok]
            mu = np.mean(yy)
            return float(np.polyfit(x, yy, 1)[0] / abs(mu)) if abs(mu) > 1e-9 else np.nan
        out[name] = mo.groupby("customer_id")[col].apply(sl)

    # category entropy per customer and its recent shift
    def entropy(s):
        p = s / s.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum()) if len(p) else np.nan

    cat = ec[ec.done == 1].groupby(["customer_id", "product_category"]).spend.sum()
    out["ed_cat_entropy"] = cat.groupby("customer_id").apply(entropy)
    recent = ec[(ec.done == 1) & (ec.days_ago <= 90)]
    c2 = recent.groupby(["customer_id", "product_category"]).spend.sum()
    out["ed_cat_entropy_90"] = c2.groupby("customer_id").apply(entropy)
    out["ed_cat_entropy_shift"] = out.ed_cat_entropy_90 - out.ed_cat_entropy

    # absolute unit price against declared income, and spend against income
    inc = pd.read_csv(
        D + "customer_profiles.csv",
        usecols=["customer_id", "declared_monthly_income"],
    ).set_index("customer_id").declared_monthly_income
    out["ed_lunit"] = ec.groupby("customer_id").lunit.mean()
    out["ed_unit"] = ec.groupby("customer_id").unit.mean()
    out["ed_unit_to_income"] = out.ed_unit / out.index.map(inc).astype(float)
    out["ed_spend_to_income"] = (
        od.groupby("customer_id").order_total.sum()
        / out.index.map(inc).astype(float) / (out.ed_span / 30.44 + 1)
    )
    out["ed_basket_to_income"] = out.ed_basket_mean / out.index.map(inc).astype(float)

    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out


def block_ratefit(app):
    """Within-grade residuals and grade interactions. Uses no labels."""
    f = app[["application_id", "customer_id", "annual_interest_rate"]].copy()
    f["rf_grade"] = f.annual_interest_rate.round(2).rank(method="dense").astype(float)

    # features we want to residualise within grade
    tr = pd.read_pickle(D + "train_applications.csv").iloc[0:0]  # placeholder
    return f


BLOCKS = {
    "monthstress": block_monthstress,
    "spendstress": block_spendstress,
    "loanstatus": block_loanstatus,
    "ecomdepth": block_ecomdepth,
}


def build(names=None, app=None):
    names = list(names) if names else list(BLOCKS)
    if app is None:
        tr = pd.read_csv(D + "train_applications.csv",
                         parse_dates=["application_datetime", "decision_datetime"])
        te = pd.read_csv(D + "test_applications.csv",
                         parse_dates=["application_datetime", "decision_datetime"])
        app = pd.concat([tr, te], ignore_index=True)
    cmap = app[["customer_id", "application_id"]].drop_duplicates("customer_id")
    parts = []
    for name in names:
        p = BLOCKS[name](app)
        if "application_id" not in p.columns:
            p = p.merge(cmap, on="customer_id", how="left").drop(columns=["customer_id"])
        parts.append(p[["application_id"] + [c for c in p.columns if c != "application_id"]])
    out = app[["application_id"]].copy()
    for p in parts:
        out = out.merge(p, on="application_id", how="left")
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
    import paths

    out = build()
    print(out.shape)
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(paths.CACHE / "v70_blocks.pkl")
