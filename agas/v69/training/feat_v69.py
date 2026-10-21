"""Feature blocks for v69.

Three sequence blocks that the earlier rounds did not express:

`cashflow` — recent cash-flow shocks and buffers. Existing account features
summarise levels, averages and a few last-three-month means. This block asks
whether the last quarter is worse than the customer's own baseline, whether
income is regular, and how many months of instalments the observed balance can
cover.

`repayseq` — repayment transitions rather than counts. Existing features count
late instalments and rates. This block measures the probability of rolling from
late to late, curing, worsening or improving, the current on-time streak, and
the payment timing distribution.

`prevseq` — the sequence of previous application outcomes. Existing features
count approvals and rejections. This block measures recent approval rate,
approval streaks, the terms of the last approved loan, and how the current
request compares with that last approved loan.

Every block cuts at `decision_datetime`; no future outcomes, no test labels and
no current-application target are used.
"""

import numpy as np
import pandas as pd

D = "data/"


def _cut(app):
    return app[["application_id", "customer_id", "decision_datetime"]].rename(
        columns={"decision_datetime": "cut"}
    )


def _cv(s):
    m = s.mean()
    return float(s.std() / abs(m)) if m and np.isfinite(m) else np.nan


def _slope(s):
    v = s.to_numpy(dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(v), dtype=float)[ok]
    m = np.nanmean(v[ok])
    if not np.isfinite(m) or abs(m) < 1e-9:
        return np.nan
    return float(np.polyfit(x, v[ok], 1)[0] / abs(m))


def _streak_last(s):
    v = s.to_numpy(dtype=bool)[::-1]
    n = 0
    for x in v:
        if x:
            n += 1
        else:
            break
    return float(n)


def _streak_max(s):
    best = cur = 0
    for x in s.to_numpy(dtype=bool):
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return float(best)


def block_cashflow(app):
    ab = pd.read_csv(
        D + "account_behavior.csv",
        parse_dates=["month"],
        usecols=["customer_id", "month", "opening_balance", "closing_balance",
                 "inflows", "outflows", "income_like_credits", "average_balance",
                 "minimum_balance", "low_balance_days"],
    )
    cut = _cut(app)[["customer_id", "cut"]]
    ab = ab.merge(cut, on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut].copy()
    ab = ab.sort_values(["customer_id", "month"])
    inst = app.set_index("customer_id").proposed_monthly_installment
    ab["inst"] = ab.customer_id.map(inst).replace(0, np.nan)
    ab["net"] = ab.inflows - ab.outflows
    ab["margin"] = (ab.inflows - ab.outflows) / ab.inflows.replace(0, np.nan)
    ab["buffer"] = ab.average_balance / ab.inst
    ab["min_buffer"] = ab.minimum_balance / ab.inst
    ab["rev_rank"] = ab.groupby("customer_id").cumcount(ascending=False)

    g = ab.groupby("customer_id")
    out = pd.DataFrame({
        "cf_months": g.size(),
        "cf_inc_mean": g.income_like_credits.mean(),
        "cf_inc_cv": g.income_like_credits.apply(_cv),
        "cf_inc_slope": g.income_like_credits.apply(_slope),
        "cf_inflow_cv": g.inflows.apply(_cv),
        "cf_outflow_cv": g.outflows.apply(_cv),
        "cf_margin_mean": g.margin.mean(),
        "cf_margin_std": g.margin.std(),
        "cf_net_neg_share": g.net.apply(lambda s: float((s < 0).mean())),
        "cf_bal_mean": g.average_balance.mean(),
        "cf_bal_cv": g.average_balance.apply(_cv),
        "cf_bal_slope": g.average_balance.apply(_slope),
        "cf_bal_min": g.average_balance.min(),
        "cf_bal_max": g.average_balance.max(),
        "cf_lowdays_mean": g.low_balance_days.mean(),
        "cf_lowdays_max": g.low_balance_days.max(),
        "cf_buffer_mean": g.buffer.mean(),
        "cf_buffer_min": g.buffer.min(),
        "cf_min_buffer_mean": g.min_buffer.mean(),
        "cf_inc_zero_share": g.income_like_credits.apply(lambda s: float((s <= 0).mean())),
    })

    for n in (1, 3, 6):
        r = ab[ab.rev_rank < n].groupby("customer_id")
        out[f"cf_inc_mean{n}"] = r.income_like_credits.mean()
        out[f"cf_margin_mean{n}"] = r.margin.mean()
        out[f"cf_buffer_mean{n}"] = r.buffer.mean()
        out[f"cf_min_buffer_mean{n}"] = r.min_buffer.mean()
        out[f"cf_lowdays_mean{n}"] = r.low_balance_days.mean()
        out[f"cf_bal_mean{n}"] = r.average_balance.mean()
        out[f"cf_net_neg_share{n}"] = r.net.apply(lambda s: float((s < 0).mean()))
        out[f"cf_inc_zero_n{n}"] = r.income_like_credits.apply(lambda s: float((s <= 0).sum()))

    prior = ab[ab.rev_rank >= 3].groupby("customer_id").income_like_credits.agg(["mean", "size"])
    prior_mean = prior["mean"].reindex(out.index)
    out["cf_inc_mean3_over_prior"] = out.cf_inc_mean3 / prior_mean.replace(0, np.nan)
    prior3 = ab[(ab.rev_rank >= 3) & (ab.rev_rank < 6)].groupby("customer_id").income_like_credits.mean()
    out["cf_inc_mean3_over_prior3"] = out.cf_inc_mean3 / prior3.replace(0, np.nan)
    out["cf_inc_last_over_mean"] = out.cf_inc_mean1 / out.cf_inc_mean.replace(0, np.nan)
    out["cf_bal_last_over_mean"] = out.cf_bal_mean1 / out.cf_bal_mean.replace(0, np.nan)
    out["cf_bal_drawdown"] = (out.cf_bal_max - out.cf_bal_mean1) / out.cf_bal_max.replace(0, np.nan)
    out["cf_margin_last3_delta"] = out.cf_margin_mean3 - out.cf_margin_mean
    out["cf_lowdays_last3_delta"] = out.cf_lowdays_mean3 - out.cf_lowdays_mean
    out["cf_buffer_last3_delta"] = out.cf_buffer_mean3 - out.cf_buffer_mean

    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out


def block_repayseq(app):
    rh = pd.read_csv(
        D + "repayment_history.csv",
        parse_dates=["due_date", "last_payment_datetime"],
        usecols=["loan_id", "customer_id", "due_date", "amount_due", "amount_paid",
                 "last_payment_datetime", "days_past_due", "max_days_past_due"],
    )
    cut = _cut(app)[["customer_id", "cut"]]
    rh = rh.merge(cut, on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.total_seconds() / 86400
    rh = rh[rh.days_ago >= 0].sort_values(["customer_id", "loan_id", "due_date"]).copy()
    rh["paid_ratio"] = rh.amount_paid / rh.amount_due.replace(0, np.nan)
    rh["pay_delay"] = (rh.last_payment_datetime - rh.due_date).dt.total_seconds() / 86400
    rh["late"] = (rh.max_days_past_due > 0).astype(float)
    rh["late30"] = (rh.max_days_past_due >= 30).astype(float)
    rh["late90"] = (rh.max_days_past_due >= 90).astype(float)
    rh["ontime"] = ((rh.max_days_past_due == 0) & (rh.paid_ratio >= 0.95)).astype(float)
    rh["state"] = np.select(
        [rh.max_days_past_due == 0, rh.max_days_past_due < 30,
         rh.max_days_past_due < 90],
        [0, 1, 2], default=3,
    )
    rh["rev_rank"] = rh.groupby("customer_id").cumcount(ascending=False)
    grp = rh.groupby(["customer_id", "loan_id"], sort=False)
    rh["prev_late"] = grp.late.shift()
    rh["prev_state"] = grp.state.shift()
    rh["prev_delay"] = grp.pay_delay.shift()
    rh["roll"] = rh.late.where(rh.prev_late == 1)
    rh["cure"] = (1.0 - rh.late).where(rh.prev_late == 1)
    rh["onset"] = rh.late.where(rh.prev_late == 0)
    rh["worsen"] = (rh.state > rh.prev_state).where(rh.prev_state.notna()).astype(float)
    rh["improve"] = (rh.state < rh.prev_state).where(rh.prev_state.notna()).astype(float)
    rh["cure_flag"] = ((rh.prev_late == 1) & (rh.late == 0)).astype(float)
    rh["cured_cum"] = grp.cure_flag.cumsum()
    rh["cured_before"] = rh.groupby(["customer_id", "loan_id"], sort=False).cured_cum.shift(1).fillna(0) > 0
    rh["late_after_cure"] = (rh.late * rh.cured_before).where(rh.cured_before)

    g = rh.groupby("customer_id")
    out = pd.DataFrame({
        "rq_n": g.size(),
        "rq_loans": g.loan_id.nunique(),
        "rq_late_rate": g.late.mean(),
        "rq_late30_rate": g.late30.mean(),
        "rq_late90_rate": g.late90.mean(),
        "rq_ontime_rate": g.ontime.mean(),
        "rq_roll_rate": g.roll.mean(),
        "rq_cure_rate": g.cure.mean(),
        "rq_onset_rate": g.onset.mean(),
        "rq_worsen_rate": g.worsen.mean(),
        "rq_improve_rate": g.improve.mean(),
        "rq_pay_early_rate": g.pay_delay.apply(lambda s: float((s < 0).mean())),
        "rq_pay_late5_rate": g.pay_delay.apply(lambda s: float((s > 5).mean())),
        "rq_delay_mean": g.pay_delay.mean(),
        "rq_delay_max": g.pay_delay.max(),
        "rq_paid_ratio_mean": g.paid_ratio.mean(),
        "rq_paid_ratio_min": g.paid_ratio.min(),
        "rq_streak_ontime_last": g.ontime.apply(_streak_last),
        "rq_streak_late_max": g.late.apply(_streak_max),
        "rq_late_after_cure": g.late_after_cure.mean(),
        "rq_late_n": g.late.sum(),
        "rq_late30_n": g.late30.sum(),
        "rq_late90_n": g.late90.sum(),
    })

    for flag, name in [("late", "late"), ("late30", "late30"), ("late90", "late90")]:
        m = rh[rh[flag] == 1]
        out[f"rq_days_since_{name}"] = m.groupby("customer_id").days_ago.min()
    r3 = rh[rh.rev_rank < 3].groupby("customer_id")
    out["rq_late_rate_last3"] = r3.late.mean()
    out["rq_ontime_rate_last3"] = r3.ontime.mean()
    out["rq_delay_mean_last3"] = r3.pay_delay.mean()
    out["rq_paid_ratio_last3"] = r3.paid_ratio.mean()
    out["rq_late90_last3"] = r3.late90.mean()

    last_loan = rh.sort_values("due_date").groupby("customer_id").loan_id.last()
    lm = rh[rh.loan_id.eq(rh.customer_id.map(last_loan))]
    lg = lm.groupby("customer_id")
    out["rq_lastloan_n"] = lg.size()
    out["rq_lastloan_late_rate"] = lg.late.mean()
    out["rq_lastloan_ontime_rate"] = lg.ontime.mean()
    out["rq_lastloan_paid_ratio"] = lg.paid_ratio.mean()
    out["rq_lastloan_dpd_max"] = lg.max_days_past_due.max()

    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out


def block_prevseq(app):
    pa = pd.read_csv(
        D + "previous_applications.csv",
        parse_dates=["application_datetime", "decision_datetime"],
        usecols=["customer_id", "application_datetime", "decision_datetime",
                 "requested_amount", "term_months", "decision", "approved_amount",
                 "annual_interest_rate", "scheduled_monthly_installment"],
    )
    cut = _cut(app)[["customer_id", "cut"]]
    pa = pa.merge(cut, on="customer_id")
    pa = pa[(pa.application_datetime <= pa.cut) & (pa.decision_datetime <= pa.cut)].copy()
    pa = pa.sort_values(["customer_id", "decision_datetime"])
    pa["appr"] = (pa.decision == "approved").astype(float)
    pa["rej"] = (pa.decision == "rejected").astype(float)
    pa["can"] = (pa.decision == "cancelled").astype(float)
    pa["days_since"] = (pa.cut - pa.decision_datetime).dt.total_seconds() / 86400
    pa["decision_lag"] = (pa.decision_datetime - pa.application_datetime).dt.total_seconds() / 86400
    pa["appr_ratio"] = pa.approved_amount / pa.requested_amount.replace(0, np.nan)
    pa["rev_rank"] = pa.groupby("customer_id").cumcount(ascending=False)

    g = pa.groupby("customer_id")
    out = pd.DataFrame({
        "pv_n": g.size(),
        "pv_appr_rate": g.appr.mean(),
        "pv_rej_rate": g.rej.mean(),
        "pv_can_rate": g.can.mean(),
        "pv_n90": g.apply(lambda d: float((d.days_since <= 90).sum()), include_groups=False),
        "pv_n180": g.apply(lambda d: float((d.days_since <= 180).sum()), include_groups=False),
        "pv_n365": g.apply(lambda d: float((d.days_since <= 365).sum()), include_groups=False),
        "pv_appr_n90": g.apply(lambda d: float(((d.appr == 1) & (d.days_since <= 90)).sum()), include_groups=False),
        "pv_rej_n90": g.apply(lambda d: float(((d.rej == 1) & (d.days_since <= 90)).sum()), include_groups=False),
        "pv_decision_lag_mean": g.decision_lag.mean(),
        "pv_gap_last": g.days_since.min(),
        "pv_req_mean": g.requested_amount.mean(),
        "pv_req_max": g.requested_amount.max(),
        "pv_appr_amount_sum": g.approved_amount.sum(),
        "pv_appr_ratio_mean": g.appr_ratio.mean(),
        "pv_rej_streak_last": g.rej.apply(_streak_last),
        "pv_appr_streak_last": g.appr.apply(_streak_last),
    })

    for n in (3, 5):
        r = pa[pa.rev_rank < n].groupby("customer_id")
        out[f"pv_appr_rate_last{n}"] = r.appr.mean()
        out[f"pv_rej_rate_last{n}"] = r.rej.mean()
    older = pa[pa.rev_rank >= 3].groupby("customer_id").appr.mean()
    out["pv_appr_rate_delta"] = out.pv_appr_rate_last3 - older

    appr = pa[pa.appr == 1]
    last = appr.sort_values("decision_datetime").groupby("customer_id").tail(1).set_index("customer_id")
    out["pv_last_appr_days"] = last.days_since
    out["pv_last_appr_amount"] = last.approved_amount
    out["pv_last_appr_inst"] = last.scheduled_monthly_installment
    out["pv_last_appr_rate"] = last.annual_interest_rate
    out["pv_last_appr_term"] = last.term_months
    out["pv_last_appr_ratio"] = last.appr_ratio

    rej = pa[pa.rej == 1].groupby("customer_id").days_since.min()
    can = pa[pa.can == 1].groupby("customer_id").days_since.min()
    out["pv_days_since_rej"] = rej
    out["pv_days_since_can"] = can

    last_req = pa.sort_values("decision_datetime").groupby("customer_id").tail(1).set_index("customer_id")
    out["pv_last_req_amount"] = last_req.requested_amount
    out["pv_last_req_days"] = last_req.days_since
    out["pv_last_req_growth"] = last_req.requested_amount / out.pv_req_mean.replace(0, np.nan)
    out["pv_last_req_vs_max"] = last_req.requested_amount / out.pv_req_max.replace(0, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan).reset_index()
    return out


BLOCKS = {
    "cashflow": block_cashflow,
    "repayseq": block_repayseq,
    "prevseq": block_prevseq,
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
    out.to_pickle(paths.CACHE / "v69_blocks.pkl")
