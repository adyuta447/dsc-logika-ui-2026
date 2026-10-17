import numpy as np, pandas as pd
# New candidates from the non-ecommerce tables, angles features.py does not cover: bureau scores standardised per provider, score velocity and
# reports not yet available at decision; previous decisions (approval cuts, last rate vs current rate); account drawdowns and zero-income months;
# early and partial repayments. Writes work/oth_cand.parquet keyed by application_id.
D = "data/"
app = pd.concat([pd.read_csv(D + f"{s}_applications.csv", parse_dates=["decision_datetime"]) for s in ("train", "test")])
cut = app[["application_id", "customer_id", "decision_datetime", "annual_interest_rate", "loan_amount"]]
out = pd.DataFrame(index=app.application_id)

bu = pd.read_csv(D + "bureau_observations.csv", parse_dates=["report_datetime", "available_datetime"])
bu["z"] = bu.groupby("bureau_provider").score.transform(lambda s: (s - s.mean()) / s.std())
bu = bu.merge(cut[["application_id", "customer_id", "decision_datetime"]], on="customer_id", suffixes=("", "_sel"))
bu = bu[bu.report_datetime < bu.decision_datetime]
bu["avail"] = bu.available_datetime <= bu.decision_datetime
bu["t"] = (bu.decision_datetime - bu.report_datetime).dt.days
g = bu.groupby("application_id_sel")
out["od_bu_z_mean"], out["od_bu_z_std"] = g.z.mean(), g.z.std()
out["od_bu_unavail_share"] = 1 - g.avail.mean()
av = bu[bu.avail].sort_values("report_datetime").groupby("application_id_sel")
out["od_bu_avail_last_z"], out["od_bu_avail_last_age"] = av.z.last(), av.t.last()
out["od_bu_z_last_vs_mean"] = bu.sort_values("report_datetime").groupby("application_id_sel").z.last() - out.od_bu_z_mean
def slope(d, c): return np.polyfit(-d.t, d[c], 1)[0] * 365 if len(d) > 2 and d.t.nunique() > 1 else np.nan
for c in ("z", "utilization", "inquiries_last_6m"):
    out[f"od_bu_slope_{c}"] = bu.dropna(subset=[c]).groupby("application_id_sel")[["t", c]].apply(lambda d: slope(d, c))
out["od_bu_n90"] = (bu.t <= 90).groupby(bu.application_id_sel).sum()
out["od_bu_util_gap"] = (bu.revolving_outstanding_balance / bu.revolving_credit_limit.replace(0, np.nan) - bu.utilization).groupby(bu.application_id_sel).mean()

pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
pa = pa.merge(cut, on="customer_id", suffixes=("", "_sel"))
pa = pa[pa.decision_datetime < pa.decision_datetime_sel].sort_values("decision_datetime")
pa["days"] = (pa.decision_datetime_sel - pa.decision_datetime).dt.days
ap = pa[pa.decision == "approved"]; ga = ap.groupby("application_id_sel")
out["od_pa_cut_mean"] = (ap.approved_amount / ap.requested_amount).groupby(ap.application_id_sel).mean()
out["od_pa_cut_share"] = (ap.approved_amount < ap.requested_amount).groupby(ap.application_id_sel).mean()
out["od_pa_last_rate_diff"] = ga.annual_interest_rate_sel.last() - ga.annual_interest_rate.last()
out["od_pa_last_appr_days"] = ga.days.last()
out["od_pa_loan_vs_last_req"] = pa.groupby("application_id_sel").loan_amount.last() / pa.groupby("application_id_sel").requested_amount.last()
rj = pa[pa.decision == "rejected"].groupby("application_id_sel")
out["od_pa_last_rej_days"] = rj.days.last(); out["od_pa_rej30"] = ((pa.days <= 30) & (pa.decision == "rejected")).groupby(pa.application_id_sel).sum()
out["od_pa_last_decision"] = pa.groupby("application_id_sel").decision.last().map({"approved": 0, "cancelled": 1, "rejected": 2})

ab = pd.read_csv(D + "account_behavior.csv", parse_dates=["month"]).merge(cut[["application_id", "customer_id", "decision_datetime"]], on="customer_id")
ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.decision_datetime].sort_values("month")
ab["k"] = ab.groupby("application_id").cumcount(ascending=False)
gb = ab.groupby("application_id"); l6 = ab[ab.k < 6].groupby("application_id")
out["od_ab_zero_inc6"] = l6.income_like_credits.apply(lambda s: (s == 0).sum())
out["od_ab_min_zero6"] = l6.minimum_balance.apply(lambda s: (s <= 0).sum())
out["od_ab_drawdown"] = gb.closing_balance.last() / gb.closing_balance.max().replace(0, np.nan)
out["od_ab_low_days_trend"] = ab[ab.k < 3].groupby("application_id").low_balance_days.mean() - ab[ab.k.between(3, 11)].groupby("application_id").low_balance_days.mean()
out["od_ab_inc_cv6"] = l6.income_like_credits.std() / l6.income_like_credits.mean().replace(0, np.nan)
out["od_ab_out_spike"] = ab[ab.k == 0].groupby("application_id").outflows.mean() / l6.outflows.median().replace(0, np.nan)

rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date", "last_payment_datetime"]).merge(cut[["application_id", "customer_id", "decision_datetime"]], on="customer_id")
rh = rh[rh.due_date < rh.decision_datetime]
early = (rh.due_date - rh.last_payment_datetime).dt.total_seconds() / 86400
gr = rh.groupby("application_id")
out["od_rh_early_share"] = (early > 0).groupby(rh.application_id).mean(); out["od_rh_early_days"] = early.clip(lower=0).groupby(rh.application_id).mean()
out["od_rh_partial_share"] = ((rh.amount_paid > 0) & (rh.amount_paid < rh.amount_due)).groupby(rh.application_id).mean()
out["od_rh_active_loans"] = rh[(rh.decision_datetime - rh.due_date).dt.days <= 35].groupby("application_id").loan_id.nunique()
la = rh.groupby(["application_id", "loan_id"]).agg(a=("amount_due", "mean"), e=("due_date", "max")).sort_values("e").groupby(level=0).a
out["od_rh_amt_growth"] = la.last() / la.first()
out = out.reset_index()
assert out.application_id.is_unique and len(out) == len(app)
out.to_parquet("work/oth_cand.parquet"); print(out.shape, out.notna().mean().round(2).to_dict())
