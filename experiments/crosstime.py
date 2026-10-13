import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.argv = ["bag.py", "run_v4_s42", "run_v4_s7", "run_v4_s13"]
exec(open("bag.py").read().split("for v in")[0])
p = np.mean([B[r][0]["top_mix"] for r in runs], 0); lo = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
f = pd.read_parquet("features.parquet", columns=["application_id", "customer_id", "target"]); t = f[f.target.notna()].reset_index(drop=True)
allapp = pd.concat([pd.read_csv("data/train_applications.csv", parse_dates=["decision_datetime"]), pd.read_csv("data/test_applications.csv", parse_dates=["decision_datetime"])])
cut = allapp.set_index("customer_id").decision_datetime
pa = pd.read_csv("data/previous_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
ec = pd.read_csv("data/ecommerce_transactions.csv", parse_dates=["transaction_datetime"])
ec = ec[ec.order_status == "completed"].drop_duplicates("order_id")[["customer_id", "transaction_datetime", "order_total"]]
rh = pd.read_csv("data/repayment_history.csv", parse_dates=["due_date"])
ab = pd.read_csv("data/account_behavior.csv", parse_dates=["month"])
bu = pd.read_csv("data/bureau_observations.csv", parse_dates=["available_datetime"])
C = {}
# 1. spending in the 30 days after an approved loan, relative to the customer's usual 30 day spending
appr = pa[pa.decision == "approved"][["customer_id", "decision_datetime"]]
span = (ec.groupby("customer_id").transaction_datetime.max() - ec.groupby("customer_id").transaction_datetime.min()).dt.days.clip(lower=30)
base30 = ec.groupby("customer_id").order_total.sum() / span * 30
m = appr.merge(ec, on="customer_id"); m = m[(m.transaction_datetime > m.decision_datetime) & (m.transaction_datetime <= m.decision_datetime + pd.Timedelta(days=30))]
after = m.groupby(["customer_id", "decision_datetime"]).order_total.sum().groupby("customer_id").mean()
C["spend_after_loan_vs_usual"] = (after.reindex(appr.customer_id.unique()).fillna(0) / base30.reindex(appr.customer_id.unique())).dropna()
# 2. lateness in months with a low balance against lateness in other months
rh["m"] = rh.due_date.dt.to_period("M").dt.to_timestamp(); rh["late"] = (rh.max_days_past_due > 0).astype(float)
j = rh.merge(ab[["customer_id", "month", "low_balance_days", "minimum_balance"]], left_on=["customer_id", "m"], right_on=["customer_id", "month"])
j["low"] = j.low_balance_days > 0
gl = j.groupby(["customer_id", "low"]).late.mean().unstack()
C["late_low_minus_normal"] = (gl.get(True) - gl.get(False)).dropna()
C["late_share_in_low_months"] = j[j.late == 1].groupby("customer_id").low.mean()
# 3. how fast a customer comes back after a rejection
pa = pa.sort_values(["customer_id", "application_datetime"])
pa["next_gap"] = pa.groupby("customer_id").application_datetime.shift(-1) - pa.application_datetime
C["days_to_reapply_after_reject"] = pa[pa.decision == "rejected"].groupby("customer_id").next_gap.mean().dt.days
C["reapply_within_30d_after_reject"] = (pa[pa.decision == "rejected"].next_gap.dt.days <= 30).groupby(pa[pa.decision == "rejected"].customer_id).mean()
# 4. bureau score change after an approved loan
bu["z"] = (bu.score - bu.groupby("bureau_provider").score.transform("mean")) / bu.groupby("bureau_provider").score.transform("std")
bb = appr.merge(bu[["customer_id", "available_datetime", "z"]].dropna(), on="customer_id")
pre = bb[(bb.available_datetime < bb.decision_datetime) & (bb.available_datetime > bb.decision_datetime - pd.Timedelta(days=180))].groupby(["customer_id", "decision_datetime"]).z.mean()
post = bb[(bb.available_datetime > bb.decision_datetime) & (bb.available_datetime < bb.decision_datetime + pd.Timedelta(days=180))].groupby(["customer_id", "decision_datetime"]).z.mean()
C["bureau_z_change_after_loan"] = (post - pre).dropna().groupby("customer_id").mean()
# 5. spending in the month before each repayment that was late, against spending before on time ones
def ll(X):
    mm = LogisticRegression(C=1e6, max_iter=3000).fit(X, y); q = np.clip(mm.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9); return (y * np.log(q) + (1 - y) * np.log(1 - q)).sum()
base = ll(lo[:, None]); rows = []
for k, s in C.items():
    v = t.customer_id.map(s).astype(float).replace([np.inf, -np.inf], np.nan); ok = v.notna().values
    rr = v.rank(pct=True).fillna(0.5).values; na = v.isna().astype(float).values
    a = roc_auc_score(y[ok], v[ok]) if ok.sum() > 100 and v.nunique() > 2 else np.nan
    rows.append((k, round(ok.mean(), 2), round(max(a, 1 - a), 4) if a == a else None, round(2 * (ll(np.column_stack([lo, rr, rr ** 2, na])) - base), 1)))
print(pd.DataFrame(rows, columns=["feature", "cov", "auc_alone", "LR_over_v4"]).sort_values("LR_over_v4", ascending=False).to_string(index=False))
