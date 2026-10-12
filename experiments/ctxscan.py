import numpy as np, pandas as pd, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
sys.argv = ["bag.py", "run_fac"]
exec(open("bag.py").read().split("for v in")[0])
p = B["run_fac"][0]["top_mix"]; lo = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
f = pd.read_parquet("features.parquet"); ftr = f[f.target.notna()].reset_index(drop=True)
tr = pd.read_csv("data/train_applications.csv", parse_dates=["application_datetime", "decision_datetime"]).set_index("application_id").loc[ftr.application_id].reset_index()
def ll(X):
    m = LogisticRegression(C=1e6, max_iter=3000).fit(X, y); q = m.predict_proba(X)[:, 1]; return (y * np.log(q) + (1 - y) * np.log(1 - q)).sum()
base = ll(lo[:, None])
sp = lambda v: SplineTransformer(n_knots=6, degree=3).fit_transform(pd.Series(v).rank(pct=True).values[:, None])
dum = lambda v: pd.get_dummies(pd.Series(v).astype(str)).values[:, 1:].astype(float)
d, a = tr.decision_datetime, tr.application_datetime
cut = d.values
ec = pd.read_csv("data/ecommerce_transactions.csv", parse_dates=["transaction_datetime"]).groupby("customer_id").transaction_datetime.min()
ab = pd.read_csv("data/account_behavior.csv", parse_dates=["month"]).groupby("customer_id").month.min()
bu = pd.read_csv("data/bureau_observations.csv", parse_dates=["report_datetime"]).groupby("customer_id").report_datetime.min()
pa = pd.read_csv("data/previous_applications.csv", parse_dates=["application_datetime"]).groupby("customer_id").application_datetime.min()
first = pd.concat([ec, ab, bu, pa], axis=1).min(axis=1)
cand = {
    "application month": dum(a.dt.month), "application day of year spline": sp(a.dt.dayofyear),
    "decision minute": dum(d.dt.minute // 10), "decision second nonzero": (d.dt.second > 0).astype(float).values[:, None],
    "lag decision minus application (spline)": sp((d - a).dt.total_seconds()),
    "relationship age: first record anywhere (spline)": sp((d - tr.customer_id.map(first)).dt.days),
    "first ecommerce age (spline)": sp((d - tr.customer_id.map(ec)).dt.days.fillna(-1)),
    "first account month age (spline)": sp((d - tr.customer_id.map(ab)).dt.days.fillna(-1)),
    "first bureau report age (spline)": sp((d - tr.customer_id.map(bu)).dt.days.fillna(-1)),
    "term x quarter": dum(tr.term_months.astype(str) + "_" + d.dt.quarter.astype(str)),
    "rate grade x quarter": dum(tr.annual_interest_rate.round(2).astype(str) + "_" + d.dt.quarter.astype(str)),
    "loan purpose x quarter": dum(tr.loan_purpose.astype(str) + "_" + d.dt.quarter.astype(str)),
    "maturity month (decision + term) ": dum(((d.dt.month - 1 + tr.term_months) % 12)),
    "end of month decision": (d.dt.day >= 25).astype(float).values[:, None],
}
for k, X in cand.items():
    print(f"{k:48s} LR gain {2 * (ll(np.column_stack([lo, X])) - base):7.1f}  df {X.shape[1]}", flush=True)
