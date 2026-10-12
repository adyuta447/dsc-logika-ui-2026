import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
S = sys.argv[1]
sys.argv = ["bag.py", "run_fac"]
exec(open("bag.py").read().split("for v in")[0])
p = B["run_fac"][0]["top_mix"]; lo = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
f = pd.read_parquet("features.parquet", columns=["application_id", "customer_id", "target"]); t = f[f.target.notna()].reset_index(drop=True)
app = pd.read_csv("data/train_applications.csv", parse_dates=["decision_datetime"]).set_index("application_id").loc[t.application_id.astype(str)].reset_index()
cut = app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"})
M = pd.get_dummies(app.decision_datetime.dt.month).values[:, 1:].astype(float)
def ll(X):
    m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y); q = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9); return (y * np.log(q) + (1 - y) * np.log(1 - q)).sum()
X0 = np.column_stack([lo, M]); base = ll(X0); rows = []
def scan(df, time, cols, tag):
    df = df.sort_values(["customer_id", time]); df["k"] = df.groupby("customer_id").cumcount(ascending=False)
    df["tt"] = (df[time] - df.groupby("customer_id")[time].transform("min")).dt.days
    g = df.groupby("customer_id")
    for c in cols:
        x = df[c].astype(float); gg = x.groupby(df.customer_id)
        stats = {"mean": gg.mean(), "std": gg.std(), "last": x[df.k == 0].groupby(df.customer_id[df.k == 0]).mean(),
                 "first": gg.first(), "min": gg.min(), "max": gg.max(),
                 "recent3_minus_mean": x[df.k < 3].groupby(df.customer_id[df.k < 3]).mean() - gg.mean(),
                 "slope": (df.assign(xt=x * df.tt).groupby("customer_id").xt.mean() - gg.mean() * g.tt.mean()) / (g.tt.var(ddof=0) + 1e-9)}
        for s, v in stats.items():
            v = t.customer_id.map(v).astype(float).replace([np.inf, -np.inf], np.nan)
            if v.notna().mean() < 0.05 or v.nunique() < 3: continue
            r = v.rank(pct=True).fillna(0.5).values; na = v.isna().astype(float).values
            rows.append((f"{tag}_{c}_{s}", round(v.notna().mean(), 2), round(2 * (ll(np.column_stack([X0, r, r ** 2, na])) - base), 1)))
        print(tag, c, "done", flush=True)
ab = pd.read_csv("data/account_behavior.csv", parse_dates=["month"]).merge(cut, on="customer_id")
ab["net"] = ab.inflows - ab.outflows; ab["ilc_share"] = ab.income_like_credits / ab.inflows.replace(0, np.nan)
ab["bal_change"] = ab.closing_balance - ab.opening_balance; ab["min_to_avg"] = ab.minimum_balance / ab.average_balance.replace(0, np.nan)
scan(ab, "month", ["opening_balance", "closing_balance", "inflows", "outflows", "income_like_credits", "average_balance", "minimum_balance", "low_balance_days", "net", "ilc_share", "bal_change", "min_to_avg"], "ab")
bu = pd.read_csv("data/bureau_observations.csv", parse_dates=["available_datetime"]).merge(cut, on="customer_id")
scan(bu, "available_datetime", ["score", "active_accounts", "outstanding_balance", "revolving_credit_limit", "revolving_outstanding_balance", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m"], "bu")
rh = pd.read_csv("data/repayment_history.csv", parse_dates=["due_date", "last_payment_datetime"]).merge(cut, on="customer_id")
rh["delay"] = (rh.last_payment_datetime - rh.due_date).dt.total_seconds() / 86400; rh["paid_ratio"] = rh.amount_paid / rh.amount_due
scan(rh, "due_date", ["amount_due", "amount_paid", "days_past_due", "max_days_past_due", "delay", "paid_ratio"], "rh")
d = pd.DataFrame(rows, columns=["feature", "cov", "LR"]).sort_values("LR", ascending=False)
d.to_csv(S + "sysscan.csv", index=False); print(len(d), "candidates"); print(d.head(30).to_string(index=False))
