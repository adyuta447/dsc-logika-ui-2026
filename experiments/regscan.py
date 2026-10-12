import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
S = sys.argv[1]
sys.argv = ["bag.py", "run_v4_s42"]
exec(open("bag.py").read().split("for v in")[0])
p = B["run_v4_s42"][0]["top_mix"]; lo = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
f = pd.read_parquet("features.parquet", columns=["application_id", "customer_id", "target"]); t = f[f.target.notna()].reset_index(drop=True)
app = pd.read_csv("data/train_applications.csv", parse_dates=["decision_datetime"]).set_index("application_id").loc[t.application_id.astype(str)].reset_index()
allapp = pd.concat([pd.read_csv("data/train_applications.csv", parse_dates=["decision_datetime"]), pd.read_csv("data/test_applications.csv", parse_dates=["decision_datetime"])])
cut = allapp[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"})
def ll(X):
    m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y); q = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9); return (y * np.log(q) + (1 - y) * np.log(1 - q)).sum()
base = ll(lo[:, None]); rows = []; C = {}
def test(name, s):
    v = app.customer_id.map(s).astype(float).replace([np.inf, -np.inf], np.nan); C[name] = s
    if v.notna().mean() < 0.05 or v.nunique() < 3: return
    r = v.rank(pct=True).fillna(0.5).values; na = v.isna().astype(float).values
    rows.append((name, round(v.notna().mean(), 2), round(2 * (ll(np.column_stack([lo, r, r ** 2, na])) - base), 1)))
ab = pd.read_csv("data/account_behavior.csv", parse_dates=["month"]).merge(cut, on="customer_id"); ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut].sort_values(["customer_id", "month"])
ab["share"] = ab.income_like_credits / ab.inflows.replace(0, np.nan); g = ab.groupby("customer_id")
test("ab_inflow_cv", g.inflows.std() / g.inflows.mean()); test("ab_ilc_cv", g.income_like_credits.std() / g.income_like_credits.mean())
test("ab_share_std", g.share.std()); test("ab_share_recent3_minus", ab.groupby("customer_id").tail(3).groupby("customer_id").share.mean() - g.share.mean())
test("ab_zero_ilc_share", (ab.income_like_credits == 0).groupby(ab.customer_id).mean()); test("ab_outflow_cv", g.outflows.std() / g.outflows.mean())
test("ab_ilc_to_outflow", g.income_like_credits.sum() / g.outflows.sum().replace(0, np.nan))
pa = pd.read_csv("data/previous_applications.csv", parse_dates=["application_datetime"]).merge(cut, on="customer_id"); pa = pa[pa.application_datetime < pa.cut].sort_values(["customer_id", "application_datetime"])
pa["gap"] = pa.groupby("customer_id").application_datetime.diff().dt.days; gp = pa.groupby("customer_id").gap
test("pa_gap_std", gp.std()); test("pa_gap_min", gp.min()); test("pa_gap_last_vs_mean", gp.last() / gp.mean())
test("pa_since_last_vs_gap", (pa.groupby("customer_id").cut.first() - pa.groupby("customer_id").application_datetime.max()).dt.days / gp.mean())
ec = pd.read_csv("data/ecommerce_transactions.csv", parse_dates=["transaction_datetime"]).merge(cut, on="customer_id")
od = ec[ec.transaction_datetime < ec.cut].drop_duplicates("order_id").sort_values(["customer_id", "transaction_datetime"])
od["gap"] = od.groupby("customer_id").transaction_datetime.diff().dt.total_seconds() / 86400; go = od.groupby("customer_id").gap
test("ec_gap_cv", go.std() / go.mean()); test("ec_gap_max_vs_mean", go.max() / go.mean())
test("ec_since_last_vs_gap", (od.groupby("customer_id").cut.first() - od.groupby("customer_id").transaction_datetime.max()).dt.total_seconds() / 86400 / go.mean())
ent = lambda s: -(s.value_counts(normalize=True) * np.log(s.value_counts(normalize=True))).sum()
test("ec_dow_entropy", od.groupby("customer_id").transaction_datetime.apply(lambda s: ent(s.dt.dayofweek)))
test("ec_hour_entropy", od.groupby("customer_id").transaction_datetime.apply(lambda s: ent(s.dt.hour // 3)))
test("ec_gap_last3_vs_first3", go.apply(lambda s: s.dropna().tail(3).mean() / s.dropna().head(3).mean() if s.notna().sum() >= 6 else np.nan))
rh = pd.read_csv("data/repayment_history.csv", parse_dates=["due_date", "last_payment_datetime"]).merge(cut, on="customer_id"); rh = rh[rh.due_date < rh.cut]
rh["delay"] = (rh.last_payment_datetime - rh.due_date).dt.total_seconds() / 86400; gr = rh.groupby("customer_id")
test("rh_delay_std", gr.delay.std()); test("rh_payday_std", rh.last_payment_datetime.dt.day.groupby(rh.customer_id).std())
test("rh_on_due_day", (rh.last_payment_datetime.dt.date == rh.due_date.dt.date).groupby(rh.customer_id).mean())
test("rh_early_share", (rh.delay < -1).groupby(rh.customer_id).mean())
d = pd.DataFrame(rows, columns=["feature", "cov", "LR"]).sort_values("LR", ascending=False); print(d.to_string(index=False))
pd.DataFrame({k: v for k, v in C.items()}).to_parquet(S + "cand_reg.parquet")
