import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.argv = ["bag.py", "run_v4_s42", "run_v4_s7", "run_v4_s13"]
exec(open("bag.py").read().split("for v in")[0])
p = np.mean([B[r][0]["top_mix"] for r in runs], 0); lo = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
f = pd.read_parquet("features.parquet", columns=["application_id", "customer_id", "target", "pa_active_inst", "declared_monthly_income"]); t = f[f.target.notna()].reset_index(drop=True)
app = pd.read_csv("data/train_applications.csv", parse_dates=["decision_datetime"]).set_index("application_id").loc[t.application_id.astype(str)].reset_index()
allapp = pd.concat([pd.read_csv("data/train_applications.csv", parse_dates=["decision_datetime"]), pd.read_csv("data/test_applications.csv", parse_dates=["decision_datetime"])])
info = allapp.set_index("customer_id")[["decision_datetime", "proposed_monthly_installment"]]
act = f.set_index("customer_id").pa_active_inst.fillna(0)
ab = pd.read_csv("data/account_behavior.csv", parse_dates=["month"])
ab["inst"] = ab.customer_id.map(info.proposed_monthly_installment); ab["oblig"] = ab.inst + ab.customer_id.map(act).fillna(0)
ab["cut"] = ab.customer_id.map(info.decision_datetime); ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut]
ab["k"] = ab.sort_values("month").groupby("customer_id").cumcount(ascending=False)
g = ab.groupby("customer_id"); r3 = ab[ab.k < 3].groupby("customer_id")
C = {
  "min_bal_below_inst": (ab.minimum_balance < ab.inst).groupby(ab.customer_id).mean(),
  "min_bal_below_oblig": (ab.minimum_balance < ab.oblig).groupby(ab.customer_id).mean(),
  "close_below_inst": (ab.closing_balance < ab.inst).groupby(ab.customer_id).mean(),
  "net_below_inst": ((ab.inflows - ab.outflows) < ab.inst).groupby(ab.customer_id).mean(),
  "ilc_below_oblig": (ab.income_like_credits < ab.oblig).groupby(ab.customer_id).mean(),
  "surplus_after_oblig": (g.income_like_credits.mean() - g.outflows.mean() - g.oblig.mean()) / g.inst.mean(),
  "months_of_cover": g.average_balance.mean() / g.oblig.mean(),
  "min_cover_recent3": r3.minimum_balance.min() / r3.oblig.mean(),
  "net_cover_recent3": (r3.inflows.mean() - r3.outflows.mean()) / r3.oblig.mean(),
  "low_days_and_inst": g.low_balance_days.mean() * g.inst.mean() / g.income_like_credits.mean().replace(0, np.nan),
}
def ll(X):
    m = LogisticRegression(C=1e6, max_iter=3000).fit(X, y); q = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9); return (y * np.log(q) + (1 - y) * np.log(1 - q)).sum()
base = ll(lo[:, None]); rows = []
for k, s in C.items():
    v = t.customer_id.map(s).astype(float).replace([np.inf, -np.inf], np.nan)
    rr = v.rank(pct=True).fillna(0.5).values; na = v.isna().astype(float).values; ok = v.notna().values
    a = roc_auc_score(y[ok], v[ok]) if v.nunique() > 2 else np.nan
    rows.append((k, round(ok.mean(), 2), round(max(a, 1 - a), 4), round(2 * (ll(np.column_stack([lo, rr, rr ** 2, na])) - base), 1)))
print(pd.DataFrame(rows, columns=["feature", "cov", "auc_alone", "LR_over_v4"]).sort_values("LR_over_v4", ascending=False).to_string(index=False))
