import numpy as np, pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score
from features import load_apps, D
app = load_apps(); cut = app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"})
yv = app.set_index("customer_id").target
src = {}
ec = pd.read_csv(D + "ecommerce_transactions.csv").merge(cut, on="customer_id"); ec = ec[ec.quantity > 0]
u = ec.item_total / ec.quantity; src["price"] = pd.DataFrame({"c": ec.customer_id, "x": np.log(u / u.groupby(ec.product_category).transform("median"))})
bu = pd.read_csv(D + "bureau_observations.csv", parse_dates=["available_datetime"]).merge(cut, on="customer_id"); bu = bu[bu.available_datetime <= bu.cut]
g = bu.groupby("bureau_provider").score; bu["z"] = (bu.score - g.transform("mean")) / g.transform("std")
src["bu_z"] = bu[["customer_id", "z"]].dropna().set_axis(["c", "x"], axis=1)
src["bu_util"] = bu[["customer_id", "utilization"]].dropna().set_axis(["c", "x"], axis=1)
ab = pd.read_csv(D + "account_behavior.csv", parse_dates=["month"]).merge(cut, on="customer_id"); ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut]
src["ab_low"] = pd.DataFrame({"c": ab.customer_id, "x": ab.low_balance_days})
src["ab_outin"] = pd.DataFrame({"c": ab.customer_id, "x": np.log1p(ab.outflows) - np.log1p(ab.inflows)})
src["ab_ilc"] = pd.DataFrame({"c": ab.customer_id, "x": np.log1p(ab.income_like_credits)})
rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date"]).merge(cut, on="customer_id"); rh = rh[(rh.cut - rh.due_date).dt.days >= 1]
src["ldpd"] = pd.DataFrame({"c": rh.customer_id, "x": np.log1p(rh.max_days_past_due)})
tr = app[app.target.notna()]
for j, t in src.items():
    g = t.groupby("c").x; n = g.transform("size"); r = (t.x - g.transform("mean"))[n >= 8]
    r = r / r.std()
    est = {"mean": g.mean(), "median": g.median(), "trim20": g.apply(lambda s: s.sort_values().iloc[int(len(s)*.2):max(int(len(s)*.8), int(len(s)*.2)+1)].mean()),
           "midrange": (g.max() + g.min()) / 2}
    aucs = {}
    for k, e in est.items():
        v = tr.customer_id.map(e); ok = v.notna().values
        a = roc_auc_score(tr.target.values[ok], v.values[ok]); aucs[k] = round(max(a, 1 - a), 4)
    print(f"{j:9s} skew {skew(r):+.2f} excess kurtosis {kurtosis(r):+.2f}  AUC by estimator {aucs}", flush=True)
