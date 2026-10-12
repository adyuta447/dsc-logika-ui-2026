import numpy as np, pandas as pd
from features import D


def extra_features(app, M=None):
    cut = app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"})
    out = pd.DataFrame(index=pd.Index(app.customer_id, name="customer_id"))
    ab = pd.read_csv(D + "account_behavior.csv", parse_dates=["month"]).merge(cut, on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut]
    g = ab.groupby("customer_id")
    out["ab_inc_share"] = g.income_like_credits.sum() / g.inflows.sum().replace(0, np.nan)
    out["ab_open_avg"] = (g.opening_balance.mean() / g.average_balance.mean().replace(0, np.nan))
    pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["application_datetime"]).merge(cut, on="customer_id")
    pa = pa[pa.application_datetime < pa.cut].sort_values(["customer_id", "application_datetime"])
    pa["gap"] = pa.groupby("customer_id").application_datetime.diff().dt.total_seconds() / 86400
    out["pa_gap_mean"] = pa.groupby("customer_id").gap.mean()
    ec = pd.read_csv(D + "ecommerce_transactions.csv", parse_dates=["transaction_datetime"]).merge(cut, on="customer_id")
    od = ec[ec.transaction_datetime < ec.cut].drop_duplicates("order_id").sort_values(["customer_id", "transaction_datetime"])
    od["gap"] = od.groupby("customer_id").transaction_datetime.diff().dt.total_seconds() / 86400
    go = od.groupby("customer_id").gap
    out["ec_gap_mean"] = go.mean()
    out["ec_gap_last3"] = go.apply(lambda s: s.tail(3).mean()) / out.ec_gap_mean.reindex(go.mean().index)
    if M is not None:
        # unlabelled one factor fit (principal axis), a second view of the shared risk next to the label based factor
        R = M.cov().values; a = np.full(len(R), .5)
        for _ in range(200):
            R2 = R.copy(); np.fill_diagonal(R2, a ** 2); val, vec = np.linalg.eigh(R2); a = vec[:, -1] * np.sqrt(val[-1])
        a = np.clip(a * np.sign(a[list(M.columns).index("rate")]), -.97, .97)
        out["factor_unsup"] = np.nansum(np.where(M.notna(), a * M.values / (1 - a ** 2), 0), 1) / (1 + np.nansum(np.where(M.notna(), a ** 2 / (1 - a ** 2), 0), 1))
    return out.reset_index(drop=True)
