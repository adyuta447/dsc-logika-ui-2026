import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from features import D


def factor_inputs(app):
    cut = app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"})
    items = {}

    rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date"]).merge(cut, on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.days
    rh = rh[rh.days_ago >= 1].reset_index(drop=True)
    rh["late"] = (rh.max_days_past_due > 0).astype(float)
    rh["ldpd"] = np.log1p(rh.max_days_past_due)
    d = rh.due_date
    rh["f_dow"], rh["f_dom"], rh["f_ym"] = d.dt.dayofweek, d.dt.day // 8, d.dt.to_period("M").astype(str)
    rh["f_seq"] = rh.groupby("loan_id").due_date.rank().clip(upper=12)
    rh["f_term"] = rh.groupby("loan_id").due_date.transform("size").clip(upper=36)
    rh["f_amt"] = pd.qcut(np.log(rh.amount_due), 10, labels=False, duplicates="drop")
    rh["f_age"] = pd.cut(rh.days_ago, [0, 30, 90, 180, 365, 730, 5000], labels=False)
    fx = ["f_dow", "f_dom", "f_ym", "f_seq", "f_term", "f_amt", "f_age"]
    # installment level effects are removed with alternating projections next to a customer effect
    for col in ["late", "ldpd"]:
        eff = {f: np.zeros(len(rh)) for f in fx}
        for _ in range(8):
            cust = (rh[col] - sum(eff.values())).groupby(rh.customer_id).transform("mean").values
            for f in fx:
                rest = rh[col].values - cust - sum(v for k, v in eff.items() if k != f)
                eff[f] = pd.Series(rest).groupby(rh[f].values).transform("mean").values
        items[col] = pd.DataFrame({"c": rh.customer_id, "x": rh[col] - sum(eff.values())})

    ec = pd.read_csv(D + "ecommerce_transactions.csv").merge(cut, on="customer_id")
    ec = ec[ec.quantity > 0]
    u = ec.item_total / ec.quantity
    items["price"] = pd.DataFrame({"c": ec.customer_id, "x": np.log(u / u.groupby(ec.product_category).transform("median"))})

    bu = pd.read_csv(D + "bureau_observations.csv", parse_dates=["available_datetime"]).merge(cut, on="customer_id")
    bu = bu[bu.available_datetime <= bu.cut]
    g = bu.groupby("bureau_provider").score
    bu["z"] = (bu.score - g.transform("mean")) / g.transform("std")
    for c in ["z", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m"]:
        b = bu[bu[c].notna()]
        items["bu_" + c] = pd.DataFrame({"c": b.customer_id, "x": b[c]})

    ab = pd.read_csv(D + "account_behavior.csv", parse_dates=["month"]).merge(cut, on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut]
    items["ab_low"] = pd.DataFrame({"c": ab.customer_id, "x": ab.low_balance_days})
    items["ab_outin"] = pd.DataFrame({"c": ab.customer_id, "x": np.log1p(ab.outflows) - np.log1p(ab.inflows)})
    items["ab_ilc"] = pd.DataFrame({"c": ab.customer_id, "x": np.log1p(ab.income_like_credits)})

    pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["decision_datetime"]).merge(cut, on="customer_id")
    pa = pa[pa.decision_datetime < pa.cut]
    items["pa_rej"] = pd.DataFrame({"c": pa.customer_id, "x": (pa.decision == "rejected").astype(float)})
    items["rate"] = pd.DataFrame({"c": app.customer_id, "x": app.annual_interest_rate})

    # each source mean is standardized by its between customer spread, and its noise shrinks with the item count
    M, E = {}, {}
    for j, t in items.items():
        g = t.groupby("c").x
        m, k = g.mean(), g.size()
        w = ((t.x - g.transform("mean")) ** 2).sum() / max(len(t) - len(m), 1)
        b = m.var() - (w / k).mean()
        M[j] = ((m - m.mean()) / np.sqrt(b)).reindex(app.customer_id).values
        E[j] = (w / (k * b)).reindex(app.customer_id).values
    return pd.DataFrame(M), pd.DataFrame(E)


def factor_features(M, E, y, rows):
    """Loadings come from the labelled rows given, and the common scale c is picked by AUC on the same rows."""
    corr = np.array([pd.Series(M[j].values[rows]).corr(pd.Series(y[rows].astype(float))) for j in M.columns])
    best = None
    for c in np.arange(0.30, 0.95, 0.05):
        a = np.clip(corr / c, -0.97, 0.97)
        den = (1 - a ** 2) + E.values
        ok = M.notna().values
        prec = 1 + np.where(ok, a ** 2 / den, 0).sum(1)
        z = np.where(ok, a * M.values / den, 0).sum(1) / prec
        auc = roc_auc_score(y[rows], z[rows])
        if best is None or auc > best[0]:
            best = (auc, z, prec, np.where(ok, M.values * (1 - a ** 2) / den, np.nan))
    out = pd.DataFrame({"factor_z": best[1], "factor_sd": 1 / np.sqrt(best[2])})
    for i, j in enumerate(M.columns):
        if j in ("price", "late", "ldpd", "bu_z"):
            out["fs_" + j] = best[3][:, i]
    return out
