import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from features import D


def factor_inputs(app, split=None, extra=False):
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
        items[col] = pd.DataFrame({"c": rh.customer_id, "x": rh[col] - sum(eff.values()), "d": rh.days_ago})

    ec = pd.read_csv(D + "ecommerce_transactions.csv", parse_dates=["transaction_datetime"]).merge(cut, on="customer_id")
    ec["d"] = (ec.cut - ec.transaction_datetime).dt.days
    ec = ec[ec.quantity > 0]
    u = ec.item_total / ec.quantity
    items["price"] = pd.DataFrame({"c": ec.customer_id, "x": np.log(u / u.groupby(ec.product_category).transform("median")), "d": ec.d})

    bu = pd.read_csv(D + "bureau_observations.csv", parse_dates=["available_datetime"]).merge(cut, on="customer_id")
    bu = bu[bu.available_datetime <= bu.cut]
    bu["d"] = (bu.cut - bu.available_datetime).dt.days
    g = bu.groupby("bureau_provider").score
    bu["z"] = (bu.score - g.transform("mean")) / g.transform("std")
    for c in ["z", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m"]:
        b = bu[bu[c].notna()]
        items["bu_" + c] = pd.DataFrame({"c": b.customer_id, "x": b[c], "d": b.d})

    ab = pd.read_csv(D + "account_behavior.csv", parse_dates=["month"]).merge(cut, on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut]
    ab["d"] = (ab.cut - ab.month).dt.days
    items["ab_low"] = pd.DataFrame({"c": ab.customer_id, "x": ab.low_balance_days, "d": ab.d})
    items["ab_outin"] = pd.DataFrame({"c": ab.customer_id, "x": np.log1p(ab.outflows) - np.log1p(ab.inflows), "d": ab.d})
    items["ab_ilc"] = pd.DataFrame({"c": ab.customer_id, "x": np.log1p(ab.income_like_credits), "d": ab.d})

    pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["decision_datetime"]).merge(cut, on="customer_id")
    pa = pa[pa.decision_datetime < pa.cut]
    pa["d"] = (pa.cut - pa.decision_datetime).dt.days
    items["pa_rej"] = pd.DataFrame({"c": pa.customer_id, "x": (pa.decision == "rejected").astype(float), "d": pa.d})
    items["rate"] = pd.DataFrame({"c": app.customer_id, "x": app.annual_interest_rate})

    if extra:
        items["ec_cancel"] = pd.DataFrame({"c": ec.customer_id, "x": (ec.order_status == "cancelled").astype(float), "d": ec.d})
        items["ec_cod"] = pd.DataFrame({"c": ec.customer_id, "x": (ec.payment_method == "COD").astype(float), "d": ec.d})
        items["bu_out"] = pd.DataFrame({"c": bu.customer_id, "x": np.log1p(bu.outstanding_balance), "d": bu.d}).dropna()
        items["ab_minavg"] = pd.DataFrame({"c": ab.customer_id, "x": (ab.minimum_balance / ab.average_balance.where(ab.average_balance > 0)).clip(-1, 1), "d": ab.d}).dropna()
        rr = rh.amount_paid / rh.amount_due
        items["rh_paid"] = pd.DataFrame({"c": rh.customer_id, "x": rr.clip(0, 1.5), "d": rh.days_ago})
        items["pa_req"] = pd.DataFrame({"c": pa.customer_id, "x": np.log(pa.requested_amount), "d": pa.d})
    if split:
        for j in [j for j, t in items.items() if "d" in t]:
            t = items.pop(j)
            items[j + "_r"], items[j + "_o"] = t[t.d <= split], t[t.d > split]
    # each source mean is standardized by its between customer spread, and its noise shrinks with the item count
    M, E = {}, {}
    for j, t in items.items():
        g = t.groupby("c").x
        m, k = g.mean(), g.size()
        w = ((t.x - g.transform("mean")) ** 2).sum() / max(len(t) - len(m), 1)
        b = m.var() - (w / k).mean()
        if b <= 0:
            continue
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
        if j.removesuffix("_r").removesuffix("_o") in ("price", "late", "ldpd", "bu_z"):
            out["fs_" + j] = best[3][:, i]
    return out
