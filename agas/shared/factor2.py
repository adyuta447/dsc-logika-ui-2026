"""Extended factor model.

`features.factor_inputs` builds a shrinkage factor score from twelve noisy
sources. This module reuses that machinery verbatim and appends six sources
built from the new signal blocks: income regularity, disposable cash flow,
installment affordability stress, decayed lateness, absolute basket price
deviation, and discretionary share of spend.

Every source is a (customer_id, value) long table aggregated with the same
between/within decomposition, so the combination step and the per-fold loading
fit stay identical to the original factor model. No label is used here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# `features.py` lives at the repository root and this module is imported by
# version scripts that only guarantee `agas/shared` on the path, so the root is
# added here rather than at every call site.
_ROOT = None
for _p in Path(__file__).resolve().parents:
    if (_p / "features.py").exists():
        _ROOT = _p
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
if _ROOT is None:
    raise RuntimeError("features.py not found above shared/factor2.py")

import features as ft  # noqa: E402


def _new_items(app):
    cut = app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"})
    items = {}

    ab = pd.read_csv(ft.D + "account_behavior.csv", parse_dates=["month"]).merge(cut, on="customer_id")
    ab = ab[ab.month + pd.offsets.MonthBegin(1) <= ab.cut]
    inst = app.set_index("customer_id").proposed_monthly_installment
    ab["inst"] = ab.customer_id.map(inst).fillna(0)
    ab = ab.sort_values(["customer_id", "month"])
    ab["stress"] = (ab.minimum_balance < ab.inst).astype(float)
    ab["cash_gap"] = np.log1p(ab.income_like_credits) - np.log1p(ab.inst.clip(lower=0))
    items["ab_ilc_log"] = pd.DataFrame({"c": ab.customer_id, "x": -np.log1p(ab.income_like_credits)})
    items["ab_disposable"] = pd.DataFrame({"c": ab.customer_id, "x": ab.inflows - ab.outflows})
    items["ab_stress"] = pd.DataFrame({"c": ab.customer_id, "x": ab.stress})
    items["ab_cashgap"] = pd.DataFrame({"c": ab.customer_id, "x": ab.cash_gap})

    rh = pd.read_csv(ft.D + "repayment_history.csv", parse_dates=["due_date"]).merge(cut, on="customer_id")
    rh["days_ago"] = (rh.cut - rh.due_date).dt.days
    rh = rh[rh.days_ago >= 1]
    w = np.exp(-rh.days_ago.clip(lower=0) * np.log(2) / 180)
    rh["wlate"] = (w * (rh.max_days_past_due > 0)).astype(float)
    items["rx_wlate"] = pd.DataFrame({"c": rh.customer_id, "x": rh.wlate})

    ec = pd.read_csv(ft.D + "ecommerce_transactions.csv", usecols=[
        "customer_id", "product_category", "quantity", "item_total", "order_status"])
    ec = ec[(ec.order_status == "completed") & (ec.quantity > 0)]
    u = ec.item_total / ec.quantity
    rel = u / u.groupby(ec.product_category).transform("median")
    items["ec_absrel"] = pd.DataFrame({"c": ec.customer_id, "x": np.log(rel).abs()})
    disc = ec.product_category.isin(["electronics", "home", "personal_care", "fashion"]).astype(float)
    items["ec_disc"] = pd.DataFrame({"c": ec.customer_id, "x": disc})
    return items


def factor_inputs_v2(app):
    M, E = ft.factor_inputs(app)
    items = _new_items(app)
    for j, t in items.items():
        g = t.groupby("c").x
        m, k = g.mean(), g.size()
        w = ((t.x - g.transform("mean")) ** 2).sum() / max(len(t) - len(m), 1)
        b = m.var() - (w / k).mean()
        if not np.isfinite(b) or b <= 0:
            b = m.var() if m.var() > 0 else 1.0
        M[j] = ((m - m.mean()) / np.sqrt(b)).reindex(app.customer_id).values
        E[j] = (w / (k * b)).reindex(app.customer_id).values
    return M, E


def factor_features_v2(M, E, y, rows, keep):
    fac = ft.factor_features(M, E, y, rows)
    for c in fac.columns:
        if c.startswith("fs_") and c[3:] not in keep:
            fac = fac.drop(columns=[c])
    return fac
