"""Build leak-free monthly sequence tensors for v72.

The existing v66--v71 models compress every source into static aggregates.  This
script keeps the last 24 calendar months as a sequence: account cash flow,
repayment behaviour, e-commerce activity, bureau snapshots, and application
velocity.  Every event is aligned to the selected application's decision month
and only events at or before the decision are used.

Output is a compact npz file with one array per source plus a per-source mask.
No target value is used while building sequences.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = HERE.parent / "output"
DATA = ROOT / "data"
L = 24


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def month_index(s):
    return s.dt.year * 12 + s.dt.month


def app_frames():
    dt = ["application_datetime", "decision_datetime"]
    tr = pd.read_csv(DATA / "train_applications.csv", parse_dates=dt)
    te = pd.read_csv(DATA / "test_applications.csv", parse_dates=dt)
    te["target"] = np.nan
    app = pd.concat([tr, te], ignore_index=True)
    prof = pd.read_csv(DATA / "customer_profiles.csv",
                       usecols=["customer_id", "declared_monthly_income"])
    app = app.merge(prof, on="customer_id", how="left")
    app["cut_month"] = month_index(app.decision_datetime)
    return app


def empty(n, k):
    return np.zeros((n, L, k), dtype=np.float32)


def assign(arr, rows, t, vals):
    """Assign customer rows at integer time index t; vals is [m, k]."""
    ok = (t >= 0) & (t < L) & (rows >= 0)
    arr[rows[ok], t[ok], :] = vals[ok].astype(np.float32)


def channel(name, n, cols, rows, t, vals):
    a = empty(n, len(cols))
    assign(a, rows, t, vals)
    m = empty(n, 1)
    assign(m, rows, t, np.ones((len(rows), 1), dtype=np.float32))
    return a, m


def clean(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -20.0, 20.0).astype(np.float32)


def build_account(app, cid_pos):
    ab = pd.read_csv(DATA / "account_behavior.csv", parse_dates=["month"])
    ab = ab.merge(app[["customer_id", "cut_month", "declared_monthly_income"]], on="customer_id", how="left")
    ab["moff"] = ab.cut_month - month_index(ab.month)
    ab = ab[(ab.moff >= 0) & (ab.moff < L)].copy()
    ab["t"] = L - 1 - ab.moff
    inc = ab.declared_monthly_income.replace(0, np.nan)
    vals = np.column_stack([
        np.log1p(ab.inflows.clip(lower=0)),
        np.log1p(ab.outflows.clip(lower=0)),
        np.log1p(ab.income_like_credits.clip(lower=0)),
        (ab.inflows - ab.outflows) / inc,
        np.log1p(ab.average_balance.clip(lower=0)),
        np.log1p(ab.minimum_balance.clip(lower=0)),
        np.log1p(ab.closing_balance.clip(lower=0)),
        ab.low_balance_days / 30.0,
        ab.outflows / (ab.inflows + 1.0),
        ab.income_like_credits / (ab.inflows + 1.0),
    ])
    rows = cid_pos.reindex(ab.customer_id).to_numpy()
    return channel("acct", len(app), ["acct_in", "acct_out", "acct_inc", "acct_net_inc",
                                      "acct_bal", "acct_min", "acct_close", "acct_low",
                                      "acct_out_in", "acct_inc_share"],
                   rows, ab.t.to_numpy(), clean(vals))


def build_repay(app, cid_pos):
    rh = pd.read_csv(DATA / "repayment_history.csv", parse_dates=["due_date"])
    rh = rh.merge(app[["customer_id", "cut_month", "declared_monthly_income"]], on="customer_id", how="left")
    rh["due_month"] = month_index(rh.due_date)
    rh["moff"] = rh.cut_month - rh.due_month
    rh = rh[(rh.moff >= 0) & (rh.moff < L)].copy()
    rh["t"] = L - 1 - rh.moff
    rh["late"] = (rh.max_days_past_due > 0).astype(float)
    rh["late30"] = (rh.max_days_past_due >= 30).astype(float)
    rh["late90"] = (rh.max_days_past_due >= 90).astype(float)
    rh["open"] = (rh.days_past_due > 0).astype(float)
    rh["unpaid"] = (rh.amount_due - rh.amount_paid).clip(lower=0)
    rh["payratio"] = rh.amount_paid / rh.amount_due.replace(0, np.nan)
    g = rh.groupby(["customer_id", "t"]).agg(
        due=("amount_due", "sum"), paid=("amount_paid", "sum"), unpaid=("unpaid", "sum"),
        n=("amount_due", "size"), loans=("loan_id", "nunique"),
        late=("late", "mean"), late30=("late30", "mean"), late90=("late90", "mean"),
        dpd=("max_days_past_due", "max"), dpd_now=("days_past_due", "max"),
        open=("open", "mean"), payratio=("payratio", "mean"),
        inc=("declared_monthly_income", "first"),
    ).reset_index()
    vals = np.column_stack([
        np.log1p(g.due.clip(lower=0)), np.log1p(g.paid.clip(lower=0)),
        np.log1p(g.unpaid), g.n / 5.0, g.loans / 3.0,
        g.late, g.late30, g.late90, g.dpd / 90.0, g.dpd_now / 90.0,
        g.open, g.payratio, g.due / g.inc.replace(0, np.nan),
    ])
    rows = cid_pos.reindex(g.customer_id).to_numpy()
    return channel("repay", len(app), ["rep_due", "rep_paid", "rep_unpaid", "rep_n",
                                      "rep_loans", "rep_late", "rep_late30", "rep_late90",
                                      "rep_dpd", "rep_dpd_now", "rep_open", "rep_payratio",
                                      "rep_due_inc"], rows, g.t.to_numpy(), clean(vals))


def build_ecom(app, cid_pos):
    ec = pd.read_csv(DATA / "ecommerce_transactions.csv", parse_dates=["transaction_datetime"])
    ec = ec.merge(app[["customer_id", "cut_month", "declared_monthly_income"]], on="customer_id", how="left")
    ec["moff"] = ec.cut_month - month_index(ec.transaction_datetime)
    ec = ec[(ec.moff >= 0) & (ec.moff < L)].copy()
    ec["t"] = L - 1 - ec.moff
    ec["done"] = ec.order_status == "completed"
    ec["spend"] = ec.item_total.where(ec.done, 0.0)

    # order-level panel: deduplicate repeated order_total
    od = ec.sort_values("transaction_datetime").drop_duplicates("order_id").copy()
    od["spend"] = od.order_total.where(od.order_status == "completed", 0.0)
    od["cancel"] = (od.order_status != "completed").astype(float)
    go = od.groupby(["customer_id", "t"]).agg(
        spend=("spend", "sum"), orders=("order_id", "size"), cancel=("cancel", "mean"),
        max_order=("spend", "max"), inc=("declared_monthly_income", "first"),
    ).reset_index()

    # item-level category mix and unit-price relative to category median
    it = ec[ec.done & (ec.quantity > 0)].copy()
    it["unit"] = it.item_total / it.quantity
    it["lrel"] = np.log(it.unit / it.groupby("product_category").unit.transform("median"))
    it["need"] = it.product_category.isin(
        ["groceries", "home", "health", "education", "personal_care"]).astype(float)
    gi = it.groupby(["customer_id", "t"]).agg(
        items=("quantity", "sum"), lrel=("lrel", "mean"), need=("need", "mean"),
    ).reset_index()
    cat = it.pivot_table(index=["customer_id", "t"], columns="product_category",
                         values="item_total", aggfunc="sum", fill_value=0.0)
    share = cat.div(cat.sum(axis=1), axis=0)
    ent = -(share * np.log(share.replace(0, np.nan))).sum(axis=1).rename("entropy")
    g = go.merge(gi, on=["customer_id", "t"], how="left").merge(
        pd.concat([share.add_prefix("share_"), ent], axis=1).reset_index(),
        on=["customer_id", "t"], how="left")
    for c in [c for c in g.columns if c.startswith("share_")]:
        g[c] = g[c].fillna(0.0)
    g = g.fillna({"items": 0.0, "lrel": 0.0, "need": 0.0, "entropy": 0.0})
    vals = np.column_stack([
        np.log1p(g.spend.clip(lower=0)), np.log1p(g.orders),
        g.cancel, np.log1p(g.max_order.clip(lower=0)),
        np.log1p(g["items"]), g.lrel, g.need, g.entropy,
        g.spend / g.inc.replace(0, np.nan),
    ] + [g[c].to_numpy() for c in g.columns if c.startswith("share_")])
    cols = ["ec_spend", "ec_orders", "ec_cancel", "ec_max_order", "ec_items",
            "ec_lrel", "ec_need", "ec_entropy", "ec_spend_inc"] + [
        "ec_" + c.replace("share_", "share_") for c in g.columns if c.startswith("share_")]
    rows = cid_pos.reindex(g.customer_id).to_numpy()
    return channel("ecom", len(app), cols, rows, g.t.to_numpy(), clean(vals))


def build_bureau(app, cid_pos):
    bu = pd.read_csv(DATA / "bureau_observations.csv",
                     parse_dates=["report_datetime", "available_datetime"])
    bu = bu.merge(app[["customer_id", "cut_month", "decision_datetime"]], on="customer_id", how="left")
    # available_datetime is the point-in-time guard; report_datetime is used for month alignment
    bu = bu[(bu.available_datetime <= bu.decision_datetime) & bu.report_datetime.notna()].copy()
    bu["moff"] = bu.cut_month - month_index(bu.report_datetime)
    bu = bu[(bu.moff >= 0) & (bu.moff < L)].copy()
    bu["t"] = L - 1 - bu.moff
    bu = bu.sort_values(["customer_id", "t", "report_datetime", "bureau_provider"])
    latest = bu.groupby(["customer_id", "t"]).tail(1)
    vals = np.column_stack([
        (latest.score - 300.0) / 550.0,
        latest.score.isna().astype(float),
        latest.utilization.fillna(0.0).clip(0, 5),
        latest.inquiries_last_6m / 10.0,
        latest.delinquent_accounts_last_12m / 5.0,
        latest.active_accounts / 10.0,
        np.log1p(latest.outstanding_balance.clip(lower=0)),
        np.log1p(latest.revolving_outstanding_balance.clip(lower=0)),
    ])
    rows = cid_pos.reindex(latest.customer_id).to_numpy()
    return channel("bureau", len(app), ["bu_score", "bu_missing", "bu_util", "bu_inq",
                                        "bu_delinq", "bu_active", "bu_out", "bu_rev"],
                   rows, latest.t.to_numpy(), clean(vals))


def build_prev(app, cid_pos):
    pa = pd.read_csv(DATA / "previous_applications.csv",
                     parse_dates=["application_datetime", "decision_datetime"])
    pa = pa.merge(app[["customer_id", "cut_month", "decision_datetime"]].rename(
        columns={"decision_datetime": "cut"}), on="customer_id", how="left")
    pa = pa[pa.decision_datetime < pa.cut].copy()
    pa["moff"] = pa.cut_month - month_index(pa.decision_datetime)
    pa = pa[(pa.moff >= 0) & (pa.moff < L)].copy()
    pa["t"] = L - 1 - pa.moff
    pa["approved"] = (pa.decision == "approved").astype(float)
    pa["rejected"] = (pa.decision == "rejected").astype(float)
    pa["cancelled"] = (pa.decision == "cancelled").astype(float)
    pa["lag"] = (pa.decision_datetime - pa.application_datetime).dt.total_seconds() / 86400
    g = pa.groupby(["customer_id", "t"]).agg(
        n=("decision", "size"), approved=("approved", "sum"),
        rejected=("rejected", "sum"), cancelled=("cancelled", "sum"),
        req=("requested_amount", "sum"), appr=("approved_amount", "sum"),
        rate=("annual_interest_rate", "mean"), inst=("scheduled_monthly_installment", "sum"),
        lag=("lag", "mean"),
    ).reset_index()
    vals = np.column_stack([
        g.n / 3.0, g.approved, g.rejected, g.cancelled,
        np.log1p(g.req.clip(lower=0)), np.log1p(g.appr.clip(lower=0)),
        g.rate, np.log1p(g.inst.clip(lower=0)), g.lag / 30.0,
    ])
    rows = cid_pos.reindex(g.customer_id).to_numpy()
    return channel("prev", len(app), ["prev_n", "prev_appr", "prev_rej", "prev_can",
                                      "prev_req", "prev_amt", "prev_rate", "prev_inst",
                                      "prev_lag"], rows, g.t.to_numpy(), clean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "sequences_v72.npz"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    app = app_frames()
    cid_pos = pd.Series(np.arange(len(app)), index=app.customer_id)
    log(f"apps={len(app)} train={app.target.notna().sum()} test={app.target.isna().sum()}")

    blocks = {}
    for fn in [build_account, build_repay, build_ecom, build_bureau, build_prev]:
        a, m = fn(app, cid_pos)
        name = fn.__name__.replace("build_", "")
        blocks[name] = a
        blocks[name + "_mask"] = m
        log(f"  {name}: {a.shape} nonmissing={float(m.mean()):.3f}")

    X = np.concatenate([blocks[k] for k in ["account", "repay", "ecom", "bureau", "prev"]], axis=2)
    M = np.concatenate([blocks[k + "_mask"] for k in ["account", "repay", "ecom", "bureau", "prev"]], axis=2)
    channels = (["acct_in", "acct_out", "acct_inc", "acct_net_inc", "acct_bal", "acct_min",
                 "acct_close", "acct_low", "acct_out_in", "acct_inc_share"]
                + ["rep_due", "rep_paid", "rep_unpaid", "rep_n", "rep_loans", "rep_late",
                   "rep_late30", "rep_late90", "rep_dpd", "rep_dpd_now", "rep_open",
                   "rep_payratio", "rep_due_inc"]
                + ["ec_spend", "ec_orders", "ec_cancel", "ec_max_order", "ec_items",
                   "ec_lrel", "ec_need", "ec_entropy", "ec_spend_inc", "ec_share_education",
                   "ec_share_electronics", "ec_share_fashion", "ec_share_groceries",
                   "ec_share_health", "ec_share_home", "ec_share_personal_care"]
                + ["bu_score", "bu_missing", "bu_util", "bu_inq", "bu_delinq", "bu_active",
                   "bu_out", "bu_rev"]
                + ["prev_n", "prev_appr", "prev_rej", "prev_can", "prev_req", "prev_amt",
                   "prev_rate", "prev_inst", "prev_lag"])
    assert X.shape[2] == len(channels), (X.shape, len(channels))
    y = app.target.to_numpy(np.float32)
    np.savez_compressed(args.out,
                        application_id=app.application_id.values,
                        customer_id=app.customer_id.values,
                        y=y, X=X.astype(np.float32), M=M.astype(np.float32),
                        channels=np.array(channels), L=np.array([L]))
    summary = {"apps": len(app), "train": int(app.target.notna().sum()),
               "test": int(app.target.isna().sum()), "L": L,
               "channels": channels, "X_shape": list(X.shape),
               "mask_shape": list(M.shape), "out": str(args.out),
               "minutes": (time.time() - t0) / 60.0}
    (OUT / "summary_build_sequences_v72.json").write_text(json.dumps(summary, indent=2))
    log(f"saved {args.out} X={X.shape} M={M.shape} in {summary['minutes']:.2f} min")


if __name__ == "__main__":
    main()
