import numpy as np, pandas as pd
# New ecommerce candidates from angles features.py does not cover: payment method x cancellation, payday calendar, basket shape,
# and unit prices normalised by (category, quantity) and (category, month). Writes work/ec_cand.parquet keyed by application_id.
D = "data/"
dt = ["decision_datetime"]
app = pd.concat([pd.read_csv(D + f"{s}_applications.csv", usecols=["application_id", "customer_id", "decision_datetime"], parse_dates=dt) for s in ("train", "test")])
ec = pd.read_csv(D + "ecommerce_transactions.csv", parse_dates=["transaction_datetime"]).merge(app, on="customer_id")
ec = ec[ec.transaction_datetime < ec.decision_datetime]
ec["days"] = (ec.decision_datetime - ec.transaction_datetime).dt.total_seconds() / 86400
ec["done"] = ec.order_status == "completed"
ec = ec[ec.quantity > 0].copy()
unit = np.log(ec.item_total / ec.quantity)
ec["lrel"] = unit - unit.groupby(ec.product_category).transform("median")
ec["lrel_q"] = unit - unit.groupby([ec.product_category, ec.quantity.clip(upper=4)]).transform("median")
ec["lrel_m"] = unit - unit.groupby([ec.product_category, ec.transaction_datetime.dt.to_period("M")]).transform("median")
g = ec.groupby("application_id"); out = pd.DataFrame(index=app.application_id)
for c in ("lrel_q", "lrel_m"):
    out[f"ed_{c}"] = g[c].mean(); out[f"ed_{c}_90"] = ec[ec.days <= 90].groupby("application_id")[c].mean()
q = g.lrel.quantile([0.1, 0.9]).unstack(); out["ed_lrel_p10"], out["ed_lrel_p90"] = q[0.1], q[0.9]
out["ed_lrel_iqr"] = out.ed_lrel_p90 - out.ed_lrel_p10
out["ed_premium_share"] = g.lrel.apply(lambda s: (s > np.log(1.5)).mean())
out["ed_lrel_cancel_gap"] = ec[~ec.done].groupby("application_id").lrel.mean() - ec[ec.done].groupby("application_id").lrel.mean()
for hl in (30, 90):
    w = np.exp(-ec.days * np.log(2) / hl)
    out[f"ed_lrel_decay{hl}"] = (w * ec.lrel).groupby(ec.application_id).sum() / w.groupby(ec.application_id).sum()
for m in ("COD", "credit_card", "e_wallet"):
    out[f"ed_lrel_{m}"] = ec[ec.payment_method == m].groupby("application_id").lrel.mean()

od = ec.drop_duplicates("order_id").copy()
od["cancel"] = (~od.done).astype(float)
go = od.groupby("application_id")
for m in ("COD", "credit_card", "e_wallet", "bank_transfer", "debit_card"):
    s = od[od.payment_method == m].groupby("application_id").cancel
    out[f"ed_cancel_{m}"] = s.mean(); out[f"ed_cancel_{m}_n"] = s.sum()
out["ed_cancel_cod_gap"] = out.ed_cancel_COD - od[od.payment_method != "COD"].groupby("application_id").cancel.mean()
r = od[od.days <= 90]; out["ed_cancel_cod_90"] = r[r.payment_method == "COD"].groupby("application_id").cancel.mean()
out["ed_cancel_30"] = od[od.days <= 30].groupby("application_id").cancel.mean()
out["ed_cancel_trend"] = r.groupby("application_id").cancel.mean() - od[od.days > 90].groupby("application_id").cancel.mean()
out["ed_pay_methods"] = go.payment_method.nunique()
for m in ("COD", "credit_card"):
    out[f"ed_{m}_shift"] = (r.payment_method == m).groupby(r.application_id).mean() - (od[od.days > 90].payment_method == m).groupby(od[od.days > 90].application_id).mean()

dom = od.transaction_datetime.dt.day
od["payday"] = ((dom >= 25) | (dom <= 3)).astype(float); od["midmonth"] = dom.between(10, 20).astype(float)
od["night"] = (od.transaction_datetime.dt.hour < 6).astype(float); od["weekend"] = (od.transaction_datetime.dt.dayofweek >= 5).astype(float)
done = od[od.done]; gd = done.groupby("application_id")
out["ed_payday_share"] = gd.payday.mean(); out["ed_payday_spend_share"] = (done.order_total * done.payday).groupby(done.application_id).sum() / gd.order_total.sum()
out["ed_payday_vs_mid"] = (gd.payday.mean() + 0.05) / (gd.midmonth.mean() + 0.05)
out["ed_night_share"] = go.night.mean(); out["ed_night_90"] = od[od.days <= 90].groupby("application_id").night.mean(); out["ed_weekend_share"] = go.weekend.mean()
out["ed_hour_std"] = go.transaction_datetime.apply(lambda s: s.dt.hour.std())

bk = ec.groupby("order_id").agg(application_id=("application_id", "first"), items=("order_item_id", "size"), cats=("product_category", "nunique"), qty=("quantity", "sum"), total=("order_total", "first"))
gb = bk.groupby("application_id")
out["ed_items_mean"], out["ed_cats_mean"], out["ed_qty_mean"] = gb["items"].mean(), gb.cats.mean(), gb.qty.mean()
out["ed_multi_share"] = gb["items"].apply(lambda s: (s > 1).mean())
out["ed_order_max_vs_med"] = gb.total.max() / gb.total.median()
ms = done.assign(m=done.transaction_datetime.dt.to_period("M")).groupby(["application_id", "m"]).order_total.sum().groupby(level=0)
out["ed_month_spend_cv"] = ms.std() / ms.mean()
out = out.reset_index()
assert out.application_id.is_unique and len(out) == len(app)
out.to_parquet("work/ec_cand.parquet"); print(out.shape, out.notna().mean().round(2).to_dict())
