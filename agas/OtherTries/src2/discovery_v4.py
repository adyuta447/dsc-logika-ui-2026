"""Eksperimen v4: fitur kohort-relatif (double-demeaning) & agregat robust.

Motifasi: fitur `lrel` (harga satuan relatif median kategori) adalah sinyal terkuat, tetapi
mengandung drift harga bersama antar waktu. Menetralkan efek waktu dengan menghitung
selisih terhadap rata-rata populasi pada bulan & kategori yang sama membuat sinyal
risiko个人 lebih bersih. Ditambah versi median (robust terhadap order ekstrem).
"""
import sys, time
sys.path.insert(0, 'src2')
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

DATA = 'data/'
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

dt = ["application_datetime", "decision_datetime"]
train_apps = pd.read_csv(DATA+'train_applications.csv', parse_dates=dt)
test_apps = pd.read_csv(DATA+'test_applications.csv', parse_dates=dt)
profiles = pd.read_csv(DATA+'customer_profiles.csv')
apps = pd.concat([train_apps, test_apps], ignore_index=True).merge(profiles, on='customer_id', how='left').reset_index(drop=True)
ecom = pd.read_csv(DATA+'ecommerce_transactions.csv', parse_dates=["transaction_datetime"])
acct = pd.read_csv(DATA+'account_behavior.csv', parse_dates=["month"])

items = ecom[ecom.order_status == 'completed'].merge(
    apps[['customer_id', 'decision_datetime']], on='customer_id', how='inner')
items = items[items.transaction_datetime <= items.decision_datetime].copy()
items['days_ago'] = (items.decision_datetime - items.transaction_datetime).dt.total_seconds() / 86400
items['unit'] = items.item_total / items.quantity.replace(0, np.nan)
items['cat_med'] = items.unit.groupby(items.product_category).transform('median')
items['lrel'] = np.log(items.unit / items.cat_med)
items['ym'] = items.transaction_datetime.dt.to_period('M').astype(str)

# double-demeaning: lrel dikurangi rata-rata populasi pada bulan & kategori yang sama
pop_mean = items.groupby(['ym', 'product_category'])['lrel'].transform('mean')
items['lrel_dm'] = items.lrel - pop_mean
pop_median = items.groupby(['ym', 'product_category'])['lrel'].transform('median')
items['lrel_dm_med'] = items.lrel - pop_median

out = {}
idx = apps.customer_id
for w, name in [(30, 'd30'), (90, 'd90'), (365, 'd365'), (10**6, 'all')]:
    m = items.days_ago <= w
    out[f'cd_lrel_dm_{name}'] = items[m].groupby('customer_id').lrel_dm.mean().reindex(idx).to_numpy()
    out[f'cd_lrel_dmmed_{name}'] = items[m].groupby('customer_id').lrel_dm_med.mean().reindex(idx).to_numpy()
    out[f'cd_lrel_med_{name}'] = items[m].groupby('customer_id').lrel.median().reindex(idx).to_numpy()
    out[f'cd_spend_med_{name}'] = items[m].groupby('customer_id').item_total.median().reindex(idx).to_numpy()
    out[f'cd_unit_med_{name}'] = items[m].groupby('customer_id').unit.median().reindex(idx).to_numpy()
    sub = items[m].assign(_d=items.loc[m, 'transaction_datetime'].dt.normalize())
    out[f'cd_days_active_{name}'] = sub.groupby('customer_id')['_d'].nunique().reindex(idx).to_numpy()

# keterability: belanja bulanan dibanding income deklaratif & Compared to baseline
month_spend = items.assign(m=items.days_ago // 30).groupby(['customer_id', 'm']).item_total.sum()
INC = apps.set_index('customer_id')['declared_monthly_income']
ms_mean = month_spend.groupby('customer_id').mean().reindex(idx).to_numpy()
out['aff_spend_per_income_m'] = ms_mean / INC.reindex(idx).to_numpy()
last3 = month_spend.groupby(level=0).apply(lambda s: s.iloc[-3:].mean())
out['aff_spend3_per_income'] = last3.reindex(idx).to_numpy() / INC.reindex(idx).to_numpy()
out['aff_spend_ratio_3vbase'] = last3.reindex(idx).to_numpy() / pd.Series(ms_mean).replace(0, np.nan).to_numpy()

# akun: income_share dinormalisasi terhadap rata-rata populasi pada bulan yang sama
ab = acct.merge(apps[['customer_id', 'decision_datetime']], on='customer_id', how='inner')
ab = ab[ab.month < ab.decision_datetime.dt.to_period('M').dt.to_timestamp()].copy()
ab['net'] = ab.inflows - ab.outflows
ab['inc_share'] = ab.income_like_credits / ab.inflows.replace(0, np.nan)
ab['flow_ratio'] = ab.outflows / ab.inflows.replace(0, np.nan)
ab['ym'] = ab.month.dt.to_period('M').astype(str)
for c in ['inc_share', 'flow_ratio']:
    ab[c + '_dm'] = ab[c] - ab.groupby('ym')[c].transform('mean')
g = ab.groupby('customer_id')
out['cd_income_share_dm_m6'] = g.inc_share_dm.mean().reindex(idx).to_numpy()
out['cd_flow_ratio_dm_m6'] = g.flow_ratio_dm.mean().reindex(idx).to_numpy()
ab['_r3'] = ab.month >= ab.groupby('customer_id').month.transform('max') - pd.DateOffset(months=3)
out['cd_flow_ratio_dm_m3'] = ab[ab._r3].groupby('customer_id').flow_ratio_dm.mean().reindex(idx).to_numpy()
out['cd_zero_outflow_months'] = g.outflows.apply(lambda s: (s == 0).sum()).reindex(idx).to_numpy()

X4 = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)
X4.columns = ['cd__' + c for c in X4.columns]
log(f'fitur kohort: {X4.shape}')

y = train_apps.target.to_numpy()
rows = []
for c in X4.columns:
    v = X4[c].iloc[:len(y)].astype('float64')
    if v.notna().sum() < 3000: continue
    vv = v.fillna(v.median()).to_numpy()
    rows.append((c, ks_2samp(vv[y == 1], vv[y == 0]).statistic))
rank = pd.Series(dict(rows)).sort_values(ascending=False)
print('\nKS univariat fitur kohort (10 teratas):'); print(rank.head(10).round(4).to_string())

old = pd.read_pickle('artifacts2/X_v3.pkl')
pd.concat([old.reset_index(drop=True), X4], axis=1).to_pickle('artifacts2/X_v4.pkl')
X4.to_pickle('artifacts2/X_cd.pkl')
log(f'X_v4: {(old.shape[1] + X4.shape[1])} fitur')

from scipy.stats import ks_2samp