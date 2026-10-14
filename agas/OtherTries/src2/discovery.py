"""Eksperimen feature discovery: batch fitur perilaku baru + alternatif definisi
sinyal terkuat (harga relatif e-commerce). Hasilnya digabung ke X_v2 untuk diuji."""
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
repay = pd.read_csv(DATA+'repayment_history.csv', parse_dates=["due_date", "last_payment_datetime"])
bureau = pd.read_csv(DATA+'bureau_observations.csv', parse_dates=["report_datetime", "available_datetime"])
log('loaded')

row = pd.Series(np.arange(len(apps)), index=apps.customer_id.to_numpy())
INC = apps.set_index('customer_id')['declared_monthly_income']
out = {}

# ---------------- e-commerce: alternatif definisi harga relatif ----------------
items = ecom[ecom.order_status == 'completed'].merge(
    apps[['customer_id', 'decision_datetime']], on='customer_id', how='inner')
items = items[items.transaction_datetime <= items.decision_datetime].copy()
items['days_ago'] = (items.decision_datetime - items.transaction_datetime).dt.total_seconds()/86400
items['unit'] = items.item_total / items.quantity.replace(0, np.nan)
cat_med = items.unit.groupby(items.product_category).transform('median')
cat_mean = items.unit.groupby(items.product_category).transform('mean')
items['lrel_med'] = np.log(items.unit / cat_med)
items['lrel_mean'] = np.log(items.unit / cat_mean)
items['above_med'] = (items.unit > cat_med).astype(float)

# (1) ratio-of-means, bukan mean-of-ratios
agg = items.groupby('customer_id').unit.mean() / items.groupby('customer_id').apply(
    lambda d: np.nan, include_groups=False) if False else None
unit_by_cat_mean = items.groupby(['customer_id', 'product_category']).unit.mean().groupby('customer_id').mean()
out['px_unit_mean'] = items.groupby('customer_id').unit.mean().reindex(apps.customer_id).to_numpy()
out['px_lrel_mean_of_logs'] = items.groupby('customer_id').lrel_med.mean().reindex(apps.customer_id).to_numpy()
out['px_lrel_mean_catmean'] = items.groupby('customer_id').lrel_mean.mean().reindex(apps.customer_id).to_numpy()
out['px_frac_above_med'] = items.groupby('customer_id').above_med.mean().reindex(apps.customer_id).to_numpy()
out['px_unit_mean_vs_income'] = (out['px_unit_mean'] / INC.reindex(apps.customer_id).to_numpy())
out['px_spend_per_order_vs_catmed'] = (
    items.groupby('customer_id').item_total.mean().reindex(apps.customer_id).to_numpy()
    / items.groupby('product_category').item_total.mean().groupby(items.product_category).median()
       .reindex(items.groupby('customer_id').product_category.first().reindex(apps.customer_id).to_numpy()).to_numpy())

# (2) window pendek dari harga relatif
for lo, hi in [(0, 14), (0, 30), (14, 60), (30, 90), (90, 180)]:
    m = (items.days_ago > lo) & (items.days_ago <= hi)
    out[f'px_lrel_{lo}_{hi}'] = items[m].groupby('customer_id').lrel_med.mean().reindex(apps.customer_id).to_numpy()

# (3) frekuensi & kebiasaan belanja
items['_day'] = items.transaction_datetime.dt.normalize()
active_days = items.groupby('customer_id')['_day'].nunique()
out['ec_active_days'] = active_days.reindex(apps.customer_id).to_numpy()
out['ec_spend_per_active_day'] = (items.groupby('customer_id').item_total.sum().reindex(apps.customer_id).to_numpy()
                                  / active_days.reindex(apps.customer_id).replace(0, np.nan).to_numpy())
out['ec_distinct_products'] = items.groupby('customer_id').product_category.nunique().reindex(apps.customer_id).to_numpy()
out['ec_orders_per_month'] = (items.groupby('customer_id').order_id.nunique().reindex(apps.customer_id).to_numpy()
                              / (items.groupby('customer_id').days_ago.max().reindex(apps.customer_id).to_numpy()/30.44 + 1))
night = (items.transaction_datetime.dt.hour >= 22) | (items.transaction_datetime.dt.hour < 5)
out['ec_night_share'] = night.groupby(items.customer_id).mean().reindex(apps.customer_id).to_numpy()
out['ec_weekend_share'] = (items.transaction_datetime.dt.dayofweek >= 5).groupby(items.customer_id).mean().reindex(apps.customer_id).to_numpy()
months = items.assign(m=items.days_ago//30).groupby(['customer_id','m']).item_total.sum()
out['ec_months_with_spend'] = months.groupby('customer_id').count().reindex(apps.customer_id).to_numpy()
out['ec_month_max_share'] = (months.groupby('customer_id').max().reindex(apps.customer_id).to_numpy()
                             / months.groupby('customer_id').sum().reindex(apps.customer_id).replace(0, np.nan).to_numpy())
out['ec_month_zero_share'] = 1 - out['ec_months_with_spend'] / (items.groupby('customer_id').days_ago.max().reindex(apps.customer_id).to_numpy()/30.44 + 1)

# ---------------- repayment: peluruhan waktu & durasi tunggakan ----------------
cut = repay.customer_id.map(apps.set_index('customer_id')['decision_datetime'])
rh = repay[repay.due_date <= cut].copy()
rh['days_ago'] = (cut.reindex(rh.index) - rh.due_date).dt.total_seconds()/86400
for hl in [60, 180]:
    w = np.exp(-rh.days_ago*np.log(2)/hl)
    out[f'rh_decay_dpd{hl}'] = (w*rh.max_days_past_due).groupby(rh.customer_id).sum().reindex(apps.customer_id).to_numpy()
    out[f'rh_decay_late{hl}'] = (w*(rh.max_days_past_due>0)).groupby(rh.customer_id).sum().reindex(apps.customer_id).to_numpy()
settled = rh.amount_paid >= rh.amount_due
rh['unpaid'] = np.where(settled, 0.0, rh.amount_due.astype(float))
lo = rh.groupby(['customer_id','loan_id']).unpaid.sum()
out['rh_loans_ever_unpaid'] = (lo>0).groupby(level=0).sum().reindex(apps.customer_id).to_numpy()
out['rh_unpaid_share_loans'] = (lo>0).groupby(level=0).mean().reindex(apps.customer_id).to_numpy()
out['rh_first_due_days'] = rh.groupby('customer_id').due_date.min().pipe(lambda s: (cut.reindex(s.index) - s).dt.total_seconds()/86400).reindex(apps.customer_id).to_numpy()
out['rh_inst_per_loan'] = rh.groupby(['customer_id','loan_id']).size().groupby(level=0).mean().reindex(apps.customer_id).to_numpy()
out['rh_paid_ratio_raw'] = (rh.amount_paid/rh.amount_due.replace(0,np.nan)).groupby(rh.customer_id).mean().reindex(apps.customer_id).to_numpy()
out['rh_days_since_any_unpaid'] = (rh[~settled].groupby('customer_id').due_date.max()
                                   .pipe(lambda s: (cut.reindex(s.index)-s).dt.total_seconds()/86400)
                                   .reindex(apps.customer_id).to_numpy())

# ---------------- bureau: usia sistem kredit & konsistensi ----------------
bu = bureau.merge(apps[['customer_id','decision_datetime']], on='customer_id', how='inner')
bu = bu[bu.available_datetime <= bu.decision_datetime]
g = bu.groupby('customer_id')
out['bu_providers'] = bu.groupby('customer_id').bureau_provider.nunique().reindex(apps.customer_id).to_numpy()
out['bu_reports'] = g.size().reindex(apps.customer_id).to_numpy()
out['bu_first_report_age'] = (bu.groupby('customer_id').report_datetime.min()
                              .pipe(lambda s: (cut.reindex(s.index)-s).dt.total_seconds()/86400)
                              .reindex(apps.customer_id).to_numpy())
def _slope(d):
    dd = d[['age_days', 'score']].dropna()
    if len(dd) < 3 or dd.age_days.nunique() < 3:
        return np.nan
    return float(np.polyfit(dd.age_days.to_numpy(), dd.score.to_numpy(), 1)[0])


bu['age_days'] = (bu.decision_datetime - bu.report_datetime).dt.total_seconds() / 86400
out['bu_score_slope'] = bu.groupby('customer_id').apply(_slope, include_groups=False).reindex(apps.customer_id).to_numpy()
last3 = bu.sort_values('available_datetime').groupby('customer_id').tail(3)
out['bu_score_last3_std'] = last3.groupby('customer_id').score.std().reindex(apps.customer_id).to_numpy()
out['bu_score_last3_mean'] = last3.groupby('customer_id').score.mean().reindex(apps.customer_id).to_numpy()

X_new = pd.DataFrame(out, index=apps.customer_id.to_numpy()).reset_index(drop=True)
X_new = X_new.replace([np.inf,-np.inf], np.nan)
X_new.columns = [f'disc__{c}' for c in X_new.columns]
log(f'fitur baru: {X_new.shape}')

# uji kekuatan univariat
y = train_apps.target.to_numpy(); ntr = len(y)
rows = []
for c in X_new.columns:
    v = X_new[c].iloc[:ntr].astype('float64')
    if v.notna().sum() < 2000: continue
    vv = v.fillna(v.median()).to_numpy()
    rows.append((c, ks_2samp(vv[y==1], vv[y==0]).statistic))
rank = pd.Series(dict(rows)).sort_values(ascending=False)
print('\nKS univariat fitur baru (10 teratas):'); print(rank.head(10).round(4).to_string())
print('\nKS univariat fitur baru (10 terbawah):'); print(rank.tail(10).round(4).to_string())

X_new.to_pickle('artifacts2/X_disc.pkl')
old = pd.read_pickle('artifacts2/X_v2.pkl')
merged = pd.concat([old.reset_index(drop=True), X_new], axis=1)
merged.to_pickle('artifacts2/X_v3.pkl')
log(f'X_v3: {merged.shape}')