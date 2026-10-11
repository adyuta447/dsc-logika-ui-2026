import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
import sys; sys.path.insert(0, '.')
from features import load_apps, D
S = sys.argv[1]
app = load_apps()
r = pd.read_pickle(S + 'res4.pkl')
rk = lambda v: rankdata(v) / len(v)
oof = np.mean([rk(r[k][0]) for k in r], axis=0)
tr = app[app.target.notna()].reset_index(drop=True)
y = tr.target.values
resid = y - oof
C = pd.DataFrame({'application_id': app.application_id, 'customer_id': app.customer_id})
cut = app[['customer_id', 'decision_datetime']].rename(columns={'decision_datetime': 'cut'})

# application: installment vs annuity formula
m = app.annual_interest_rate / 12
ann = app.loan_amount * m / (1 - (1 + m) ** -app.term_months)
C['a_inst_ratio'] = app.proposed_monthly_installment / ann
C['a_amt_round'] = (app.loan_amount % 1_000_000 == 0).astype(int)
C['a_app_min'] = app.application_datetime.dt.minute
C['a_dec_hour'] = app.decision_datetime.dt.hour
C['a_app_dom'] = app.application_datetime.dt.day

# ecommerce
ec = pd.read_csv(D + 'ecommerce_transactions.csv', parse_dates=['transaction_datetime']).merge(cut, on='customer_id')
ec['days'] = (ec.cut - ec.transaction_datetime).dt.days
ec['hour'] = ec.transaction_datetime.dt.hour
ec['night'] = ec.hour.isin([0, 1, 2, 3, 4]).astype(int)
ec['dom'] = ec.transaction_datetime.dt.day
ec['eom'] = (ec.dom >= 25).astype(int)
ec['wkend'] = (ec.transaction_datetime.dt.dayofweek >= 5).astype(int)
q = ec[ec.quantity > 0]
ec['lq'] = np.log1p(ec.quantity) - np.log1p(ec.quantity).groupby(ec.product_category).transform('median')
unit = ec.item_total / ec.quantity.replace(0, np.nan)
ec['lrel'] = np.log(unit / unit.groupby(ec.product_category).transform('median'))
ec['lrel_hi'] = (ec.lrel > np.log(2)).astype(float).where(ec.lrel.notna())
ec['lrel_lo'] = (ec.lrel < -np.log(2)).astype(float).where(ec.lrel.notna())
ec['canc'] = (ec.order_status == 'cancelled').astype(int)
ec['zeroq'] = (ec.quantity == 0).astype(int)
ec['items_in_order'] = ec.groupby('order_id').order_item_id.transform('size')
ec['sec'] = ec.transaction_datetime.dt.second + ec.transaction_datetime.dt.minute
g = ec.groupby('customer_id')
e = pd.DataFrame({
    'e_night': g.night.mean(), 'e_hour_mean': g.hour.mean(), 'e_eom': g.eom.mean(), 'e_wkend': g.wkend.mean(),
    'e_lq': g.lq.mean(), 'e_lrel_hi': g.lrel_hi.mean(), 'e_lrel_lo': g.lrel_lo.mean(),
    'e_lrel_med': g.lrel.median(), 'e_lrel_q90': g.lrel.quantile(.9), 'e_lrel_q10': g.lrel.quantile(.1),
    'e_zeroq': g.zeroq.mean(), 'e_items_order': g.items_in_order.mean(),
    'e_lrel_canc': ec[ec.canc == 1].groupby('customer_id').lrel.mean(),
    'e_lrel_done': ec[ec.canc == 0].groupby('customer_id').lrel.mean(),
    'e_lrel_cc': ec[ec.payment_method == 'credit_card'].groupby('customer_id').lrel.mean(),
    'e_lrel_cod': ec[ec.payment_method == 'COD'].groupby('customer_id').lrel.mean(),
    'e_night_r': ec[ec.days <= 90].groupby('customer_id').night.mean(),
})
for cat in ec.product_category.unique():
    e[f'e_lq_{cat}'] = ec[ec.product_category == cat].groupby('customer_id').lq.mean()
# order-level total vs sum of item totals
od = ec.groupby('order_id').agg(customer_id=('customer_id', 'first'), ot=('order_total', 'first'), it=('item_total', 'sum'))
e['e_order_mismatch'] = (od.ot != od.it).groupby(od.customer_id).mean()
C = C.merge(e.reset_index(), on='customer_id', how='left')

# account
ab = pd.read_csv(D + 'account_behavior.csv', parse_dates=['month'])
ab['flow_gap'] = (ab.closing_balance - ab.opening_balance - ab.inflows + ab.outflows)
ab['flow_gap_rel'] = ab.flow_gap / (ab.inflows + 1)
ab['min_avg'] = ab.minimum_balance / ab.average_balance.replace(0, np.nan)
ab['nonincome'] = (ab.inflows - ab.income_like_credits) / (ab.inflows + 1)
ab = ab.sort_values(['customer_id', 'month'])
ab['open_vs_prevclose'] = ab.opening_balance - ab.groupby('customer_id').closing_balance.shift()
g = ab.groupby('customer_id')
a = pd.DataFrame({'b_flow_gap': g.flow_gap.mean(), 'b_flow_gap_abs': g.flow_gap.apply(lambda s: s.abs().mean()),
                  'b_flow_gap_rel': g.flow_gap_rel.mean(), 'b_min_avg': g.min_avg.mean(),
                  'b_nonincome': g.nonincome.mean(), 'b_open_jump': g.open_vs_prevclose.apply(lambda s: s.abs().mean()),
                  'b_inc_cv': g.income_like_credits.std() / g.income_like_credits.mean()})
C = C.merge(a.reset_index(), on='customer_id', how='left')

# repayment
rh = pd.read_csv(D + 'repayment_history.csv', parse_dates=['due_date', 'last_payment_datetime'])
rh['early'] = (rh.due_date - rh.last_payment_datetime).dt.total_seconds() / 86400
rh['pay_hour'] = rh.last_payment_datetime.dt.hour
rh['ontime_early'] = rh.early.clip(lower=0).where(rh.max_days_past_due == 0)
rh['due_dom'] = rh.due_date.dt.day
rh['partial'] = ((rh.amount_paid > 0) & (rh.amount_paid < rh.amount_due)).astype(int)
rh['over'] = (rh.amount_paid > rh.amount_due).astype(int)
g = rh.groupby('customer_id')
rr = pd.DataFrame({'r_early_mean': g.early.mean(), 'r_ontime_early': g.ontime_early.mean(), 'r_pay_hour': g.pay_hour.mean(),
                   'r_partial': g.partial.mean(), 'r_over': g.over.mean(),
                   'r_exact_due': (rh.early.abs() < 1/24).groupby(rh.customer_id).mean()})
C = C.merge(rr.reset_index(), on='customer_id', how='left')

# bureau: provider standardised scores and report lags
bu = pd.read_csv(D + 'bureau_observations.csv', parse_dates=['report_datetime', 'available_datetime'])
bu['z'] = (bu.score - bu.groupby('bureau_provider').score.transform('mean')) / bu.groupby('bureau_provider').score.transform('std')
bu['lag'] = (bu.available_datetime - bu.report_datetime).dt.days
bu = bu.merge(app[['customer_id']], on='customer_id')
g = bu.groupby('customer_id')
bb = pd.DataFrame({'u_z_mean': g.z.mean(), 'u_z_min': g.z.min(), 'u_z_std': g.z.std(), 'u_lag': g.lag.mean(), 'u_lag_max': g.lag.max(),
                   'u_rev_util_pool': g.revolving_outstanding_balance.sum() / g.revolving_credit_limit.sum().replace(0, np.nan)})
C = C.merge(bb.reset_index(), on='customer_id', how='left')

# previous applications
pa = pd.read_csv(D + 'previous_applications.csv', parse_dates=['application_datetime', 'decision_datetime'])
pa['lag'] = (pa.decision_datetime - pa.application_datetime).dt.total_seconds() / 3600
pa['appr_ratio'] = pa.approved_amount / pa.requested_amount
mm = pa.annual_interest_rate / 12
pa['inst_ratio'] = pa.scheduled_monthly_installment / (pa.approved_amount * mm / (1 - (1 + mm) ** -pa.term_months))
g = pa.groupby('customer_id')
pp = pd.DataFrame({'p_lag': g.lag.mean(), 'p_appr_ratio': g.appr_ratio.mean(), 'p_inst_ratio': g.inst_ratio.mean(),
                   'p_term_var': g.term_months.nunique()})
C = C.merge(pp.reset_index(), on='customer_id', how='left')
C.to_parquet(S + 'cand.parquet')

Ct = C.iloc[:len(tr)]
rows = []
for c in C.columns[2:]:
    v = Ct[c].values.astype(float); ok = ~np.isnan(v)
    if ok.sum() < 1000 or np.nanstd(v) == 0: continue
    auc = roc_auc_score(y[ok], v[ok]); rc = spearmanr(v[ok], resid[ok]).correlation
    rows.append((c, ok.mean().round(3), round(max(auc, 1 - auc), 4), round(rc, 4)))
print(pd.DataFrame(rows, columns=['f', 'cov', 'auc', 'resid_corr']).sort_values('resid_corr', key=abs, ascending=False).to_string())
