import sys, time
sys.path.insert(0, 'src2')
import numpy as np, pandas as pd
import features_v2 as F

DATA = 'data/'
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

dt = ["application_datetime", "decision_datetime"]
train_apps = pd.read_csv(DATA+'train_applications.csv', parse_dates=dt)
test_apps = pd.read_csv(DATA+'test_applications.csv', parse_dates=dt)
profiles = pd.read_csv(DATA+'customer_profiles.csv')
prev = pd.read_csv(DATA+'previous_applications.csv', parse_dates=dt)
repay = pd.read_csv(DATA+'repayment_history.csv', parse_dates=["due_date", "last_payment_datetime"])
acct = pd.read_csv(DATA+'account_behavior.csv', parse_dates=["month"])
bureau = pd.read_csv(DATA+'bureau_observations.csv', parse_dates=["report_datetime", "available_datetime"])
ecom = pd.read_csv(DATA+'ecommerce_transactions.csv', parse_dates=["transaction_datetime"])
apps = pd.concat([train_apps, test_apps], ignore_index=True).merge(profiles, on='customer_id', how='left')
apps = apps.reset_index(drop=True)
log('loaded')

blocks = {}
blocks['app'] = F.app_features(apps.drop(columns=['age','region','employment_type','employment_tenure_months','declared_monthly_income','dependents','housing_status']), profiles); log('app %s' % (blocks['app'].shape,))
blocks['bu'] = F.bureau_features(apps, bureau); log('bu %s' % (blocks['bu'].shape,))
blocks['pa'] = F.previous_features(apps, prev); log('pa %s' % (blocks['pa'].shape,))
blocks['rh'] = F.repayment_features(apps, repay); log('rh %s' % (blocks['rh'].shape,))
blocks['ab'] = F.account_features(apps, acct); log('ab %s' % (blocks['ab'].shape,))
blocks['ec'] = F.ecommerce_features(apps, ecom); log('ec %s' % (blocks['ec'].shape,))

X = pd.concat(list(blocks.values()), axis=1)
X.columns = [f'{k}__{c}' for k, v in blocks.items() for c in v.columns]
X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
X = pd.concat([X, F.cross_features(apps, X)], axis=1)
X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
assert not any(str(d).startswith('datetime') for d in X.dtypes)
log('X %s nan %.3f' % (X.shape, X.isna().mean().mean()))
X.to_pickle('artifacts2/X_v2.pkl')
apps[['application_id','customer_id','decision_datetime','target']].to_pickle('artifacts2/apps_meta.pkl')
log('saved')