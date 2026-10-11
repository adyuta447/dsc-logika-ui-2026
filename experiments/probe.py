import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D='data/'
tr=pd.read_csv(D+'train_applications.csv',parse_dates=['decision_datetime'])
ec=pd.read_csv(D+'ecommerce_transactions.csv',parse_dates=['transaction_datetime']).drop_duplicates('order_id')
ec=ec.merge(tr[['customer_id','decision_datetime','target']],on='customer_id')
ec['d']=(ec.decision_datetime-ec.transaction_datetime).dt.days
ec['spend']=ec.order_total.where(ec.order_status=='completed',0)
print(ec.d.describe())
# monthly spend profile by target
ec['m']=ec.d//30
t=ec.groupby(['m','target']).spend.sum().unstack()/tr.groupby('target').size()
print(t.head(14))
t=ec.groupby(['m','target']).size().unstack()/tr.groupby('target').size()
print(t.head(14))
ec['cc']=ec.payment_method=='credit_card'
print(ec.groupby(['m','target']).cc.mean().unstack().head(8))
ab=pd.read_csv(D+'account_behavior.csv',parse_dates=['month']).merge(tr[['customer_id','decision_datetime','target']],on='customer_id')
ab['k']=(ab.decision_datetime.dt.to_period('M')-ab.month.dt.to_period('M')).apply(lambda x:x.n)
print(ab.groupby(['k','target'])[['inflows','outflows','income_like_credits','average_balance','low_balance_days']].mean().unstack().head(12).round(0).to_string())
print('has ab by target', tr.customer_id.isin(ab.customer_id).groupby(tr.target).mean())
