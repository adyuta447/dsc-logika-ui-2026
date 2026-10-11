import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D='data/'
tr=pd.read_csv(D+'train_applications.csv',parse_dates=['decision_datetime'])
t=tr[['customer_id','decision_datetime','target']]
def scan(df,tcol,cols,name,split=90):
    df=df.merge(t,on='customer_id'); df['d']=(df.decision_datetime-df[tcol]).dt.days
    rec=df[df.d<=split].groupby('customer_id')[cols].mean(); old=df[df.d>split].groupby('customer_id')[cols].mean()
    allm=df.groupby('customer_id')[cols].mean(); y=t.set_index('customer_id').target
    for c in cols:
        out=[]
        for lab,s in [('all',allm[c]),('rec',rec[c]),('delta',rec[c]-old[c]),('ratio',rec[c]/old[c].replace(0,np.nan))]:
            s=s.replace([np.inf,-np.inf],np.nan).dropna(); yy=y.loc[s.index]
            out.append(f"{lab}={abs(roc_auc_score(yy,s)-.5)+.5:.3f}")
        print(name,c,' '.join(out))
ab=pd.read_csv(D+'account_behavior.csv',parse_dates=['month']); ab['net']=ab.inflows-ab.outflows; ab['oi']=ab.outflows/ab.inflows; ab['ilc_share']=ab.income_like_credits/ab.inflows; ab['nonincome_in']=ab.inflows-ab.income_like_credits
ab['bal_range']=ab.average_balance-ab.minimum_balance; ab['bal_chg']=ab.closing_balance-ab.opening_balance
#scan(ab,'month',['inflows','outflows','income_like_credits','average_balance','minimum_balance','low_balance_days','net','oi','ilc_share','nonincome_in','bal_range','bal_chg'],'ab',100)
rh=pd.read_csv(D+'repayment_history.csv',parse_dates=['due_date','last_payment_datetime'])
rh['delay']=(rh.last_payment_datetime-rh.due_date).dt.total_seconds()/86400; rh['ph']=rh.last_payment_datetime.dt.hour; rh['pr']=rh.amount_paid/rh.amount_due
rh['early']=(rh.delay<=0).astype(float)
#scan(rh,'due_date',['max_days_past_due','delay','ph','pr','early','amount_due'],'rh',180)
bu=pd.read_csv(D+'bureau_observations.csv',parse_dates=['report_datetime'])
#scan(bu,'report_datetime',['score','utilization','inquiries_last_6m','delinquent_accounts_last_12m','outstanding_balance','active_accounts','revolving_credit_limit'],'bu',180)
ec=pd.read_csv(D+'ecommerce_transactions.csv',parse_dates=['transaction_datetime'])
ec['unit']=ec.item_total/ec.quantity.replace(0,np.nan)
ec['lrel']=np.log(ec.unit/ec.groupby('product_category').unit.transform('median'))
ec['lrel_c']=np.log(ec.unit/ec.groupby(['customer_id','product_category']).unit.transform('median'))
ec['n_items']=ec.groupby('order_id').order_id.transform('size')
ec['cancel']=(ec.order_status=='cancelled').astype(float)
ec['cod']=(ec.payment_method=='COD').astype(float)
ec['hour']=ec.transaction_datetime.dt.hour
ec['night']=((ec.hour<6)|(ec.hour>=22)).astype(float)
ec['wkend']=(ec.transaction_datetime.dt.dayofweek>=5).astype(float)
ec['big']=(ec.lrel>np.log(2)).astype(float)
scan(ec,'transaction_datetime',['lrel','lrel_c','quantity','n_items','order_total','cancel','cod','night','wkend','big'],'ec',90)
