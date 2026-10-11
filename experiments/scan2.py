import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from features import load_apps, D
S=sys.argv[1]; app=load_apps(); n=int(app.target.notna().sum()); y=app.target.values[:n].astype(int)
r=pd.read_pickle(S+'res4.pkl'); p=np.mean([r[k][0] for k in ['lgb0','lgb1','lgb2','xgb']],0).clip(1e-4,1-1e-4); lo=np.log(p/(1-p))
def ll(X):
    m=LogisticRegression(C=1e6,max_iter=500).fit(X,y); q=m.predict_proba(X)[:,1].clip(1e-9,1-1e-9); return (y*np.log(q)+(1-y)*np.log(1-q)).sum()
base=ll(lo[:,None]); out={}; res=[]
def test(name,s):
    v=app.customer_id.map(s).astype(float).replace([np.inf,-np.inf],np.nan); out[name]=v.values
    t=v[:n]
    if t.notna().mean()<0.3 or t.std()==0: return
    t=t.rank(pct=True); na=t.isna().astype(float); t=t.fillna(0.5)
    res.append((name,round(t.notna().mean(),2),round(2*(ll(np.column_stack([lo,t,na]))-base),1)))
cut=app[['customer_id','decision_datetime']].rename(columns={'decision_datetime':'cut'})
ec=pd.read_csv(D+'ecommerce_transactions.csv',parse_dates=['transaction_datetime']).merge(cut,on='customer_id')
od=ec.drop_duplicates('order_id').sort_values(['customer_id','transaction_datetime']).copy()
od['days']=(od.cut-od.transaction_datetime).dt.total_seconds()/86400
od['gap']=od.groupby('customer_id').transaction_datetime.diff().dt.total_seconds()/86400
g=od.groupby('customer_id')
test('ec_gap_mean',g.gap.mean()); test('ec_gap_cv',g.gap.std()/g.gap.mean()); test('ec_gap_max',g.gap.max())
test('ec_since_last_vs_gap',g.days.min()/g.gap.median()); test('ec_gap_last3',g.gap.apply(lambda s:s.tail(3).mean())/g.gap.mean())
od['pm_switch']=(od.payment_method!=od.groupby('customer_id').payment_method.shift()).astype(float)
test('ec_pm_switch',g.pm_switch.mean())
od['canc']=(od.order_status=='cancelled').astype(float); od['cc']=(od.payment_method=='credit_card').astype(float); od['cod']=(od.payment_method=='COD').astype(float)
rec=od[od.days<=90].groupby('customer_id'); old=od[od.days>90].groupby('customer_id')
for c in ['canc','cc','cod']: test(f'ec_{c}_trend',rec[c].mean()-old[c].mean())
test('ec_order_size_trend',rec.order_total.mean()/old.order_total.mean())
pc=ec.groupby(['customer_id','product_category']).size().unstack(fill_value=0); pr=pc.div(pc.sum(1),axis=0)
test('ec_cat_entropy',-(pr*np.log(pr.replace(0,np.nan))).sum(1)); test('ec_n_cats',(pc>0).sum(1))
ec['wk']=ec.transaction_datetime.dt.dayofweek; ec['hr']=ec.transaction_datetime.dt.hour
test('ec_hour_std',ec.groupby('customer_id').hr.std())
rh=pd.read_csv(D+'repayment_history.csv',parse_dates=['due_date','last_payment_datetime']).merge(cut,on='customer_id')
rh['days']=(rh.cut-rh.due_date).dt.days; rh['late']=(rh.max_days_past_due>0).astype(float)
rh['delay']=(rh.last_payment_datetime-rh.due_date).dt.total_seconds()/86400
for hl in [90,180,365]:
    w=np.exp(-rh.days.clip(lower=0)*np.log(2)/hl)
    test(f'rh_decay_late{hl}',(w*rh.late).groupby(rh.customer_id).sum()/w.groupby(rh.customer_id).sum())
    test(f'rh_decay_dpd{hl}',(w*rh.max_days_past_due).groupby(rh.customer_id).sum()/w.groupby(rh.customer_id).sum())
gr=rh.groupby('customer_id')
test('rh_delay_slope',gr.apply(lambda d:np.polyfit(-d.days,d.delay.fillna(d.delay.max()),1)[0] if len(d)>3 and d.delay.notna().sum()>2 else np.nan))
test('rh_delay_std',gr.delay.std()); test('rh_delay_med',gr.delay.median()); test('rh_delay_q90',gr.delay.quantile(.9))
pa=pd.read_csv(D+'previous_applications.csv',parse_dates=['application_datetime'])
due=rh.groupby('loan_id').amount_due.median(); pa['due_ratio']=pa.loan_id.map(due)/pa.scheduled_monthly_installment
test('rh_due_vs_sched',pa.groupby('customer_id').due_ratio.mean())
pa=pa.sort_values(['customer_id','application_datetime']); pa['gap']=pa.groupby('customer_id').application_datetime.diff().dt.days
gp=pa.groupby('customer_id'); test('pa_gap_mean',gp.gap.mean()); test('pa_gap_min',gp.gap.min())
test('pa_req_trend',gp.requested_amount.last()/gp.requested_amount.mean())
inc=pd.read_csv(D+'customer_profiles.csv').set_index('customer_id').declared_monthly_income
test('pa_req_to_income',gp.requested_amount.mean()/inc)
pd.DataFrame(out).assign(application_id=app.application_id.values).to_parquet(S+'cand2.parquet')
print(pd.DataFrame(res,columns=['f','cov','LR']).sort_values('LR',ascending=False).to_string())
