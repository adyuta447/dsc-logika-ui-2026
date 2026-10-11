import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from features import load_apps, D
S=sys.argv[1]; app=load_apps(); n=int(app.target.notna().sum()); y=app.target.values[:n].astype(int)
cut=app[['customer_id','decision_datetime']].rename(columns={'decision_datetime':'cut'})
ec=pd.read_csv(D+'ecommerce_transactions.csv',parse_dates=['transaction_datetime']).merge(cut,on='customer_id')
ec=ec[ec.quantity>0].reset_index(drop=True)
u=ec.item_total/ec.quantity; ec['l']=np.log(u/u.groupby(ec.product_category).transform('median'))
t=ec.transaction_datetime
ec['f_pm']=ec.payment_method; ec['f_q']=ec.quantity.clip(upper=5).astype(str); ec['f_hour']=(t.dt.hour//3).astype(str)
ec['f_dow']=t.dt.dayofweek.astype(str); ec['f_ym']=t.dt.to_period('M').astype(str); ec['f_status']=ec.order_status
ec['f_nitems']=ec.groupby('order_id').order_item_id.transform('size').clip(upper=5).astype(str)
ec['f_age']=pd.cut((ec.cut-t).dt.days,[-1,30,90,180,365,730,5000]).astype(str)
ec['f_cat_q']=ec.product_category+'_'+ec.f_q; ec['f_dom']=(t.dt.day//8).astype(str)
F=['f_pm','f_cat_q','f_hour','f_dow','f_ym','f_status','f_nitems','f_age','f_dom']
eff={f:np.zeros(len(ec)) for f in F}; cust=np.zeros(len(ec))
for it in range(8):
    cust=(ec.l-sum(eff.values())).groupby(ec.customer_id).transform('mean').values
    for f in F:
        rest=ec.l.values-cust-sum(v for k,v in eff.items() if k!=f)
        eff[f]=pd.Series(rest).groupby(ec[f].values).transform('mean').values
resid=ec.l.values-cust-sum(eff.values())
print('total var',round(ec.l.var(),4),'customer effect var',round(np.var(cust),4),'residual var',round(np.var(resid),4))
for f in F: print(f'  {f:10s} effect var {np.var(eff[f]):.5f}  range {eff[f].min():+.3f}..{eff[f].max():+.3f}')
ec['clean']=ec.l-sum(eff.values())
# noise level by category and quantity, for precision weighting
ec['w']=1/pd.Series(resid**2).groupby(ec.f_cat_q.values).transform('mean').values
g=ec.groupby('customer_id')
cand={'raw_mean':g.l.mean(),'clean_mean':g.clean.mean(),
      'clean_wmean':(ec.clean*ec.w).groupby(ec.customer_id).sum()/g.w.sum(),
      'clean_shrunk':g.clean.sum()/(g.size()+10)}
for c in ['f_pm','f_status','f_age']:
    print(ec.groupby(c).l.agg(['mean','size']).round(3).T.to_string())
r=pd.read_pickle(S+'res4.pkl'); p=np.mean([r[k][0] for k in ['lgb0','lgb1','lgb2','xgb']],0).clip(1e-4,1-1e-4); lo=np.log(p/(1-p))
def ll(X):
    m=LogisticRegression(C=1e6,max_iter=500).fit(X,y); q=m.predict_proba(X)[:,1].clip(1e-9,1-1e-9); return (y*np.log(q)+(1-y)*np.log(1-q)).sum()
base=ll(lo[:,None]); out=pd.DataFrame({'application_id':app.application_id})
for k,s in cand.items():
    v=app.customer_id.map(s).astype(float); out['pd_'+k]=v.values; t_=v[:n].fillna(v[:n].mean())
    z=(t_-t_.mean())/t_.std()
    print(k,'auc',round(roc_auc_score(y,t_),4),'LR over model',round(2*(ll(np.column_stack([lo,z]))-base),1))
out.to_parquet(S+'cand_denoise.parquet')
