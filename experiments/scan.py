import pandas as pd, numpy as np, sys, itertools, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from features import load_apps, D
S=sys.argv[1]; app=load_apps(); n=int(app.target.notna().sum()); y=app.target.values[:n].astype(int)
r=pd.read_pickle(S+'res4.pkl'); p=np.mean([r[k][0] for k in ['lgb0','lgb1','lgb2','xgb']],0).clip(1e-4,1-1e-4); lo=np.log(p/(1-p))
def ll(X):
    m=LogisticRegression(C=1e6,max_iter=500).fit(X,y); q=m.predict_proba(X)[:,1].clip(1e-9,1-1e-9)
    return (y*np.log(q)+(1-y)*np.log(1-q)).sum()
base=ll(lo[:,None])
def test(name,s,out):
    v=app.customer_id[:n].map(s).astype(float).replace([np.inf,-np.inf],np.nan)
    if v.notna().mean()<0.3 or v.std()==0: return
    v=v.rank(pct=True); na=v.isna().astype(float); v=v.fillna(0.5)
    out.append((name,round(2*(ll(np.column_stack([lo,v,na]))-base),1)))
ab=pd.read_csv(D+'account_behavior.csv',parse_dates=['month']).merge(app[['customer_id','decision_datetime']],on='customer_id')
ab['k']=(ab.decision_datetime.dt.year*12+ab.decision_datetime.dt.month)-(ab.month.dt.year*12+ab.month.dt.month)
cols=['opening_balance','closing_balance','inflows','outflows','income_like_credits','average_balance','minimum_balance','low_balance_days']
out=[]
for lo_k,hi_k,tag in [(1,1,'k1'),(1,3,'l3'),(1,6,'l6'),(1,99,'all')]:
    g=ab[(ab.k>=lo_k)&(ab.k<=hi_k)].groupby('customer_id')[cols].mean()
    for a,b in itertools.permutations(cols,2):
        test(f'ab_{tag}_{a}/{b}',g[a]/g[b].replace(0,np.nan),out)
b4=ab[(ab.k>=4)&(ab.k<=12)].groupby('customer_id')[cols].mean(); k1=ab[ab.k<=3].groupby('customer_id')[cols].mean()
for a in cols: test(f'ab_l3_vs_base_{a}',k1[a]/b4[a].replace(0,np.nan),out)
g=ab.groupby('customer_id')
for a in cols:
    test(f'ab_cv_{a}',g[a].std()/g[a].mean().replace(0,np.nan),out)
    test(f'ab_slope_{a}',g.apply(lambda d:np.polyfit(-d.k,d[a],1)[0]/(abs(d[a].mean())+1) if len(d)>2 else np.nan),out)
bu=pd.read_csv(D+'bureau_observations.csv',parse_dates=['report_datetime'])
num=['score','active_accounts','outstanding_balance','revolving_credit_limit','revolving_outstanding_balance','utilization','inquiries_last_6m','delinquent_accounts_last_12m']
for prov,d in bu.groupby('bureau_provider'):
    gm=d.groupby('customer_id')[num].mean()
    for c in num: test(f'bu_{prov}_{c}',gm[c],out)
bs=bu.sort_values('report_datetime'); gg=bs.groupby('customer_id')
for c in num:
    test(f'bu_trend_{c}',gg[c].last()-gg[c].first(),out); test(f'bu_std_{c}',gg[c].std(),out)
res=pd.DataFrame(out,columns=['f','LR']).sort_values('LR',ascending=False)
print(len(res),'tested'); print(res.head(25).to_string())
