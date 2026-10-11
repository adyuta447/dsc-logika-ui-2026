import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from features import load_apps, D
S=sys.argv[1]; app=load_apps(); n=int(app.target.notna().sum()); y=app.target.values[:n].astype(int)
cut=app[['customer_id','decision_datetime']].rename(columns={'decision_datetime':'cut'})
r=pd.read_pickle(S+'res4.pkl'); p=np.mean([r[k][0] for k in ['lgb0','lgb1','lgb2','xgb']],0).clip(1e-4,1-1e-4); lo=np.log(p/(1-p))
def ll(X):
    m=LogisticRegression(C=1e6,max_iter=500).fit(X,y); q=m.predict_proba(X)[:,1].clip(1e-9,1-1e-9); return (y*np.log(q)+(1-y)*np.log(1-q)).sum()
base=ll(lo[:,None]); res=[]; out=pd.DataFrame({'application_id':app.application_id})
def test(name,s):
    v=app.customer_id.map(s).astype(float); out[name]=v.values; t=v[:n]
    a=roc_auc_score(y[t.notna().values],t.dropna()); t=t.rank(pct=True); na=t.isna().astype(float); t=t.fillna(0.5)
    res.append((name,round(t.notna().mean(),3),round(max(a,1-a),4),round(2*(ll(np.column_stack([lo,t,na]))-base),1)))

# 3. repayment lateness: within customer fixed effects, same method as the price denoising
rh=pd.read_csv(D+'repayment_history.csv',parse_dates=['due_date']).merge(cut,on='customer_id')
rh['days_ago']=(rh.cut-rh.due_date).dt.days
print('repayment rows due after cut',(rh.days_ago<0).mean())
rh=rh[rh.days_ago>=1].reset_index(drop=True)
rh['late']=(rh.max_days_past_due>0).astype(float); rh['ldpd']=np.log1p(rh.max_days_past_due)
d=rh.due_date; rh['f_dow']=d.dt.dayofweek; rh['f_dom']=d.dt.day//8; rh['f_ym']=d.dt.to_period('M').astype(str)
rh['f_seq']=rh.groupby('loan_id').due_date.rank().clip(upper=12); rh['f_term']=rh.groupby('loan_id').due_date.transform('size').clip(upper=36)
rh['f_amt']=pd.qcut(np.log(rh.amount_due),10,labels=False,duplicates='drop'); rh['f_age']=pd.cut(rh.days_ago,[0,30,90,180,365,730,5000],labels=False)
F=['f_dow','f_dom','f_ym','f_seq','f_term','f_amt','f_age']
for col in ['late','ldpd']:
    eff={f:np.zeros(len(rh)) for f in F}
    for it in range(8):
        cust=(rh[col]-sum(eff.values())).groupby(rh.customer_id).transform('mean').values
        for f in F: eff[f]=pd.Series(rh[col].values-cust-sum(v for k,v in eff.items() if k!=f)).groupby(rh[f].values).transform('mean').values
    resid=rh[col].values-cust-sum(eff.values())
    print(col,'total var',round(rh[col].var(),4),'customer var',round(np.var(cust),4),'resid var',round(np.var(resid),4))
    for f in F: print(f'   {f:7s} effect var {np.var(eff[f]):.5f} range {eff[f].min():+.3f}..{eff[f].max():+.3f}')
    rh[col+'_c']=rh[col]-sum(eff.values())
    test('rh_raw_'+col,rh.groupby('customer_id')[col].mean()); test('rh_clean_'+col,rh.groupby('customer_id')[col+'_c'].mean())

# 2. one factor model over all sources, with per customer measurement noise from item counts
items={}
ec=pd.read_csv(D+'ecommerce_transactions.csv',parse_dates=['transaction_datetime']).merge(cut,on='customer_id')
print('ec rows after cut',(ec.transaction_datetime>=ec.cut).mean())
ec=ec[(ec.quantity>0)&(ec.transaction_datetime<ec.cut)]; u=ec.item_total/ec.quantity
items['price']=pd.DataFrame({'c':ec.customer_id,'x':np.log(u/u.groupby(ec.product_category).transform('median'))})
items['late']=pd.DataFrame({'c':rh.customer_id,'x':rh.late_c}); items['dpd']=pd.DataFrame({'c':rh.customer_id,'x':rh.ldpd_c})
bu=pd.read_csv(D+'bureau_observations.csv',parse_dates=['available_datetime']).merge(cut,on='customer_id')
print('bureau rows available after cut',(bu.available_datetime>bu.cut).mean())
bu=bu[bu.available_datetime<=bu.cut]; g=bu.groupby('bureau_provider').score
bu['z']=(bu.score-g.transform('mean'))/g.transform('std')
for c in ['z','utilization','inquiries_last_6m','delinquent_accounts_last_12m']:
    b=bu[bu[c].notna()]; items['bu_'+c]=pd.DataFrame({'c':b.customer_id,'x':b[c]})
ab=pd.read_csv(D+'account_behavior.csv',parse_dates=['month']).merge(cut,on='customer_id')
print('account months not finished at cut',(ab.month+pd.offsets.MonthBegin(1)>ab.cut).mean())
ab=ab[ab.month+pd.offsets.MonthBegin(1)<=ab.cut]
items['ab_low']=pd.DataFrame({'c':ab.customer_id,'x':ab.low_balance_days})
items['ab_outin']=pd.DataFrame({'c':ab.customer_id,'x':np.log1p(ab.outflows)-np.log1p(ab.inflows)})
items['ab_ilc']=pd.DataFrame({'c':ab.customer_id,'x':np.log1p(ab.income_like_credits)})
pa=pd.read_csv(D+'previous_applications.csv',parse_dates=['decision_datetime']).merge(cut,on='customer_id')
pa=pa[pa.decision_datetime<pa.cut]; items['pa_rej']=pd.DataFrame({'c':pa.customer_id,'x':(pa.decision=='rejected').astype(float)})
items['rate']=pd.DataFrame({'c':app.customer_id,'x':app.interest_rate if 'interest_rate' in app else app.annual_interest_rate})

C=pd.Index(app.customer_id); M={}; E={}
for j,t in items.items():
    g=t.groupby('c').x; m=g.mean(); k=g.size(); dev=t.x-g.transform('mean')
    w=(dev**2).sum()/max(len(t)-len(m),1)
    b=m.var()-(w/k).mean()
    M[j]=((m-m.mean())/np.sqrt(b)).reindex(C); E[j]=(w/(k*b)).reindex(C)
    print(f'{j:30s} customers {len(m):6d} items/cust {k.mean():6.1f} item noise share {w/(w+b):.3f}')
M=pd.DataFrame(M); E=pd.DataFrame(E)
pt=np.mean([r[k][1] for k in ['lgb0','lgb1','lgb2','xgb']],0).clip(1e-4,1-1e-4); lt=np.log(pt/(1-pt)); zl=pd.Series((lt-lt.mean())/lt.std(),index=C[n:]); a=np.array([M[j].iloc[n:].corr(zl)/0.92 for j in M.columns])
a=np.clip(a,-.97,.97)
print(pd.Series(a,index=M.columns).round(3).to_string())
den=(1-a**2)+E.values; num=np.nansum(np.where(M.notna(),a*M.values/den,0),1); prec=1+np.nansum(np.where(M.notna(),a**2/den,0),1)
z=pd.Series(num/prec,index=C); test('factor_z',z); test('factor_sd',pd.Series(1/np.sqrt(prec),index=C))
# same posterior without the precision weighting, to see whether the weighting is what helps
den0=(1-a**2)+E.mean().values; z0=np.nansum(np.where(M.notna(),a*M.values/den0,0),1)/(1+np.nansum(np.where(M.notna(),a**2/den0,0),1))
test('factor_z_flat',pd.Series(z0,index=C))
sh=np.where(M.notna(),M.values*(1-a**2)/den,np.nan)
for i,j in enumerate(M.columns): out['fs_'+j]=app.customer_id.map(pd.Series(sh[:,i],index=C)).values
print('base auc',round(roc_auc_score(y,lo),4))
print(pd.DataFrame(res,columns=['f','cov','auc','LR']).to_string())
out.to_parquet(S+'cand_fa3.parquet')
