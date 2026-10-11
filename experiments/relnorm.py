import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from features import load_apps, D
S=sys.argv[1]; app=load_apps(); n=int(app.target.notna().sum()); y=app.target.values[:n].astype(int)
r=pd.read_pickle(S+'res4.pkl'); p=np.mean([r[k][0] for k in ['lgb0','lgb1','lgb2','xgb']],0).clip(1e-4,1-1e-4); lo=np.log(p/(1-p))
def ll(X):
    m=LogisticRegression(C=1e6,max_iter=500).fit(X,y); q=m.predict_proba(X)[:,1].clip(1e-9,1-1e-9); return (y*np.log(q)+(1-y)*np.log(1-q)).sum()
base=ll(lo[:,None]); res=[]; out={}
def test(name,s):
    v=app.customer_id.map(s).astype(float).replace([np.inf,-np.inf],np.nan); out[name]=v.values; t=v[:n]
    if t.notna().mean()<0.2 or t.std()==0: return
    a=roc_auc_score(y[t.notna().values],t.dropna()); t=t.rank(pct=True); na=t.isna().astype(float); t=t.fillna(0.5)
    res.append((name,round(max(a,1-a),4),round(2*(ll(np.column_stack([lo,t,na]))-base),1)))
ab=pd.read_csv(D+'account_behavior.csv',parse_dates=['month'])
for c in ['inflows','outflows','income_like_credits','average_balance','minimum_balance']:
    lv=np.log1p(ab[c].clip(lower=0)); rel=lv-lv.groupby(ab.month).transform('median')
    test(f'ab_monthrel_{c}',rel.groupby(ab.customer_id).mean())
inc=pd.read_csv(D+'customer_profiles.csv').set_index('customer_id').declared_monthly_income
ab['inc']=ab.customer_id.map(inc); ab['ilc_decl']=np.log1p(ab.income_like_credits)-np.log(ab.inc)
test('ab_ilc_vs_declared_mean',ab.groupby('customer_id').ilc_decl.mean()); test('ab_ilc_vs_declared_std',ab.groupby('customer_id').ilc_decl.std())
bu=pd.read_csv(D+'bureau_observations.csv',parse_dates=['report_datetime'])
bu['m']=bu.report_datetime.dt.to_period('M')
for c in ['score','utilization','outstanding_balance','inquiries_last_6m']:
    v=bu[c] if c!='outstanding_balance' else np.log1p(bu[c])
    rel=v-v.groupby([bu.bureau_provider,bu.m]).transform('median'); test(f'bu_provmonth_rel_{c}',rel.groupby(bu.customer_id).mean())
pa=pd.read_csv(D+'previous_applications.csv')
pa['req_rel']=np.log(pa.requested_amount)-np.log(pa.requested_amount).groupby(pa.term_months).transform('median')
pa['rate_rel']=pa.annual_interest_rate-pa.annual_interest_rate.groupby(pa.term_months).transform('median')
test('pa_req_rel',pa.groupby('customer_id').req_rel.mean()); test('pa_rate_rel',pa.groupby('customer_id').rate_rel.mean())
pa['req_inc']=np.log(pa.requested_amount)-np.log(pa.customer_id.map(inc)); test('pa_req_vs_income',pa.groupby('customer_id').req_inc.mean())
rh=pd.read_csv(D+'repayment_history.csv'); rh['due_rel']=np.log(rh.amount_due)-np.log(rh.customer_id.map(inc))
test('rh_due_vs_income',rh.groupby('customer_id').due_rel.mean())
a=app.copy(); a['amt_rel']=np.log(a.loan_amount)-np.log(a.loan_amount).groupby(a.loan_purpose).transform('median')
test('app_amt_rel_purpose',pd.Series(a.amt_rel.values,index=a.customer_id))
print(pd.DataFrame(res,columns=['f','auc','LR']).sort_values('LR',ascending=False).to_string())
