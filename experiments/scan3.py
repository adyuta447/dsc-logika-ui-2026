import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from features import load_apps, D
S=sys.argv[1]; app=load_apps(); n=int(app.target.notna().sum()); y=app.target.values[:n].astype(int)
F=pd.read_parquet('features.parquet')
r=pd.read_pickle(S+'res4.pkl'); p=np.mean([r[k][0] for k in ['lgb0','lgb1','lgb2','xgb']],0).clip(1e-4,1-1e-4); lo=np.log(p/(1-p))
def ll(X):
    m=LogisticRegression(C=1e6,max_iter=500).fit(X,y); q=m.predict_proba(X)[:,1].clip(1e-9,1-1e-9); return (y*np.log(q)+(1-y)*np.log(1-q)).sum()
base=ll(lo[:,None]); out={}; res=[]
def test(name,v):
    v=pd.Series(np.asarray(v,dtype=float)).replace([np.inf,-np.inf],np.nan); out[name]=v.values; t=v[:n]
    if t.notna().mean()<0.2 or t.std()==0: return
    t=t.rank(pct=True); na=t.isna().astype(float); t=t.fillna(0.5)
    res.append((name,round(2*(ll(np.column_stack([lo,t,na]))-base),1)))
cm=lambda s:app.customer_id.map(s).values
# 1. per grade
grade=app.annual_interest_rate.values
o=np.r_[rankdata(p)/n,np.full(len(app)-n,np.nan)]
print('model AUC within each rate grade:')
for g in sorted(set(grade[:n])):
    m=grade[:n]==g
    if y[m].sum()>30: print('  ',g,m.sum(),round(y[m].mean(),3),round(roc_auc_score(y[m],p[m]),4))
for c in ['bua_score','rh_rate1_smooth','ec_price_rel','bu_latest_score','ab_l3_income_like_credits','inst_to_income','dti_total','bua_utilization','ec_n30_vs_base','ab_k1_vs_base_outflows']:
    v=F[c].astype(float); gm=v.groupby(grade).transform('mean'); gs=v.groupby(grade).transform('std')
    test(f'g_{c}_z',(v-gm)/gs)
# 2. item level ecommerce
cut=app[['customer_id','decision_datetime']].rename(columns={'decision_datetime':'cut'})
ec=pd.read_csv(D+'ecommerce_transactions.csv',parse_dates=['transaction_datetime']).merge(cut,on='customer_id')
ec=ec[ec.quantity>0].copy(); u=ec.item_total/ec.quantity
ec['l']=np.log(u/u.groupby(ec.product_category).transform('median')); ec['d']=(ec.cut-ec.transaction_datetime).dt.days
ec['done']=ec.order_status=='completed'
g=ec.groupby('customer_id')
test('ec_l_spendw',cm((ec.l*ec.item_total).groupby(ec.customer_id).sum()/g.item_total.sum()))
for w in [30,90]:
    rec=ec[ec.d<=w].groupby('customer_id')
    test(f'ec_l_max{w}',cm(rec.l.max())); test(f'ec_l_hi_n{w}',cm(ec[(ec.d<=w)&(ec.l>0.7)].groupby('customer_id').size().reindex(g.size().index,fill_value=0)))
own=ec[ec.d>90].groupby('customer_id').l.agg(['mean','std']); rec=ec[ec.d<=90].groupby('customer_id').l.mean()
test('ec_l_recent_z_own',cm((rec-own['mean'])/own['std']))
for cat in ['electronics','fashion','home']:
    test(f'ec_l_{cat}_r90',cm(ec[(ec.product_category==cat)&(ec.d<=90)].groupby('customer_id').l.mean()))
test('ec_l_done_cc',cm(ec[ec.done&(ec.payment_method=='credit_card')].groupby('customer_id').l.mean()))
test('ec_l_x_qty',cm((ec.l*ec.quantity).groupby(ec.customer_id).sum()/g.quantity.sum()))
test('ec_l_last10',cm(ec.sort_values('transaction_datetime').groupby('customer_id').l.apply(lambda s:s.tail(10).mean())))
test('ec_l_first10',cm(ec.sort_values('transaction_datetime').groupby('customer_id').l.apply(lambda s:s.head(10).mean())))
test('ec_l_iqr',cm(g.l.quantile(.75)-g.l.quantile(.25)))
# 3. monthly cross source
ec['ym']=ec.transaction_datetime.dt.to_period('M')
sp=ec[ec.done].groupby(['customer_id','ym']).item_total.sum().rename('spend').reset_index()
ab=pd.read_csv(D+'account_behavior.csv',parse_dates=['month']); ab['ym']=ab.month.dt.to_period('M')
mm=ab.merge(sp,on=['customer_id','ym'],how='left').fillna({'spend':0}).merge(cut,on='customer_id')
mm['k']=(mm.cut.dt.to_period('M')-mm.ym).apply(lambda x:x.n)
mm['sp_out']=mm.spend/mm.outflows.replace(0,np.nan); mm['sp_inc']=mm.spend/mm.income_like_credits.replace(0,np.nan)
r3=mm[mm.k<=3].groupby('customer_id'); b=mm[mm.k>3].groupby('customer_id')
test('x_sp_out_r3',cm(r3.sp_out.mean())); test('x_sp_out_trend',cm(r3.sp_out.mean()-b.sp_out.mean()))
test('x_sp_inc_r3',cm(r3.sp_inc.mean())); test('x_sp_inc_trend',cm(r3.sp_inc.mean()-b.sp_inc.mean()))
test('x_corr_sp_inflow',cm(mm.groupby('customer_id').apply(lambda d:d.spend.corr(d.inflows) if len(d)>4 else np.nan)))
test('x_corr_sp_bal',cm(mm.groupby('customer_id').apply(lambda d:d.spend.corr(d.average_balance) if len(d)>4 else np.nan)))
pd.DataFrame(out).assign(application_id=app.application_id.values).to_parquet(S+'cand3.parquet')
print(pd.DataFrame(res,columns=['f','LR']).sort_values('LR',ascending=False).to_string())
