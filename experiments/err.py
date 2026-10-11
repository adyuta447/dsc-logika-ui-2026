import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
S=sys.argv[1]
f=pd.read_parquet('features.parquet'); tr=f[f.target.notna()].reset_index(drop=True); y=tr.target.astype(int).values
C=pd.read_parquet(S+'cand.parquet'); tr=tr.merge(C.drop(columns='customer_id'),on='application_id',how='left')
r=pd.read_pickle(S+'res4.pkl'); o=np.mean([rankdata(r[k][0]) for k in r],0)/len(y)
seg={'no_repay':tr.rh_n.isna(),'has_repay':tr.rh_n.notna(),'no_prev':tr.pa_n.isna(),'few_ec(<20 orders)':tr.ec_orders<20,
     'many_ec(>=60)':tr.ec_orders>=60,'no_account':tr.ab_months.isna(),'unscored_bureau':tr.bu_latest_score.isna()}
for c in ['loan_purpose','employment_type','region','housing_status']:
    for v in tr[c].unique(): seg[f'{c}={v}']=tr[c]==v
for q,(a,b) in enumerate([(0,.33),(.33,.66),(.66,1)]):
    seg[f'rate_tercile{q}']=tr.interest_rate.rank(pct=True).between(a,b)
rows=[(k,int(m.sum()),round(y[m].mean(),3),round(roc_auc_score(y[m],o[m]),4)) for k,m in seg.items() if m.sum()>500]
print(pd.DataFrame(rows,columns=['segment','n','rate','auc']).sort_values('auc').to_string())
dec=pd.qcut(o,20,labels=False)
out=[]
for c in tr.columns.drop(['application_id','customer_id','target']):
    v=tr[c]
    if str(v.dtype)=='category': v=v.cat.codes.astype(float)
    v=v.astype(float)
    num=den=0
    for d in range(20):
        m=(dec==d)&v.notna().values; yy=y[m]
        if yy.sum()<20 or (1-yy).sum()<20: continue
        a=roc_auc_score(yy,v.values[m]); w=yy.sum()*(1-yy).sum(); num+=(a-.5)*w; den+=w
    if den: out.append((c,round(num/den,4)))
res=pd.DataFrame(out,columns=['f','strat_auc_minus_half']); res['abs']=res.iloc[:,1].abs()
print(res.sort_values('abs',ascending=False).head(25).to_string())
