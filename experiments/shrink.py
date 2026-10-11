import pandas as pd, numpy as np, sys
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'.'); from features import load_apps, D
S=sys.argv[1]; app=load_apps(); n=app.target.notna().sum(); y=app.target.values[:n]
ec=pd.read_csv(D+'ecommerce_transactions.csv',usecols=['customer_id','product_category','quantity','item_total','order_status'])
ec=ec[ec.quantity>0]; u=ec.item_total/ec.quantity
ec['l']=np.log(u/u.groupby(ec.product_category).transform('median'))
# within-category spread vs between-customer spread
g=ec.groupby('customer_id').l; m=g.mean(); k=g.size()
within=ec.l.sub(ec.customer_id.map(m)).var(); between=m.var()-within/k.mean()
print('within var',round(within,4),'between var',round(between,4),'items/cust median',k.median(),'q10',k.quantile(.1))
print(ec.groupby('product_category').l.std().round(3).to_dict())
# per-category z then mean
ec['z']=ec.l/ec.groupby('product_category').l.transform('std')
r=pd.read_pickle(S+'res4.pkl'); p=np.mean([r[x][0] for x in ['lgb0','lgb1','lgb2','xgb']],axis=0).clip(1e-4,1-1e-4); lo=np.log(p/(1-p))
base=sm.Logit(y,sm.add_constant(lo)).fit(disp=0).llf
cid=app.customer_id[:n]
cands={'mean':m,'zmean':ec.groupby('customer_id').z.mean(),'median':g.median(),'trim':g.apply(lambda s:s.clip(s.quantile(.1),s.quantile(.9)).mean())}
for kk in [2,5,10,20]: cands[f'shrink{kk}']=g.sum()/(k+kk)
for nm,s in cands.items():
    v=cid.map(s).astype(float).fillna(0).values; v=(v-v.mean())/v.std()
    mm=sm.Logit(y,np.column_stack([np.ones(n),lo,v])).fit(disp=0)
    print(nm,'auc',round(roc_auc_score(y,v),4),'LR',round(2*(mm.llf-base),1))
