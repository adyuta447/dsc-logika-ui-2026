import pandas as pd, numpy as np, lightgbm as lgb, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, rankdata
S=sys.argv[1]
f=pd.read_parquet('features.parquet').merge(pd.read_parquet(S+'cand_fa3.parquet'),on='application_id',how='left')
tr=f[f.target.notna()].reset_index(drop=True); y=tr.target.astype(int).values
noise=["region","employment_type","housing_status","age","employment_tenure_months","dependents","tenure_to_age","income_per_dependent","loan_purpose","decision_lag_h","app_hour","app_dow"]
new=['factor_z','factor_sd','fs_price','fs_late','fs_dpd','fs_bu_z','rh_clean_late','rh_clean_ldpd']
base=[c for c in tr.columns if c not in noise+new+['application_id','customer_id','target'] and not c.startswith(('ec_share_','ec_catqty_','cs_shift','buh_','fs_','factor','rh_raw','rh_clean'))]
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y))
def run(cols,seed):
    p=dict(objective="binary",learning_rate=0.01,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,max_bin=127,seed=seed,deterministic=True,force_row_wise=True,n_jobs=-1,verbose=-1)
    oof=np.zeros(len(tr))
    for a,b in folds:
        m=lgb.train(p,lgb.Dataset(tr[cols].iloc[a],y[a]),10000,valid_sets=[lgb.Dataset(tr[cols].iloc[b],y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
        oof[b]=m.predict(tr[cols].iloc[b])
    return oof
ks=lambda p: ks_2samp(p[y==1],p[y==0]).statistic
for name,cols in [('base',base),('+factor',base+new),('+factor_z only',base+['factor_z','factor_sd'])]:
    o=[run(cols,s) for s in (0,1)]
    for s in (0,1): print(name,'seed',s,round(roc_auc_score(y,o[s]),5),round(ks(o[s]),5),flush=True)
    m=np.mean([rankdata(v) for v in o],0); print(name,'2 seed avg',round(roc_auc_score(y,m),5),round(ks(m),5),flush=True)
