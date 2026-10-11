import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
ks=lambda p:ks_2samp(p[y==1],p[y==0]).statistic
p=dict(objective='binary',learning_rate=0.02,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127,seed=0)
def run(cols):
    oof=np.zeros(len(y))
    for a,b in folds:
        m=lgb.train(p,lgb.Dataset(X.iloc[a][cols],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b][cols],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
        oof[b]=m.predict(X.iloc[b][cols])
    return roc_auc_score(y,oof),ks(oof)
allc=list(X.columns)
noise=['region','employment_type','housing_status','age','employment_tenure_months','dependents','tenure_to_age','income_per_dependent','loan_purpose','decision_lag_h','app_hour','app_dow']
noise+= [c for c in allc if c.startswith(('ec_share_','ec_catqty_','cs_shift','buh_'))]
keep=[c for c in allc if c not in noise]
for s in [0,1]:
    p['seed']=s; print('seed',s,'all',*run(allc),'drop4',len(keep),*run(keep),flush=True)
pd.Series(noise).to_csv(S+'noise.csv',index=False) if False else None
