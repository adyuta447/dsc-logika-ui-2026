import pandas as pd, numpy as np, lightgbm as lgb, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
S=sys.argv[1]; add=sys.argv[2].split(',')
f=pd.read_parquet('features.parquet'); C=pd.read_parquet(S+'cand.parquet')
f=f.merge(C[['application_id']+add],on='application_id',how='left')
tr=f[f.target.notna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values
allc=list(X.columns)
noise=['region','employment_type','housing_status','age','employment_tenure_months','dependents','tenure_to_age','income_per_dependent','loan_purpose','decision_lag_h','app_hour','app_dow']+[c for c in allc if c.startswith(('ec_share_','ec_catqty_','cs_shift','buh_'))]
base=[c for c in allc if c not in noise and c not in add]
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
def run(cols,s):
    p=dict(objective='binary',learning_rate=0.02,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127,seed=s)
    oof=np.zeros(len(y))
    for a,b in folds:
        m=lgb.train(p,lgb.Dataset(X.iloc[a][cols],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b][cols],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
        oof[b]=m.predict(X.iloc[b][cols])
    return oof
for s in [0,1]:
    o0=run(base,s); o1=run(base+add,s)
    print(s,'base',round(roc_auc_score(y,o0),5),'add',round(roc_auc_score(y,o1),5),'ks',round(ks_2samp(o0[y==1],o0[y==0]).statistic,4),round(ks_2samp(o1[y==1],o1[y==0]).statistic,4),flush=True)
