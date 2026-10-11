import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
f=pd.read_parquet('features.parquet'); tr=f[f.target.notna()].reset_index(drop=True); te=f[f.target.isna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values; Xt=te.drop(columns=['application_id','customer_id','target'])
allc=list(X.columns)
noise=['region','employment_type','housing_status','age','employment_tenure_months','dependents','tenure_to_age','income_per_dependent','loan_purpose','decision_lag_h','app_hour','app_dow']+[c for c in allc if c.startswith(('ec_share_','ec_catqty_','cs_shift','buh_'))]
cols=[c for c in allc if c not in noise]; X=X[cols]; Xt=Xt[cols]
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
ev=lambda o:(round(roc_auc_score(y,o),5),round(ks_2samp(o[y==1],o[y==0]).statistic,5))
for s in [0,1]:
    p=dict(objective='binary',learning_rate=0.02,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127,seed=s)
    o0=np.zeros(len(y)); o1=np.zeros(len(y)); o2=np.zeros(len(y))
    for a,b in folds:
        dv=lgb.Dataset(X.iloc[b],y[b])
        m=lgb.train(p,lgb.Dataset(X.iloc[a],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
        o0[b]=m.predict(X.iloc[b]); pt=m.predict(Xt)
        # soft pseudo labels: each test row appears as a positive with weight p and a negative with weight 1-p
        Xa=pd.concat([X.iloc[a],Xt,Xt]); ya=np.r_[y[a],np.ones(len(Xt)),np.zeros(len(Xt))]; wa=np.r_[np.ones(len(a)),pt,1-pt]
        m2=lgb.train(p,lgb.Dataset(Xa,ya,weight=wa),10000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
        o1[b]=m2.predict(X.iloc[b])
        # hard pseudo labels only for confident test rows
        conf=(pt<np.quantile(pt,0.5))|(pt>np.quantile(pt,0.95))
        Xh=pd.concat([X.iloc[a],Xt[conf]]); yh=np.r_[y[a],(pt[conf]>np.quantile(pt,0.95)).astype(int)]
        m3=lgb.train(p,lgb.Dataset(Xh,yh),10000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
        o2[b]=m3.predict(X.iloc[b])
    print('seed',s,'base',*ev(o0),'soft',*ev(o1),'hard',*ev(o2),flush=True)
