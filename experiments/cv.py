import pandas as pd, numpy as np, lightgbm as lgb, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int)
p=dict(objective='binary',learning_rate=0.02,num_leaves=31,min_child_samples=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,n_jobs=-1)
oof=np.zeros(len(tr)); imp=0
for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
    m=lgb.train(p,lgb.Dataset(X.iloc[a],y[a]),5000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(X.iloc[b]); imp+=m.feature_importance('gain'); print(k,m.best_iteration,flush=True)
ks=lambda p:ks_2samp(p[y==1],p[y==0]).statistic
print('AUC',roc_auc_score(y,oof),'KS',ks(oof))
print(pd.Series(imp,X.columns).sort_values(ascending=False).head(40))
