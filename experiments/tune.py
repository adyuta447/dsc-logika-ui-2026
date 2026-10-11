import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
ks=lambda p:ks_2samp(p[y==1],p[y==0]).statistic
base=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127,seed=0)
def run(p,cols):
    oof=np.zeros(len(y)); imp=0
    for a,b in folds:
        m=lgb.train(p,lgb.Dataset(X.iloc[a][cols],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b][cols],y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
        oof[b]=m.predict(X.iloc[b][cols]); imp+=m.feature_importance('gain')
    return oof, pd.Series(imp,cols)
cols=list(X.columns)
o,imp=run(base,cols); print('base',len(cols),roc_auc_score(y,o),ks(o),flush=True)
for name,upd in [('leaves7',dict(num_leaves=7,min_child_samples=300)),('leaves31',dict(num_leaves=31,min_child_samples=300,lambda_l2=50)),('ff015',dict(feature_fraction=0.15)),('depth4',dict(max_depth=4,num_leaves=15))]:
    o2,_=run({**base,**upd},cols); print(name,roc_auc_score(y,o2),ks(o2),flush=True)
keep=list(imp.sort_values(ascending=False).index[:int(len(cols)*0.6)])
o3,_=run(base,keep); print('top60%',len(keep),roc_auc_score(y,o3),ks(o3),flush=True)
pd.Series(keep).to_csv('/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/keep.csv',index=False)
