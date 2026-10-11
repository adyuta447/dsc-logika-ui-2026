import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True); te=f[f.target.isna()].reset_index(drop=True)
drop=['application_id','customer_id','target']
X=tr.drop(columns=drop); y=tr.target.astype(int).values; Xt=te.drop(columns=drop)
p=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127,seed=42)
oof=np.zeros(len(y)); pt=np.zeros(len(te))
for a,b in StratifiedKFold(5,shuffle=True,random_state=42).split(X,y):
    m=lgb.train(p,lgb.Dataset(X.iloc[a],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
    oof[b]=m.predict(X.iloc[b]); pt+=m.predict(Xt)/5
from sklearn.metrics import roc_auc_score; print('AUC',roc_auc_score(y,oof),'KS',ks_2samp(oof[y==1],oof[y==0]).statistic)
ss=pd.read_csv('data/sample_submission.csv')
sub=ss[['application_id']].merge(pd.DataFrame({'application_id':te.application_id,'default_probability':pt}),how='left')
assert sub.default_probability.notna().all() and len(sub)==30000
sub.to_csv('/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/sub_lgb.csv',index=False)
