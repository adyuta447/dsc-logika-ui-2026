import pandas as pd, numpy as np, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, rankdata
S='/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/'
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True); te=f[f.target.isna()].reset_index(drop=True)
drop=['application_id','customer_id','target']
X=tr.drop(columns=drop); y=tr.target.astype(int).values; Xt=te.drop(columns=drop)
cats=[c for c in X.columns if str(X[c].dtype)=='category']
ks=lambda p:ks_2samp(p[y==1],p[y==0]).statistic
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
res={}
for seed in [0,1,2]:
    p=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127,seed=seed)
    oof=np.zeros(len(y)); pt=np.zeros(len(te))
    for a,b in folds:
        m=lgb.train(p,lgb.Dataset(X.iloc[a],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
        oof[b]=m.predict(X.iloc[b]); pt+=m.predict(Xt)/5
    res[f'lgb{seed}']=(oof,pt); print('lgb',seed,roc_auc_score(y,oof),ks(oof),flush=True)
Xc=X.copy(); Xc[cats]=Xc[cats].astype(str); Xtc=Xt.copy(); Xtc[cats]=Xtc[cats].astype(str)
oof=np.zeros(len(y)); pt=np.zeros(len(te))
for a,b in folds:
    m=CatBoostClassifier(iterations=6000,learning_rate=0.03,depth=6,l2_leaf_reg=5,eval_metric='AUC',od_type='Iter',od_wait=300,verbose=0,cat_features=cats,thread_count=-1,random_seed=0)
    m.fit(Xc.iloc[a],y[a],eval_set=(Xc.iloc[b],y[b])); oof[b]=m.predict_proba(Xc.iloc[b])[:,1]; pt+=m.predict_proba(Xtc)[:,1]/5
res['cat']=(oof,pt); print('cat',roc_auc_score(y,oof),ks(oof),flush=True)
pd.to_pickle(res,S+'res.pkl')
r=lambda v:rankdata(v)/len(v)
lo=np.mean([r(res[f'lgb{s}'][0]) for s in range(3)],0); print('lgb avg',roc_auc_score(y,lo),ks(lo))
for w in [0.3,0.5,0.7]:
    b=(1-w)*lo+w*r(res['cat'][0]); print('blend w',w,roc_auc_score(y,b),ks(b))
