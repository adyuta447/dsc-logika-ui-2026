import pandas as pd, numpy as np, lightgbm as lgb, sys
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, rankdata
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values
cats=[c for c in X.columns if str(X[c].dtype)=='category']
ks=lambda p:ks_2samp(p[y==1],p[y==0]).statistic
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
def run_lgb(p):
    oof=np.zeros(len(y))
    for a,b in folds:
        m=lgb.train(p,lgb.Dataset(X.iloc[a],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
        oof[b]=m.predict(X.iloc[b])
    return oof
base=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127)
o1=run_lgb(base); print('lgb small', roc_auc_score(y,o1), ks(o1), flush=True)
oc=np.zeros(len(y))
for a,b in folds:
    m=CatBoostClassifier(iterations=5000,learning_rate=0.03,depth=6,l2_leaf_reg=5,eval_metric='AUC',od_type='Iter',od_wait=300,verbose=0,cat_features=cats,thread_count=-1)
    Xa=X.copy(); Xa[cats]=Xa[cats].astype(str)
    m.fit(Xa.iloc[a],y[a],eval_set=(Xa.iloc[b],y[b])); oc[b]=m.predict_proba(Xa.iloc[b])[:,1]
print('cat', roc_auc_score(y,oc), ks(oc), flush=True)
bl=rankdata(o1)+rankdata(oc); print('blend', roc_auc_score(y,bl), ks(bl))
np.save('/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/oof.npy',np.vstack([o1,oc]))
