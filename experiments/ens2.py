import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
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
res=pd.read_pickle(S+'res.pkl')
oof=np.zeros(len(y)); pt=np.zeros(len(te))
for a,b in folds:
    m=xgb.XGBClassifier(n_estimators=10000,learning_rate=0.01,max_depth=4,min_child_weight=20,subsample=0.8,colsample_bytree=0.3,reg_lambda=20,tree_method='hist',enable_categorical=True,eval_metric='auc',early_stopping_rounds=300,n_jobs=-1,max_bin=128)
    m.fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],verbose=False); oof[b]=m.predict_proba(X.iloc[b])[:,1]; pt+=m.predict_proba(Xt)[:,1]/5
res['xgb']=(oof,pt); print('xgb',roc_auc_score(y,oof),ks(oof),flush=True)
Xc=X.copy(); Xc[cats]=Xc[cats].astype(str); Xtc=Xt.copy(); Xtc[cats]=Xtc[cats].astype(str)
oof=np.zeros(len(y)); pt=np.zeros(len(te))
for a,b in folds:
    m=CatBoostClassifier(iterations=6000,learning_rate=0.04,depth=8,l2_leaf_reg=10,rsm=0.3,eval_metric='AUC',od_type='Iter',od_wait=300,verbose=0,cat_features=cats,thread_count=-1,random_seed=7)
    m.fit(Xc.iloc[a],y[a],eval_set=(Xc.iloc[b],y[b])); oof[b]=m.predict_proba(Xc.iloc[b])[:,1]; pt+=m.predict_proba(Xtc)[:,1]/5
res['cat8']=(oof,pt); print('cat8',roc_auc_score(y,oof),ks(oof),flush=True)
pd.to_pickle(res,S+'res2.pkl')
r=lambda v:rankdata(v)/len(v)
def blend(w,i): return sum(w[k]*r(res[k][i]) for k in w)/sum(w.values())
lgbm=lambda i:np.mean([r(res[f'lgb{s}'][i]) for s in range(3)],0)
res['lgb']=(lgbm(0),lgbm(1))
for w in [dict(lgb=1,cat=1),dict(lgb=1,cat=1,xgb=1),dict(lgb=1,cat=1,cat8=1),dict(lgb=1,cat=1,xgb=1,cat8=1),dict(lgb=2,cat=1,xgb=1,cat8=1)]:
    b=blend(w,0); print(w,round(roc_auc_score(y,b),5),round(ks(b),5),flush=True)
