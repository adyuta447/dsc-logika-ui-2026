import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb, sys
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, rankdata
S=sys.argv[1]
f=pd.read_parquet('features.parquet'); tr=f[f.target.notna()].reset_index(drop=True); te=f[f.target.isna()].reset_index(drop=True)
cols=pd.read_csv(S+'imp_order.csv').iloc[:80,0].tolist()
X=tr[cols]; Xt=te[cols]; y=tr.target.astype(int).values
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
ev=lambda o:(round(roc_auc_score(y,o),5),round(ks_2samp(o[y==1],o[y==0]).statistic,5))
res={}
for s in range(5):
    p=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=200,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,max_bin=127,seed=s,deterministic=True,force_row_wise=True,n_jobs=-1,verbose=-1)
    oof=np.zeros(len(y)); pt=np.zeros(len(te))
    for a,b in folds:
        m=lgb.train(p,lgb.Dataset(X.iloc[a],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
        oof[b]=m.predict(X.iloc[b]); pt+=m.predict(Xt)/5
    res[f'lgb{s}']=(oof,pt); print('lgb',s,*ev(oof),flush=True)
oof=np.zeros(len(y)); pt=np.zeros(len(te))
for a,b in folds:
    m=CatBoostClassifier(iterations=6000,learning_rate=0.03,depth=6,l2_leaf_reg=5,eval_metric='AUC',od_type='Iter',od_wait=300,random_seed=0,verbose=0)
    m.fit(X.iloc[a],y[a],eval_set=(X.iloc[b],y[b])); oof[b]=m.predict_proba(X.iloc[b])[:,1]; pt+=m.predict_proba(Xt)[:,1]/5
res['cat']=(oof,pt); print('cat',*ev(oof),flush=True)
oof=np.zeros(len(y)); pt=np.zeros(len(te))
for a,b in folds:
    m=xgb.XGBClassifier(n_estimators=10000,learning_rate=0.01,max_depth=4,min_child_weight=20,subsample=0.8,colsample_bytree=0.3,reg_lambda=20,tree_method='hist',max_bin=128,eval_metric='auc',early_stopping_rounds=300,random_state=0,n_jobs=-1)
    m.fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],verbose=False); oof[b]=m.predict_proba(X.iloc[b])[:,1]; pt+=m.predict_proba(Xt)[:,1]/5
res['xgb']=(oof,pt); print('xgb',*ev(oof),flush=True)
pd.to_pickle(res,S+'res_top80.pkl')
r=lambda v:rankdata(v)/len(v)
R={f'r{i}':pd.read_pickle(S+f'res{i}.pkl') for i in [3,4]}; L=pd.read_pickle(S+'lin_0.03.pkl')
ens=lambda R,i,k=3:(np.mean([r(R[f'lgb{s}'][i]) for s in range(k)],0)+r(R['cat'][i])+r(R['xgb'][i]))/3
nb=lambda i:0.8*(ens(R['r3'],i)+ens(R['r4'],i)+r(R['r3']['cat'][i])+r(R['r4']['cat'][i]))/4+0.2*r(L[i])
print('notebook-like blend',*ev(nb(0)))
print('top80 ensemble',*ev(ens(res,0,5)))
for w in [0.2,0.33,0.5]:
    print('blend + top80 w',w,*ev((1-w)*nb(0)+w*ens(res,0,5)),flush=True)
