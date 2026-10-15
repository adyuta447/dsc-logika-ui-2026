import sys, time; sys.path.insert(0,'src2')
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from models import *
X=pd.read_pickle('artifacts2/X_v3.pkl'); meta=pd.read_pickle('artifacts2/apps_meta.pkl')
is_tr=meta.target.notna().to_numpy()
Xa,y=X[is_tr].reset_index(drop=True), meta.target[is_tr].astype(int).to_numpy()
Xb=X[~is_tr].reset_index(drop=True)
for c in [c for c in X.columns if str(X[c].dtype)=='category']: Xa[c]=Xa[c].cat.codes; Xb[c]=Xb[c].cat.codes
asof=[c for c in Xa.columns if (c.startswith('rh__rh_') and any(k in c for k in ['asof','dpd_ge_','dpd_asof','pay_ratio'])) or c.startswith('rh__rha_')]
cols=[c for c in Xa.columns if c not in asof]
folds=list(make_folds(Xa,y))
ksv=lambda yv,p: ks_2samp(p[yv==1],p[yv==0]).statistic
P=dict(LGB_RND,num_threads=10)
for k in [0,2]:
    a,b=folds[k]; yv=y[b]
    ds=lgb.Dataset(Xa[cols].iloc[a],y[a]); dv=lgb.Dataset(Xa[cols].iloc[b],yv,reference=ds)
    m=lgb.train(P,ds,12000,valid_sets=[dv],callbacks=[lgb.early_stopping(400,verbose=False)])
    base=m.predict(Xa[cols].iloc[b],num_iteration=m.best_iteration); pt=m.predict(Xb[cols],num_iteration=m.best_iteration)
    print(f'fold{k} base KS={ksv(yv,base):.5f}',flush=True)
    # soft pseudo-label: seluruh test, label=prob, bobot=keyakinan
    Xp=pd.concat([Xa[cols].iloc[a],Xb[cols]],ignore_index=True)
    for name,lab,w in [('hard_conf',(pt>0.5).astype(int),np.abs(pt-0.5)*2)]:
        yp=np.concatenate([y[a],lab]); wp=np.concatenate([np.ones(len(a)),w*0.5])
        ds2=lgb.Dataset(Xp,yp,weight=wp); dv2=lgb.Dataset(Xa[cols].iloc[b],yv,reference=ds2)
        m2=lgb.train(P,ds2,12000,valid_sets=[dv2],callbacks=[lgb.early_stopping(400,verbose=False)])
        pr=m2.predict(Xa[cols].iloc[b],num_iteration=m2.best_iteration)
        print(f'fold{k} pseudo {name} KS={ksv(yv,pr):.5f} AUC={roc_auc_score(yv,pr):.5f}',flush=True)
