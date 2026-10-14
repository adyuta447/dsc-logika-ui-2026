import sys, time; sys.path.insert(0,'src2')
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score
from models import *
X=pd.read_pickle('artifacts2/X_v3.pkl'); meta=pd.read_pickle('artifacts2/apps_meta.pkl')
is_tr=meta.target.notna().to_numpy()
Xa,y=X[is_tr].reset_index(drop=True), meta.target[is_tr].astype(int).to_numpy()
for c in [c for c in X.columns if str(X[c].dtype)=='category']: Xa[c]=Xa[c].cat.codes
asof=[c for c in Xa.columns if (c.startswith('rh__rh_') and any(k in c for k in ['asof','dpd_ge_','dpd_asof','pay_ratio'])) or c.startswith('rh__rha_')]
cols=[c for c in Xa.columns if c not in asof]
folds=list(make_folds(Xa,y))
ksv=lambda yv,p: ks_2samp(p[yv==1],p[yv==0]).statistic
base=dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.3,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, max_bin=127,
          verbose=-1, deterministic=True, force_row_wise=True, num_threads=10, seed=42)

def groups(n, size=1000, seed=0):
    idx=np.random.default_rng(seed).permutation(n); g=[]
    for i in range(0,n,size): g.append(min(size,n-i))
    return np.array(g), idx

for k in [0,2]:
    a,b=folds[k]; yv=y[b]
    for obj,rounds in [('lambdarank',1500),('rank_xendcg',1500)]:
        t0=time.time()
        ga,ia=groups(len(a),1000,seed=k); gb,ib=groups(len(b),1000,seed=100+k)
        p=dict(base,objective=obj,lambdarank_truncation_level=30,label_gain=[0,1])
        ds=lgb.Dataset(Xa[cols].iloc[a[ia]],y[a[ia]],group=ga)
        dv=lgb.Dataset(Xa[cols].iloc[b[ib]],y[b[ib]],group=gb,reference=ds)
        m=lgb.train(p,ds,rounds,valid_sets=[dv],callbacks=[lgb.early_stopping(150,verbose=False)])
        pr=np.zeros(len(b)); pr[ib]=m.predict(Xa[cols].iloc[b[ib]],num_iteration=m.best_iteration)
        print(f'fold{k} {obj:12s} KS={ksv(yv,pr):.5f} AUC={roc_auc_score(yv,pr):.5f} iter={m.best_iteration} {time.time()-t0:.0f}s',flush=True)
