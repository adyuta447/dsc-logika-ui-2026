import sys, time; sys.path.insert(0,'src2')
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from models import *
X=pd.read_pickle('artifacts2/X_v4.pkl'); meta=pd.read_pickle('artifacts2/apps_meta.pkl')
is_tr=meta.target.notna().to_numpy()
Xa,y=X[is_tr].reset_index(drop=True), meta.target[is_tr].astype(int).to_numpy()
for c in [c for c in X.columns if str(X[c].dtype)=='category']: Xa[c]=Xa[c].cat.codes
asof=[c for c in Xa.columns if (c.startswith('rh__rh_') and any(k in c for k in ['asof','dpd_ge_','dpd_asof','pay_ratio'])) or c.startswith('rh__rha_')]
cols=[c for c in Xa.columns if c not in asof]
folds=list(make_folds(Xa,y)); P=dict(LGB_RND,num_threads=10)
ksv=lambda yv,p: ks_2samp(p[yv==1],p[yv==0]).statistic
# (1) apakah fitur kohort menambah KS?
for k in [0,2]:
    a,b=folds[k]; yv=y[b]
    ds=lgb.Dataset(Xa[cols].iloc[a],y[a]); dv=lgb.Dataset(Xa[cols].iloc[b],yv,reference=ds)
    m=lgb.train(P,ds,12000,valid_sets=[dv],callbacks=[lgb.early_stopping(400,verbose=False)])
    pr=m.predict(Xa[cols].iloc[b],num_iteration=m.best_iteration)
    print(f'fold{k} LGB v4( cohort) KS={ksv(yv,pr):.5f} AUC={roc_auc_score(yv,pr):.5f} (v3=0.51973/0.54131)',flush=True)
# (2) model berbeda: kNN pada rank fitur terkuat
g=pd.read_csv('artifacts2/gain_fold0.csv',index_col=0).iloc[:,0]
top=[c for c in g.index[:40] if c in cols]
Z=Xa[top].astype('float64').rank(pct=True).fillna(0.5).to_numpy()
t0=time.time(); oof=np.zeros(len(y))
for a,b in folds:
    kn=KNeighborsClassifier(n_neighbors=300, weights='distance', n_jobs=10)
    kn.fit(Z[a],y[a]); oof[b]=kn.predict_proba(Z[b])[:,1]
print(f'kNN(k=300, rank-40 fitur)      KS={ks_score(y,oof):.5f} AUC={roc_auc_score(y,oof):.5f} {time.time()-t0:.0f}s',flush=True)
np.save('artifacts2/oof_knn.npy',oof)
