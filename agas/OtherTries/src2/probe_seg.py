import sys, time; sys.path.insert(0,'src2')
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from models import *
X=pd.read_pickle('artifacts2/X_v3.pkl'); meta=pd.read_pickle('artifacts2/apps_meta.pkl')
is_tr=meta.target.notna().to_numpy()
Xa,y=X[is_tr].reset_index(drop=True), meta.target[is_tr].astype(int).to_numpy()
for c in [c for c in X.columns if str(X[c].dtype)=='category']: Xa[c]=Xa[c].cat.codes
asof=[c for c in Xa.columns if (c.startswith('rh__rh_') and any(k in c for k in ['asof','dpd_ge_','dpd_asof','pay_ratio'])) or c.startswith('rh__rha_')]
cols=[c for c in Xa.columns if c not in asof]
pa=pd.read_csv('data/previous_applications.csv',usecols=['customer_id'])
has_pa=Xa.iloc[:,0].index  # placeholder
cust=Xa.index
tr_meta=pd.read_csv('data/train_applications.csv',usecols=['customer_id','target'])
has_pa=tr_meta.customer_id.isin(set(pa.customer_id)).to_numpy()
folds=list(make_folds(Xa,y)); P=dict(LGB_RND,num_threads=10)
ksv=lambda yv,p: ks_2samp(p[yv==1],p[yv==0]).statistic
oof_global=np.zeros(len(y)); oof_seg=np.zeros(len(y))
for k,(a,b) in enumerate(folds):
    ds=lgb.Dataset(Xa[cols].iloc[a],y[a]); dv=lgb.Dataset(Xa[cols].iloc[b],y[b],reference=ds)
    m=lgb.train(P,ds,12000,valid_sets=[dv],callbacks=[lgb.early_stopping(400,verbose=False)])
    oof_global[b]=m.predict(Xa[cols].iloc[b],num_iteration=m.best_iteration)
    # model spesialis untuk subgroup tanpa histori previous_applications
    a2=a[~has_pa[a]]; b2=b[~has_pa[b]]
    if len(a2)>500:
        ds2=lgb.Dataset(Xa[cols].iloc[a2],y[a2]); dv2=lgb.Dataset(Xa[cols].iloc[b2],y[b2],reference=ds2)
        m2=lgb.train(dict(P,seed=7),ds2,12000,valid_sets=[dv2],callbacks=[lgb.early_stopping(400,verbose=False)])
        oof_seg[b2]=m2.predict(Xa[cols].iloc[b2],num_iteration=m2.best_iteration)
print(f'global        ALL KS={ksv(y,oof_global):.5f}  no-pa KS={ksv(y[~has_pa],oof_global[~has_pa]):.5f}  pa KS={ksv(y[has_pa],oof_global[has_pa]):.5f}')
mix=oof_global.copy(); mix[~has_pa]=oof_seg[~has_pa]
print(f'segment-model no-pa KS={ksv(y[~has_pa],oof_seg[~has_pa]):.5f}')
print(f'mixed         ALL KS={ksv(y,mix):.5f}')
from scipy.stats import rankdata
rk=lambda v: rankdata(v)/len(v)
print(f'rank-blend 50/50 ALL KS={ksv(y, 0.5*rk(oof_global)+0.5*rk(mix)):.5f}')
np.save('artifacts2/oof_seg.npy', mix); np.save('artifacts2/oof_glob.npy', oof_global)
