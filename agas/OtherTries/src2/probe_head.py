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
folds=list(make_folds(Xa,y)); P=dict(LGB_RND,num_threads=10)
ksv=lambda yv,p: ks_2samp(p[yv==1],p[yv==0]).statistic
oof_base=np.zeros(len(y)); oof_spec=np.zeros(len(y)); oof_mix=np.zeros(len(y))
for a,b in folds:
    ds=lgb.Dataset(Xa[cols].iloc[a],y[a]); dv=lgb.Dataset(Xa[cols].iloc[b],y[b],reference=ds)
    m=lgb.train(P,ds,12000,valid_sets=[dv],callbacks=[lgb.early_stopping(400,verbose=False)])
    pa=m.predict(Xa[cols].iloc[a],num_iteration=m.best_iteration)
    pb=m.predict(Xa[cols].iloc[b],num_iteration=m.best_iteration)
    oof_base[b]=pb
    for frac in [0.25,0.4]:
        thr_t=np.quantile(pa,1-frac); sel=a[pa>=thr_t]
        thr_b=np.quantile(pb,1-frac); mask=pb>=thr_b
        # spesialis dilatih HANYA pada baris latih di wilayah atas
        ds2=lgb.Dataset(Xa[cols].iloc[sel],y[sel]); dv2=lgb.Dataset(Xa[cols].iloc[b][mask],y[b][mask],reference=ds2)
        m2=lgb.train(dict(P,seed=11,learning_rate=0.03),ds2,4000,valid_sets=[dv2],
                     callbacks=[lgb.early_stopping(200,verbose=False)])
        ps=m2.predict(Xa[cols].iloc[b],num_iteration=m2.best_iteration)
        # skor spesialis hanya di dalam wilayah atas, diskalakan ke rentang wilayah atas
        sub=pb[mask]; lo,hi=sub.min(),sub.max()
        pr=pb.copy()
        if mask.sum()>10:
            rs=rankdata(ps[mask])/mask.sum()
            pr[mask]=lo+(hi-lo)*rs
            if frac==0.25: oof_mix[b]=pr
            if frac==0.4: oof_spec[b]=pr
        print(f'  frac={frac}: KS dalam wilayah atas base={ks_2samp(sub[y[b][mask]==1],sub[y[b][mask]==0]).statistic:.4f} '
              f'spesialis={ks_2samp(pr[mask][y[b][mask]==1],pr[mask][y[b][mask]==0]).statistic:.4f}',flush=True)
print(f'\nbase   ALL KS={ks_score(y,oof_base):.5f}')
print(f'spec40 ALL KS={ks_score(y,oof_spec):.5f}')
print(f'spec25 ALL KS={ks_score(y,oof_mix):.5f}')
# blend base + spesialis
rk=lambda v: rankdata(v)/len(v)
for w in [0.2,0.35,0.5]:
    print(f'  base+{w}*spec25 KS={ks_score(y,(1-w)*rk(oof_base)+w*rk(oof_mix)):.5f}')
