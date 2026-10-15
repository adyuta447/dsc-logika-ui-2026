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
base=dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.3,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, max_bin=127,
          verbose=-1, deterministic=True, force_row_wise=True, num_threads=10, seed=42)

for k in [0,2]:
    a,b=folds[k]; yv=y[b]
    # (0) baseline binary
    t0=time.time()
    m=lgb.train(dict(base,objective='binary',metric='auc'),lgb.Dataset(Xa[cols].iloc[a],y[a]),6000,
                valid_sets=[lgb.Dataset(Xa[cols].iloc[b],yv)],callbacks=[lgb.early_stopping(200,verbose=False)])
    pr=m.predict(Xa[cols].iloc[b],num_iteration=m.best_iteration)
    print(f'fold{k} binary          KS={ksv(yv,pr):.5f} AUC={roc_auc_score(yv,pr):.5f} iter={m.best_iteration} {time.time()-t0:.0f}s',flush=True)
    # (A1) lambdarank, satu grup
    for obj in ['lambdarank','rank_xendcg']:
        t0=time.time()
        p=dict(base,objective=obj,metric='auc',lambdarank_truncation_level=30,label_gain=[0,1])
        ds=lgb.Dataset(Xa[cols].iloc[a],y[a],group=[len(a)])
        dv=lgb.Dataset(Xa[cols].iloc[b],yv,group=[len(b)],reference=ds)
        m=lgb.train(p,ds,4000,valid_sets=[dv],callbacks=[lgb.early_stopping(200,verbose=False)])
        pr=m.predict(Xa[cols].iloc[b],num_iteration=m.best_iteration)
        print(f'fold{k} {obj:14s} KS={ksv(yv,pr):.5f} AUC={roc_auc_score(yv,pr):.5f} iter={m.best_iteration} {time.time()-t0:.0f}s',flush=True)
    # (A2) binary dengan fokus kepala: bobot latih lebih besar pada baris berisiko tinggi
    t0=time.time()
    prev=rankdata(m.predict(Xa[cols].iloc[a]))/len(a)
    w=np.where(prev>0.7,3.0,1.0)
    m2=lgb.train(dict(base,objective='binary',metric='auc'),lgb.Dataset(Xa[cols].iloc[a],y[a],weight=w),6000,
                 valid_sets=[lgb.Dataset(Xa[cols].iloc[b],yv)],callbacks=[lgb.early_stopping(200,verbose=False)])
    pr2=m2.predict(Xa[cols].iloc[b],num_iteration=m2.best_iteration)
    print(f'fold{k} binary+headw   KS={ksv(yv,pr2):.5f} AUC={roc_auc_score(yv,pr2):.5f} iter={m2.best_iteration} {time.time()-t0:.0f}s',flush=True)
    # (B) pemurnian kepala: 2 putaran latih dengan label baru diwilayah atas
    t0=time.time(); p_top=pr.copy(); cur=np.zeros(len(yv))
    for rnd in [1,2]:
        r_prev=rankdata(p_top)/len(p_top)
        q=0.35
        thr=np.quantile(r_prev,1-q)
        top=r_prev>=thr
        # diwilayah atas, tandai sebagai positives hanya yang benar-benar default
        y_new=np.zeros_like(yv); y_new[top & (yv==1)]=1
        # latih pada data latih: mimic dengan benar-benar positives diwilayah atas prediksi
        rt=rankdata(m.predict(Xa[cols].iloc[a]))/len(a); thr_t=np.quantile(rt,1-q)
        tt=rt>=thr_t
        ya=np.zeros(len(a)); ya[tt & (y[a]==1)]=1
        m3=lgb.train(dict(base,objective='binary',metric='auc'),lgb.Dataset(Xa[cols].iloc[a],ya),4000,
                     valid_sets=[lgb.Dataset(Xa[cols].iloc[b],yv)],callbacks=[lgb.early_stopping(150,verbose=False)])
        p_top=m3.predict(Xa[cols].iloc[b],num_iteration=m3.best_iteration)
        print(f'fold{k} head-purify r{rnd} KS={ksv(yv,p_top):.5f} AUC={roc_auc_score(yv,p_top):.5f}',flush=True)
