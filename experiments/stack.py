import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata, ks_2samp
S=sys.argv[1]
f=pd.read_parquet('features.parquet').merge(pd.read_parquet(S+'cand.parquet').drop(columns='customer_id'),on='application_id',how='left')
n=int(f.target.notna().sum()); y=f.target.values[:n].astype(int)
R={f'r{i}':pd.read_pickle(S+f'res{i}.pkl') for i in [2,3,4]}; lin=pd.read_pickle(S+'lin_0.03.pkl')
rk=lambda v:rankdata(v)/len(v)
ens=lambda R,i:(np.mean([rk(R[f'lgb{s}'][i]) for s in range(3)],0)+rk(R['cat'][i])+rk(R['xgb'][i]))/3
T=lambda i:(ens(R['r3'],i)+ens(R['r4'],i)+rk(R['r3']['cat'][i])+rk(R['r4']['cat'][i])+rk(R['r2']['cat8'][i]))/5
bl=lambda i:0.8*T(i)+0.2*rk(lin[i])
ks=lambda p:ks_2samp(p[y==1],p[y==0]).statistic
from scipy.special import logit
def design(i,feats):
    s=bl(i); s=rk(s).clip(1e-4,1-1e-4); cols=[logit(s)]
    part=f.iloc[:n] if i==0 else f.iloc[n:]
    for c in feats:
        v=part[c].astype(float).values; na=np.isnan(v)
        # rank against the training distribution so train and test share one scale
        ref=np.sort(f.iloc[:n][c].dropna().values); q=np.searchsorted(ref,v)/len(ref); q[na]=0.5
        cols+=[q,na.astype(float)]
    return np.column_stack(cols)
b0=bl(0); print('blend',round(roc_auc_score(y,b0),5),round(ks(b0),5))
sets={'rh':['rh_rate1_smooth','rh_rate15_smooth','rh_rate60_smooth','rh_maxdpd_mean'],
      'rh+inc':['rh_rate1_smooth','rh_rate15_smooth','rh_rate60_smooth','rh_maxdpd_mean','x_inc_share','b_nonincome'],
      'rh+inc+ec':['rh_rate1_smooth','rh_rate15_smooth','rh_rate60_smooth','rh_maxdpd_mean','x_inc_share','b_nonincome','ec_price_rel','x_price_and_late']}
for name,fe in sets.items():
    X=design(0,fe); oof=np.zeros(n)
    for a,b in StratifiedKFold(5,shuffle=True,random_state=7).split(X,y):
        oof[b]=LogisticRegression(C=1.0,max_iter=2000).fit(X[a],y[a]).predict_proba(X[b])[:,1]
    print(name,round(roc_auc_score(y,oof),5),round(ks(oof),5),flush=True)
