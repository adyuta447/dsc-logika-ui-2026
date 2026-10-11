import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
S='/private/tmp/claude-501/-Users-yutaaa-Developments-dsc-logikaui/58ebb857-fe45-4d65-a85d-951e7d9dcdc2/scratchpad/'
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values
ks=lambda p:ks_2samp(p[y==1],p[y==0]).statistic
glob=pd.read_pickle(S+'res3.pkl')['lgb0'][0]
seg=tr.rh_n.isna().values
p=dict(objective='binary',learning_rate=0.01,num_leaves=15,min_child_samples=100,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127,seed=0)
cols=[c for c in X.columns if not c.startswith(('rh_','sq_','x_price_and_late'))]
Xs=X[seg][cols]; ys=y[seg]; o=np.zeros(seg.sum())
# same fold assignment as the global model keeps the comparison fair
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
idx=np.where(seg)[0]; pos={v:i for i,v in enumerate(idx)}
for a,b in folds:
    a2=[pos[i] for i in a if seg[i]]; b2=[pos[i] for i in b if seg[i]]
    m=lgb.train(p,lgb.Dataset(Xs.iloc[a2],ys[a2]),10000,valid_sets=[lgb.Dataset(Xs.iloc[b2],ys[b2])],callbacks=[lgb.early_stopping(300,verbose=False)])
    o[b2]=m.predict(Xs.iloc[b2])
print('segment no_rh: global',roc_auc_score(ys,glob[seg]),'segment model',roc_auc_score(ys,o),'avg',roc_auc_score(ys,(o+glob[seg])/2))
for w in [0,0.5,1]:
    q=glob.copy(); q[seg]=(1-w)*glob[seg]+w*o; print('overall w',w,roc_auc_score(y,q),ks(q))
