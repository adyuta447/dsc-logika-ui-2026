import pandas as pd, numpy as np, lightgbm as lgb, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
f=pd.read_parquet('features.parquet'); tr=f[f.target.notna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values
allc=list(X.columns)
noise=['region','employment_type','housing_status','age','employment_tenure_months','dependents','tenure_to_age','income_per_dependent','loan_purpose','decision_lag_h','app_hour','app_dow']+[c for c in allc if c.startswith(('ec_share_','ec_catqty_','cs_shift','buh_'))]
lean=[c for c in allc if c not in noise]
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
def run(cols,ff=0.3,s=0,imp=False):
    p=dict(objective='binary',learning_rate=0.02,num_leaves=15,min_child_samples=200,feature_fraction=ff,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=-1,max_bin=127,seed=s)
    oof=np.zeros(len(y)); gi=pd.Series(0.0,index=cols)
    for a,b in folds:
        # importance only from the training part of each fold to keep the ranking honest
        m=lgb.train(p,lgb.Dataset(X.iloc[a][cols],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b][cols],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
        oof[b]=m.predict(X.iloc[b][cols]); gi+=m.feature_importance('gain')
    return oof,gi
ev=lambda o:(round(roc_auc_score(y,o),5),round(ks_2samp(o[y==1],o[y==0]).statistic,5))
o,gi=run(lean); print('lean',len(lean),*ev(o),flush=True)
order=gi.sort_values(ascending=False).index.tolist(); pd.Series(order).to_csv(sys.argv[1]+'imp_order.csv',index=False)
for k in [30,50,80,120]:
    for ff in [0.3,0.6]:
        print('top',k,'ff',ff,*ev(run(order[:k],ff)[0]),flush=True)
