import pandas as pd, numpy as np, sys, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from scipy.stats import ks_2samp
S,mode=sys.argv[1],sys.argv[2]
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True); te=f[f.target.isna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values; Xt=te.drop(columns=['application_id','customer_id','target'])
allc=list(X.columns)
noise=['region','employment_type','housing_status','age','employment_tenure_months','dependents','tenure_to_age','income_per_dependent','loan_purpose','decision_lag_h','app_hour','app_dow']+[c for c in allc if c.startswith(('ec_share_','ec_catqty_','cs_shift','buh_'))]
lean=[c for c in allc if c not in noise]
num=[c for c in lean if str(X[c].dtype)!='category']; flag=[c for c in num if X[c].isna().mean()>0.01]
mat=lambda D,med:pd.concat([D[num].astype(float).fillna(med),D[flag].isna().astype(float).add_suffix('_na')],axis=1)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
oof=np.zeros(len(y)); pt=np.zeros(len(te))
for k,(a,b) in enumerate(folds):
    if mode.startswith('mlp'):
        med=X.iloc[a][num].median(); preds_b=[]; preds_t=[]
        for s in range(3):
            m=make_pipeline(QuantileTransformer(n_quantiles=200,output_distribution='normal',subsample=10**6,random_state=0),
                MLPClassifier(hidden_layer_sizes=(64,32),alpha=float(mode[3:]),batch_size=512,learning_rate_init=1e-3,early_stopping=True,validation_fraction=0.1,n_iter_no_change=8,max_iter=200,random_state=s))
            m.fit(mat(X.iloc[a],med),y[a]); preds_b.append(m.predict_proba(mat(X.iloc[b],med))[:,1]); preds_t.append(m.predict_proba(mat(Xt,med))[:,1])
        oof[b]=np.mean(preds_b,0); pt+=np.mean(preds_t,0)/5
    elif mode=='spline8':
        med=X.iloc[a][num].median()
        m=make_pipeline(QuantileTransformer(n_quantiles=200,subsample=10**6,random_state=0),SplineTransformer(n_knots=8,degree=3),LogisticRegression(C=0.01,max_iter=4000))
        m.fit(mat(X.iloc[a],med),y[a]); oof[b]=m.predict_proba(mat(X.iloc[b],med))[:,1]; pt+=m.predict_proba(mat(Xt,med))[:,1]/5
    elif mode=='lgbd2':
        p=dict(objective='binary',learning_rate=0.02,max_depth=2,num_leaves=4,min_child_samples=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10,verbose=-1,n_jobs=4,max_bin=127,seed=0)
        m=lgb.train(p,lgb.Dataset(X.iloc[a][lean],y[a]),20000,valid_sets=[lgb.Dataset(X.iloc[b][lean],y[b])],callbacks=[lgb.early_stopping(400,verbose=False)])
        oof[b]=m.predict(X.iloc[b][lean]); pt+=m.predict(Xt[lean])/5
    print(mode,'fold',k,flush=True)
print(mode,'AUC',round(roc_auc_score(y,oof),5),'KS',round(ks_2samp(oof[y==1],oof[y==0]).statistic,5),flush=True)
pd.to_pickle((oof,pt),S+f'new_{mode}.pkl')
