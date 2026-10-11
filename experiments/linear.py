import pandas as pd, numpy as np, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy.stats import ks_2samp, rankdata
S=sys.argv[1]
f=pd.read_parquet('features.parquet')
tr=f[f.target.notna()].reset_index(drop=True); te=f[f.target.isna()].reset_index(drop=True)
X=tr.drop(columns=['application_id','customer_id','target']); y=tr.target.astype(int).values; Xt=te.drop(columns=['application_id','customer_id','target'])
allc=list(X.columns)
noise=['region','employment_type','housing_status','age','employment_tenure_months','dependents','tenure_to_age','income_per_dependent','loan_purpose','decision_lag_h','app_hour','app_dow']+[c for c in allc if c.startswith(('ec_share_','ec_catqty_','cs_shift','buh_'))]
num=[c for c in allc if c not in noise and str(X[c].dtype)!='category']
def prep(D,med):
    A=D[num].astype(float); M=A.isna()
    keep=[c for c in num if M[c].mean()>0.01]
    return pd.concat([A.fillna(med),M[keep].astype(float).add_suffix('_na')],axis=1)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
ks=lambda p:ks_2samp(p[y==1],p[y==0]).statistic
for C in [float(sys.argv[2])]:
    oof=np.zeros(len(y)); pt=np.zeros(len(te))
    for a,b in folds:
        med=X.iloc[a][num].median()
        A=prep(X.iloc[a],med); B=prep(X.iloc[b],med); T=prep(Xt,med)
        m=make_pipeline(QuantileTransformer(n_quantiles=200,output_distribution='uniform'),SplineTransformer(n_knots=4,degree=2),LogisticRegression(C=C,max_iter=3000))
        m.fit(A,y[a]); oof[b]=m.predict_proba(B)[:,1]; pt+=m.predict_proba(T)[:,1]/5
    print('C',C,round(roc_auc_score(y,oof),5),round(ks(oof),5),flush=True)
pd.to_pickle((oof,pt),S+f'lin_{C}.pkl')
