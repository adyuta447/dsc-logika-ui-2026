import pandas as pd, numpy as np, sys
import statsmodels.api as sm
S=sys.argv[1]; C=pd.read_parquet(S+'cand.parquet'); r=pd.read_pickle(S+'res4.pkl')
p=np.mean([r[k][0] for k in ['lgb0','lgb1','lgb2','xgb']],axis=0).clip(1e-4,1-1e-4)
y=pd.read_parquet('features.parquet').query('target.notna()').target.values
lo=np.log(p/(1-p)); n=len(y); Ct=C.iloc[:n]
base=sm.Logit(y,sm.add_constant(lo)).fit(disp=0).llf
rows=[]
for c in C.columns[2:]:
    v=Ct[c].astype(float); na=v.isna().astype(float); v=(v-v.mean())/(v.std()+1e-12); v=v.fillna(0).clip(-5,5)
    X=np.column_stack([np.ones(n),lo,v,na]) if na.mean()>0.01 else np.column_stack([np.ones(n),lo,v])
    try: m=sm.Logit(y,X).fit(disp=0,method="bfgs",maxiter=200)
    except Exception: continue
    rows.append((c,round(2*(m.llf-base),1),round(m.tvalues[2],2)))
print(pd.DataFrame(rows,columns=['f','LR','z']).sort_values('LR',ascending=False).head(25).to_string())
