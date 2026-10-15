import sys, time
sys.path.insert(0,'src2')
import numpy as np, pandas as pd
from models import *

t0=time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

X = pd.read_pickle('artifacts2/X_v2.pkl'); meta = pd.read_pickle('artifacts2/apps_meta.pkl')
is_tr = meta.target.notna().to_numpy()
Xa = X[is_tr].reset_index(drop=True); y = meta.target[is_tr].astype(int).to_numpy()
Xb = X[~is_tr].reset_index(drop=True)
cat = [c for c in X.columns if str(X[c].dtype)=='category']
Xa_np = Xa.copy()
for c in cat: Xa_np[c]=Xa_np[c].cat.codes
Xb_np = Xb.copy()
for c in cat: Xb_np[c]=Xb_np[c].cat.codes

asof = [c for c in Xa.columns if (c.startswith('rh__rh_') and any(k in c for k in ['asof','dpd_ge_','dpd_asof','pay_ratio'])) or c.startswith('rh__rha_')]
noise = [c for c in Xa.columns if c in ('app__region','app__employment_type','app__housing_status','app__age',
         'app__employment_tenure_months','app__dependents','app__tenure_to_age','app__income_per_dependent',
         'app__loan_purpose','app__decision_lag_h','app__app_hour','app__app_dow')
         or c.startswith(('ec__ec_share_','ec__ec_catqty_','ec__cs_shift','x__x_'))]
FULL = [c for c in Xa.columns if c not in asof]
LEAN = [c for c in FULL if c not in noise]
print('full',len(FULL),'lean',len(LEAN))

folds = make_folds(Xa_np, y)
res = pd.read_pickle('artifacts2/preds_dev.pkl') if __import__('os').path.exists('artifacts2/preds_dev.pkl') else {}
def run(name, cols, kind, params, seed_note=''):
    oof = np.zeros(len(Xa)); pred = np.zeros(len(Xb))
    Xtr_s = Xa_np[cols]; Xte_s = Xb_np[cols]
    for k,(a,b) in enumerate(folds):
        t1=time.time()
        if kind=='lgb':
            va,te,m = fit_lgb(Xtr_s.iloc[a], y[a], Xtr_s.iloc[b], y[b], Xte_s, params)
        elif kind=='xgb':
            va,te,m = fit_xgb(Xtr_s.iloc[a].to_numpy(dtype=np.float32), y[a],
                             Xtr_s.iloc[b].to_numpy(dtype=np.float32), y[b],
                             Xte_s.to_numpy(dtype=np.float32), params)
        else:
            va,te,m = fit_cat(Xa[cols].iloc[a], y[a], Xa[cols].iloc[b], y[b], Xb[cols], params)
        oof[b]=va; pred+=te/len(folds)
        log(f'  {name} fold{k} ks={ks_score(y[b],va):.4f} {time.time()-t1:.0f}s')
    res[name]=(oof,pred)
    pd.to_pickle(res,'artifacts2/preds_dev.pkl')
    log(f'{name}: OOF KS={ks_score(y,oof):.5f} AUC={roc_auc_score(y,oof):.5f}')
    return oof, pred

for nm, cl, pr in [('lgb_full_s0', FULL, dict(LGB_RND, num_threads=10)),
                   ('lgb_full_s1', FULL, dict(LGB_ET, num_threads=10)),
                   ('lgb_lean_s0', LEAN, dict(LGB_RND, num_threads=10))]:
    if nm not in res:
        run(nm, cl, 'lgb', pr)
if 'xgb_full' not in res:
    run('xgb_full', FULL, 'xgb', XGB_BASE)
if 'cat_full' not in res:
    run('cat_full', FULL, 'cat', CAT_BASE)

if 'linear' not in res:
  num = [c for c in LEAN if str(Xa[c].dtype) != 'category']
  flags = [c for c in num if Xa[c].isna().mean() > 0.01]
  print('linear matrix', len(num), '+', len(flags), flush=True)
  oof=np.zeros(len(Xa)); pred=np.zeros(len(Xb))
  for k,(a,b) in enumerate(folds):
      va,te,m = fit_linear(Xa[LEAN].iloc[a], y[a], Xa[LEAN].iloc[b], Xb[LEAN], num, flags)
      oof[b]=va; pred+=te/len(folds)
  res['linear']=(oof,pred)
  log(f"linear: OOF KS={ks_score(y,oof):.5f} AUC={roc_auc_score(y,oof):.5f}")
pd.to_pickle(res,'artifacts2/preds_dev.pkl')

pd.to_pickle(res,'artifacts2/preds_dev.pkl')
pd.to_pickle({'full':FULL,'lean':LEAN},'artifacts2/colsets.pkl')
log('saved')