import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from factor import factor_inputs

GROUPS = {"beh": ["price", "late", "ldpd", "ab_low", "ab_outin", "ab_ilc"],
          "bu": ["bu_z", "bu_utilization", "bu_inquiries_last_6m", "bu_delinquent_accounts_last_12m"],
          "app": ["rate", "pa_rej"]}


def posterior(M, E, y, rows):
    corr = np.array([pd.Series(M[j].values[rows]).corr(pd.Series(y[rows].astype(float))) for j in M.columns])
    best = None
    for c in np.arange(0.30, 0.95, 0.05):
        a = np.clip(corr / c, -0.97, 0.97)
        den = (1 - a ** 2) + E.values
        ok = M.notna().values
        prec = 1 + np.where(ok, a ** 2 / den, 0).sum(1)
        z = np.where(ok, a * M.values / den, 0).sum(1) / prec
        auc = roc_auc_score(y[rows], z[rows])
        if best is None or auc > best[0]:
            best = (auc, z, prec, np.where(ok, M.values * (1 - a ** 2) / den, np.nan))
    return best[1:]


def factor_features(M, E, y, rows, groups=False, all_fs=False):
    z, prec, sh = posterior(M, E, y, rows)
    out = pd.DataFrame({"factor_z": z, "factor_sd": 1 / np.sqrt(prec)})
    for i, j in enumerate(M.columns):
        if all_fs or j in ("price", "late", "ldpd", "bu_z"):
            out["fs_" + j] = sh[:, i]
    if groups:
        for g, cols in GROUPS.items():
            out["factor_z_" + g] = posterior(M[cols], E[cols], y, rows)[0]
    return out
