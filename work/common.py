import numpy as np, pandas as pd
from scipy.stats import rankdata
rank = lambda v: rankdata(v) / len(v)
def variants(r, i):
    res = pd.read_pickle(f"{r}/models/predictions.pkl")
    pick = lambda pre: np.mean([rank(res[k][i]) for k in res if k.startswith(pre)], 0)
    c = {f"{m}_{t}": pick(f"{m}_{t}") for m in ("lgb", "cat", "xgb") for t in ("full", "lean", "top")}
    lin = rank(res["linear"][i]); tr = {t: (c[f"lgb_{t}"] + c[f"cat_{t}"] + c[f"xgb_{t}"]) / 3 for t in ("full", "lean", "top")}
    core = (tr["full"] + tr["lean"] + c["cat_full"] + c["cat_lean"]) / 4
    return {"top_mix": 0.8 * (core + tr["top"]) / 2 + 0.2 * lin, "trees_only": (core + tr["top"]) / 2, "linear30": 0.7 * (core + tr["top"]) / 2 + 0.3 * lin,
            "main": 0.8 * core + 0.2 * lin, "top_only": 0.8 * tr["top"] + 0.2 * lin, "lgb_only": (c["lgb_full"] + c["lgb_lean"] + c["lgb_top"]) / 3,
            "cat_only": (c["cat_full"] + c["cat_lean"] + c["cat_top"]) / 3, "xgb_only": (c["xgb_full"] + c["xgb_lean"] + c["xgb_top"]) / 3,
            "full_only": tr["full"], "lean_only": tr["lean"], "linear_only": lin, "lgb_full": c["lgb_full"], "cat_full": c["cat_full"], "xgb_full": c["xgb_full"]}
