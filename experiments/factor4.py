import numpy as np, pandas as pd
from features import D, factor_inputs as base_inputs


def counts(app):
    cut = app[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"})
    rh = pd.read_csv(D + "repayment_history.csv", parse_dates=["due_date"]).merge(cut, on="customer_id")
    rh = rh[(rh.cut - rh.due_date).dt.days >= 1]
    late = rh.assign(k=(rh.max_days_past_due > 0).astype(int)).groupby("customer_id").k.agg(["sum", "size"])
    pa = pd.read_csv(D + "previous_applications.csv", parse_dates=["decision_datetime"]).merge(cut, on="customer_id")
    pa = pa[pa.decision_datetime < pa.cut]
    rej = pa.assign(k=(pa.decision == "rejected").astype(int)).groupby("customer_id").k.agg(["sum", "size"])
    return {"late": late, "pa_rej": rej}


def factor_inputs(app, mode="elogit"):
    M, E = base_inputs(app)
    for j, kn in counts(app).items():
        k, n = kn["sum"], kn["size"]
        # empirical logit: variance stabilised estimate of each customer's own rate, with its own sampling variance
        el = np.log((k + 0.5) / (n - k + 0.5)); v = 1 / (k + 0.5) + 1 / (n - k + 0.5)
        b = el.var() - v.mean()
        M[j] = ((el - el.mean()) / np.sqrt(b)).reindex(app.customer_id).values
        E[j] = (v / b).reindex(app.customer_id).values
    return M, E
