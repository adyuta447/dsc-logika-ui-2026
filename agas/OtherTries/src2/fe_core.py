"""Temporal aggregation helpers for as-of-cutoff feature engineering (v2)."""

import numpy as np
import pandas as pd

DAY = 86400.0
SEC_SPAN = 1 << 31


def _as_seconds(s):
    s = pd.to_datetime(s)
    if s.isna().any():
        raise ValueError("null timestamps are not supported")
    return s.dt.as_unit("s").astype("int64").to_numpy().astype(np.float64)


def _prepare(events, queries, customer_col="customer_id", time_col="event_time"):
    ev = events.sort_values([customer_col, time_col], kind="mergesort").reset_index(drop=True)
    if ev.empty:
        raise ValueError("event table is empty")
    q = queries.copy()
    q["_orig"] = np.arange(len(q))
    q = q.sort_values([customer_col, "cutoff"], kind="mergesort").reset_index(drop=True)

    codes, uniques = pd.factorize(ev[customer_col], sort=True)
    lookup = pd.Series(np.arange(len(uniques)), index=uniques)
    q_codes = lookup.reindex(q[customer_col]).to_numpy()
    ev_key = codes.astype(np.int64) * SEC_SPAN + _as_seconds(ev[time_col])
    q_code_i = np.where(np.isnan(q_codes), len(uniques), q_codes).astype(np.int64)
    q_key = q_code_i * SEC_SPAN + _as_seconds(q["cutoff"])
    cust_start = np.searchsorted(ev_key, q_code_i * SEC_SPAN, side="left")
    return ev, q, ev_key, q_key, cust_start


def _bounds(ev_key, q_key, cust_start, window):
    end = np.searchsorted(ev_key, q_key, side="right")
    if window is None:
        start = cust_start.copy()
    else:
        start = np.maximum(np.searchsorted(ev_key, q_key - window * DAY, side="right"), cust_start)
    return start, end


def _reduceat(vals, start, end, how):
    out = np.full(len(start), np.nan, dtype=np.float64)
    valid = end > start
    if not valid.any():
        return out
    s, e = start[valid], end[valid]
    keep = np.zeros(vals.size, dtype=bool)
    for a, b in zip(s, e):
        keep[a:b] = True
    idx = np.flatnonzero(keep)
    rank = np.cumsum(keep) - 1
    out[valid] = getattr(np, how).reduceat(vals[idx], rank[s])
    return out


def window_stats(events, queries, value_cols, windows,
                 ops=("sum", "mean", "std", "min", "max", "last"),
                 customer_col="customer_id", time_col="event_time"):
    ev, q, ev_key, q_key, cust_start = _prepare(events, queries, customer_col, time_col)
    names, cols = [], []
    q_sec, times = _as_seconds(q["cutoff"]), _as_seconds(ev[time_col])

    for w_name, window in windows.items():
        s, e = _bounds(ev_key, q_key, cust_start, window)
        names.append(f"n_events__{w_name}")
        cols.append((e - s).astype(np.float64))

    for c in value_cols:
        matrix = ev[c].to_numpy(dtype=np.float64)
        ps = np.concatenate([[0.0], np.cumsum(matrix)])
        psq = np.concatenate([[0.0], np.cumsum(matrix ** 2)])
        for w_name, window in windows.items():
            start, end = _bounds(ev_key, q_key, cust_start, window)
            cnt = (end - start).astype(np.float64)
            tot = ps[end] - ps[start]
            mean = np.divide(tot, cnt, out=np.full_like(tot, np.nan), where=cnt > 0)
            if "sum" in ops:
                names.append(f"sum_{c}__{w_name}"); cols.append(tot)
            if "mean" in ops:
                names.append(f"mean_{c}__{w_name}"); cols.append(mean)
            if "std" in ops:
                totsq = psq[end] - psq[start]
                var = np.divide(totsq, cnt, out=np.zeros_like(tot), where=cnt > 0) - np.nan_to_num(mean) ** 2
                names.append(f"std_{c}__{w_name}")
                cols.append(np.where(cnt > 0, np.sqrt(np.clip(var, 0, None)), np.nan))
            if "min" in ops:
                names.append(f"min_{c}__{w_name}"); cols.append(_reduceat(matrix, start, end, "minimum"))
            if "max" in ops:
                names.append(f"max_{c}__{w_name}"); cols.append(_reduceat(matrix, start, end, "maximum"))
            if "last" in ops:
                last_idx = np.maximum(end - 1, 0)
                names.append(f"last_{c}__{w_name}")
                cols.append(np.where(cnt > 0, matrix[last_idx], np.nan))
                names.append(f"days_since_last_{c}__{w_name}")
                cols.append(np.where(cnt > 0, (q_sec - times[last_idx]) / DAY, np.nan))

    out = pd.DataFrame(np.column_stack(cols), columns=names)
    out.index = q["_orig"].to_numpy()
    return out.sort_index().reset_index(drop=True), names


def asof_point(events, queries, value_cols, customer_col="customer_id", time_col="event_time"):
    ev, q, _, _, _ = _prepare(events, queries, customer_col, time_col)
    left = q[[customer_col, "cutoff", "_orig"]]
    right = ev[[customer_col, time_col] + value_cols]
    merged = pd.merge_asof(
        left.sort_values("cutoff"), right.sort_values(time_col),
        left_on="cutoff", right_on=time_col, by=customer_col, direction="backward",
    ).sort_values("_orig").reset_index(drop=True)
    ts = merged[time_col]
    out = merged[value_cols].copy()
    out["days_since_ts"] = (_as_seconds(q.sort_values("_orig")["cutoff"])
                            - ts.dt.as_unit("s").astype("int64").to_numpy()) / DAY
    names = list(value_cols) + ["days_since_ts"]
    return out[names], names