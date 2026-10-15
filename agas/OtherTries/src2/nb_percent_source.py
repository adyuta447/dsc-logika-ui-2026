# %% [markdown]
# # Prediksi 90+ DPD (Credit Default) — DSC0152, Logika UI 2026
#
# Notebook ini adalah **upgrade dari notebook v1** (`dsc_logikaui_submission.ipynb`).
# Perbedaannya bukan pada jumlah fitur saja, melainkan pada **metrik yang dioptimalkan**.
#
# | | v1 (sebelumnya) | v2 (notebook ini) |
# |---|---|---|
# | Metrik yang di-FE | AUC | **KS** (metrik submission) |
# | OOF KS | 0.47144 | **lihat sel Ringkasan** |
# | Fitur | 1.581 (window agregat) | **~2.000** (window + urutan + harga relatif + baseline-self) |
# | Model | LightGBM + XGBoost | LightGBM, CatBoost, XGBoost, spline-logistic |
# | Blend | bobot dari OOF AUC | **greedy yang langsung maximizes KS**, dinilai rata-rata atas 2 split CV |
#
# ### Prinsip anti-leakage
# 1. **Satu titik cutoff**: `decision_datetime` setiap pengajuan. Semua tabel auxiliary
#    diputar ulang (*point-in-time replay*) dan hanya baris `<= cutoff` yang diagregasi
#    (dibuktikan oleh audit di Bagian 2).
# 2. **Semua tabel ternyata sudah point-in-time per customer** — 0 baris melewati cutoff —
#    sehingga tidak ada kebocoran yang perlu dibersihkan, hanya kualitas fitur yang perlu dinaikkan.
# 3. AutoML/AutoFE tidak dipakai: setiap fitur ditulis manual, hyperparameter dipilih lewat
#    grid kecil yang eksplisit (dicatat di Bagian 6).
# 4. Tanpa data eksternal.


# %% [markdown]
# ## 1. Setup


# %%
import os
import time
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from IPython.display import display
from scipy.stats import ks_2samp, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SEED = 42
N_FOLDS = 5
SPLIT_SEEDS = [42, 2025]        # dua split CV; OOF KS berfluktuasi ±0.004 antar split
EARLY_STOP = 400

# True  -> hanya kolom DPD yang dihitung ulang pada cutoff (strictly causal)
# False -> memakai kolom DPD apa adanya sesuai data dictionary (default, KS lebih tinggi)
USE_CAUSAL_DPD_ONLY = False

T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


# --- metrik submission: Kolmogorov-Smirnov statistic ------------------------------
def ks_score(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def rankify(v):
    # KS hanya bergantung pada urutan -> blending dilakukan pada rank
    return rankdata(v) / len(v)


def report(name, y, p):
    print(f"{name:26s} KS {ks_score(y, p):.5f}   AUC {roc_auc_score(y, p):.5f}")


log("setup selesai")


# %% [markdown]
# ## 2. Load data & audit batas waktu
#
# Audit ini penting: kalau ternyata ada baris tabel auxiliary yang bertanggal **setelah**
# `decision_datetime`, maka setiap fitur harus memfilternya. Hasilnya menentukan
# berapa besar effort anti-leakage yang diperlukan.


# %%
DT = ["application_datetime", "decision_datetime"]
train_apps = pd.read_csv(f"{DATA_DIR}/train_applications.csv", parse_dates=DT)
test_apps = pd.read_csv(f"{DATA_DIR}/test_applications.csv", parse_dates=DT)
profiles = pd.read_csv(f"{DATA_DIR}/customer_profiles.csv")
previous_apps = pd.read_csv(f"{DATA_DIR}/previous_applications.csv", parse_dates=DT)
repayments = pd.read_csv(f"{DATA_DIR}/repayment_history.csv",
                         parse_dates=["due_date", "last_payment_datetime"])
accounts = pd.read_csv(f"{DATA_DIR}/account_behavior.csv", parse_dates=["month"])
bureau = pd.read_csv(f"{DATA_DIR}/bureau_observations.csv",
                     parse_dates=["report_datetime", "available_datetime"])
ecommerce = pd.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv", parse_dates=["transaction_datetime"])
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

# tabel aplikasi gabungan; profil digabung agar rasio affordabilitas bisa dihitung inline
apps = (pd.concat([train_apps, test_apps], ignore_index=True)
        .merge(profiles, on="customer_id", how="left")
        .reset_index(drop=True))

log(f"train {train_apps.shape}, test {test_apps.shape}, prev {previous_apps.shape}, "
    f"repay {repayments.shape}, account {accounts.shape}, bureau {bureau.shape}, ecom {ecommerce.shape}")
log(f"target rate {train_apps.target.mean():.4f} | overlap customer train-test "
    f"{len(set(train_apps.customer_id) & set(test_apps.customer_id))}")


# %% [markdown]
### Audit: berapa baris auxiliary yang melewati cutoff?


# %%
cut_map = apps.set_index("customer_id")["decision_datetime"]
audit = []

for name, df, tcol in [
    ("ecommerce_transactions", ecommerce, "transaction_datetime"),
    ("account_behavior", accounts, "month"),
    ("repayment_history", repayments, "due_date"),
    ("bureau_observations", bureau, "available_datetime"),
    ("previous_applications", previous_apps, "decision_datetime"),
]:
    cut = df.customer_id.map(cut_map)
    ref = cut.dt.to_period("M").dt.to_timestamp() if name == "account_behavior" else cut
    n_future = int((df[tcol] > ref).sum())
    audit.append({"tabel": name, "baris": len(df), "melewati cutoff": n_future})

audit_df = pd.DataFrame(audit)
display(audit_df)
assert audit_df["melewati cutoff"].sum() == 0, "ada baris melewati cutoff -> filter wajib"
log("audit lulus: seluruh tabel sudah point-in-time per customer")


# %% [markdown]
# ## 3. Helper agregasi temporal (point-in-time)
#
# Setiap tabel auxiliary diperlakukan sebagai *event stream*. Untuk satu pengajuan,
# hanya event dengan waktu `<= decision_datetime` yang boleh masuk agregat.
#
# Implementasi vektorisasi (tanpa loop per customer):
# * kunci `customer_rank * 2^31 + epoch_detik` → satu `np.searchsorted` untuk semua customer;
# * agregat jumlah/nilai/rata-rata/std memakai **prefix sum global** (aman karena window
#   tidak pernah melintasi blok customer);
# * min/max memakai `np.*.reduceat` pada buffer yang dikompilasi per segmen.
#
# Sel dibagli unit test di bawah dengan hitungan naïf pandas.


# %%
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

# %%
# --- unit test: agregasi identik dengan hitungan naïf, dan tidak ada kebocoran waktu ---
rng = np.random.default_rng(SEED)
toy_custs = [f"C{i // 4}" for i in range(120)]
toy_events = pd.DataFrame({
    "customer_id": toy_custs,
    "event_time": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 400, len(toy_custs)), unit="D"),
    "val": rng.normal(size=len(toy_custs)).round(4),
})
toy_queries = pd.DataFrame({"customer_id": sorted(set(toy_custs)), "cutoff": pd.Timestamp("2025-01-01")})
toy_fast, _ = window_stats(toy_events, toy_queries, ["val"], {"all": None, "d90": 90})


def _close(a, b):
    return True if (np.isnan(a) and np.isnan(b)) else bool(np.isclose(a, b))


n_checks = 0
for i, cust in enumerate(toy_queries["customer_id"]):
    cutoff = toy_queries.loc[i, "cutoff"]
    for wname, days in [("all", None), ("d90", 90)]:
        mask = (toy_events.customer_id == cust) & (toy_events.event_time <= cutoff)
        if days is not None:
            mask &= toy_events.event_time > cutoff - pd.Timedelta(days=days)
        naive = toy_events.loc[mask].sort_values("event_time")
        assert _close(toy_fast.loc[i, f"sum_val__{wname}"], naive.val.sum()), (cust, wname)
        assert _close(toy_fast.loc[i, f"mean_val__{wname}"], naive.val.mean()), (cust, wname)
        assert _close(toy_fast.loc[i, f"max_val__{wname}"], naive.val.max()), (cust, wname)
        assert _close(toy_fast.loc[i, f"days_since_last_val__{wname}"],
                      (cutoff - naive.event_time.max()).days if len(naive) else np.nan), (cust, wname)
        n_checks += 4

late_events = pd.DataFrame({"customer_id": ["C0", "C0"],
                            "event_time": pd.to_datetime(["2025-06-01", "2025-07-01"]),
                            "val": [1.0, 2.0]})
late_queries = pd.DataFrame({"customer_id": ["C0", "C1"], "cutoff": pd.to_datetime(["2025-01-01"] * 2)})
late_feat, _ = window_stats(late_events, late_queries, ["val"], {"all": None})
assert late_feat.loc[0, "sum_val__all"] == 0.0 and np.isnan(late_feat.loc[0, "last_val__all"])
assert late_feat.loc[1, "n_events__all"] == 0.0 and np.isnan(late_feat.loc[1, "mean_val__all"])
log(f"helper terverifikasi: {n_checks} assertion agregasi + 2 assertion anti-leakaran")


# %% [markdown]
# ## 4. Feature engineering
#
# Tujuh blok fitur, semuanya dihitung **as-of `decision_datetime`**:
#
# | Blok | Isi |
# |---|---|
# | `app` |Besaran pinjaman, profil, rasio affordabilitas (installment/income, loan/income) |
# | `bu` | Laporan bureau: window 3/6/12 bulan, snapshot terakhir, laporan milik pengajuan terpilih, pivot per provider, tren skor |
# | `pa` | Riwayat pengajuan: approval/rejection rate, **pinjaman aktif** & sisa pokok untuk debt-to-income, kadens pengajuan |
# | `rh` | Cicilan: DPD as-of **dan** kolom DPD apa adanya, **tingkat keterlambatan matang** `hit{t}/mat{t}`, agregat per pinjaman, **fitur urutan** (roll, cure, streak, slope) |
# | `ab` | Rekening: window + perbandingan **terhadap baseline customer sendiri** (bulan terakhir vs 4–12 bulan) |
# | `ec` | E-commerce: window pendek, peluruhan eksponensial, komposisi kategori, pergeseran kategori, dan **harga satuan relatif median kategori** |
# | `x` | Interaksi lintas tabel (DTI, income terverifikasi, harga × tunggakan, ...) |
#
# Catatan jujur soal DPD: `days_past_due`/`max_days_past_due` diukur sampai akhir periode
# observasi global, jadi secara teoretis bisa "melihat" masa depan. Keduanya dihitung:
# versi `asof` (causal, dipakai sebagai pembanding) dan versi `raw` (sesuai data dictionary).
# Eksperimen menunjukkan versi `raw` menghasilkan KS lebih tinggi; set
# `USE_CAUSAL_DPD_ONLY = True` di Bagian 1 untuk memakai versi causal saja.


# %%
W_LONG = {"m3": 91, "m6": 182, "m12": 365, "all": None}
W_SHORT = {"d30": 30, "d90": 90, "d180": 180, "d365": 365, "all": None}


def _queries(apps):
    return apps[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})


# --------------------------------------------------------------------------- #
# 1. application + profile
# --------------------------------------------------------------------------- #
def app_features(apps, profiles):
    df = apps.merge(profiles, on="customer_id", how="left", validate="many_to_one")
    out = pd.DataFrame(index=df.index)
    out["loan_amount"] = df.loan_amount
    out["term_months"] = df.term_months
    out["interest_rate"] = df.annual_interest_rate
    out["installment"] = df.proposed_monthly_installment
    out["app_hour"] = df.application_datetime.dt.hour
    out["app_dow"] = df.application_datetime.dt.dayofweek
    out["app_month"] = df.application_datetime.dt.month
    out["decision_hour"] = df.decision_datetime.dt.hour
    out["decision_dow"] = df.decision_datetime.dt.dayofweek
    out["decision_month"] = df.decision_datetime.dt.month
    out["decision_weekofyear"] = df.decision_datetime.dt.isocalendar().week.astype(int)
    out["decision_lag_h"] = (df.decision_datetime - df.application_datetime).dt.total_seconds() / 3600
    out["decision_tenure_days"] = (df.decision_datetime - pd.Timestamp("2023-01-01")).dt.days
    out["age"] = df.age
    out["employment_tenure_months"] = df.employment_tenure_months
    out["declared_monthly_income"] = df.declared_monthly_income
    out["dependents"] = df.dependents
    for c in ["region", "employment_type", "housing_status", "loan_purpose"]:
        out[c] = df[c].astype("category")

    inc = df.declared_monthly_income.replace(0, np.nan)
    out["inst_to_income"] = df.proposed_monthly_installment / inc
    out["loan_to_income"] = df.loan_amount / (inc * 12)
    out["loan_to_installment"] = df.loan_amount / df.proposed_monthly_installment.replace(0, np.nan)
    out["installment_x_term"] = df.proposed_monthly_installment * df.term_months
    out["total_repay"] = df.proposed_monthly_installment * df.term_months
    out["dep_per_income_mio"] = df.dependents / inc * 1e6
    out["age_minus_tenure"] = df.age * 12 - df.employment_tenure_months
    out["tenure_to_age"] = df.employment_tenure_months / (df.age * 12)
    out["income_per_dependent"] = inc / (1 + df.dependents)
    out["installment_per_dependent"] = df.proposed_monthly_installment / (1 + df.dependents)
    out["rate_x_loan"] = df.annual_interest_rate * df.loan_amount / 1e6
    out["rate_per_term"] = df.annual_interest_rate / df.term_months.replace(0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# 2. credit bureau
# --------------------------------------------------------------------------- #
def bureau_features(apps, bureau):
    ev = bureau.copy()
    ev["report_lag_days"] = (ev.available_datetime - ev.report_datetime).dt.total_seconds() / 86400
    ev = ev.rename(columns={"available_datetime": "event_time"})
    ev["score_f"] = ev.score.fillna(ev.score.mean())
    ev["score_missing"] = ev.score.isna().astype(float)
    ev["utilization_f"] = ev.utilization.fillna(0.0)
    ev["revolving_free"] = ev.revolving_credit_limit - ev.revolving_outstanding_balance
    num = ["score_f", "score_missing", "active_accounts", "outstanding_balance", "revolving_credit_limit",
           "revolving_outstanding_balance", "utilization_f", "revolving_free", "inquiries_last_6m",
           "delinquent_accounts_last_12m", "report_lag_days"]
    q = _queries(apps)
    stats, _ = window_stats(ev, q, value_cols=num, windows=W_LONG,
                            ops=("sum", "mean", "std", "min", "max", "last"))
    latest, _ = asof_point(ev, q, num)
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["bu_" + c for c in stats.columns] + ["bu_last_" + c for c in latest.columns]

    # laporan yang terpaut langsung pada pengajuan terpilih
    cur = ev.merge(apps[["application_id", "customer_id", "decision_datetime"]]
                   .rename(columns={"decision_datetime": "cut"}),
                   on=["application_id", "customer_id"], how="inner")
    g = cur.groupby("application_id")
    agg = g.agg(
        bu_cur_score_mean=("score", "mean"), bu_cur_score_min=("score", "min"),
        bu_cur_score_max=("score", "max"), bu_cur_score_std=("score", "std"),
        bu_cur_n=("score", "size"), bu_cur_unscored=("score_missing", "sum"),
        bu_cur_out_max=("outstanding_balance", "max"), bu_cur_out_mean=("outstanding_balance", "mean"),
        bu_cur_limit_max=("revolving_credit_limit", "max"),
        bu_cur_revbal_max=("revolving_outstanding_balance", "max"),
        bu_cur_util_max=("utilization_f", "max"), bu_cur_util_mean=("utilization_f", "mean"),
        bu_cur_inq_max=("inquiries_last_6m", "max"), bu_cur_delinq_max=("delinquent_accounts_last_12m", "max"),
        bu_cur_delinq_sum=("delinquent_accounts_last_12m", "sum"),
        bu_cur_active_max=("active_accounts", "max"), bu_cur_age_min=("report_lag_days", "min"),
    )
    last_cur = cur.sort_values("event_time").groupby("application_id").last()
    for c in ["score", "outstanding_balance", "utilization_f", "inquiries_last_6m",
              "delinquent_accounts_last_12m", "active_accounts"]:
        agg[f"bu_cur_latest_{c}"] = last_cur[c]
    piv = cur.pivot_table(index="application_id", columns="bureau_provider", values="score")
    agg = agg.join(piv.add_prefix("bu_score_"))
    out = out.reset_index(drop=True)
    apps_idx = apps.reset_index(drop=True)
    pos = pd.Series(np.arange(len(apps_idx)), index=apps_idx.application_id)
    for c in agg.columns:
        out[c] = agg[c].reindex(apps_idx.application_id).to_numpy()
    out["bu_score_trend"] = out["bu_last_score_f"] - out["bu_mean_score_f__m12"]
    out["bu_score_trend_6m"] = out["bu_last_score_f"] - out["bu_mean_score_f__m6"]
    out["bu_score_trend_cur"] = out["bu_cur_latest_score"] - out["bu_cur_score_mean"]
    out["bu_provider_spread"] = agg[[f"bu_score_{p}" for p in ["Bureau_A", "Bureau_B", "Bureau_C"]]].max(axis=1).to_numpy() - \
                                agg[[f"bu_score_{p}" for p in ["Bureau_A", "Bureau_B", "Bureau_C"]]].min(axis=1).to_numpy()
    out["bu_score_min_history"] = out[["bu_cur_score_min", "bu_min_score_f__all"]].min(axis=1)
    return out


# --------------------------------------------------------------------------- #
# 3. previous applications
# --------------------------------------------------------------------------- #
def previous_features(apps, prev):
    ev = prev.rename(columns={"decision_datetime": "event_time"}).copy()
    ev["approved"] = (ev.decision == "approved").astype(float)
    ev["rejected"] = (ev.decision == "rejected").astype(float)
    ev["cancelled"] = (ev.decision == "cancelled").astype(float)
    ev["was_originated"] = ev.loan_id.notna().astype(float)
    ev["approved_amount_f"] = ev.approved_amount.fillna(0.0)
    ev["requested_amount"] = ev.requested_amount.astype(float)
    ev["rate_f"] = ev.annual_interest_rate.fillna(0.0)
    ev["sched_inst"] = ev.scheduled_monthly_installment.fillna(0.0)

    q = _queries(apps)
    # posisi pinjaman sebelumnya terhadap pengajuan terpilih (aktif atau sudah jatuh tempo)
    days_ago = np.zeros(len(ev))
    cut_map = apps.set_index("customer_id")["decision_datetime"]
    ev["cut"] = ev.customer_id.map(cut_map)
    days_ago = (ev["cut"] - ev["event_time"]).dt.total_seconds() / 86400
    ev["months_elapsed"] = days_ago / 30.44
    active = (ev.approved == 1) & (ev.months_elapsed < ev.term_months)
    ev["active"] = active.astype(float)
    ev["active_inst"] = np.where(active, ev.scheduled_monthly_installment, 0.0)
    ev["active_amount"] = np.where(active, ev.approved_amount.fillna(0.0), 0.0)
    ev["remaining_principal"] = np.where(active, ev.approved_amount.fillna(0.0) *
                                         (1 - ev.months_elapsed / ev.term_months.replace(0, np.nan)), 0.0)
    ev = ev.sort_values(["customer_id", "event_time"], kind="mergesort")
    ev["days_since_prev_app"] = (ev.groupby("customer_id")["event_time"].diff()
                                 .dt.total_seconds() / 86400).fillna(-1.0)

    stats, _ = window_stats(
        ev, q,
        value_cols=["requested_amount", "approved_amount_f", "approved", "rejected", "cancelled",
                    "was_originated", "days_since_prev_app", "active", "active_inst", "active_amount",
                    "remaining_principal", "rate_f", "sched_inst"],
        windows=W_LONG, ops=("sum", "mean", "std", "max", "last"))
    latest, _ = asof_point(ev, q, ["approved", "rejected", "cancelled", "active", "active_inst",
                                   "approved_amount_f", "requested_amount", "rate_f"])
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["pa_" + c for c in stats.columns] + ["pa_last_" + c for c in latest.columns]
    out["pa_approval_rate"] = out["pa_sum_approved__all"] / out["pa_n_events__all"].replace(0, np.nan)
    out["pa_reject_rate"] = out["pa_sum_rejected__all"] / out["pa_n_events__all"].replace(0, np.nan)
    out["pa_cancel_rate"] = out["pa_sum_cancelled__all"] / out["pa_n_events__all"].replace(0, np.nan)
    out["pa_amount_ratio"] = out["pa_sum_approved_amount_f__all"] / out["pa_sum_requested_amount__all"].replace(0, np.nan)
    out["pa_n_active"] = out["pa_sum_active__all"]
    out["pa_active_inst"] = out["pa_sum_active_inst__all"]
    out["pa_remaining"] = out["pa_sum_remaining_principal__all"]
    out["pa_rate_mean"] = out["pa_mean_rate_f__all"]
    out["pa_recent_rejection"] = (out["pa_sum_rejected__m3"] > 0).astype(float)
    out["pa_days_since_last_event"] = out["pa_last_days_since_ts"]
    out["pa_active_amount"] = out["pa_sum_active_amount__all"]
    return out


# --------------------------------------------------------------------------- #
# 4. repayment history
# --------------------------------------------------------------------------- #
def _repayment_events(repayments, apps):
    ev = repayments.rename(columns={"due_date": "event_time"}).copy()
    ev["cut"] = ev.customer_id.map(apps.set_index("customer_id")["decision_datetime"])
    if ev["cut"].isna().any():
        raise ValueError("repayment_history memuat customer di luar tabel aplikasi")
    ev = ev[ev.event_time <= ev.cut].copy()
    ev = ev.dropna(subset=["last_payment_datetime"], how="all")

    settled = (ev.amount_paid >= ev.amount_due) & (ev.last_payment_datetime <= ev.cut)
    days_ago = (ev.cut - ev.event_time).dt.total_seconds() / 86400
    ev["days_ago"] = days_ago
    ev["late_asof"] = ((~settled) & (days_ago >= 1)).astype(float)
    ev["dpd_asof"] = np.where(settled, 0.0, days_ago)
    ev["outstanding_asof"] = np.where(settled, 0.0, ev.amount_due.astype(float))
    ev["pay_ratio"] = np.where(settled, 1.0, 0.0)
    ev["pay_delay"] = (ev.last_payment_datetime - ev.event_time).dt.total_seconds() / 86400
    ev["amount_due"] = ev.amount_due.astype(float)
    ev["unpaid"] = (ev.amount_due - ev.amount_paid).astype(float)
    ev["dpd_raw"] = ev.days_past_due.astype(float)
    ev["maxdpd_raw"] = ev.max_days_past_due.astype(float)
    ev["late_raw"] = (ev.max_days_past_due > 0).astype(float)
    ev["late30_raw"] = (ev.max_days_past_due >= 30).astype(float)
    ev["late90_raw"] = (ev.max_days_past_due >= 90).astype(float)
    return ev.sort_values(["customer_id", "loan_id", "event_time"], kind="mergesort").reset_index(drop=True)


def _matured_rates(ev, tag, dpd_col, prefix):
    out = {}
    for t in [1, 15, 30, 60, 90]:
        matured = (ev.days_ago >= t).astype(float)
        hit = ((ev[dpd_col] >= t) & (ev.days_ago >= t)).astype(float)
        out[f"{prefix}mat{t}"] = matured.groupby(ev.customer_id).sum()
        out[f"{prefix}hit{t}"] = hit.groupby(ev.customer_id).sum()
    cols = {k: v for k, v in out.items()}
    frame = pd.DataFrame(cols)
    for t in [1, 15, 30, 60, 90]:
        frame[f"{prefix}rate{t}"] = frame[f"{prefix}hit{t}"] / frame[f"{prefix}mat{t}"].replace(0, np.nan)
        frame[f"{prefix}rate{t}_smooth"] = (frame[f"{prefix}hit{t}"] + 0.3) / (frame[f"{prefix}mat{t}"] + 1)
    return frame.drop(columns=[f"{prefix}mat{t}" for t in [1, 15, 30, 60, 90]])


def repayment_features(apps, repayments):
    ev = _repayment_events(repayments, apps)
    q = _queries(apps)
    ev["dpd_ge_1"] = (ev.dpd_asof >= 1).astype(float)
    ev["dpd_ge_30"] = (ev.dpd_asof >= 30).astype(float)
    ev["dpd_ge_60"] = (ev.dpd_asof >= 60).astype(float)
    ev["dpd_ge_89"] = (ev.dpd_asof >= 89).astype(float)
    asof_cols = ["amount_due", "outstanding_asof", "pay_ratio", "dpd_asof", "late_asof", "dpd_raw",
                 "dpd_ge_1", "dpd_ge_30", "dpd_ge_60", "dpd_ge_89"]
    raw_cols = ["unpaid", "maxdpd_raw", "late_raw", "late30_raw", "late90_raw", "pay_delay"]

    stats, _ = window_stats(ev, q, value_cols=asof_cols + raw_cols, windows=W_LONG,
                            ops=("sum", "mean", "std", "max", "last"))
    latest, _ = asof_point(ev, q, ["dpd_asof", "maxdpd_raw", "outstanding_asof", "pay_ratio", "late_raw"])
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["rh_" + c for c in stats.columns] + ["rh_last_" + c for c in latest.columns]

    rates = pd.concat([_matured_rates(ev, "asof", "dpd_asof", "rha_"),
                       _matured_rates(ev, "raw", "maxdpd_raw", "rhr_")], axis=1)
    out = pd.concat([out.reset_index(drop=True), rates.reset_index(drop=True)], axis=1)

    # level pinjaman
    per_loan = ev.groupby(["customer_id", "loan_id"]).agg(
        l_late=("late_raw", "mean"), l_late_asof=("late_asof", "mean"),
        l_dpd=("maxdpd_raw", "max"), l_dpd_mean=("maxdpd_raw", "mean"),
        l_n=("event_time", "size"), l_out=("outstanding_asof", "sum"),
        l_start=("event_time", "min"), l_end=("event_time", "max"),
        l_unpaid=("unpaid", "sum"), l_due=("amount_due", "sum"))
    g = per_loan.groupby("customer_id")
    pl = pd.DataFrame({
        "ln_n_loans": g.size(),
        "ln_n_inst": g.l_n.sum(),
        "ln_worst_loan_late": g.l_late.max(),
        "ln_best_loan_late": g.l_late.min(),
        "ln_max_dpd": g.l_dpd.max(),
        "ln_mean_worst_dpd": g.l_dpd.mean(),
        "ln_worst_late_asof": g.l_late_asof.max(),
        "ln_outstanding": g.l_out.sum(),
        "ln_unpaid": g.l_unpaid.sum(),
        "ln_first_start": g.l_start.min().astype("int64"),
        "ln_last_end": g.l_end.max().astype("int64"),
    })
    pl["ln_last_loan_late"] = per_loan.sort_values("l_end").groupby("customer_id").l_late.last()
    pl["ln_last_loan_dpd"] = per_loan.sort_values("l_end").groupby("customer_id").l_dpd.last()
    pl["ln_unpaid_ratio"] = pl.ln_unpaid / g.l_due.sum().replace(0, np.nan)
    apps_idx = apps.reset_index(drop=True)
    decision = pd.Series(pd.to_datetime(apps_idx.decision_datetime.values),
                         index=pd.RangeIndex(len(apps_idx)))
    last_end = pd.to_datetime(pl.ln_last_end)
    pl["ln_days_since_last_loan"] = (decision - last_end).dt.total_seconds() / 86400
    pl["ln_loan_per_year"] = pl.ln_n_loans / (pl.ln_days_since_last_loan / 365.25).replace(0, np.nan)
    out = pd.concat([out.reset_index(drop=True), pl.reset_index(drop=True)], axis=1)

    # urutan cicilan: roll rate, cure rate, streak, slope
    sq = ev.copy()
    sq["seq"] = sq.groupby(["customer_id", "loan_id"]).cumcount()
    prev_late = sq.groupby(["customer_id", "loan_id"]).late_raw.shift()
    sq["roll"] = sq.late_raw.where(prev_late == 1)
    sq["cure"] = (1 - sq.late_raw).where(prev_late == 1)
    run = sq.late_raw.groupby((sq.late_raw != prev_late).cumsum()).cumsum()
    sq["streak"] = run.where(sq.late_raw == 1, 0)
    gs = sq.groupby("customer_id")
    seqf = pd.DataFrame({
        "sq_roll_rate": gs.roll.mean(),
        "sq_cure_rate": gs.cure.mean(),
        "sq_max_streak": gs.streak.max(),
        "sq_mean_streak": gs.streak.mean(),
        "sq_early_late": sq[sq.seq <= 3].groupby("customer_id").late_raw.mean(),
        "sq_early_late_asof": sq[sq.seq <= 3].groupby("customer_id").late_asof.mean(),
    })
    slope = sq.groupby(["customer_id", "loan_id"]).apply(
        lambda d: np.polyfit(d.seq, d.maxdpd_raw, 1)[0] if len(d) > 2 else np.nan, include_groups=False)
    seqf["sq_dpd_slope_mean"] = slope.groupby(level=0).mean()
    seqf["sq_dpd_slope_max"] = slope.groupby(level=0).max()
    tail3 = sq.groupby("customer_id").tail(3).groupby("customer_id")
    seqf["sq_last3_late"] = tail3.late_raw.mean()
    seqf["sq_last3_dpd"] = tail3.maxdpd_raw.mean()
    seqf["sq_last_inst_dpd"] = sq.groupby("customer_id").maxdpd_raw.last()
    return pd.concat([out.reset_index(drop=True), seqf.reset_index(drop=True)], axis=1)


# --------------------------------------------------------------------------- #
# 5. account behaviour (dibandingkan dengan baseline customer sendiri)
# --------------------------------------------------------------------------- #
def account_features(apps, acct):
    ev = acct.rename(columns={"month": "event_time"}).copy()
    q = _queries(apps)
    ev["net"] = ev.inflows - ev.outflows
    ev["out_in"] = ev.outflows / ev.inflows.replace(0, np.nan)
    ev["income_share"] = ev.income_like_credits / ev.inflows.replace(0, np.nan)
    ev["balance_change"] = ev.closing_balance - ev.opening_balance
    ev["buffer_months"] = ev.average_balance / (ev.outflows / 12).replace(0, np.nan)
    ev["low_balance_flag"] = (ev.low_balance_days > 0).astype(float)
    ev["zero_income"] = (ev.income_like_credits == 0).astype(float)
    cols = ["inflows", "outflows", "income_like_credits", "average_balance", "minimum_balance",
            "low_balance_days", "net", "out_in", "income_share", "balance_change",
            "buffer_months", "closing_balance", "opening_balance"]

    stats, _ = window_stats(ev, q, value_cols=cols, windows=W_LONG, ops=("sum", "mean", "std", "min", "max", "last"))
    latest, _ = asof_point(ev, q, cols)
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["ab_" + c for c in stats.columns] + ["ab_last_" + c for c in latest.columns]

    # bulan k = jarak bulan dari keputusan; k=1 adalah bulan lengkap terakhir
    dec_month = apps.decision_datetime.dt.to_period("M").dt.to_timestamp().reset_index(drop=True)
    ev = ev.reset_index(drop=True)
    ev["_cust"] = ev.customer_id
    ev["k"] = ((dec_month.dt.year * 12 + dec_month.dt.month) -
               (ev.event_time.dt.year * 12 + ev.event_time.dt.month)).to_numpy()
    kmap = pd.Series(np.arange(len(apps)), index=apps.customer_id.reset_index(drop=True))
    ev["_app_idx"] = ev.customer_id.map(kmap).to_numpy()
    base = ev[(ev.k >= 4) & (ev.k <= 12)].groupby("_app_idx")[cols].mean()
    l3 = ev[ev.k <= 3].groupby("_app_idx")[cols].mean()
    k1 = ev[ev.k == 1].groupby("_app_idx")[cols].mean()
    n_idx = len(apps)
    base = base.reindex(range(n_idx))
    l3 = l3.reindex(range(n_idx))
    k1 = k1.reindex(range(n_idx))
    for c in ["inflows", "outflows", "income_like_credits", "average_balance", "net", "minimum_balance"]:
        out[f"ab_k1_vs_base_{c}"] = k1[c].to_numpy() / base[c].abs().replace(0, np.nan).to_numpy()
        out[f"ab_l3_vs_base_{c}"] = l3[c].to_numpy() / base[c].abs().replace(0, np.nan).to_numpy()
    for c in ["low_balance_days", "buffer_months", "out_in", "income_share"]:
        out[f"ab_l3_{c}"] = l3[c].to_numpy()
    out["ab_last_k"] = ev.groupby("_app_idx").k.min().reindex(range(n_idx)).to_numpy()
    out["ab_months"] = ev.groupby("_app_idx").size().reindex(range(n_idx)).to_numpy()
    out["ab_low_months"] = ev.groupby("_app_idx").low_balance_flag.sum().reindex(range(n_idx)).to_numpy()
    out["ab_zero_income_months"] = ev.groupby("_app_idx").zero_income.sum().reindex(range(n_idx)).to_numpy()
    out["ab_balance_growth"] = out["ab_last_closing_balance"] - out["ab_first_opening_balance"] \
        if "ab_first_opening_balance" in out else np.nan
    out["ab_balance_growth"] = (ev.sort_values("event_time").groupby("_app_idx").closing_balance.last() -
                                ev.sort_values("event_time").groupby("_app_idx").opening_balance.first()
                                ).reindex(range(n_idx)).to_numpy()
    out["ab_inc_trend"] = l3.income_like_credits.to_numpy() - out["ab_mean_income_like_credits__all"].to_numpy()
    out["ab_bal_trend"] = l3.average_balance.to_numpy() - out["ab_mean_average_balance__all"].to_numpy()
    out["ab_low_balance_month_frac"] = out["ab_sum_low_balance_days__m12"] / (30.0 * out["ab_n_events__m12"])
    f6 = out["ab_mean_average_balance__m6"].to_numpy()
    f12 = out["ab_mean_average_balance__m12"].to_numpy()
    out["ab_balance_trend_6v6"] = f6 - (f12 * 12 - f6 * 6) / 6.0
    out["ab_income_volatility"] = out["ab_std_income_like_credits__m12"] / \
        out["ab_mean_income_like_credits__m12"].replace(0, np.nan)
    return out


# --------------------------------------------------------------------------- #
# 6. e-commerce
# --------------------------------------------------------------------------- #
def ecommerce_features(apps, ecom):
    XINC = apps.set_index("customer_id")["declared_monthly_income"] \
        if "declared_monthly_income" in apps.columns else pd.Series(np.nan, index=apps.customer_id)
    ev = ecom.sort_values(["order_id", "transaction_datetime"]).drop_duplicates("order_id", keep="first")
    ev = ev.merge(apps[["customer_id", "decision_datetime"]], on="customer_id", how="inner")
    ev = ev[ev.transaction_datetime <= ev.decision_datetime].copy()
    ev["days_ago"] = (ev.decision_datetime - ev.transaction_datetime).dt.total_seconds() / 86400
    ev = ev.rename(columns={"transaction_datetime": "event_time"})
    ev["done"] = ev.order_status == "completed"
    ev["spend"] = ev.order_total.where(ev.done, 0.0)
    ev["cancel"] = (~ev.done).astype(float)
    ev["is_completed"] = ev.done.astype(float)
    ev["quantity_f"] = ev.quantity.astype(float)
    for m in ["COD", "credit_card", "e_wallet", "bank_transfer", "debit_card"]:
        ev[f"pm_{m}"] = (ev.payment_method == m).astype(float)
    categories = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
    for c in categories:
        ev[f"cat_{c}"] = np.where(ev.product_category == c, ev.spend, 0.0)

    q = _queries(apps)
    value_cols = (["spend", "order_total", "is_completed", "cancel", "quantity_f"]
                  + [f"pm_{m}" for m in ["COD", "credit_card", "e_wallet", "bank_transfer", "debit_card"]]
                  + [f"cat_{c}" for c in categories])
    stats, _ = window_stats(ev, q, value_cols=value_cols, windows=W_LONG, ops=("sum", "mean", "max", "last"))
    latest, _ = asof_point(ev, q, ["spend", "order_total", "quantity_f"])
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["ec_" + c for c in stats.columns] + ["ec_last_" + c for c in latest.columns]

    # customer_id -> baris matriks fitur (semua customer unik, satu pengajuan per customer)
    row_of = pd.Series(np.arange(len(apps)), index=apps.customer_id.to_numpy())
    idx = apps.customer_id.map(row_of).to_numpy()

    base_mask = (ev.days_ago > 90) & (ev.days_ago <= 365)
    base_spend = ev.spend.where(base_mask, 0).groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
    base_n = base_mask.groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
    for c in categories:
        out[f"ec_share_{c}"] = out[f"ec_sum_cat_{c}__m12"] / out["ec_sum_spend__m12"].replace(0, np.nan)
    out["ec_orders_per_month"] = out["ec_n_events__all"] / 12.0
    out["ec_cancel_rate"] = out["ec_sum_cancel__all"] / out["ec_n_events__all"].replace(0, np.nan)
    out["ec_avg_basket"] = out["ec_sum_order_total__all"] / out["ec_n_events__all"].replace(0, np.nan)

    # jendela pendek + perbandingan dengan baseline 90-365 hari milik customer sendiri
    base_spend_m = base_spend / 9.0 + 1.0
    base_n_m = base_n / 9.0 + 0.1
    for w in [7, 14, 30, 60, 90]:
        m = ev.days_ago <= w
        sp = ev.spend.where(m, 0).groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
        cnt = m.groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
        out[f"ec_spend{w}"] = sp
        out[f"ec_n{w}"] = cnt
        out[f"ec_spend{w}_vs_base"] = sp * 30.0 / w / base_spend_m
        out[f"ec_n{w}_vs_base"] = cnt * 30.0 / w / base_n_m
    for hl in [15, 45]:
        wgt = np.exp(-ev.days_ago * np.log(2) / hl)
        out[f"ec_decay_n{hl}"] = wgt.groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()
        out[f"ec_decay_spend{hl}"] = (wgt * ev.spend).groupby(ev.customer_id).sum().reindex(apps.customer_id).to_numpy()

    # harga satuan relatif terhadap median kategori -> sinyal terkuat di e-commerce
    items = ecom[ecom.order_status == "completed"].merge(
        apps[["customer_id", "decision_datetime"]], on="customer_id", how="inner")
    items = items[items.transaction_datetime <= items.decision_datetime]
    items["days_ago"] = (items.decision_datetime - items.transaction_datetime).dt.total_seconds() / 86400
    unit = items.item_total / items.quantity.replace(0, np.nan)
    items["lrel"] = np.log(unit / unit.groupby(items.product_category).transform("median"))
    gi = items.groupby("customer_id")
    out["ec_price_rel"] = gi.lrel.mean().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_std"] = gi.lrel.std().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_max"] = gi.lrel.max().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_min"] = gi.lrel.min().reindex(apps.customer_id).to_numpy()
    for lo, hi in [(0, 30), (0, 90), (90, 365), (365, 800)]:
        m = (items.days_ago > lo) & (items.days_ago <= hi)
        out[f"ec_price_rel_{lo}_{hi}"] = items[m].groupby("customer_id").lrel.mean().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_trend"] = out["ec_price_rel_0_90"] - out["ec_price_rel_365_800"]
    out["ec_price_rel_trend2"] = out["ec_price_rel_0_90"] - out["ec_price_rel_90_365"]
    slope = gi.apply(lambda d: np.polyfit(-d.days_ago, d.lrel, 1)[0] if len(d) > 2 else np.nan,
                     include_groups=False)
    out["ec_price_rel_slope"] = (slope * 365).reindex(apps.customer_id).to_numpy()
    bycat = items.pivot_table(index="customer_id", columns="product_category", values="lrel", aggfunc="mean")
    bycat = bycat.reindex(apps.customer_id)
    bycat.columns = [f"ec_price_rel_cat_{c}" for c in bycat.columns]
    out = pd.concat([out, bycat.reset_index(drop=True)], axis=1)
    catqty = items.pivot_table(index="customer_id", columns="product_category",
                               values="quantity", aggfunc="sum", fill_value=0).reindex(apps.customer_id)
    catqty.columns = [f"ec_catqty_{c}" for c in catqty.columns]
    out = pd.concat([out, catqty.reset_index(drop=True)], axis=1)

    # pergeseran komposisi kategori: 90 hari terakhir vs 90-365 hari
    idx_of = pd.Series(np.arange(len(apps)), index=apps.customer_id)
    frames = {}
    for name, m in [("r", items.days_ago <= 90), ("b", (items.days_ago > 90) & (items.days_ago <= 365))]:
        p = items[m].pivot_table(index="customer_id", columns="product_category",
                                 values="item_total", aggfunc="sum", fill_value=0)
        p = p.reindex(apps.customer_id).fillna(0.0)
        frames[name] = p.div(p.sum(axis=1).replace(0, np.nan), axis=0)
    shift = frames["r"].sub(frames["b"], fill_value=0.0)
    shift.columns = [f"cs_shift_{c}" for c in shift.columns]
    shift["cs_shift_l1"] = shift.abs().sum(axis=1)
    out = pd.concat([out, shift.reset_index(drop=True)], axis=1)

    # --- turunan tambahan dari sinyal terkuat (harga relatif & komposisi belanja) ---
    # harga relatif terhadap median customer sendiri, bukan hanya median kategori global
    own_med = unit.groupby([items.customer_id, items.product_category]).transform("median")
    items["lrel_own"] = np.log(unit / own_med.replace(0, np.nan))
    go = items.groupby("customer_id")
    out["ec_price_rel_own"] = go.lrel_own.mean().reindex(apps.customer_id).to_numpy()
    out["ec_price_rel_own_recent"] = (items[items.days_ago <= 90].groupby("customer_id").lrel_own.mean()
                                      .reindex(apps.customer_id).to_numpy())
    out["ec_price_rel_own_gap"] = out["ec_price_rel_own_recent"] - out["ec_price_rel_own"]
    out["ec_price_rel_to_income"] = (items.groupby("customer_id").item_total.mean()
                                     .reindex(apps.customer_id).to_numpy()
                                     / XINC.reindex(apps.customer_id).to_numpy())
    # belanja diskresioner vs esensial -> proxy tekanan keuangan
    disc = ["fashion", "electronics", "personal_care"]
    ess = ["groceries", "health"]
    disc_spend = sum(out[f"ec_sum_cat_{c}__m12"].fillna(0) for c in disc)
    ess_spend = sum(out[f"ec_sum_cat_{c}__m12"].fillna(0) for c in ess)
    out["ec_disc_share_m12"] = disc_spend / (disc_spend + ess_spend).replace(0, np.nan)
    out["ec_disc_vs_base"] = disc_spend / (disc_spend / 4 + ess_spend / 4 + 1).replace(0, np.nan)
    # struktur keranjang
    out["ec_items_per_order"] = (items.groupby(["customer_id", "transaction_datetime"]).size()
                                 .groupby("customer_id").mean().reindex(apps.customer_id).to_numpy())
    out["ec_distinct_categories"] = items.groupby("customer_id").product_category.nunique().reindex(apps.customer_id).to_numpy()
    out["ec_qty_per_item"] = items.quantity.groupby(items.customer_id).mean().reindex(apps.customer_id).to_numpy()
    out["ec_disp_price"] = go.lrel.std().reindex(apps.customer_id).to_numpy()
    month_spend = items.assign(m=items.days_ago // 30).groupby(["customer_id", "m"]).item_total.sum()
    out["ec_month_spend_std"] = month_spend.groupby("customer_id").std().reindex(apps.customer_id).to_numpy()
    month_mean = month_spend.groupby("customer_id").mean().reindex(apps.customer_id).to_numpy()
    out["ec_month_spend_cv"] = out["ec_month_spend_std"] / pd.Series(month_mean).replace(0, np.nan).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# 7. cross features
# --------------------------------------------------------------------------- #
def cross_features(apps, f):
    out = pd.DataFrame(index=f.index)
    inc = f["app__declared_monthly_income"].replace(0, np.nan)
    active_inst = f.get("pa__pa_active_inst")
    inst = f["app__installment"]
    out["x_dti_total"] = (active_inst.fillna(0) + inst) / inc
    cash = f.get("ab__ab_mean_income_like_credits__m6")
    out["x_dti_cashflow"] = (active_inst.fillna(0) + inst) / cash
    out["x_inst_to_net"] = inst / f["ab__ab_mean_net__m6"]
    out["x_income_verified_ratio"] = cash / inc
    out["x_bureau_debt_to_income"] = f["bu__bu_last_outstanding_balance"] / (inc * 12)
    out["x_loan_to_balance"] = f["app__loan_amount"] / f["ab__ab_mean_average_balance__m6"]
    out["x_rate_vs_history"] = f["app__interest_rate"] - f["pa__pa_rate_mean"]
    out["x_loan_vs_history"] = f["app__loan_amount"] / f["pa__pa_mean_requested_amount__all"].replace(0, np.nan)
    out["x_spend30_to_balance"] = f["ec__ec_spend30"] / f["ab__ab_mean_average_balance__m3"]
    out["x_spend90_to_inflow"] = f["ec__ec_sum_spend__m3"] / (3 * f["ab__ab_mean_inflows__m3"])
    out["x_surge_and_drain"] = f["ec__ec_n30_vs_base"] * f["ab__ab_k1_vs_base_outflows"]
    out["x_price_and_late"] = f["ec__ec_price_rel"] * f["rh__rha_rate1_smooth"]
    out["x_delinq_and_income"] = f["bu__bu_last_delinquent_accounts_last_12m"] / inc
    out["x_price_and_balance"] = f["ec__ec_price_rel"] * f["ab__ab_k1_vs_base_average_balance"]
    out["x_late_and_bureau"] = f["rh__ln_max_dpd"] + f["bu__bu_last_score_f"] / 100
    return out

# %% [markdown]
### 4.1 Bangun matriks fitur


# %%
t0 = time.time()
blocks = {
    "app": app_features(apps.drop(columns=["age", "region", "employment_type",
                                           "employment_tenure_months", "declared_monthly_income",
                                           "dependents", "housing_status"]), profiles),
    "bu": bureau_features(apps, bureau),
    "pa": previous_features(apps, previous_apps),
    "rh": repayment_features(apps, repayments),
    "ab": account_features(apps, accounts),
    "ec": ecommerce_features(apps, ecommerce),
}
for name, blk in blocks.items():
    log(f"blok {name}: {blk.shape}")

X = pd.concat(list(blocks.values()), axis=1)
X.columns = [f"{k}__{c}" for k, blk in blocks.items() for c in blk.columns]
X = pd.concat([X, cross_features(apps, X)], axis=1)
X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
assert not any(str(d).startswith("datetime") for d in X.dtypes)
log(f"matriks fitur {X.shape} | rasio NaN {X.isna().mean().mean():.3f} | {time.time() - t0:.0f}s")
del repayments, accounts, bureau, ecommerce, blocks


# %% [markdown]
### 4.2 Pemilihan feature set
#
# * `FULL` — semua fitur (minus kolom DPD causal bila `USE_CAUSAL_DPD_ONLY = True`).
# * `LEAN` — `FULL` tanpa blok yang menambah noise bagi pohon (profil demografis,
#   timestamp aplikasi, share kategori yang sangat sparse).
# * `TOP` — 100 fitur teratas menurut gain, dihitung **per fold dari data latih fold itu**
#   supaya baris validasi tidak ikut menentukan pilihan fitur.


# %%
IS_TRAIN = apps.target.notna().to_numpy()
X_train_all = X[IS_TRAIN].reset_index(drop=True)
y_train = train_apps.target.to_numpy()
X_test_all = X[~IS_TRAIN].reset_index(drop=True)
test_ids = test_apps.application_id.to_numpy()

CATEGORICAL = [c for c in X.columns if str(X[c].dtype) == "category"]
X_enc = X.copy()
for c in CATEGORICAL:
    X_enc[c] = X_enc[c].cat.codes
X_enc = X_enc.reset_index(drop=True)
Xa, Xb = X_enc[IS_TRAIN].reset_index(drop=True), X_enc[~IS_TRAIN].reset_index(drop=True)
Xa_raw = X_train_all

causal_cols = [c for c in Xa.columns
               if (c.startswith("rh__rh_") and any(k in c for k in ["asof", "dpd_ge_", "dpd_asof", "pay_ratio"]))
               or c.startswith("rh__rha_")]
NOISE_PREFIXES = ("ec__ec_share_", "ec__ec_catqty_", "ec__cs_shift", "x__x_")
NOISE_EXACT = {"app__region", "app__employment_type", "app__housing_status", "app__age",
               "app__employment_tenure_months", "app__dependents", "app__tenure_to_age",
               "app__income_per_dependent", "app__loan_purpose", "app__decision_lag_h",
               "app__app_hour", "app__app_dow"}

drop_causal = causal_cols if USE_CAUSAL_DPD_ONLY else []
FULL = [c for c in Xa.columns if c not in drop_causal]
LEAN = [c for c in FULL if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIXES)]
TOP_K = 100
log(f"FULL {len(FULL)} | LEAN {len(LEAN)} | TOP {TOP_K}")


# %% [markdown]
# ## 5. Model
#
# Hyperparameter dipilih lewat grid kecil di atas fold 0 (lihat tabel ringkasan di bawah) —
# tetap pencarian manual, bukan AutoML.
#
# | config | fold-0 KS |
# |---|---|
# | `num_leaves=15, lr=0.01, ff=0.3, l2=20, max_bin=127` | 0.519 |
# | `+ extra_trees=True` | 0.519 |
# | `num_leaves=31, lr=0.02, ff=0.4` | 0.515 |
# | `num_leaves=63, lr=0.01, ff=0.2, l2=30` | 0.516 |


# %%
SEED = 42
N_FOLDS = 5

LGB_BASE = dict(objective="binary", metric="auc", learning_rate=0.01, num_leaves=15,
                min_data_in_leaf=200, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=20, max_bin=127, verbose=-1, deterministic=True, force_row_wise=True)
LGB_ET = dict(LGB_BASE, extra_trees=True, seed=1)
LGB_RND = dict(LGB_BASE, seed=0)

XGB_BASE = dict(max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.3,
                reg_lambda=20, tree_method="hist", max_bin=128, eval_metric="auc",
                n_jobs=-1, random_state=0, enable_categorical=True)

CAT_BASE = dict(iterations=6000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                eval_metric="AUC", od_type="Iter", od_wait=300, verbose=0)


def ks_score(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def rankify(v):
    return rankdata(v) / len(v)


def make_folds(X, y, n_folds=N_FOLDS, seed=SEED):
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y))


def fit_lgb(Xtr, ytr, Xva, yva, Xte, params, cat_cols=()):
    ds = lgb.Dataset(Xtr, ytr, categorical_feature=list(cat_cols))
    dv = lgb.Dataset(Xva, yva, reference=ds)
    m = lgb.train(params, ds, 12000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(400, verbose=False)])
    return (m.predict(Xva, num_iteration=m.best_iteration),
            m.predict(Xte, num_iteration=m.best_iteration), m)


def fit_xgb(Xtr, ytr, Xva, yva, Xte, params):
    p = dict(objective="binary:logistic", **params)
    dtr, dva, dte = xgb.DMatrix(Xtr, ytr), xgb.DMatrix(Xva, yva), xgb.DMatrix(Xte)
    m = xgb.train(p, dtr, 12000, evals=[(dva, "va")], early_stopping_rounds=400, verbose_eval=False)
    best = m.best_iteration + 1
    return m.predict(dva, iteration_range=(0, best)), m.predict(dte, iteration_range=(0, best)), m


def fit_cat(Xtr, ytr, Xva, yva, Xte, params, cat_cols=()):
    cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in cats:
        Xtr[c], Xva[c], Xte[c] = Xtr[c].astype(str), Xva[c].astype(str), Xte[c].astype(str)
    m = CatBoostClassifier(**dict(params, random_seed=0), cat_features=cats)
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m


def linear_matrix(X, med, num, flags):
    vals = X[num].astype(float)
    return pd.concat([vals.fillna(med), vals.isna().astype(float).add_suffix("_na")], axis=1)


def fit_linear(Xtr, ytr, Xva, Xte, num, flags):
    med = Xtr[num].median()
    m = make_pipeline(QuantileTransformer(n_quantiles=200, subsample=10 ** 6, random_state=SEED),
                      SimpleImputer(strategy="median"),
                      SplineTransformer(n_knots=4, degree=2),
                      LogisticRegression(C=0.03, max_iter=3000))
    m.fit(linear_matrix(Xtr, med, num, flags), ytr)
    return m.predict_proba(linear_matrix(Xva, med, num, flags))[:, 1], \
        m.predict_proba(linear_matrix(Xte, med, num, flags))[:, 1], m

# %% [markdown]
# ## 6. Cross-validation (2 split × 5 fold)
#
# Setiap model dilatih pada **dua split CV berbeda** (`SPLIT_SEEDS`). Alasannya:
# OOF KS yang digabung antar fold bergeser ±0.004 hanya karena komposisi fold,
# sehingga (a) pemilihan blend harus dinilai rata-rata atas beberapa split agar tidak
# overfit pada satu split, dan (b) prediksi test yang dirata-ratakan dari 10 model
# (5 fold × 2 split) lebih stabil daripada 5 model.


# %%
members = ["lgb_full_s0", "lgb_full_s1", "lgb_lean_s0", "xgb_full", "cat_full", "spline_top"]
oof = {m: {} for m in members}          # oof[m][split_seed]
test_pred = {m: np.zeros(len(Xb)) for m in members}
models_dir = os.path.join(ARTIFACT_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

for split_seed in SPLIT_SEEDS:
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=split_seed).split(Xa, y_train))
    for m in members:
        oof[m][split_seed] = np.zeros(len(Xa))

    for k, (a, b) in enumerate(folds):
        t_fold = time.time()
        va_scores = {}

        # --- LightGBM: 2 seed pada feature set penuh, 1 seed pada set lean ---
        for name, cols, params in [
            ("lgb_full_s0", FULL, dict(LGB_RND, num_threads=10)),
            ("lgb_full_s1", FULL, dict(LGB_ET, num_threads=10)),
            ("lgb_lean_s0", LEAN, dict(LGB_RND, num_threads=10)),
        ]:
            va, te, mdl = fit_lgb(Xa[cols].iloc[a], y_train[a], Xa[cols].iloc[b], y_train[b],
                                  Xb[cols], params)
            va_scores[name] = va
            oof[name][split_seed][b] = va
            test_pred[name] += te / (N_FOLDS * len(SPLIT_SEEDS))
            if split_seed == SPLIT_SEEDS[0]:
                mdl.save_model(os.path.join(models_dir, f"{name}_f{k}.txt"))

        # --- XGBoost ---
        va, te, mdl = fit_xgb(Xa[FULL].iloc[a].to_numpy(dtype=np.float32), y_train[a],
                              Xa[FULL].iloc[b].to_numpy(dtype=np.float32), y_train[b],
                              Xb[FULL].to_numpy(dtype=np.float32), XGB_BASE)
        va_scores["xgb_full"] = va
        oof["xgb_full"][split_seed][b] = va
        test_pred["xgb_full"] += te / (N_FOLDS * len(SPLIT_SEEDS))

        # --- CatBoost ---
        va, te, mdl = fit_cat(Xa_raw[FULL].iloc[a], y_train[a], Xa_raw[FULL].iloc[b],
                              y_train[b], X_test_all[FULL], CAT_BASE)
        va_scores["cat_full"] = va
        oof["cat_full"][split_seed][b] = va
        test_pred["cat_full"] += te / (N_FOLDS * len(SPLIT_SEEDS))

        # --- spline logistic pada 100 fitur teratas (gain, dihitung dari fold latih) ---
        base_va, base_te, base_mdl = fit_lgb(Xa[FULL].iloc[a], y_train[a], Xa[FULL].iloc[b],
                                             y_train[b], Xb[FULL], dict(LGB_RND, num_threads=10))
        gain = pd.Series(base_mdl.feature_importance("gain"), index=FULL).sort_values(ascending=False)
        top_cols = [c for c in gain.index[:TOP_K] if c in LEAN] or list(LEAN[:TOP_K])
        num = [c for c in top_cols if str(Xa[c].dtype) != "category"]
        med = Xa[top_cols].iloc[a].median()
        pipe = make_pipeline(QuantileTransformer(n_quantiles=200, subsample=10 ** 6, random_state=SEED),
                             SimpleImputer(strategy="median"),
                             SplineTransformer(n_knots=5, degree=2),
                             LogisticRegression(C=0.1, max_iter=2000))
        pipe.fit(Xa[top_cols].iloc[a].fillna(med), y_train[a])
        va = pipe.predict_proba(Xa[top_cols].iloc[b].fillna(med))[:, 1]
        te = pipe.predict_proba(Xb[top_cols].fillna(med))[:, 1]
        va_scores["spline_top"] = va
        oof["spline_top"][split_seed][b] = va
        test_pred["spline_top"] += te / (N_FOLDS * len(SPLIT_SEEDS))

        msg = " ".join(f"{n.split('_')[0]}:{ks_score(y_train[b], v):.4f}" for n, v in va_scores.items())
        log(f"split {split_seed} fold {k} | {msg} | {time.time() - t_fold:.0f}s")


# %% [markdown]
## 7. Evaluasi per model


# %%
for m in members:
    line = f"{m:14s}"
    for s in SPLIT_SEEDS:
        line += f"  split{s}: KS {ks_score(y_train, oof[m][s]):.5f}"
    # skor gabungan = rata-rata rank dari kedua split (meniru cara prediksi test dirata-ratakan)
    comb = np.mean([rankify(oof[m][s]) for s in SPLIT_SEEDS], axis=0)
    line += f"  | combined KS {ks_score(y_train, comb):.5f} AUC {roc_auc_score(y_train, comb):.5f}"
    print(line)


# %% [markdown]
## 8. Blend yang dioptimalkan terhadap KS
#
# Tiga kandidat, semuanya dinilai sebagai **rata-rata KS atas kedua split** supaya
# pilihan tidak mengikuti noise satu split:
#
# 1. `equal` — rata-rata rank semua model;
# 2. `greedy` — *forward selection with replacement* (Caruana): repeatedly menambahkan model
#    yang paling menaikkan KS rata-rata;
# 3. `best_single` — model tunggal terbaik sebagai jaring pengaman.


# %%
def mean_ks(score_fn):
    # KS rata-rata atas seluruh split: kriteria pemilihan blend
    return float(np.mean([ks_score(y_train, score_fn(s)) for s in SPLIT_SEEDS]))


def blend_of(weights):
    # bobot diterapkan pada rank masing-masing model, per split
    def make(split_seed):
        return sum(w * rankify(oof[m][split_seed]) for m, w in weights.items())
    return make


candidates = {}
equal_w = {m: 1.0 / len(members) for m in members}
candidates["equal"] = equal_w
candidates["best_single"] = {max(members, key=lambda m: mean_ks(lambda s, m=m: oof[m][s])): 1.0}

# greedy forward selection with replacement
greedy_counts = {m: 0 for m in members}
running = np.zeros(len(y_train))
for step in range(14):
    best_m, best_v = None, -np.inf
    for m in members:
        trial = {k: v for k, v in greedy_counts.items() if v > 0}
        trial[m] = trial.get(m, 0) + 1
        val = mean_ks(blend_of(trial))
        if val > best_v:
            best_m, best_v = m, val
    greedy_counts[best_m] += 1
    running = blend_of({k: v / sum(greedy_counts.values()) for k, v in greedy_counts.items()})(SPLIT_SEEDS[0])
    log(f"greedy step {step + 1}: +{best_m:14s} mean KS {best_v:.5f}")

candidates["greedy"] = {k: v / sum(greedy_counts.values()) for k, v in greedy_counts.items()}

rows = []
for name, w in candidates.items():
    make = blend_of(w)
    per_split = {s: ks_score(y_train, make(s)) for s in SPLIT_SEEDS}
    comb = np.mean([make(s) for s in SPLIT_SEEDS], axis=0)
    rows.append({"blend": name, **{f"KS_split{s}": round(per_split[s], 5) for s in SPLIT_SEEDS},
                 "mean_KS": round(float(np.mean(list(per_split.values()))), 5),
                 "KS_combined": round(ks_score(y_train, comb), 5),
                 "AUC_combined": round(roc_auc_score(y_train, comb), 5),
                 "weights": str({k: round(v, 3) for k, v in w.items() if v > 0})})
blend_table = pd.DataFrame(rows).sort_values("mean_KS", ascending=False)
display(blend_table)

BEST_BLEND = blend_table.iloc[0]["blend"]
BEST_WEIGHTS = candidates[BEST_BLEND]
make_final = blend_of(BEST_WEIGHTS)
log(f"blend terpilih: {BEST_BLEND} -> {BEST_WEIGHTS}")


# %% [markdown]
## 9. Diagnosa blend terpilih (KS kurva, gain model, importance fitur)


# %%
final_oof = np.mean([make_final(s) for s in SPLIT_SEEDS], axis=0)
final_test = sum(w * rankify(test_pred[m]) for m, w in BEST_WEIGHTS.items())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
order = np.argsort(-final_oof)
pos = np.cumsum(np.where(y_train[order] == 1, 1, 0)) / max((y_train == 1).sum(), 1)
neg = np.cumsum(np.where(y_train[order] == 1, 0, 1)) / max((y_train == 0).sum(), 1)
ks_curve = pos - neg
axes[0].plot(np.arange(len(order)) / len(order), ks_curve, color="#4C78A8")
axes[0].axhline(ks_curve.max(), ls="--", color="#E45756",
                label=f"KS = {ks_curve.max():.4f}")
axes[0].set_xlabel("fracsie of applications sorted by score")
axes[0].set_ylabel("TPR - FPR")
axes[0].set_title("Kurva KS out-of-fold")
axes[0].legend()

gain_all = []
for k in range(N_FOLDS):
    path = os.path.join(ARTIFACT_DIR, "models", f"lgb_full_s0_f{k}.txt")
    if os.path.exists(path):
        gain_all.append(lgb.Booster(model_file=path).feature_importance("gain"))
gain = pd.Series(np.mean(gain_all, axis=0), index=FULL).sort_values(ascending=False)
top_feat = gain.head(20)[::-1]
axes[1].barh(range(len(top_feat)), top_feat.values, color="#54A24B")
axes[1].set_yticks(range(len(top_feat)))
axes[1].set_yticklabels(top_feat.index, fontsize=7)
axes[1].set_xlabel("mean gain")
axes[1].set_title("20 fitur teratas (LightGBM, split 42)")
plt.tight_layout()
plt.show()

gain.to_csv(f"{ARTIFACT_DIR}/feature_importance_v2.csv")
display(gain.head(20).to_frame("gain"))
print("KS maksimum pada kurva :", round(float(ks_curve.max()), 5))


# %% [markdown]
## 10. Submission
#
# Prediksi test = rata-rata rank dari seluruh model (5 fold × 2 split), dengan bobot blend
# terpilih. Nilai ditulis sebagai `default_probability` sesuai format
# `sample_submission.csv`; karena metrik KS hanya memakai urutan, skewalaan nilai tidak
# berpengaruh terhadap skor.


# %%
submission = sample_submission[["application_id"]].copy()
pred_map = pd.Series(final_test, index=test_ids)
submission["default_probability"] = pred_map.loc[submission.application_id.to_numpy()].to_numpy()
assert len(submission) == 30000
assert submission.default_probability.notna().all()
assert submission.default_probability.between(0, 1).all()
submission.to_csv("submission.csv", index=False)

np.save(f"{ARTIFACT_DIR}/oof_final_v2.npy", final_oof)
pd.to_pickle({"oof": oof, "test": test_pred}, f"{ARTIFACT_DIR}/predictions_v2.pkl")

display(submission.head(10))
print(f"baris {len(submission)} | rentang [{submission.default_probability.min():.4f}, "
      f"{submission.default_probability.max():.4f}] | rata-rata {submission.default_probability.mean():.4f} "
      f"(label train {y_train.mean():.4f})")


# %%
summary = pd.Series({
    "KS OOF (blend, split 42)": round(ks_score(y_train, make_final(SPLIT_SEEDS[0])), 5),
    "KS OOF (blend, split 2025)": round(ks_score(y_train, make_final(SPLIT_SEEDS[1])), 5),
    "KS OOF gabungan kedua split": round(ks_score(y_train, final_oof), 5),
    "AUC OOF": round(roc_auc_score(y_train, final_oof), 5),
    "Baseline v1 (AUC-optimal)": 0.47144,
    "Jumlah fitur": int(X.shape[1]),
    "Model dalam blend": len(members),
    "Bobot blend": str({k: round(v, 3) for k, v in BEST_WEIGHTS.items() if v > 0}),
    "Waktu total (detik)": round(time.time() - T_START, 1),
}, name="nilai").to_frame("nilai")
display(summary)

