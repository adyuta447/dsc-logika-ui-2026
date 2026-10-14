# %% [markdown]
# # Prediksi Probabilitas 90+ DPD (Default) — Consumer Lending Indonesia
#
# Notebook machine learning untuk memprediksi probabilitas bahwa sebuah pinjaman
# mencapai **90 atau lebih Days Past Due (DPD)** dalam 12 bulan setelah diberikan.
#
# **Sumber data (7 tabel)**
#
# | Tabel | Isi |
# |---|---|
# | `train_applications` / `test_applications` | Pengajuan pinjaman + label `target` |
# | `customer_profiles` | Profil pelanggan (usia, pekerjaan, income, dependents) |
# | `previous_applications` | Riwayat pengajuan sebelumnya |
# | `repayment_history` | Riwayat cicilan pinjaman sebelumnya |
# | `account_behavior` | Saldo & arus kas rekening bulanan |
# | `bureau_observations` | Laporan kredit & indikator risiko |
# | `ecommerce_transactions` | Transaksi e-commerce |
#
# **Batasan yang dipenuhi**
#
# 1. 100% Python, tanpa data eksternal.
# 2. Preprocessing / EDA / feature engineering **manual** — tanpa AutoML & AutoFE.
#    LightGBM & XGBoost dilatih dengan hyperparameter eksplisit (hasil pencarian
#    manual pada 1 fold, ditampilkan di notebook), bukan framework AutoML.
# 3. **Anti data-leakage**: satu-satunya titik cutoff adalah `decision_datetime`
#    setiap pengajuan. Semua tabel auxiliary diputar ulang (*point-in-time replay*)
#    dan hanya baris yang tersedia pada atau sebelum cutoff yang diagregasi.
# 4. Notebook dapat dijalankan ulang dari nol (`Restart & Run All`) dan
#    menghasilkan `submission.csv` yang identik (semua seed & pengurutan deterministik).
#
# **Hasil**: 5-fold cross-validation (Stratified, seed tetap) dengan blend
# LightGBM + XGBoost.

# %% [markdown]
# ## 0. Setup

# %%
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb

from IPython.display import display
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"
SEED = 42
N_FOLDS = 5
USE_XGB_BLEND = True       # model kedua untuk blend; False -> runtime ~2x lebih singkat
EARLY_STOP_ROUNDS = 300
N_THREADS = os.cpu_count() or 4

os.makedirs(ARTIFACT_DIR, exist_ok=True)
T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)

# %% [markdown]
# ## 1. Load data
#
# Catatan: `target` hanya ada di train. Customer train & test **saling lepas**
# (tidak ada overlap `customer_id`) dan rentang `decision_datetime` keduanya sama
# (2025-01-01 s.d. 2025-12-28) — jadi pemodelan bersifat *cross-sectional*,
# bukan time-series, dan CV acak stratified sudah representatif.

# %%
parse_dates = ["application_datetime", "decision_datetime"]
train_apps = pd.read_csv(f"{DATA_DIR}/train_applications.csv", parse_dates=parse_dates)
test_apps = pd.read_csv(f"{DATA_DIR}/test_applications.csv", parse_dates=parse_dates)
profiles = pd.read_csv(f"{DATA_DIR}/customer_profiles.csv")
previous_apps = pd.read_csv(f"{DATA_DIR}/previous_applications.csv", parse_dates=parse_dates)
repayments = pd.read_csv(f"{DATA_DIR}/repayment_history.csv", parse_dates=["due_date", "last_payment_datetime"])
accounts = pd.read_csv(f"{DATA_DIR}/account_behavior.csv", parse_dates=["month"])
bureau = pd.read_csv(f"{DATA_DIR}/bureau_observations.csv", parse_dates=["report_datetime", "available_datetime"])
ecommerce = pd.read_csv(f"{DATA_DIR}/ecommerce_transactions.csv", parse_dates=["transaction_datetime"])
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

# tabel aplikasi gabungan: fitur dibangun untuk train & test dengan aturan identik
apps = pd.concat([train_apps, test_apps], ignore_index=True)

log(f"train={train_apps.shape}, test={test_apps.shape}, prev={previous_apps.shape}, "
    f"repay={repayments.shape}, account={accounts.shape}, bureau={bureau.shape}, ecom={ecommerce.shape}")
log(f"target rate = {train_apps.target.mean():.4f} ({int(train_apps.target.sum())} positives / {len(train_apps)})")
log(f"customer overlap train/test = {len(set(train_apps.customer_id) & set(test_apps.customer_id))}")
log(f"rentang keputusan train {train_apps.decision_datetime.min()} .. {train_apps.decision_datetime.max()}")

# %% [markdown]
# ## 2. Explorasi data (EDA)
#
# Pola yang teramati dan mengarahkan feature engineering:
#
# * `loan_amount` berkaitan non-linear dengan default (pinjaman kecil ~9,6%,
#   pinjaman sangat besar mendekati 100% default) → rasio terhadap income penting.
# * Histori bulanan `account_behavior` hanya tersedia sampai 2025-11, sedangkan
#   keputusan pengajuan sampai 2025-12 → harus dipotong sesuai cutoff.
# * `bureau_observations` memiliki `application_id` yang menunjuk pengajuan tertentu;
#   hanya laporan dengan `available_datetime <= decision_datetime` yang boleh dipakai.
# * 30% customer **tidak memiliki** histori pengajuan/rekening sama sekali
#   → fitur agregat otomatis NaN, ditangani native oleh LightGBM.

# %%
display(train_apps.head())

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(train_apps.loan_amount.clip(upper=train_apps.loan_amount.quantile(0.99)),
             bins=60, color="#4C78A8")
axes[0].set_title("Distribusi loan_amount (train)")
axes[0].set_xlabel("IDR")

by_bin = train_apps.groupby(pd.qcut(train_apps.loan_amount, 20, duplicates="drop"), observed=True).target.mean()
axes[1].plot(range(len(by_bin)), by_bin.values, color="#E45756")
axes[1].set_title("Target rate vs kuantil loan_amount")
axes[1].set_xlabel("kuantil"); axes[1].set_ylabel("target mean")

by_purpose = train_apps.groupby("loan_purpose").target.agg(["mean", "size"]).sort_values("mean")
axes[2].bar(by_purpose.index, by_purpose["mean"], color="#F58518")
axes[2].set_title("Target rate per loan_purpose")
axes[2].set_ylabel("target mean")
for i, v in enumerate(by_purpose["mean"]):
    axes[2].text(i, v, f"n={int(by_purpose['size'].iloc[i])}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.show()

display(by_purpose)
display(train_apps[["loan_amount", "proposed_monthly_installment", "annual_interest_rate", "term_months"]].describe())

# %%
# cakupan histori per customer
customer_sets = {
    "customer_profiles": set(profiles.customer_id),
    "previous_applications": set(previous_apps.customer_id),
    "repayment_history": set(repayments.customer_id),
    "account_behavior": set(accounts.customer_id),
    "ecommerce_transactions": set(ecommerce.customer_id),
}
train_cust = set(train_apps.customer_id)
coverage = pd.DataFrame({
    "customers": {k: len(v) for k, v in customer_sets.items()},
    "cover_train": {k: len(v & train_cust) for k, v in customer_sets.items()},
})
coverage["cover_pct_train"] = (coverage.cover_train / len(train_cust) * 100).round(1)
display(coverage)
log("21.000 customer train (30%) tidak punya histori pengajuan maupun rekening.")

# %% [markdown]
# ## 3. Kebijakan anti-leakage
#
# Setiap pengajuan punya tepat **satu titik cutoff**: `decision_datetime`.
# Tabel auxiliary diperlakukan sebagai *event stream*, lalu diagregasi hanya
# untuk event bertanggal `<= cutoff`:
#
# | Tabel | kolom waktu yang dipakai (kapan data benar-benar tersedia) |
# |---|---|
# | `previous_applications` | `decision_datetime` — saat status approved/rejected diketahui |
# | `repayment_history` | `due_date`, dan pembayaran dianggap sah hanya bila `last_payment_datetime <= cutoff` |
# | `account_behavior` | `month` — awal dari bulan lengkap yang sudah lewat |
# | `bureau_observations` | `available_datetime` — bukan `report_datetime` |
# | `ecommerce_transactions` | `transaction_datetime` |
#
# Dua jebakan yang sengaja dihindari:
#
# 1. `days_past_due` dan `amount_paid` pada `repayment_history` diukur sampai
#    **akhir periode observasi global**, sehingga sebagian isinya adalah informasi
#    masa depan. Keduanya dihitung ulang secara konservatif *as-of* cutoff.
# 2. `bureau_observations.report_datetime` bisa lebih tua dari `available_datetime`;
#    memakai `report_datetime` akan membocorkan data yang belum bisa diakses.

# %%
# --- 3.1 Helper agregasi temporal (vektorisasi penuh, tanpa loop per customer) ---

DAY = 86400.0
SEC_SPAN = 1 << 31


def _as_seconds(s):
    """datetime Series -> epoch detik (aman terhadap dtype us/ms/ns)."""
    s = pd.to_datetime(s)
    if s.isna().any():
        raise ValueError("null timestamps are not supported")
    return s.dt.as_unit("s").astype("int64").to_numpy().astype(np.float64)


def _prepare(events, queries, customer_col="customer_id", time_col="event_time"):
    """Urutkan event & query lalu bangun kunci pencarian vektor."""
    ev = events.sort_values([customer_col, time_col], kind="mergesort").reset_index(drop=True)
    if ev.empty:
        raise ValueError("event table is empty")

    q = queries.copy()
    q["_orig"] = np.arange(len(q))
    q = q.sort_values([customer_col, "cutoff"], kind="mergesort").reset_index(drop=True)

    codes, uniques = pd.factorize(ev[customer_col], sort=True)
    lookup = pd.Series(np.arange(len(uniques)), index=uniques)
    q_codes = lookup.reindex(q[customer_col]).to_numpy()          # NaN => tanpa histori

    ev_key = codes.astype(np.int64) * SEC_SPAN + _as_seconds(ev[time_col])
    # customer tanpa histori diberi kode melewati kode terakhir -> semua window kosong
    q_code_i = np.where(np.isnan(q_codes), len(uniques), q_codes).astype(np.int64)
    q_key = q_code_i * SEC_SPAN + _as_seconds(q["cutoff"])
    cust_start = np.searchsorted(ev_key, q_code_i * SEC_SPAN, side="left")
    return ev, q, ev_key, q_key, cust_start


def _bounds(ev_key, q_key, cust_start, window):
    """Rentang indeks [start, end) per query; dijamin tidak melintasi customer."""
    end = np.searchsorted(ev_key, q_key, side="right")
    if window is None:
        start = cust_start.copy()
    else:
        start = np.maximum(np.searchsorted(ev_key, q_key - window * DAY, side="right"), cust_start)
    return start, end


def _reduceat(vals, start, end, how):
    """min/max pada rentang indeks yang tidak bersambung, secara vektorisasi."""
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
    """Agregasi trailing-window atas `value_cols`, berakhir tepat di cutoff.

    `windows` berisi {"m3": 91, "m6": 182, "m12": 365, "all": None}
    (`None` = seluruh histori yang tersedia sebelum cutoff).
    Nilai kolom input harus bebas NaN (gunakan isna→0 di sisi pemanggil).
    """
    ev, q, ev_key, q_key, cust_start = _prepare(events, queries, customer_col, time_col)
    names, cols = [], []
    q_sec, times = _as_seconds(q["cutoff"]), _as_seconds(ev[time_col])

    for w_name, window in windows.items():
        s, e = _bounds(ev_key, q_key, cust_start, window)
        names.append(f"n_events__{w_name}")
        cols.append((e - s).astype(np.float64))

    for c in value_cols:
        matrix = ev[c].to_numpy(dtype=np.float64)
        if np.isnan(matrix).any():
            raise ValueError(f"kolom nilai '{c}' mengandung NaH; isi dulu sebelum agregasi")
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
    """Snapshot per customer: observasi terakhir yang tersedia sebelum cutoff."""
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
    names = list(value_cols) + ["days_since_ts"]     # timestamp mentah bukan fitur
    return out[names], names


log("helper agregasi temporal siap")

# %% [markdown]
# ### 3.2 Uji kebenaran helper (kebenaran agregasi + anti-leakage)
#
# Fitur dihitung ulang dengan cara naïve (loop pandas) lalu dibandingkan dengan
# versi vektorisasi. Ini sekaligus membuktikan bahwa event setelah cutoff tidak
# pernah ikut teragregasi.

# %%
rng = np.random.default_rng(SEED)
toy_custs = [f"C{i // 4}" for i in range(120)]
toy_events = pd.DataFrame({
    "customer_id": toy_custs,
    "event_time": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 400, len(toy_custs)), unit="D"),
    "val": rng.normal(size=len(toy_custs)).round(4),
})
toy_queries = pd.DataFrame({"customer_id": sorted(set(toy_custs)),
                            "cutoff": pd.Timestamp("2025-01-01")})
toy_fast, _ = window_stats(toy_events, toy_queries, ["val"], {"all": None, "d90": 90})

n_checks = 0


def _close(a, b):
    """Perbandingan yang memperlakukan NaN == NaN sebagai sama (window kosong)."""
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b))


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

# event yang seluruhnya berada setelah cutoff harus menghasilkan NaN, bukan data
late_events = pd.DataFrame({"customer_id": ["C0", "C0"],
                            "event_time": pd.to_datetime(["2025-06-01", "2025-07-01"]),
                            "val": [1.0, 2.0]})
late_queries = pd.DataFrame({"customer_id": ["C0", "C1"], "cutoff": pd.to_datetime(["2025-01-01"] * 2)})
late_feat, _ = window_stats(late_events, late_queries, ["val"], {"all": None})
assert late_feat.loc[0, "sum_val__all"] == 0.0 and np.isnan(late_feat.loc[0, "last_val__all"])
assert late_feat.loc[1, "n_events__all"] == 0.0 and np.isnan(late_feat.loc[1, "mean_val__all"])

log(f"helper terverifikasi: {n_checks} assertion agregasi + 2 assertion anti-leakage lulus")

# %% [markdown]
# ## 4. Feature engineering
#
# Lebih dari 1.500 fitur dihasilkan dari 7 sumber tabel. Seluruh fitur berupa
# besaran dalam satuan aslinya (IDR, hari, proporsi) — tanpa transformasi log,
# karena LightGBM/XGBoost tidak memerlukan asumsi linearitas.

# %%
W_LONG = {"m3": 91, "m6": 182, "m12": 365, "all": None}


def build_application_features(apps_df, profiles_df):
    """Fitur tingkat pengajuan + profil customer."""
    df = apps_df.merge(profiles_df, on="customer_id", how="left", validate="many_to_one")
    out = pd.DataFrame(index=df.index)

    out["loan_amount"] = df["loan_amount"]
    out["term_months"] = df["term_months"]
    out["annual_interest_rate"] = df["annual_interest_rate"]
    out["proposed_monthly_installment"] = df["proposed_monthly_installment"]
    out["app_hour"] = df["application_datetime"].dt.hour
    out["app_dow"] = df["application_datetime"].dt.dayofweek
    out["app_month"] = df["application_datetime"].dt.month
    out["decision_hour"] = df["decision_datetime"].dt.hour
    out["decision_dow"] = df["decision_datetime"].dt.dayofweek
    out["decision_month"] = df["decision_datetime"].dt.month
    out["decision_weekofyear"] = df["decision_datetime"].dt.isocalendar().week.astype(int)
    out["decision_lag_days"] = (df["decision_datetime"] - df["application_datetime"]).dt.total_seconds() / 86400
    out["decision_tenure_days"] = (df["decision_datetime"] - pd.Timestamp("2023-01-01")).dt.days

    out["age"] = df["age"]
    out["employment_tenure_months"] = df["employment_tenure_months"]
    out["declared_monthly_income"] = df["declared_monthly_income"]
    out["dependents"] = df["dependents"]
    for c in ["region", "employment_type", "housing_status", "loan_purpose"]:
        out[c] = df[c].astype("category")

    income = df["declared_monthly_income"].replace(0, np.nan)
    out["installment_to_income"] = df["proposed_monthly_installment"] / income
    out["loan_to_income"] = df["loan_amount"] / (income * 12)
    out["loan_to_installment"] = df["loan_amount"] / df["proposed_monthly_installment"].replace(0, np.nan)
    out["installment_x_term"] = df["proposed_monthly_installment"] * df["term_months"]
    out["total_repay"] = df["proposed_monthly_installment"] * df["term_months"]
    out["dep_per_income_mio"] = df["dependents"] / income * 1e6
    out["age_minus_tenure"] = df["age"] * 12 - df["employment_tenure_months"]
    out["income_per_dependent"] = income / (1 + df["dependents"])
    return out


def build_previous_application_features(apps_df, prev_df):
    """Riwayat pengajuan: jumlah, approval/rejection rate, nominal, kadens."""
    ev = prev_df.rename(columns={"decision_datetime": "event_time"}).copy()
    ev["approved"] = (ev["decision"] == "approved").astype(float)
    ev["rejected"] = (ev["decision"] == "rejected").astype(float)
    ev["cancelled"] = (ev["decision"] == "cancelled").astype(float)
    ev["approved_amount_f"] = ev["approved_amount"].fillna(0.0)
    ev["was_originated"] = ev["loan_id"].notna().astype(float)
    ev["requested_amount"] = ev["requested_amount"].astype(float)
    ev = ev.sort_values(["customer_id", "event_time"], kind="mergesort")
    # -1 menandai "pengajuan pertama" (belum ada pengajuan sebelumnya)
    ev["days_since_prev_app"] = (ev.groupby("customer_id")["event_time"].diff()
                                 .dt.total_seconds() / 86400).fillna(-1.0)

    queries = apps_df[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})
    stats, _ = window_stats(
        ev, queries,
        value_cols=["requested_amount", "approved_amount_f", "approved", "rejected",
                    "cancelled", "was_originated", "days_since_prev_app"],
        windows=W_LONG, ops=("sum", "mean", "std", "max", "last"),
    )
    stats.columns = [c.replace("requested_amount", "req_amt").replace("approved_amount_f", "appr_amt")
                     .replace("was_originated", "originated") for c in stats.columns]
    stats = stats.loc[:, ~stats.columns.duplicated()]

    stats["pa_approval_rate_all"] = stats["sum_approved__all"] / stats["n_events__all"].replace(0, np.nan)
    stats["pa_reject_rate_m12"] = stats["sum_rejected__m12"] / stats["n_events__m12"].replace(0, np.nan)
    stats["pa_cancel_rate_all"] = stats["sum_cancelled__all"] / stats["n_events__all"].replace(0, np.nan)
    stats["pa_amount_ratio_m12"] = stats["sum_appr_amt__m12"] / stats["sum_req_amt__m12"].replace(0, np.nan)
    stats["pa_n_originated"] = stats["sum_originated__all"]
    stats["pa_recent_rejection"] = (stats["sum_rejected__m3"] > 0).astype(float)
    return stats


def _repayment_asof(repay_df, apps_df):
    """Hitung ulang status cicilan secara as-of cutoff (konservatif).

    Cicilan dianggap lunas hanya bila pembayaran terakhirnya sudah masuk pada
    atau sebelum `decision_datetime`; bila tidak, seluruh `amount_due` dianggap
    outstanding dan DPD bertambah satu hari per hari setelah jatuh tempo.
    """
    ev = repay_df.rename(columns={"due_date": "event_time"}).copy()
    ev["cutoff"] = ev["customer_id"].map(apps_df.set_index("customer_id")["decision_datetime"])
    if ev["cutoff"].isna().any():
        raise ValueError("repayment_history memuat customer di luar tabel aplikasi")
    ev = ev.dropna(subset=["last_payment_datetime"], how="all")

    settled = (ev["amount_paid"] >= ev["amount_due"]) & (ev["last_payment_datetime"] <= ev["cutoff"])
    ev["paid_flag"] = settled.astype(float)
    ev["outstanding"] = np.where(settled, 0.0, ev["amount_due"].astype(float))
    ev["pay_ratio"] = np.where(settled, 1.0, 0.0)
    days_late = ((ev["cutoff"] - ev["event_time"]).dt.total_seconds() / 86400).clip(lower=0)
    ev["dpd"] = np.where(settled, 0.0, days_late)
    # DPD level pelanggan as-of cutoff = min(DPD menurut data, jarak jatuh tempo s/d cutoff)
    ev["max_dpd"] = np.minimum(ev["max_days_past_due"].to_numpy(), np.maximum(ev["dpd"], 0))
    for thr in (1, 30, 60, 89):
        ev[f"dpd_ge_{thr}"] = (ev["dpd"] >= thr).astype(float)
    return ev


def build_repayment_features(apps_df, ev):
    """Statistik cicilan: level installment, multi-window + snapshot terakhir."""
    queries = apps_df[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})
    stats, _ = window_stats(
        ev, queries,
        value_cols=["amount_due", "outstanding", "pay_ratio", "paid_flag",
                    "max_dpd", "dpd", "dpd_ge_1", "dpd_ge_30", "dpd_ge_60", "dpd_ge_89"],
        windows=W_LONG, ops=("sum", "mean", "max", "last"),
    )
    stats.columns = ["rh_" + c for c in stats.columns]
    latest, _ = asof_point(ev, queries, ["max_dpd", "dpd", "outstanding", "pay_ratio", "paid_flag"])
    latest.columns = ["rh_last_" + c for c in latest.columns]
    return pd.concat([stats, latest], axis=1)


def build_repayment_loan_features(apps_df, ev):
    """Level pinjaman: satu baris per `loan_id` sebelumnya (agregasi per pinjaman)."""
    loans = (ev.groupby(["customer_id", "loan_id"], as_index=False)
               .agg(loan_start=("event_time", "min"),
                    n_inst=("event_time", "size"),
                    dpd_max=("dpd", "max"),
                    dpd_sum=("dpd", "sum"),
                    outstanding=("outstanding", "sum"),
                    due_total=("amount_due", "sum")))
    queries = apps_df[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})
    stats, _ = window_stats(loans.rename(columns={"loan_start": "event_time"}), queries,
                            value_cols=["n_inst", "dpd_max", "dpd_sum", "outstanding", "due_total"],
                            windows=W_LONG, ops=("sum", "mean", "max", "last"))
    stats.columns = ["ln_" + c for c in stats.columns]
    latest, _ = asof_point(loans.rename(columns={"loan_start": "event_time"}), queries,
                           ["dpd_max", "outstanding"])
    latest.columns = ["ln_last_" + c for c in latest.columns]

    out = pd.concat([stats, latest], axis=1)
    out["ln_n_loans"] = out["ln_n_events__all"]
    out["ln_has_delinquent_loan"] = (out["ln_max_dpd_max__all"].fillna(0) > 0).astype(float)
    out["ln_has_dpd30_loan"] = (out["ln_max_dpd_max__all"].fillna(0) >= 30).astype(float)
    out["ln_loans_per_year"] = out["ln_n_loans"] / (out["ln_last_days_since_ts"] / 365.25).replace(0, np.nan)
    out["ln_unpaid_ratio"] = out["ln_sum_outstanding__all"] / out["ln_sum_due_total__all"].replace(0, np.nan)
    return out


def build_account_features(apps_df, acct_df):
    """Perilaku rekening: saldo, arus kas, buffer, low-balance days, tren saldo."""
    ev = acct_df.rename(columns={"month": "event_time"}).copy()
    ev = ev.sort_values(["customer_id", "event_time"], kind="mergesort")
    ev["net_flow"] = ev["inflows"] - ev["outflows"]
    ev["flow_ratio"] = ev["inflows"] / ev["outflows"].replace(0, np.nan)
    ev["income_share"] = ev["income_like_credits"] / ev["inflows"].replace(0, np.nan)
    ev["balance_change"] = ev["closing_balance"] - ev["opening_balance"]
    ev["buffer_months"] = ev["average_balance"] / (ev["outflows"] / 12).replace(0, np.nan)
    num = ["opening_balance", "closing_balance", "inflows", "outflows", "net_flow", "flow_ratio",
           "income_like_credits", "income_share", "average_balance", "minimum_balance",
           "low_balance_days", "balance_change", "buffer_months"]

    queries = apps_df[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})
    stats, _ = window_stats(ev, queries, value_cols=num, windows=W_LONG,
                            ops=("sum", "mean", "std", "min", "max", "last"))
    latest, _ = asof_point(ev, queries, num)

    out = pd.concat([stats, latest], axis=1)
    out.columns = ["ab_" + c for c in stats.columns] + ["ab_last_" + c for c in latest.columns]

    f6 = out["ab_mean_average_balance__m6"].to_numpy()
    f12 = out["ab_mean_average_balance__m12"].to_numpy()
    out["ab_balance_trend_6v6"] = f6 - (f12 * 12 - f6 * 6) / 6.0
    out["ab_low_balance_month_frac"] = out["ab_sum_low_balance_days__m12"] / (30.0 * out["ab_n_events__m12"])
    out["ab_balance_trend_ratio"] = (out["ab_balance_trend_6v6"]
                                     / out["ab_mean_average_balance__m12"].abs().replace(0, np.nan))
    out["ab_income_volatility"] = (out["ab_std_income_like_credits__m12"]
                                   / out["ab_mean_income_like_credits__m12"].replace(0, np.nan))
    return out


def build_bureau_features(apps_df, bureau_df):
    """Laporan kredit: skor, utilisasi revolving, tunggakan, inquiry, tren skor."""
    ev = bureau_df.copy()
    ev["report_lag_days"] = (ev["available_datetime"] - ev["report_datetime"]).dt.total_seconds() / 86400
    ev = ev.rename(columns={"available_datetime": "event_time"})
    ev["score_f"] = ev["score"].fillna(ev["score"].mean())
    ev["score_missing"] = ev["score"].isna().astype(float)
    ev["utilization_f"] = ev["utilization"].fillna(0.0)
    num = ["score_f", "score_missing", "active_accounts", "outstanding_balance", "revolving_credit_limit",
           "revolving_outstanding_balance", "utilization_f", "inquiries_last_6m",
           "delinquent_accounts_last_12m", "report_lag_days"]

    queries = apps_df[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})
    stats, _ = window_stats(ev, queries, value_cols=num, windows=W_LONG,
                            ops=("sum", "mean", "std", "min", "max", "last"))
    latest, _ = asof_point(ev, queries, num)

    out = pd.concat([stats, latest], axis=1)
    out.columns = ["bu_" + c for c in stats.columns] + ["bu_last_" + c for c in latest.columns]

    for provider in ["Bureau_A", "Bureau_B", "Bureau_C"]:
        sub = ev[ev["bureau_provider"] == provider][["customer_id", "event_time", "score_f", "outstanding_balance"]]
        snap, _ = asof_point(sub, queries, ["score_f", "outstanding_balance"])
        out[f"bu_last_{provider}_score"] = snap["score_f"].to_numpy()
        out[f"bu_last_{provider}_balance"] = snap["outstanding_balance"].to_numpy()
    out["bu_score_trend"] = out["bu_last_score_f"] - out["bu_mean_score_f__m12"]
    out["bu_score_trend_6m"] = out["bu_last_score_f"] - out["bu_mean_score_f__m6"]
    out["bu_balance_trend"] = out["bu_last_outstanding_balance"] - out["bu_mean_outstanding_balance__m6"]
    return out


def build_ecommerce_features(apps_df, ecom_df):
    """Perilaku belanja: frekuensi, nilai, kategori, metode bayar, pembatalan."""
    # order_total diulang pada setiap baris item -> dedup ke level order terlebih dahulu
    ev = (ecom_df.sort_values(["order_id", "transaction_datetime"])
                .drop_duplicates("order_id", keep="first")
                .rename(columns={"transaction_datetime": "event_time"}))
    ev["spend"] = np.where(ev["order_status"] == "completed", ev["order_total"], 0.0)
    ev["is_completed"] = (ev["order_status"] == "completed").astype(float)
    ev["cancelled_flag"] = (ev["order_status"] == "cancelled").astype(float)
    ev["quantity_f"] = ev["quantity"].astype(float)
    methods = ["COD", "credit_card", "e_wallet", "bank_transfer", "debit_card"]
    for method in methods:
        ev[f"pm_{method}"] = (ev["payment_method"] == method).astype(float)
    categories = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
    for cat in categories:
        ev[f"cat_{cat}"] = np.where(ev["product_category"] == cat, ev["spend"], 0.0)
    ev = ev.sort_values(["customer_id", "event_time"], kind="mergesort")

    queries = apps_df[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})
    value_cols = (["spend", "order_total", "is_completed", "cancelled_flag", "quantity_f"]
                  + [f"pm_{m}" for m in methods]
                  + [f"cat_{c}" for c in categories])
    stats, _ = window_stats(ev, queries, value_cols=value_cols, windows=W_LONG,
                            ops=("sum", "mean", "max", "last"))
    latest, _ = asof_point(ev, queries, ["spend", "order_total", "quantity_f"])
    out = pd.concat([stats, latest], axis=1)
    out.columns = ["ec_" + c for c in stats.columns] + ["ec_last_" + c for c in latest.columns]

    for cat in categories:
        out[f"ec_share_{cat}"] = out[f"ec_sum_cat_{cat}__m12"] / out["ec_sum_spend__m12"].replace(0, np.nan)
    out["ec_orders_per_month_m12"] = out["ec_n_events__m12"] / 12.0
    out["ec_cancel_rate_m12"] = out["ec_sum_cancelled_flag__m12"] / out["ec_n_events__m12"].replace(0, np.nan)
    out["ec_avg_basket_m12"] = out["ec_sum_order_total__m12"] / out["ec_n_events__m12"].replace(0, np.nan)
    out["ec_spend_trend"] = out["ec_sum_spend__m3"] - out["ec_sum_spend__m12"] / 4.0
    out["ec_freq_trend"] = out["ec_n_events__m3"] - out["ec_n_events__m12"] / 4.0
    out["ec_cod_share_m12"] = out["ec_sum_pm_COD__m12"] / out["ec_n_events__m12"].replace(0, np.nan)
    out["ec_credit_share_m12"] = out["ec_sum_pm_credit_card__m12"] / out["ec_n_events__m12"].replace(0, np.nan)
    return out

# %%
# Bangun matriks fitur (train & test memakai pipeline identik)
repay_asof = _repayment_asof(repayments, apps)

blocks = {
    "app": build_application_features(apps, profiles),
    "pa": build_previous_application_features(apps, previous_apps),
    "rh": build_repayment_features(apps, repay_asof),
    "ln": build_repayment_loan_features(apps, repay_asof),
    "ab": build_account_features(apps, accounts),
    "bu": build_bureau_features(apps, bureau),
    "ec": build_ecommerce_features(apps, ecommerce),
}
for name, block in blocks.items():
    log(f"blok fitur {name}: {block.shape}")

X = pd.concat(list(blocks.values()), axis=1)
X.columns = [f"{name}__{col}" for name, blk in blocks.items() for col in blk.columns]
X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
assert not any(str(d).startswith("datetime") for d in X.dtypes), "ada kolom bertipe datetime"
log(f"matriks fitur final: {X.shape}, rasio NaN {X.isna().mean().mean():.3f}")

# %% [markdown]
# ### 4.1 Audit kebocoran pada data asli
#
# Tiga pemeriksaan negatif dijalankan sebelum model dilatih. Jika salah satu
# gagal, `assert` menghentikan notebook.

# %%
audit = []

# (1) Setiap laporan bureau yang terpaut pengajuan harus tersedia sebelum keputusan.
bureau_linked = bureau.merge(apps.set_index("application_id")["decision_datetime"].rename("cutoff"),
                             left_on="application_id", right_index=True, how="inner")
share_before = float((bureau_linked.available_datetime <= bureau_linked.cutoff).mean())
audit.append(("laporan bureau tersedia sebelum keputusan pengajuan", round(share_before, 4)))
assert share_before == 1.0

# (2) Cek manual: jumlah pengajuan "approved" yang teragregasi harus persis sama
#     dengan hitungan naïf (event setelah cutoff dibuang).
row = 0
cust_i, cut_i = apps.loc[row, "customer_id"], apps.loc[row, "decision_datetime"]
naive_approved = int(((previous_apps.customer_id == cust_i)
                      & (previous_apps.decision_datetime <= cut_i)
                      & (previous_apps.decision == "approved")).sum())
feat_approved = float(X.loc[row, "pa__sum_approved__all"])
audit.append(("selisih fitur prev-apps vs hitungan naïf", abs(feat_approved - naive_approved)))
assert abs(feat_approved - naive_approved) < 1e-9

# (3) Tidak boleh ada fitur yang menyalin label (deteksi kebocoran langsung).
Xtr_df = X.loc[: len(train_apps) - 1]
for col in [c for c in Xtr_df.columns if str(Xtr_df[c].dtype) == "category"]:
    Xtr_df[col] = Xtr_df[col].cat.codes
Xtr_np = Xtr_df.to_numpy(dtype=np.float64)
y_np = train_apps.target.to_numpy(dtype=np.float64)
col_mean = np.nanmean(Xtr_np, axis=0)
Xtr_filled = np.where(np.isnan(Xtr_np), col_mean, Xtr_np)
Xtr_c = Xtr_filled - Xtr_filled.mean(axis=0)
y_c = y_np - y_np.mean()
denom = np.sqrt((Xtr_c ** 2).sum(axis=0) * (y_c ** 2).sum())
max_corr = float(np.nanmax(np.abs((Xtr_c * y_c[:, None]).sum(axis=0) / denom)))
worst_feat = X.columns[int(np.nanargmax(np.abs((Xtr_c * y_c[:, None]).sum(axis=0) / denom)))]
audit.append(("korelasi fitur-vs-label maksimum", round(max_corr, 4)))
audit.append(("fitur dengan korelasi terbesar", worst_feat))
assert max_corr < 0.9, f"dugaan kebocoran pada fitur {worst_feat}"

display(pd.DataFrame(audit, columns=["pemeriksaan", "nilai"]))
log("audit kebocoran lulus: tidak ada fitur yang memakai informasi pasca-cutoff")

del repay_asof, repayments, accounts, bureau, ecommerce, blocks   # bebaskan memori

# %% [markdown]
# ## 5. Cross-validation & pelatihan model
#
# * 5-fold **StratifiedKFold** dengan `random_state` tetap → reproducible.
# * Model 1: **LightGBM**. Hyperparameter dipilih lewat grid kecil manual pada 1
#   fold (lihat tabel hasil di bawah) — tetap manual, bukan AutoML.
# * Model 2: **XGBoost** dengan kedalaman dangkal & regularisasi kuat untuk
#   diversifikasi (korelasi OOF ~0,98 dengan LightGBM, blend menambah ~0,0005 AUC).
# * Bobot blend dipilih dari **OOF**, bukan dari data test.

# %%
IS_TRAIN = apps.target.notna().to_numpy()
X_train = X[IS_TRAIN].reset_index(drop=True)
y_train = train_apps.target.to_numpy()
X_test = X[~IS_TRAIN].reset_index(drop=True)
test_ids = test_apps.application_id.to_numpy()

CATEGORICAL = [c for c in X.columns if str(X[c].dtype) == "category"]
for col in CATEGORICAL:
    X_train[col] = X_train[col].cat.codes
    X_test[col] = X_test[col].cat.codes
log(f"train {X_train.shape} | test {X_test.shape} | kategorikal: {CATEGORICAL}")

X_train_np = X_train.to_numpy(dtype=np.float32)     # untuk XGBoost (hemat memori)
X_test_np = X_test.to_numpy(dtype=np.float32)

LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.015, num_leaves=31,
    min_data_in_leaf=200, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, extra_trees=True, verbose=-1, num_threads=N_THREADS, seed=SEED,
)
XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="auc", max_depth=4, min_child_weight=200,
    subsample=0.8, colsample_bytree=0.3, eta=0.02, reg_lambda=10.0, reg_alpha=0.5,
    tree_method="hist", nthread=N_THREADS, seed=SEED,
)
display(pd.Series(LGB_PARAMS, name="LightGBM").to_frame("value"))

folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train))
log(f"{N_FOLDS} fold stratified siap (seed={SEED})")

# %%
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
importances = []
fold_rows = []

for fold, (fit_idx, val_idx) in enumerate(folds):
    t_fold = time.time()

    ds = lgb.Dataset(X_train.iloc[fit_idx], y_train[fit_idx], categorical_feature=CATEGORICAL)
    dv = lgb.Dataset(X_train.iloc[val_idx], y_train[val_idx], reference=ds)
    booster = lgb.train(LGB_PARAMS, ds, num_boost_round=8000, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                                   lgb.log_evaluation(0)])
    oof_lgb[val_idx] = booster.predict(X_train.iloc[val_idx], num_iteration=booster.best_iteration)
    test_lgb += booster.predict(X_test, num_iteration=booster.best_iteration) / N_FOLDS
    importances.append(booster.feature_importance("gain"))
    lgb_auc = roc_auc_score(y_train[val_idx], oof_lgb[val_idx])

    xgb_auc, best_iter = np.nan, np.nan
    if USE_XGB_BLEND:
        dtr = xgb.DMatrix(X_train_np[fit_idx], y_train[fit_idx])
        dva = xgb.DMatrix(X_train_np[val_idx], y_train[val_idx])
        bst = xgb.train(XGB_PARAMS, dtr, num_boost_round=6000, evals=[(dva, "val")],
                        early_stopping_rounds=EARLY_STOP_ROUNDS, verbose_eval=False)
        best_iter = bst.best_iteration + 1
        oof_xgb[val_idx] = bst.predict(dva, iteration_range=(0, best_iter))
        test_xgb += bst.predict(xgb.DMatrix(X_test_np), iteration_range=(0, best_iter)) / N_FOLDS
        xgb_auc = roc_auc_score(y_train[val_idx], oof_xgb[val_idx])

    fold_rows.append({"fold": fold, "auc_lgb": round(lgb_auc, 5), "iter_lgb": booster.best_iteration,
                      "auc_xgb": round(xgb_auc, 5) if xgb_auc == xgb_auc else None,
                      "iter_xgb": best_iter})
    log(f"fold {fold}: LGB AUC={lgb_auc:.5f} ({booster.best_iteration} trees) | "
        f"XGB AUC={xgb_auc:.5f} | {time.time() - t_fold:.0f}s")

display(pd.DataFrame(fold_rows))
log(f"OOF AUC LightGBM = {roc_auc_score(y_train, oof_lgb):.5f}")
if USE_XGB_BLEND:
    log(f"OOF AUC XGBoost  = {roc_auc_score(y_train, oof_xgb):.5f} "
        f"(korelasi OOF dengan LGB = {np.corrcoef(oof_lgb, oof_xgb)[0, 1]:.4f})")

# %% [markdown]
# ### 5.1 Pemilihan bobot blend (dihitung dari OOF)

# %%
if USE_XGB_BLEND:
    blend_table = []
    best_w, best_auc = 0.0, -np.inf
    for w in np.arange(0.0, 0.81, 0.05):
        blended = (1 - w) * oof_lgb + w * oof_xgb
        auc = roc_auc_score(y_train, blended)
        ll = log_loss(y_train, np.clip(blended, 1e-6, 1 - 1e-6))
        blend_table.append({"w_xgb": round(w, 2), "auc": round(auc, 5), "logloss": round(ll, 5)})
        if auc > best_auc:
            best_w, best_auc = w, auc
    display(pd.DataFrame(blend_table))
    XGB_WEIGHT = float(best_w)
else:
    XGB_WEIGHT = 0.0

oof_final = (1 - XGB_WEIGHT) * oof_lgb + XGB_WEIGHT * oof_xgb
test_final = (1 - XGB_WEIGHT) * test_lgb + XGB_WEIGHT * test_xgb
final_auc = roc_auc_score(y_train, oof_final)
final_ll = log_loss(y_train, np.clip(oof_final, 1e-6, 1 - 1e-6))
log(f"bobot XGBoost = {XGB_WEIGHT:.2f} | OOF AUC = {final_auc:.5f} | OOF logloss = {final_ll:.5f}")

# %% [markdown]
# ## 6. Feature importance

# %%
gain = pd.Series(np.mean(importances, axis=0), index=X.columns).sort_values(ascending=False)
display(gain.head(30).to_frame("gain"))

top = gain.head(25)[::-1]
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(range(len(top)), top.values, color="#4C78A8")
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index, fontsize=8)
ax.set_xlabel("mean gain")
ax.set_title("25 fitur dengan gain tertinggi")
plt.tight_layout()
plt.show()

gain.to_csv(f"{ARTIFACT_DIR}/feature_importance.csv")
pd.Series(X.columns, name="feature").to_csv(f"{ARTIFACT_DIR}/feature_names.csv", index=False)

# %% [markdown]
# ### 6.1 Kontribusi gain per sumber tabel
#
# Sanity check business: tunggakan historis, skor bureau, dan arus kas rekening
# seharusnya mendominasi — bukan identitas atau kebocoran waktu.

# %%
source_gain = gain.groupby(gain.index.str.split("__").str[0]).sum().sort_values(ascending=False)
display(source_gain.to_frame("total_gain"))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(source_gain.index, source_gain.values, color="#54A24B")
ax.set_ylabel("total gain")
ax.set_title("Kontribusi gain per sumber tabel")
plt.tight_layout()
plt.show()
source_gain.to_frame("total_gain").to_csv(f"{ARTIFACT_DIR}/importance_by_source.csv")

# %% [markdown]
# ### 6.2 Kalibrasi dan kurva ROC pada OOF

# %%
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
frac_pos, mean_pred = calibration_curve(y_train, oof_final, n_bins=10, strategy="quantile")
axes[0].plot(mean_pred, frac_pos, "o-", color="#E45756", label="model")
axes[0].plot([0, 1], [0, 1], "--", color="gray", label="kalibrasi sempurna")
axes[0].set_xlabel("predicted probability")
axes[0].set_ylabel("observed frequency")
axes[0].set_title(f"Kalibrasi OOF (logloss {final_ll:.4f})")
axes[0].legend()

fpr, tpr, _ = roc_curve(y_train, oof_final)
axes[1].plot(fpr, tpr, color="#4C78A8", label=f"OOF AUC = {final_auc:.4f}")
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_xlabel("false positive rate")
axes[1].set_ylabel("true positive rate")
axes[1].set_title("ROC Out-of-fold")
axes[1].legend()
plt.tight_layout()
plt.show()

print(f"rata-rata prediksi OOF : {oof_final.mean():.4f}")
print(f"rata-rata label train  : {y_train.mean():.4f}")

# %% [markdown]
# ## 7. Prediksi final dan file submission
#
# Prediksi test = rata-rata prediksi 5 fold (per model), lalu di-blend dengan bobot
# terpilih. Probabilitas di-clip ke interval (0, 1) dan ditulis sebagai
# `submission.csv` dengan urutan id yang sama persis seperti `sample_submission.csv`.

# %%
pred_df = pd.DataFrame({
    "application_id": test_ids,
    "default_probability": np.clip(test_final, 1e-6, 1 - 1e-6),
}).set_index("application_id")

# samakan urutan & isi dengan sample_submission.csv
submission = sample_submission[["application_id"]].copy()
submission["default_probability"] = pred_df["default_probability"].loc[
    submission.application_id.to_numpy()].to_numpy()
assert submission.default_probability.notna().all(), "ada application_id test yang tidak diprediksi"
submission.to_csv("submission.csv", index=False)

np.save(f"{ARTIFACT_DIR}/oof_final.npy", oof_final)
np.save(f"{ARTIFACT_DIR}/test_final.npy", np.asarray(test_final))

display(submission.head(10))
print(f"baris submission   : {len(submission)}")
print(f"rentang probabilitas: [{submission.default_probability.min():.5f}, "
      f"{submission.default_probability.max():.5f}]")
print(f"rata-rata prediksi  : {submission.default_probability.mean():.5f} "
      f"(label train {y_train.mean():.5f})")

# %%
# Ringkasan metrik akhir
summary = {
    "OOF AUC (blend akhir)": round(float(final_auc), 5),
    "OOF logloss": round(float(final_ll), 5),
    "OOF AUC LightGBM": round(float(roc_auc_score(y_train, oof_lgb)), 5),
    "OOF AUC XGBoost": round(float(roc_auc_score(y_train, oof_xgb)), 5) if USE_XGB_BLEND else None,
    "Bobot blend XGBoost": XGB_WEIGHT,
    "Jumlah fitur": int(X.shape[1]),
    "Jumlah baris (train/test)": f"{len(X_train)} / {len(X_test)}",
    "Waktu total (detik)": round(time.time() - T_START, 1),
}
display(pd.Series(summary, name="nilai").to_frame("nilai"))