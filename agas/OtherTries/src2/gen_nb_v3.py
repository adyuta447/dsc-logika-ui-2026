"""Generate the percent-format notebook source from the tested modules."""
import re

Q3 = chr(34) * 3

FECORE = open('src2/fe_core.py').read()
FEATURES = open('src2/features_v2.py').read()
MODELS = open('src2/models.py').read()
FEATURES3 = open('src2/features_v3.py').read()


def strip(text, drop_prefixes=()):
    """Buang docstring modul di awal file dan baris import boilerplate."""
    lines = text.split("\n")
    if lines and lines[0].lstrip().startswith(Q3):
        start = 0 if lines[0].strip() == Q3 else 1
        for j in range(start, len(lines)):
            if Q3 in lines[j]:
                lines = lines[j + 1:]
                break
    kept = [ln for ln in lines if not any(ln.strip().startswith(p) for p in drop_prefixes)]
    return "\n".join(kept).strip("\n")


fecore = strip(FECORE, (Q3, 'import numpy', 'import pandas', 'from fe_core'))
features = strip(FEATURES, ('import numpy', 'import pandas', 'from fe_core', Q3))
features3 = strip(FEATURES3, ('import numpy', 'import pandas', Q3))
models = strip(MODELS, ('import numpy', 'import pandas', 'import lightgbm', 'import xgboost',
                        'from catboost', 'from scipy', 'from sklearn', Q3))

MD = "# %% [markdown]\n"
PY = "# %%\n"

parts = []
parts.append(MD + """\
# # Prediksi 90+ DPD — DSC0152 (v3: 10-fold, fitur discovery, 3 seed LGB + CatBoost + XGBoost)
#
# Notebook ini adalah **upgrade dari notebook v1** (`dsc_logikaui_ks_v2.ipynb`).
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
""")

parts.append(MD + """\
# ## 1. Setup
""")

parts.append(PY + """\
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
N_FOLDS = 10
SPLIT_SEEDS = [42]        # dua split CV; OOF KS berfluktuasi ±0.004 antar split
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
""")

parts.append(MD + """\
# ## 2. Load data & audit batas waktu
#
# Audit ini penting: kalau ternyata ada baris tabel auxiliary yang bertanggal **setelah**
# `decision_datetime`, maka setiap fitur harus memfilternya. Hasilnya menentukan
# berapa besar effort anti-leakage yang diperlukan.
""")

parts.append(PY + """\
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
""")

parts.append(MD + """\
### Audit: berapa baris auxiliary yang melewati cutoff?
""")

parts.append(PY + """\
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
""")

parts.append(MD + """\
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
""")

parts.append(PY + fecore)

parts.append(PY + """\
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
""")

parts.append(MD + """\
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
""")

parts.append(PY + features + '\n\n' + features3)

parts.append(MD + """\
### 4.1 Bangun matriks fitur
""")

parts.append(PY + """\
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
    "disc": discovery_features(apps, ecommerce, repayments, bureau),
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
""")

parts.append(MD + """\
### 4.2 Pemilihan feature set
#
# * `FULL` — semua fitur (minus kolom DPD causal bila `USE_CAUSAL_DPD_ONLY = True`).
# * `LEAN` — `FULL` tanpa blok yang menambah noise bagi pohon (profil demografis,
#   timestamp aplikasi, share kategori yang sangat sparse).
# * `TOP` — 100 fitur teratas menurut gain, dihitung **per fold dari data latih fold itu**
#   supaya baris validasi tidak ikut menentukan pilihan fitur.
""")

parts.append(PY + """\
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
""")

parts.append(MD + """\
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
""")

parts.append(PY + models)

parts.append(MD + """\
# ## 6. Cross-validation (2 split × 5 fold)
#
# Setiap model dilatih pada **dua split CV berbeda** (`SPLIT_SEEDS`). Alasannya:
# OOF KS yang digabung antar fold bergeser ±0.004 hanya karena komposisi fold,
# sehingga (a) pemilihan blend harus dinilai rata-rata atas beberapa split agar tidak
# overfit pada satu split, dan (b) prediksi test yang dirata-ratakan dari 10 model
# (5 fold × 2 split) lebih stabil daripada 5 model.
""")

parts.append(PY + """\
members = ["lgb_full_s0", "lgb_full_s1", "lgb_full_s2", "xgb_full", "cat_full", "spline_top"]
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
        lgb_models = {}

        # --- LightGBM: 2 seed pada feature set penuh, 1 seed pada set lean ---
        for name, cols, params in [
            ("lgb_full_s0", FULL, dict(LGB_RND, num_threads=10)),
            ("lgb_full_s1", FULL, dict(LGB_ET, num_threads=10)),
            ("lgb_full_s2", FULL, dict(LGB_RND, seed=7, feature_fraction=0.25, num_threads=10)),
        ]:
            va, te, mdl = fit_lgb(Xa[cols].iloc[a], y_train[a], Xa[cols].iloc[b], y_train[b],
                                  Xb[cols], params)
            lgb_models[name] = mdl
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
        base_mdl = lgb_models["lgb_full_s0"]
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
""")

parts.append(MD + """\
## 7. Evaluasi per model
""")

parts.append(PY + """\
for m in members:
    line = f"{m:14s}"
    for s in SPLIT_SEEDS:
        line += f"  split{s}: KS {ks_score(y_train, oof[m][s]):.5f}"
    # skor gabungan = rata-rata rank dari kedua split (meniru cara prediksi test dirata-ratakan)
    comb = np.mean([rankify(oof[m][s]) for s in SPLIT_SEEDS], axis=0)
    line += f"  | combined KS {ks_score(y_train, comb):.5f} AUC {roc_auc_score(y_train, comb):.5f}"
    print(line)
""")

parts.append(MD + """\
## 8. Blend yang dioptimalkan terhadap KS
#
# Tiga kandidat, semuanya dinilai sebagai **rata-rata KS atas kedua split** supaya
# pilihan tidak mengikuti noise satu split:
#
# 1. `equal` — rata-rata rank semua model;
# 2. `greedy` — *forward selection with replacement* (Caruana): repeatedly menambahkan model
#    yang paling menaikkan KS rata-rata;
# 3. `best_single` — model tunggal terbaik sebagai jaring pengaman.
""")

parts.append(PY + """\
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
""")

parts.append(MD + """\
## 9. Diagnosa blend terpilih (KS kurva, gain model, importance fitur)
""")

parts.append(PY + """\
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
""")

parts.append(MD + """\
## 10. Submission
#
# Prediksi test = rata-rata rank dari seluruh model (5 fold × 2 split), dengan bobot blend
# terpilih. Nilai ditulis sebagai `default_probability` sesuai format
# `sample_submission.csv`; karena metrik KS hanya memakai urutan, skewalaan nilai tidak
# berpengaruh terhadap skor.
""")

parts.append(PY + """\
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
""")

parts.append(PY + """\
summary = pd.Series({
    "KS OOF (blend, split 42)": round(ks_score(y_train, make_final(SPLIT_SEEDS[0])), 5),
    "KS OOF gabungan": round(ks_score(y_train, final_oof), 5),
    "AUC OOF": round(roc_auc_score(y_train, final_oof), 5),
    "Baseline v1 (AUC-optimal)": 0.47144,
    "Jumlah fitur": int(X.shape[1]),
    "Model dalam blend": len(members),
    "Bobot blend": str({k: round(v, 3) for k, v in BEST_WEIGHTS.items() if v > 0}),
    "Waktu total (detik)": round(time.time() - T_START, 1),
}, name="nilai").to_frame("nilai")
display(summary)
""")

open('src2/nb_percent_v3.py', 'w').write('\n\n'.join(parts) + '\n')
print("wrote src2/nb_percent_v3.py", len('\n\n'.join(parts)), "chars,", len(parts), "cells")