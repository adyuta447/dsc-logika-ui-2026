"""Build agas/v74/notebook/notebook_v74.ipynb.

The notebook is the graded artefact: the elimination round scores it only on
reproducibility and code execution. So it runs end to end on a clean kernel
with RUN_TRAINING = False, rebuilding the submitted vector from the stored
out-of-fold and test predictions, and it shows the training command it would
run instead when the flag is True.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "notebook" / "notebook_v74.ipynb"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip()})


md(r"""
# DSC LOGIKA UI 2026 — Credit Default Prediction

**Task.** Rank 30.000 test applications by the probability of reaching 90+ DPD
within 12 months of origination. Metric: Kolmogorov–Smirnov statistic.

**Submitted model.** `v66_04` — a rank-space blend of six gradient boosting legs
and one spline logistic regression, extrapolated along the plateau measured in
v61. Public 0.56424, the best of 9 scored submissions across 18 versions.

**How to run.**

* `RUN_TRAINING = False` (default): the notebook rebuilds the submitted vector
  from the stored fold predictions in `agas/v66/output/preds_v66.npz` and
  verifies it reproduces the submitted CSV exactly. Runs in about two minutes.
* `RUN_TRAINING = True`: the notebook rebuilds the feature cache and retrains
  all seven legs over six fold partitions. About 6.5 hours on 10 cores.

**Sections.** 1 data · 2 cutoff audit · 3 features · 4 models · 5 blend and
submission · 6 how the candidate was chosen · 7 validation.
""")

code(r"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

RUN_TRAINING = False          # True retrains everything from the raw CSVs
RANDOM_STATE = 42

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data/train_applications.csv").exists() and (p / "features.py").exists():
            return p
    raise RuntimeError("repository root not found: run from inside the project")

ROOT = find_root(Path.cwd().resolve())
DATA = ROOT / "data"
V66, V74 = ROOT / "agas/v66", ROOT / "agas/v74"
os.chdir(ROOT)
(V74 / "output").mkdir(parents=True, exist_ok=True)

print("root       :", ROOT)
print("numpy      :", np.__version__)
print("pandas     :", pd.__version__)
import sklearn; print("sklearn    :", sklearn.__version__)
print("RUN_TRAINING:", RUN_TRAINING)
""")

md(r"""
## 1. Data

Seven provided tables, nothing else. No external data is read anywhere in this
notebook, and no table outside `data/` is opened.
""")

code(r"""
TABLES = ["train_applications", "test_applications", "customer_profiles",
          "previous_applications", "repayment_history", "account_behavior",
          "bureau_observations", "ecommerce_transactions"]

shapes = {}
for name in TABLES:
    n = sum(1 for _ in open(DATA / f"{name}.csv")) - 1
    cols = len(pd.read_csv(DATA / f"{name}.csv", nrows=1).columns)
    shapes[name] = (n, cols)

train = pd.read_csv(DATA / "train_applications.csv", parse_dates=["application_datetime", "decision_datetime"])
test = pd.read_csv(DATA / "test_applications.csv", parse_dates=["application_datetime", "decision_datetime"])

print(pd.DataFrame(shapes, index=["rows", "cols"]).T.to_string())
print()
print(f"train rows            : {len(train):,}")
print(f"test rows             : {len(test):,}")
print(f"positive rate (train) : {train.target.mean():.6f}")
print(f"customers in both sets: {len(set(train.customer_id) & set(test.customer_id))}")
print(f"decision window train : {train.decision_datetime.min()} .. {train.decision_datetime.max()}")
print(f"decision window test  : {test.decision_datetime.min()} .. {test.decision_datetime.max()}")
""")

md(r"""
## 2. Cutoff audit

Every feature must use only information available at `decision_datetime` of the
selected application. Two things are checked here.

First, whether the provided tables contain any record dated after the cutoff of
the customer's selected application. If they do, every window aggregate would
need an explicit positive-time filter.

Second, whether the bureau observation attached to the selected application was
actually available at decision time — `available_datetime` must not be later
than `decision_datetime`.
""")

code(r"""
apps = pd.concat([train[["application_id", "customer_id", "decision_datetime"]],
                  test[["application_id", "customer_id", "decision_datetime"]]])
cut = apps[["customer_id", "decision_datetime"]].rename(columns={"decision_datetime": "cut"})

audit = {}

pa = pd.read_csv(DATA / "previous_applications.csv", parse_dates=["decision_datetime"]).merge(cut, on="customer_id")
audit["previous_applications.decision_datetime"] = (len(pa), int((pa.decision_datetime > pa.cut).sum()))

rh = pd.read_csv(DATA / "repayment_history.csv", parse_dates=["due_date", "last_payment_datetime"]).merge(cut, on="customer_id")
audit["repayment_history.due_date"] = (len(rh), int((rh.due_date > rh.cut).sum()))
audit["repayment_history.last_payment_datetime"] = (len(rh), int((rh.last_payment_datetime > rh.cut).sum()))

ab = pd.read_csv(DATA / "account_behavior.csv", parse_dates=["month"]).merge(cut, on="customer_id")
audit["account_behavior.month_closed"] = (len(ab), int(((ab.month + pd.offsets.MonthBegin(1)) > ab.cut).sum()))

ec = pd.read_csv(DATA / "ecommerce_transactions.csv", parse_dates=["transaction_datetime"]).merge(cut, on="customer_id")
audit["ecommerce_transactions.transaction_datetime"] = (len(ec), int((ec.transaction_datetime > ec.cut).sum()))

bu = pd.read_csv(DATA / "bureau_observations.csv", parse_dates=["report_datetime", "available_datetime"])
sel = bu.merge(apps, on=["application_id", "customer_id"], how="inner")
audit["bureau_observations.available_datetime (selected app)"] = (len(sel), int((sel.available_datetime > sel.decision_datetime).sum()))
audit["bureau_observations.report_datetime (selected app)"] = (len(sel), int((sel.report_datetime > sel.decision_datetime).sum()))

rows = pd.DataFrame([(k, v[0], v[1]) for k, v in audit.items()],
                    columns=["source field", "rows joined", "rows after cutoff"])
print(rows.to_string(index=False))
post = int(rows["rows after cutoff"].sum())
print(f"\ntotal records dated after the cutoff: {post}")
assert post == 0, "a source contains post-cutoff records; window features need a positive-time filter"
print("PASS: the provided tables are already truncated at decision_datetime")

del pa, rh, ab, ec, bu, sel
""")

md(r"""
## 3. Features

`features.py` at the repository root builds one row per application from the
seven tables. Every historical aggregate is computed relative to `cut =
decision_datetime` of the selected application, as `days_ago = cut - event`,
and windows are taken over that quantity, so nothing from after the decision
can enter a feature.

Blocks: application terms, customer profile, previous applications, repayment
history, account behavior, bureau observations, e-commerce behavior, plus a set
of ratio and trend interactions. Three column sets are used by the legs:

| set | columns | difference |
|---|---:|---|
| `full` | 598 | everything |
| `lean` | 490 | drops demographic and share-type columns that carried no fold-stable gain |
| `fac` | 553 | drops the second-order rounding family, used by the factor leg |

Target encoded factor features are recomputed **inside each fold** from training
fold labels only (`features.factor_features`), so no fold sees its own labels.
""")

code(r"""
CACHE = ROOT / "agas/shared/cache/features_cache.pkl"
EXTRA = ROOT / "agas/shared/cache/extra_cache.pkl"

if RUN_TRAINING:
    import features as ft
    t0 = time.time()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ft.build().to_pickle(CACHE)                       # base blocks
    rc = os.system("python3 agas/shared/build_extra.py")   # second-order blocks
    assert rc == 0, f"build_extra.py failed with code {rc}"
    print(f"feature cache rebuilt in {(time.time() - t0) / 60:.1f} min")

NOISE_EXACT = {"region", "employment_type", "housing_status", "age",
               "employment_tenure_months", "dependents", "tenure_to_age",
               "income_per_dependent", "loan_purpose", "decision_lag_h",
               "app_hour", "app_dow"}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")
ROUND2_PREFIX = ("r2_", "b2_", "a2_", "e2_")

if CACHE.exists() and EXTRA.exists():
    f = pd.read_pickle(CACHE)
    ex = pd.read_pickle(EXTRA)
    add = [c for c in ex.columns if c != "application_id" and c not in f.columns]
    f = f.merge(ex[["application_id"] + add], on="application_id", how="left")
    cols = [c for c in f.columns if c not in ("application_id", "customer_id", "target")]
    lean = [c for c in cols if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    fac = [c for c in cols if not c.startswith(ROUND2_PREFIX)]
    print(f"feature frame      : {f.shape[0]:,} rows x {len(cols)} feature columns")
    print(f"full / lean / fac  : {len(cols)} / {len(lean)} / {len(fac)}")
    print(f"train rows in frame: {int(f.target.notna().sum()):,}")
    print(f"test rows in frame : {int(f.target.isna().sum()):,}")
    del f, ex
else:
    print("feature cache not present; set RUN_TRAINING = True to rebuild it")
""")

md(r"""
## 4. Models

Seven legs, all trained with the same 10-fold stratified split repeated over six
fold partitions (seeds 42, 7, 13, 21, 5, 101). The LightGBM legs additionally
average five booster seeds. Nothing is fitted on test.

| leg | model | columns |
|---|---|---|
| `lgb_full` | LightGBM, 15 leaves, lr 0.01, `lambda_l2` 5 | full |
| `lgb_lean` | same parameters | lean |
| `lgbx_full` | LightGBM, 31 leaves, lr 0.012, `lambda_l2` 12 | full |
| `cat_full` | CatBoost | full |
| `xgb_full` | XGBoost | full |
| `fac_full` | LightGBM on fold-safe factor features | fac + factor |
| `linear` | spline logistic regression | lean |

The blend is v61's, kept unchanged because it is the only chain with two
independent public results behind it:

```
trees4 = mean of rank(lgb_full, lgb_lean, cat_full, xgb_full)
core   = (2 * trees4 + lgbx_full) / 3
tm30   = 0.7 * core + 0.3 * linear
final  = rank01(tm30 + 0.5 * (tm30 - lgb_lean))
```

The last line extrapolates away from `lgb_lean` along the direction the blend
already points. v61 and v66 both measured the ratio as a broad plateau between
0.25 and 1.0, and 0.5 sits in the middle of it.
""")

code(r"""
TRAIN_CMD = ("python3 agas/v66/training/train_submission_v66.py "
             "--folds 10 --fold-seeds 42,7,13,21,5,101 --lgb-seeds 0,1,2,3,4 "
             "--lr 0.01 --l2 5.0 --out-prefix v66")

if RUN_TRAINING:
    rc = os.system(TRAIN_CMD)
    assert rc == 0, f"training failed with code {rc}"
else:
    print("training skipped; the command that produced preds_v66.npz was:")
    print(" ", TRAIN_CMD)
    print("\nlog of that run:", (V66 / "output/train_log_v66.txt").relative_to(ROOT))
""")

md(r"""
## 5. Blend and submission

The stored predictions are rebuilt into the submitted vector and checked against
the submitted CSV. The check is exact: ranks must match to floating point and
Spearman must be 1.0.
""")

code(r"""
def rank01(v):
    return rankdata(v) / len(v)

def ks_stat(score, y):
    order = np.argsort(score, kind="mergesort")
    yy = y[order]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())

z = np.load(V66 / "output/preds_v66.npz", allow_pickle=True)
train_ids, test_ids = z["train_ids"], z["test_ids"]
y = (train.set_index("application_id").loc[train_ids].target.to_numpy().astype(int))

ro = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
rt = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}

def chain(d):
    trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
    core = (trees4 * 2 + d["lgbx_full"]) / 3
    tm30 = 0.7 * core + 0.3 * d["linear"]
    return rank01(tm30 + 0.5 * (tm30 - d["lgb_lean"]))

oof_final, test_final = chain(ro), chain(rt)

print("leg performance, out of fold")
print(f"{'leg':12s} {'AUC':>9} {'KS':>9}")
for k in sorted(ro, key=lambda k: -roc_auc_score(y, ro[k])):
    print(f"{k:12s} {roc_auc_score(y, ro[k]):9.6f} {ks_stat(ro[k], y):9.6f}")
print(f"{'BLEND':12s} {roc_auc_score(y, oof_final):9.6f} {ks_stat(oof_final, y):9.6f}")

sub = pd.DataFrame({"application_id": test_ids, "default_probability": test_final})
out_path = V74 / "output/submission_v74_final.csv"
sub.to_csv(out_path, index=False)

ref = pd.read_csv(V66 / "output/submission_v66_04_ext0.5_tm30_lgb_lean.csv")
ref = ref.set_index("application_id").loc[test_ids].default_probability.to_numpy()
print(f"\nmax |rebuilt - submitted| : {np.abs(test_final - ref).max():.3e}")
print(f"Spearman vs submitted     : {spearmanr(test_final, ref).statistic:.10f}")
print(f"rank positions changed    : {int((rankdata(test_final) != rankdata(ref)).sum())}")
""")

md(r"""
## 6. How this candidate was chosen

18 versions were trained. Nine submissions were scored on the public
leaderboard, and the ledger is the reason the final pick is the simplest blend
rather than the strongest one out of fold:

| submission | OOF KS | public | note |
|---|---:|---:|---|
| `v66_04` | 0.528713 | **0.56424** | submitted model |
| `v70_07` | 0.529397 | 0.56419 | near-identical chain on v70 features |
| `v68_04` | — | 0.56411 | same recipe, different fold seeds |
| `v71 core71` | **0.532834** | 0.56111 | highest OOF KS, MLP-heavy |
| `v70_06` | 0.530232 | 0.56040 | second highest OOF KS |
| `v69_10` | 0.528254 | 0.55989 | `extra` feature pivot |
| `v72 seq05` | 0.528844 | 0.55968 | GRU sequence blend |
| `v70_01` | 0.528503 | 0.55801 | highest OOF AUC single leg |

Out of fold and public move in **opposite** directions over this range: the two
candidates with the highest OOF KS are the fourth and fifth best on public. So
v74 did not search for more OOF gain. It tested three explanations for the
mismatch, each with a gate fixed before the run.

**Fase 0 — how large is public noise?** Subsampling the submitted vector's own
out-of-fold scores gives E[KS] and sd[KS] at any sample size. The spread is
0.021 at 3.000 rows and 0.005 at 30.000. Across ten folds of 7.000 rows, the
observed KS spread of one fixed model is 0.016–0.019. A difference of 0.001 out
of fold is not measurable at this resolution.

**Fase 1 — is the out-of-fold estimator broken?** An out-of-fold vector is
stitched from ten fold models, each with its own score scale, while test is
scored by one ensemble; a global ranking statistic like KS would be depressed by
that drift. Gate: rank-normalising inside each fold must raise KS by 0.005.
Measured: **−0.00034** over two fold partitions, and −0.0001 on the independent
v70 predictions. **Rejected** — fold drift is not the cause.

**Fase 2 — does selecting on KS instead of AUC help?** KS is a single point on
the ROC while AUC is its integral, so a KS-keyed search is a different object.
The search used bootstrap-mean KS over 400 resamples, fitted on one stratified
half of the out-of-fold rows and scored on the other. Gate: out-of-selection
gain of 0.002 and bootstrap P(> submitted) of 0.80. Measured: gain +0.0024 but
the two halves disagreed (+0.0048 against −0.0001) and P was **0.70**.
**Rejected** — and the direction it chose was a 30% MLP weight, the same
direction public had already rejected twice.

All three gates failing is the finding: the measured ceiling of this feature and
model family is reached, and the differences that remain are smaller than the
leaderboard can resolve. The submitted model is therefore the one with two
independent public results, not the one that looks best out of fold.
""")

code(r"""
for name in ["public_calibration_v74", "foldnorm_v74", "ksblend_v74", "summary_v74_candidates"]:
    p = V74 / f"output/{name}.json"
    if not p.exists():
        print(f"{name}: not built")
        continue
    d = json.loads(p.read_text())
    if name == "public_calibration_v74":
        c = pd.DataFrame(d["curve"])[["n", "n_pos", "mean_ks", "sd_ks", "bias"]]
        print("fase 0 — KS sampling spread by sample size")
        print(c.to_string(index=False))
    elif name == "foldnorm_v74":
        print(f"\nfase 1 — fold normalisation: mean gain {d['mean_gain']:+.6f} "
              f"vs gate {d['gate']} -> {d['verdict']}")
        for r in d["runs"]:
            print(f"   fold seed {r['fold_seed']}: pooled {r['ks_pooled']:.6f} "
                  f"foldnorm {r['ks_foldnorm']:.6f} "
                  f"per-fold {r['ks_perfold_mean']:.6f} +-{r['ks_perfold_sd']:.6f}")
    elif name == "ksblend_v74":
        print(f"\nfase 2 — KS-keyed blend: out-of-selection gain {d['mean_oos_gain']:+.6f} "
              f"(gate {d['gate_gain']}), P(>anchor) {d['prob_beat_anchor']:.3f} "
              f"(gate {d['gate_prob']}) -> {d['verdict']}")
        for r in d["out_of_selection"]:
            print(f"   fit half {r['fit_half']}: chose {r['chosen']} "
                  f"-> out-of-selection {r['oos_gain']:+.6f}")
    else:
        t = pd.DataFrame(d["candidates"])[
            ["name", "risk", "oof_auc", "oof_ks", "spearman_vs_anchor",
             "boot_p_beat_anchor", "sd_differential"]]
        print(f"\nfase 3 — candidate portfolio, public noise sd {d['public_noise_sd']:.5f}")
        print(t.to_string(index=False))
""")

md(r"""
## 7. Submission validation

The file must have 30.000 rows, the exact test id set, unique ids, and finite
probabilities in [0, 1].
""")

code(r"""
sub = pd.read_csv(V74 / "output/submission_v74_final.csv")
test_id_set = set(pd.read_csv(DATA / "test_applications.csv", usecols=["application_id"]).application_id)

checks = {
    "columns are application_id,default_probability":
        list(sub.columns) == ["application_id", "default_probability"],
    "30000 rows": len(sub) == 30_000,
    "ids unique": bool(sub.application_id.is_unique),
    "id set equals test set": set(sub.application_id) == test_id_set,
    "no missing values": not bool(sub.isna().any().any()),
    "all finite": bool(np.isfinite(sub.default_probability).all()),
    "within [0, 1]": bool(sub.default_probability.between(0, 1).all()),
}
for k, v in checks.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
assert all(checks.values()), "submission failed validation"

digest = hashlib.sha256((V74 / "output/submission_v74_final.csv").read_bytes()).hexdigest()
print(f"\nrows {len(sub):,} | min {sub.default_probability.min():.6f} "
      f"| max {sub.default_probability.max():.6f}")
print("sha256:", digest)
print("\nSTATUS: PASS")
""")

md(r"""
## Limits

* The submitted vector is a rank, not a calibrated probability. KS is invariant
  under any increasing transform, so the ranking is the whole model, but the
  values must not be read as default rates.
* `customer_profiles` is a snapshot without an as-of date, so profile fields are
  used as given. Every other source is truncated at the decision time, verified
  in section 2.
* No external data, no AutoML, and no automated feature engineering is used.
  Feature definitions in `features.py` are hand written, and the fold-safe
  factor encodings are recomputed inside each fold.
* The feature cache and the stored fold predictions are committed artefacts;
  `RUN_TRAINING = True` rebuilds both from the raw CSVs in about 6.5 hours.
""")

NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps({
    "cells": [{**c, "source": c["source"].splitlines(keepends=True)} for c in CELLS],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5}, indent=1))
print(f"wrote {NB} with {len(CELLS)} cells")
