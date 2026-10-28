"""Build the committee notebook that reproduces the actual final submission.

Rules section 7 requires the submitted notebook to correspond to the team's
submission. The notebook sitting at the repository root reproduces
`rank01(tm30 + 0.5*(tm30 - lgb_lean))`, which scored 0.56424, but the final pick
is now `r = 0.75`, which scored 0.56520. So the root notebook does not match
what is being submitted, and this rebuilds it so that it does.
"""

import json
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "notebook" / "notebook_v79_final.ipynb"
CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip()})


md(r"""
# DSC LOGIKA UI 2026 — Credit Default Prediction

**Task.** Rank 30.000 test applications by the probability that the loan reaches
90+ days past due within 12 months of origination. Metric: the
Kolmogorov–Smirnov statistic.

**Submitted model.** A rank-space blend of six gradient boosting legs and one
spline logistic regression, then extrapolated away from the lean-column
LightGBM leg:

```
trees4 = mean of rank(lgb_full, lgb_lean, cat_full, xgb_full)
core   = (2 * trees4 + lgbx_full) / 3
tm30   = 0.7 * core + 0.3 * linear
final  = rank01(tm30 + 0.75 * (tm30 - lgb_lean))
```

Public score **0.56520**. Section 6 explains how the `0.75` was found and why
every other setting in that formula is the one it is.

**How to run.** `RUN_TRAINING = False` (the default) rebuilds the submitted
vector from the stored fold predictions and checks it reproduces the submitted
CSV exactly; about two minutes. `RUN_TRAINING = True` rebuilds the feature cache
and retrains all seven legs from the raw CSVs; about 6.5 hours on 10 cores.

**Sections.** 1 data · 2 cutoff audit · 3 features · 4 models · 5 blend and
submission · 6 how the recipe was chosen · 7 validation · 8 limits.
""")

code(r"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

RUN_TRAINING = False
RANDOM_STATE = 42
R_FINAL = 0.75          # the extrapolation ratio the public leaderboard chose

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data/train_applications.csv").exists() and (p / "features.py").exists():
            return p
    raise RuntimeError("repository root not found: run from inside the project")

ROOT = find_root(Path.cwd().resolve())
DATA = ROOT / "data"
V66, V76, V79 = ROOT / "agas/v66", ROOT / "agas/v76", ROOT / "agas/v79"
FINAL_CSV = V76 / "output/submission_v76_02_r0.75.csv"
os.chdir(ROOT)
(V79 / "output").mkdir(parents=True, exist_ok=True)

import sklearn
print("root        :", ROOT)
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("sklearn     :", sklearn.__version__)
print("RUN_TRAINING:", RUN_TRAINING)
print("final r     :", R_FINAL)
""")

md(r"""
## 1. Data

The seven provided tables and nothing else. No external data is read anywhere in
this notebook.
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

train = pd.read_csv(DATA / "train_applications.csv",
                    parse_dates=["application_datetime", "decision_datetime"])
test = pd.read_csv(DATA / "test_applications.csv",
                   parse_dates=["application_datetime", "decision_datetime"])

print(pd.DataFrame(shapes, index=["rows", "cols"]).T.to_string())
print(f"\ntrain {len(train):,} rows | test {len(test):,} rows "
      f"| positive rate {train.target.mean():.6f}")
print(f"customers appearing in both sets: "
      f"{len(set(train.customer_id) & set(test.customer_id))}")
print(f"term_months values   : {sorted(train.term_months.unique())}")
print(f"interest rate values : {sorted(train.annual_interest_rate.unique())}")
""")

md(r"""
## 2. Cutoff audit

Every feature may use only information available at the selected application's
`decision_datetime`. Two things are checked: whether any provided table holds a
record dated after that cutoff, and whether the bureau observation attached to
the application was actually available at decision time.
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
audit["bureau_observations.available_datetime"] = (len(sel), int((sel.available_datetime > sel.decision_datetime).sum()))
audit["bureau_observations.report_datetime"] = (len(sel), int((sel.report_datetime > sel.decision_datetime).sum()))

rows = pd.DataFrame([(k, v[0], v[1]) for k, v in audit.items()],
                    columns=["source field", "rows joined", "rows after cutoff"])
print(rows.to_string(index=False))
post = int(rows["rows after cutoff"].sum())
print(f"\ntotal records dated after the cutoff: {post}")
assert post == 0, "a source holds post-cutoff records; window features need a positive-time filter"
print("PASS: the provided tables are already truncated at decision_datetime")
del pa, rh, ab, ec, bu, sel
""")

md(r"""
## 3. Features

`features.py` builds one row per application from the seven tables. Every
historical aggregate is computed as `days_ago = decision_datetime - event` and
windowed over that quantity, so nothing dated after the decision can enter a
feature. Target-encoded factor features are recomputed **inside each fold** from
that fold's training labels only, so no fold sees its own labels.

Blocks: application terms, customer profile, previous applications, repayment
history, account behavior, bureau observations, e-commerce behavior, plus ratio
and trend interactions. Three column sets are used:

| set | columns | difference |
|---|---:|---|
| `full` | 598 | everything |
| `lean` | 490 | drops demographics and share-type columns with no fold-stable gain |
| `fac` | 553 | drops the second-order rounding family |
""")

code(r"""
CACHE = ROOT / "agas/shared/cache/features_cache.pkl"
EXTRA = ROOT / "agas/shared/cache/extra_cache.pkl"

if RUN_TRAINING:
    import features as ft
    t0 = time.time()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ft.build().to_pickle(CACHE)
    rc = os.system("python3 agas/shared/build_extra.py")
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
    print(f"feature frame     : {f.shape[0]:,} rows x {len(cols)} columns")
    print(f"full / lean / fac : {len(cols)} / {len(lean)} / {len(fac)}")
    print(f"positives         : {int(f.target.sum()):,}")
    del f, ex
else:
    print("feature cache not present; set RUN_TRAINING = True to rebuild it")
""")

md(r"""
## 4. Models

Seven legs, each trained with 10-fold stratified cross-validation repeated over
six fold partitions (seeds 42, 7, 13, 21, 5, 101). The LightGBM legs additionally
average five booster seeds. Nothing is ever fitted on the test set.

| leg | model | columns |
|---|---|---|
| `lgb_full` | LightGBM, 15 leaves, lr 0.01, `lambda_l2` 5 | full |
| `lgb_lean` | same parameters | lean |
| `lgbx_full` | LightGBM, 31 leaves, lr 0.012, `lambda_l2` 12 | full |
| `cat_full` | CatBoost | full |
| `xgb_full` | XGBoost | full |
| `fac_full` | LightGBM on fold-safe factor features | fac + factor |
| `linear` | spline logistic regression | lean |
""")

code(r"""
TRAIN_CMD = ("python3 agas/v66/training/train_submission_v66.py "
             "--folds 10 --fold-seeds 42,7,13,21,5,101 --lgb-seeds 0,1,2,3,4 "
             "--lr 0.01 --l2 5.0 --out-prefix v66")
if RUN_TRAINING:
    rc = os.system(TRAIN_CMD)
    assert rc == 0, f"training failed with code {rc}"
else:
    print("training skipped; the command behind the stored predictions was:")
    print(" ", TRAIN_CMD)
    print("\nlog of that run:", (V66 / "output/train_log_v66.txt").relative_to(ROOT))
""")

md(r"""
## 5. Blend and submission

The submitted vector is rebuilt from the stored fold predictions and checked
against the submitted CSV. The check is exact: ranks must match to floating
point, Spearman must be 1.0, and no rank position may differ.
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
y = train.set_index("application_id").loc[train_ids].target.to_numpy().astype(int)
ro = {k[4:]: rank01(z[k]) for k in z.files if k.startswith("oof_")}
rt = {k[5:]: rank01(z[k]) for k in z.files if k.startswith("test_")}

def chain(d, r=R_FINAL, tm=0.30, cw=2.0, ref="lgb_lean"):
    trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
    core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
    tm30 = (1 - tm) * core + tm * d["linear"]
    return rank01(tm30 + r * (tm30 - d[ref]))

oof_final, test_final = chain(ro), chain(rt)

print("leg performance, out of fold")
print(f"{'leg':12s} {'AUC':>9} {'KS':>9}")
for k in sorted(ro, key=lambda k: -roc_auc_score(y, ro[k])):
    print(f"{k:12s} {roc_auc_score(y, ro[k]):9.6f} {ks_stat(ro[k], y):9.6f}")
print(f"{'SUBMITTED':12s} {roc_auc_score(y, oof_final):9.6f} "
      f"{ks_stat(oof_final, y):9.6f}")

ref_csv = pd.read_csv(FINAL_CSV)
ref = ref_csv.set_index("application_id").loc[test_ids].default_probability.to_numpy()
changed = int((rankdata(test_final) != rankdata(ref)).sum())
print(f"\nrebuilt against {FINAL_CSV.name}")
print(f"  max |difference|       : {np.abs(test_final - ref).max():.3e}")
print(f"  Spearman               : {spearmanr(test_final, ref).statistic:.10f}")
print(f"  rank positions changed : {changed}")
assert changed == 0, "the rebuilt vector does not reproduce the submitted file"

out_path = V79 / "output/submission_final.csv"
pd.DataFrame({"application_id": test_ids,
              "default_probability": test_final}).to_csv(out_path, index=False)
print(f"\nwrote {out_path.relative_to(ROOT)}")
""")

md(r"""
## 6. How this recipe was chosen

The blend structure comes from v61 and was kept because it is the only chain
with repeated public results behind it. What changed, and what this notebook
submits, is the **extrapolation ratio**: `0.75` instead of the `0.5` used from
v61 through v76.

**Why that was not found earlier.** v61 and v66 both swept the ratio and both
picked `0.5` by out-of-fold AUC. On this axis out-of-fold points the wrong way:
`r = 0.75` is −0.000350 AUC and −0.000920 KS against `r = 0.5`, and +0.00102 on
public. The `r = 0.75` vector had in fact been built four versions before it was
ever submitted and was ranked 7th of 9 by an out-of-fold score. Selection order,
not any shortage of candidates, is what hid it.

**The axis, mapped.** Three public points fit a quadratic with curvature
−0.01264, peaking at `r = 0.786` with 0.565217. The remaining gain above the
measured `r = 0.75` is +0.000017, so the axis is closed.

| r | public |
|---:|---:|
| 0.50 | 0.56424, 0.56419, 0.56411 (three independent trainings) |
| **0.75** | **0.56520** |
| 1.00 | 0.56464 |

The three `r = 0.50` values come from different feature sets and different
fold-seed counts and agree to 0.00013, which is what makes the +0.00102 at
`r = 0.75` meaningful rather than noise.

**Every other setting was tested and is the one it is because alternatives lost.**

| change from the submitted recipe | public | cost |
|---|---:|---:|
| ratio `r` → 0.50 | 0.56424 | −0.00096 |
| ratio `r` → 1.00 | 0.56464 | −0.00056 |
| linear weight `tm` → 0.20 | 0.56158 | −0.00362 |
| reference leg → `lgb_full` | 0.55934 | −0.00586 |
| three legs retrained +0.0008 AUC each | 0.56159 | −0.00361 |

That last row is worth stating plainly: an 8.7 hour run raised three legs by
+0.00075 to +0.00083 AUC, verified as a controlled A/B in which the four
untouched legs were bit-identical to the originals, and the resulting chain
scored 0.0036 **lower**. Better legs do not make this chain better.

**What is algebraically going on.** The extrapolation is a weighted subtraction,
not an amplification:

```
tm30 + r * (tm30 - lgb_lean)  ==  tm30 - w * lgb_lean,   w = r / (1 + r)
```

so the ratio only sets how hard one leg is subtracted. An attempt to explain the
gain substantively — that subtracting the lean-column leg amplifies the 108
columns `lean` drops — was tested and withdrawn: a leg trained on those 108
columns alone moves the ranking opposite to raising `r` (Pearson −0.68), because
raising `r` subtracts rather than adds. What survives is narrower: the
subtraction amplifies the residual between the blend and `lgb_lean`, and public
rewards that. Why that residual carries signal is not established.
""")

code(r"""
LEDGER = [
    ("r=0.75  ref=lgb_lean tm=0.30 core=2:1   SUBMITTED", 0.56520),
    ("r=1.00", 0.56464),
    ("r=0.50 (v66_04)", 0.56424),
    ("r=0.50 on v70 features", 0.56419),
    ("r=0.50, ten fold seeds", 0.56411),
    ("r=0.75 on retrained legs (+0.0008 AUC each)", 0.56159),
    ("tm=0.20", 0.56158),
    ("MLP-heavy blend", 0.56111),
    ("80% MLP mix", 0.56073),
    ("MLP + v70 pair", 0.56045),
    ("anchor + 5% GRU sequence", 0.55968),
    ("reference leg = lgb_full", 0.55934),
    ("single raw leg", 0.55801),
]
led = pd.DataFrame(LEDGER, columns=["submission", "public"])
led["vs submitted"] = (led.public - 0.56520).round(5)
print(led.to_string(index=False))
print(f"\n{len(led)} scored submissions; the submitted recipe is the maximum.")
print("Out-of-fold AUC of the submitted vector is LOWER than six of the")
print("candidates below it, which is why out-of-fold was abandoned as a")
print("selection key and the public leaderboard used instead.")
""")

md(r"""
## 7. Submission validation

The file must have 30.000 rows, the exact test id set, unique ids, and finite
probabilities in [0, 1].
""")

code(r"""
sub = pd.read_csv(V79 / "output/submission_final.csv")
test_id_set = set(pd.read_csv(DATA / "test_applications.csv",
                              usecols=["application_id"]).application_id)
checks = {
    "columns are application_id,default_probability":
        list(sub.columns) == ["application_id", "default_probability"],
    "30000 rows": len(sub) == 30_000,
    "ids unique": bool(sub.application_id.is_unique),
    "id set equals the test set": set(sub.application_id) == test_id_set,
    "no missing values": not bool(sub.isna().any().any()),
    "all finite": bool(np.isfinite(sub.default_probability).all()),
    "within [0, 1]": bool(sub.default_probability.between(0, 1).all()),
}
for k, v in checks.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
assert all(checks.values()), "submission failed validation"

digest = hashlib.sha256((V79 / "output/submission_final.csv").read_bytes()).hexdigest()
submitted = hashlib.sha256(FINAL_CSV.read_bytes()).hexdigest()
print(f"\nrows {len(sub):,} | min {sub.default_probability.min():.6f} "
      f"| max {sub.default_probability.max():.6f}")
print(f"sha256 rebuilt   : {digest}")
print(f"sha256 submitted : {submitted}")
print(f"byte-identical to the submitted file: {digest == submitted}")
print("\nSTATUS: PASS")
""")

md(r"""
## 8. Limits

* The submitted values are ranks, not calibrated probabilities. KS is invariant
  under any increasing transform, so the ranking is the whole model, but the
  numbers must not be read as default rates.
* `customer_profiles` carries no as-of date, so profile fields are used as
  given. Every other source is truncated at the decision time, which section 2
  re-verifies on every run.
* `features.py` does not apply a positive `days_ago` filter in three of its main
  blocks. That is safe only because the provided tables carry no post-cutoff
  records, which is exactly what section 2 checks.
* Out-of-fold AUC and KS are reported throughout and used nowhere as a
  selection key. On the extrapolation axis they move opposite to public, and
  the submitted vector has a lower out-of-fold AUC than six candidates that
  scored below it.
* The recipe was selected against the public leaderboard, so it carries public
  selection risk: the private split is an independent sample and the measured
  +0.00102 need not reproduce there.
* No external data, no AutoML and no automated feature engineering. Feature
  definitions in `features.py` are hand written, the one tuned hyper-parameter
  was chosen from an explicitly written grid, and the fold-safe factor
  encodings are recomputed inside each fold.
""")

NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps({
    "cells": [{**c, "id": uuid.uuid4().hex[:8],
               "source": c["source"].splitlines(keepends=True)} for c in CELLS],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5}, indent=1))
print(f"wrote {NB} with {len(CELLS)} cells")
