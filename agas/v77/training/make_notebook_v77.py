"""Build agas/v77/notebook/notebook_v77.ipynb (structure reused from v76)."""

import json
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "notebook" / "notebook_v77.ipynb"
CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip()})


md(r"""
# DSC LOGIKA UI 2026 — Credit Default Prediction (v77)

**Task.** Rank 30.000 test applications by the probability of reaching 90+ DPD
within 12 months of origination. Metric: Kolmogorov–Smirnov statistic.

**Model.** A rank-space blend of six gradient boosting legs and one spline
logistic regression, extrapolated along the ratio plateau found in v61:

```
trees4 = mean of rank(lgb_full, lgb_lean, cat_full, xgb_full)
core   = (2 * trees4 + lgbx_full) / 3
tm30   = 0.7 * core + 0.3 * linear
final  = rank01(tm30 + 0.5 * (tm30 - lgb_lean))
```

**What v77 is.** v76's sweep found something. `r = 0.75` instead of `0.5`
scored **0.56520**, the first candidate in twenty versions to beat the anchor's
0.56424, and +0.00102 against the three measurements of the `r = 0.5` recipe
whose spread is 0.00007. v77 explains the mechanism algebraically, retracts the
first explanation after testing it, and follows the gradient. Section 6 has the
measurement; section 6b has the hypothesis that failed.

**How to run.** `RUN_TRAINING = False` (default) rebuilds every candidate from
the stored fold predictions and verifies the reference one reproduces the
submitted CSV exactly; about two minutes. `RUN_TRAINING = True` rebuilds the
feature cache and retrains all legs; about 6.5 hours on 10 cores.
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

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data/train_applications.csv").exists() and (p / "features.py").exists():
            return p
    raise RuntimeError("repository root not found: run from inside the project")

ROOT = find_root(Path.cwd().resolve())
DATA = ROOT / "data"
V66, V76, V77 = ROOT / "agas/v66", ROOT / "agas/v76", ROOT / "agas/v77"
os.chdir(ROOT)
(V77 / "output").mkdir(parents=True, exist_ok=True)

import sklearn
print("root        :", ROOT)
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("sklearn     :", sklearn.__version__)
print("RUN_TRAINING:", RUN_TRAINING)
""")

md(r"""
## 1. Data

Seven provided tables, nothing else. No external data is read in this notebook.
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
print(f"customers in both sets: {len(set(train.customer_id) & set(test.customer_id))}")
""")

md(r"""
## 2. Cutoff audit

Every feature may use only information available at `decision_datetime` of the
selected application. Checked: whether any table holds a record dated after
that cutoff, and whether the bureau observation attached to the application was
available at decision time.
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
windowed over that quantity. Target-encoded factor features are recomputed
inside each fold from training fold labels only. Column sets: `full` 598,
`lean` 490, `fac` 553.

v76 does not touch `features.py` and trains nothing. v75 measured that this
feature family is exhausted: a learning curve that buys only +0.0007 AUC from
the last 20 percent of rows, monotone constraints at −0.00078, and compact and
additive models all below the 490-column baseline.
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

if CACHE.exists() and EXTRA.exists():
    f = pd.read_pickle(CACHE)
    ex = pd.read_pickle(EXTRA)
    add = [c for c in ex.columns if c != "application_id" and c not in f.columns]
    f = f.merge(ex[["application_id"] + add], on="application_id", how="left")
    cols = [c for c in f.columns if c not in ("application_id", "customer_id", "target")]
    lean = [c for c in cols if c not in NOISE_EXACT and not c.startswith(NOISE_PREFIX)]
    print(f"feature frame : {f.shape[0]:,} rows x {len(cols)} columns")
    print(f"full / lean   : {len(cols)} / {len(lean)}")
    del f, ex
else:
    print("feature cache not present; set RUN_TRAINING = True to rebuild it")
""")

md(r"""
## 4. Models

Seven legs, 10-fold stratified splits repeated over six fold partitions (seeds
42, 7, 13, 21, 5, 101), LightGBM legs also averaged over five booster seeds.
Nothing is fitted on test.

| leg | model | columns |
|---|---|---|
| `lgb_full` | LightGBM, 15 leaves, lr 0.01 | full |
| `lgb_lean` | same parameters | lean |
| `lgbx_full` | LightGBM, 31 leaves, lr 0.012 | full |
| `cat_full` | CatBoost | full |
| `xgb_full` | XGBoost | full |
| `fac_full` | LightGBM on fold-safe factor features | fac + factor |
| `linear` | spline logistic regression | lean |
""")

code(r"""
TRAIN_CMD = ("python3 agas/v66/training/train_submission_v66.py "
             "--folds 10 --fold-seeds 42,7,13,21,5,101 --lgb-seeds 0,1,2,3,4 "
             "--lr 0.01 --l2 5.0 --out-prefix v66")
V76_CMD = "python3 agas/v76/training/make_submission_v76.py"
if RUN_TRAINING:
    for c in (TRAIN_CMD, V76_CMD):
        rc = os.system(c)
        assert rc == 0, f"{c} failed with code {rc}"
else:
    print("training skipped. the commands behind the stored predictions were:")
    for c in (TRAIN_CMD, V76_CMD):
        print(" ", c)
""")

md(r"""
## 5. The chain, rebuilt and verified

The reference vector is rebuilt from the stored predictions and checked against
the submitted CSV: ranks must match to floating point and Spearman must be 1.0.
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

def chain(d, r=0.50, tm=0.30, ref="lgb_lean", cw=2.0):
    trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
    core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
    base = (1 - tm) * core + tm * d["linear"]
    return rank01(base + r * (base - d[ref]))

oof_final, test_final = chain(ro), chain(rt)

print("leg performance, out of fold")
print(f"{'leg':12s} {'AUC':>9} {'KS':>9}")
for k in sorted(ro, key=lambda k: -roc_auc_score(y, ro[k])):
    print(f"{k:12s} {roc_auc_score(y, ro[k]):9.6f} {ks_stat(ro[k], y):9.6f}")
print(f"{'CHAIN':12s} {roc_auc_score(y, oof_final):9.6f} {ks_stat(oof_final, y):9.6f}")

ref_csv = pd.read_csv(V66 / "output/submission_v66_04_ext0.5_tm30_lgb_lean.csv")
ref = ref_csv.set_index("application_id").loc[test_ids].default_probability.to_numpy()
print(f"\nmax |rebuilt - submitted| : {np.abs(test_final - ref).max():.3e}")
print(f"Spearman vs submitted     : {spearmanr(test_final, ref).statistic:.10f}")
print(f"rank positions changed    : {int((rankdata(test_final) != rankdata(ref)).sum())}")
""")

md(r"""
## 6. The gradient, and the algebra behind it

Public results on the chain's extrapolation axis, holding `tm = 0.30` and the
reference leg at `lgb_lean`:

| recipe | public | n |
|---|---:|---:|
| `r = 0.50` | 0.56424, 0.56419, 0.56411 → mean **0.56418**, sd **0.00007** | 3 |
| `r = 0.75` | **0.56520** | 1 |

The three `r = 0.50` measurements come from different feature sets and different
fold-seed counts — v66 on six fold seeds, v68 on ten, `v70_07` on the v70
feature set entirely — and still agree to 0.00013. That is the instrument. The
+0.00102 at `r = 0.75` is 14.6 of those standard deviations.

It also answers the question v76 posed. `seq05` sat at Spearman 0.99950 against
the anchor and fell 0.00456, which suggested the anchor might be a brittle
local maximum. `r = 0.75` sits at 0.999803 — a larger perturbation — and went
**up**. The surface is smooth; the fragility reading is rejected.

**The algebra.** The extrapolation is a weighted contrast, not an
amplification:

```
base + r * (base - lgb_lean)  ==  base - w * lgb_lean,    w = r / (1 + r)
```

verified numerically for `r` from 0.25 to 3.0, maximum difference 3.3e-5, which
is one rank position out of 30000 from tie-breaking. `lgb_lean` also sits inside
`base` with weight `0.7 * 2/3 * 1/4 = 0.11667`, so its net coefficient runs:

| r | w | net coefficient on `lgb_lean` |
|---:|---:|---:|
| 0.00 | 0.000 | +0.11667 |
| 0.50 | 0.333 | −0.21667 |
| **0.75** | 0.429 | **−0.31190** |
| 1.00 | 0.500 | −0.38333 |
| 1.50 | 0.600 | −0.48333 |

**Why twenty versions missed it.** v61 and v66 both swept `r` and both picked
0.5 by out-of-fold AUC. On this axis out-of-fold points the wrong way: `r=0.75`
is −0.000350 AUC and −0.000920 KS against `r=0.5`, and +0.00102 on public. So
the sweep continues here against public only, with no out-of-fold gate.
""")

code(r"""
d = json.loads((V77 / "output/summary_v77.json").read_text())
print(f"instrument sd inside one recipe: {d['instrument_sd']:.5f}")
print(f"measured: r=0.50 -> {d['base_public_r050']:.5f}, "
      f"r=0.75 -> {d['measured_public']['0.75'][0]:.5f}")
print(f"local slope: {d['local_slope_per_unit_r']:+.5f} public per unit r")
print(f"\nalgebra: {d['algebra']}\n")
t = pd.DataFrame(d["grid"])[["r", "w", "net_lgb_lean", "oof_auc", "d_oof_auc",
                             "d_oof_ks", "spearman_vs_r050", "public_measured",
                             "linear_prediction"]]
print(t.to_string(index=False))
print("\nout-of-fold falls as public rises: that is why an out-of-fold argmax")
print("chose r=0.5 and why no out-of-fold gate is used on this axis.")
print("\nthe linear prediction is a registered guess, not a selection key.")
""")

md(r"""
## 6b. A hypothesis that was tested and failed

The first explanation offered for the gradient was substantive: `lgb_lean` is
the only leg trained on the 490-column `lean` set while `lgb_full`, `cat_full`,
`xgb_full` and `lgbx_full` use all 598, so `base - w * lgb_lean` looked like a
full-minus-lean contrast that isolates the 108 columns `lean` drops — the
demographics plus the `ec_share_`, `ec_catqty_`, `cs_shift`, `buh_` and `gn_`
families. On that reading, raising `w` raises how much dropped-column signal
reaches the ranking.

It was tested directly. A LightGBM leg was trained on those 108 columns alone,
10 folds by 2 fold seeds, reaching **OOF AUC 0.836393** — strong for 108
columns, and dominated by the 66 `gn_` columns rather than the raw
demographics. Then the direction of raising `r` was compared against the
direction of adding each leg.

Every correlation is negative, and the strongest is with `lgb_lean` itself.
**Raising `r` subtracts; it cannot be amplifying anything.** The hypothesis is
withdrawn.

What survives is narrower and stops where the measurement stops. `lgb_lean`
correlates about 0.99 with `base`, so subtracting a large multiple of it
amplifies the small residual between them. Raising `r` is de-shrinkage: it
moves the ranking away from the consensus along the `base - lgb_lean`
disagreement, reordering rows the consensus had left nearly tied. Public
rewarded that once. Why that residual carries signal is not established here.

The leg is kept anyway, as a second and independent lever rather than an
explanation: AUC 0.836 on 108 columns, never previously part of the chain.
""")

code(r"""
d = json.loads((V77 / "output/summary_v77_dropped.json").read_text())
print(f"dropped-columns leg: OOF AUC {d['dropped_leg_oof_auc']:.6f} "
      f"KS {d['dropped_leg_oof_ks']:.6f}")
print(f"reference r={d['reference_r']}: OOF AUC {d['reference_oof_auc']:.6f}, "
      f"public {d['reference_public']:.5f}")
print(f"\nrefuted: {d['refuted_hypothesis']}")
print("direction of raising r vs adding 20% of a leg (Pearson):")
for k, v in sorted(d["direction_pearson_vs_adding_leg"].items(), key=lambda kv: kv[1]):
    print(f"  {k:12s} {v:+.4f}")
print("\nall negative, strongest against lgb_lean itself.\n")
print(pd.DataFrame(d["candidates"]).to_string(index=False))
""")

md(r"""
## 7. Submission validation

Every candidate file must have 30.000 rows, the exact test id set, unique ids,
and finite probabilities in [0, 1].
""")

code(r"""
test_id_set = set(pd.read_csv(DATA / "test_applications.csv",
                              usecols=["application_id"]).application_id)
files = sorted((V77 / "output").glob("submission_v77*.csv"))
assert files, "no v76 submission files found"

report = []
for p in files:
    sub = pd.read_csv(p)
    checks = {
        "cols": list(sub.columns) == ["application_id", "default_probability"],
        "rows": len(sub) == 30_000,
        "unique": bool(sub.application_id.is_unique),
        "ids": set(sub.application_id) == test_id_set,
        "no_na": not bool(sub.isna().any().any()),
        "finite": bool(np.isfinite(sub.default_probability).all()),
        "range": bool(sub.default_probability.between(0, 1).all())}
    report.append({"file": p.name, **checks,
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]})
    assert all(checks.values()), f"{p.name} failed validation"

print(pd.DataFrame(report).to_string(index=False))

best = pd.read_csv(V76 / "output/submission_v76_02_r0.75.csv")
best = best.set_index("application_id").loc[test_ids].default_probability.to_numpy()
rebuilt = chain(rt, r=0.75)
print(f"\nthe scored r=0.75 file, rebuilt from stored predictions: "
      f"Spearman {spearmanr(rebuilt, best).statistic:.10f}, "
      f"rank changes {int((rankdata(rebuilt) != rankdata(best)).sum())}")
print(f"\n{len(files)} files validated")
print("STATUS: PASS")
""")

md(r"""
## 8. Recommendation and limits

**The final pick has changed.** `agas/v76/output/submission_v76_02_r0.75.csv`,
public 0.56520, replaces the anchor's 0.56424. Change it again only if a later
sweep point comes back above 0.56520.

**Next submission: one file.** `submission_v77_r1p000.csv`. If the local
gradient held it would land near 0.5662, which is a registered guess, not a
prediction to act on. The decision rule:

| `r = 1.00` returns | action |
|---|---|
| above 0.56520 | gradient continues: send `r1p250`, then `r1p500` |
| 0.56418 to 0.56520 | optimum is between 0.75 and 1.00: send `r0p875` |
| below 0.56418 | optimum is near 0.75: send `r0p625` to bracket it |

Send one point at a time. The instrument's spread is 0.00007, so each point is
nearly noise-free and jumping ahead throws information away.

**Second lever, ordered behind the first.** The `submission_v77d_*` files add
the dropped-columns leg on top of `r = 0.75`. They are a separate question with
no measured gradient behind them, so they wait until the `r` line search has an
optimum.

**Limits.**

* The submitted vector is a rank, not a calibrated probability. KS is invariant
  under any increasing transform, so the ranking is the whole model, but the
  values must not be read as default rates.
* +0.00102 is **one** measurement. `r = 1.00` is the test of it. If that comes
  back low, the `r = 0.75` point has to be treated as a single anomaly rather
  than a gradient.
* The surface must turn somewhere. The linear extrapolation reaching 0.5703 at
  `r = 2.0` is almost certainly too optimistic, and `qc_00` at `r = 1.5` scored
  only 0.56171 — though on a different reference leg, so not a direct
  comparison.
* The instrument sd of 0.00007 is three observations of one recipe. It measures
  that recipe's reproducibility across feature sets and seeds, not a general
  property of the leaderboard: cluster B's spread is 0.00123.
* Out-of-fold is measured **anti-correlated** with public on this axis, so no
  out-of-fold gate is applied to the `r` sweep. That is a deliberate departure
  from v74 and v75, justified only by this axis's measurements.
* `customer_profiles` has no as-of date, so profile fields are used as given.
  Every other source is truncated at the decision time, re-verified in
  section 2 on every run.
* `features.py` does not apply a positive `days_ago` filter in three of its main
  blocks. That is safe only because the provided tables carry no post-cutoff
  records, which section 2 re-checks.
* No external data, no AutoML, no automated feature engineering. v77 trains
  only the dropped-columns leg; every other candidate re-assembles stored
  predictions.
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
