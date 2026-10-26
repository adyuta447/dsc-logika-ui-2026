"""Build agas/v75/notebook/notebook_v75.ipynb."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "notebook" / "notebook_v75.ipynb"
CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip()})


md(r"""
# DSC LOGIKA UI 2026 — Credit Default Prediction (v75)

**Task.** Rank 30.000 test applications by the probability of reaching 90+ DPD
within 12 months of origination. Metric: Kolmogorov–Smirnov statistic.

**Model.** A rank-space blend of six gradient boosting legs and one spline
logistic regression, extrapolated along the ratio plateau measured in v61:

```
trees4 = mean of rank(lgb_full, lgb_lean, cat_full, xgb_full)
core   = (2 * trees4 + lgbx_full) / 3
tm30   = 0.7 * core + 0.3 * linear
final  = rank01(tm30 + 0.5 * (tm30 - lgb_lean))
```

**What v75 added.** v75 was asked to beat this blend's public score of 0.56424
and aim at 0.57. It tested four things, each gated before the run, and the
gates decided the outcome rather than a post-hoc reading. All four failed, and
section 6 reports the measurements. The fourth measurement — a calibration of
the public score against three results that can be mapped exactly onto
candidate classes — is the one that changed the conclusion: **0.56424 is not a
typical score for this model family, it is the maximum of the draws already
taken.** The family mean is 0.5596.

**How to run.** `RUN_TRAINING = False` (default) rebuilds every submitted vector
from the stored fold predictions and verifies the main one reproduces the
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
V66, V74, V75 = ROOT / "agas/v66", ROOT / "agas/v74", ROOT / "agas/v75"
os.chdir(ROOT)
(V75 / "output").mkdir(parents=True, exist_ok=True)

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
print(f"term_months values    : {sorted(train.term_months.unique())}")
print(f"interest rate values  : {sorted(train.annual_interest_rate.unique())}")
""")

md(r"""
## 2. Cutoff audit

Every feature may use only information available at `decision_datetime` of the
selected application. Checked here: whether any provided table holds a record
dated after the cutoff of that customer's selected application, and whether the
bureau observation attached to the application was available at decision time.
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
windowed over that quantity, so nothing after the decision can enter a feature.
Target-encoded factor features are recomputed inside each fold from training
fold labels only.

Three column sets: `full` 598, `lean` 490 (drops demographic and share-type
columns with no fold-stable gain), `fac` 553 (drops the second-order rounding
family). v75 did not change `features.py` — section 6 explains why adding
features was the wrong direction.
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
    pos = int(f.target.sum())
    print(f"feature frame     : {f.shape[0]:,} rows x {len(cols)} columns")
    print(f"full / lean / fac : {len(cols)} / {len(lean)} / {len(fac)}")
    print(f"positives         : {pos:,}  ({pos / len(lean):.1f} per lean column)")
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
V75_CMDS = ["python3 agas/v75/training/probe_v75.py",
            "python3 agas/v75/training/screen_compact_v75.py",
            "python3 agas/v75/training/make_submission_v75.py"]
if RUN_TRAINING:
    rc = os.system(TRAIN_CMD)
    assert rc == 0, f"training failed with code {rc}"
    for c in V75_CMDS:
        rc = os.system(c)
        assert rc == 0, f"{c} failed with code {rc}"
else:
    print("training skipped. the commands behind the stored predictions were:")
    for c in [TRAIN_CMD, *V75_CMDS]:
        print(" ", c)
""")

md(r"""
## 5. Blend, candidates and submission

The main vector is rebuilt from the stored predictions and checked against the
submitted CSV: ranks must match to floating point and Spearman must be 1.0.
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

ref = pd.read_csv(V66 / "output/submission_v66_04_ext0.5_tm30_lgb_lean.csv")
ref = ref.set_index("application_id").loc[test_ids].default_probability.to_numpy()
print(f"\nmax |rebuilt - submitted| : {np.abs(test_final - ref).max():.3e}")
print(f"Spearman vs submitted     : {spearmanr(test_final, ref).statistic:.10f}")
print(f"rank positions changed    : {int((rankdata(test_final) != rankdata(ref)).sum())}")
""")

md(r"""
## 6. What v75 measured

Four questions, each with a gate written down before the run.

**Is the 0.850 plateau an information ceiling or a variance ceiling?** Training
the same configuration on 20/40/60/80/100 percent of the rows gave AUC 0.8376,
0.8416, 0.8429, 0.8441, 0.8448. Still climbing, so the plateau is a variance
ceiling — but the last 20 percent of the data bought only +0.0007, which caps
what any variance reduction can return at roughly +0.001 to +0.002 AUC.

**Is the response monotone in each risk feature?** The spline logistic leg
reaches 0.8428 against 0.8484 for the best boosted leg, a gap far narrower than
real credit data gives, which suggests a near-additive response. Monotone
constraints, with directions taken inside each training fold from the sign of
that fold's Spearman correlation, were therefore expected to trade nothing for
less variance. Gate: +0.0010 AUC and positive in all three seeds. Measured
**−0.00078 AUC and −0.00196 KS, negative in all three seeds.** Rejected.

**Are 490 columns too many for 9.507 positives?** At 19 positives per column,
and with the seven sources heavily redundant — each reaching 0.62 to 0.83
standalone while all of them together reach only 0.846 — a compact model should
have matched the wide one. Columns were reranked inside every training fold and
the model refitted on the top k of that fold's own ranking. Measured: the full
490 columns were best, and the purely additive `max_depth=1` model was 0.0013
worse while `max_depth=2` was 0.0008 worse. Rejected, and the ordering says
interactions do carry signal, so the near-additive reading was wrong.

**Can a decorrelated candidate reach 0.57 on a lucky draw?** Public is a small
fixed subset, so the natural model is that a candidate's score is the anchor's
plus the unshared part of the sampling noise, with spread
`sd_public * sqrt(2 * (1 - rho))`. That model is falsified by the three public
results that map exactly onto candidate classes: every one landed below
prediction, and the miss in sigma units grew as rho grew — the opposite of what
shared noise implies. The deviations are all about −0.003 to −0.006 and barely
depend on rho.

The explanation is selection. **0.56424 is the maximum of the draws already
taken, so it carries the positive noise of that maximum.** The mean of the
three measured non-anchor results is 0.5596, and that — not 0.56424 — is what a
new candidate should be expected to score. 0.57 is three to five sigma above
it. Candidates are priced against that mean below, crediting none of a
candidate's out-of-fold gain and charging all of its loss, since every measured
gain so far arrived at public as a loss.
""")

code(r"""
for name in ["probe_v75", "compact_v75", "summary_v75_candidates"]:
    p = V75 / f"output/{name}.json"
    if not p.exists():
        print(f"{name}: not built\n")
        continue
    d = json.loads(p.read_text())
    if name == "probe_v75":
        print("learning curve")
        print(pd.DataFrame(d["learning_curve"]).to_string(index=False))
        print(f"\nmonotone constraints: mean dAUC {d['mono_mean_d_auc']:+.6f}, "
              f"mean dKS {d['mono_mean_d_ks']:+.6f}, "
              f"positive in all seeds: {d['mono_all_seeds_positive']} "
              f"-> {d['verdict']}\n")
    elif name == "compact_v75":
        print(f"compact and additive screening, baseline {d['baseline']}")
        print(pd.DataFrame(d["results"])[
            ["variant", "auc_mean", "auc_sd", "ks_mean", "d_auc_mean",
             "d_auc_all_positive"]].to_string(index=False))
        print(f"passed the gate: {d['passed'] or 'none'}\n")
    else:
        print(f"public calibration: family mean {d['family_mean_public']:.5f}, "
              f"spread {d['family_sd_public']:.5f}, "
              f"anchor {d['public_anchor']:.5f}")
        print(pd.DataFrame(d["calibration"]).to_string(index=False))
        t = pd.DataFrame(d["candidates"])
        t = t[t.quality_ok][["name", "oof_auc", "d_auc", "d_ks",
                             "spearman_vs_anchor", "expected_public",
                             "p_reach_0_57"]].head(10)
        print(f"\ncandidates that clear the quality floor, priced against "
              f"the family mean (target {d['target_public']})")
        print(t.to_string(index=False))
        print(f"\nchance at least one ticket reaches the target "
              f"(independence upper bound): {d['portfolio_p_reach_target']}")
        print(f"submit order: {d['submit_order']}")
""")

md(r"""
## 7. Submission validation

Every candidate file must have 30.000 rows, the exact test id set, unique ids,
and finite probabilities in [0, 1].
""")

code(r"""
test_id_set = set(pd.read_csv(DATA / "test_applications.csv",
                              usecols=["application_id"]).application_id)
files = sorted((V75 / "output").glob("submission_v75_*.csv"))
assert files, "no v75 submission files found"

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
anchor_file = V75 / "output/submission_v75_01_anchor.csv"
a = pd.read_csv(anchor_file).set_index("application_id").loc[test_ids].default_probability.to_numpy()
print(f"\nv75_01_anchor vs the submitted v66_04 file: "
      f"Spearman {spearmanr(a, ref).statistic:.10f}, "
      f"rank changes {int((rankdata(a) != rankdata(ref)).sum())}")
print(f"\n{len(files)} files validated")
print("STATUS: PASS")
""")

md(r"""
## 8. Recommendation and limits

**Final pick stays the anchor.** `agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv`
has a measured public score of 0.56424 and resubmitting it returns exactly that.
Every other candidate in this notebook has an expected public score of 0.5596,
because that is the measured mean of this model family, and under 1 percent
chance of reaching 0.57.

**The tickets are an option, not a model improvement.** They are worth
submitting only because the final Kaggle pick is made after public scores are
visible, so a ticket that lands low is simply not selected. They carry no
expected gain; they carry variance. Submitting them cannot lower the final
score, and `leg_mlp` is the one with the most fresh variance.

**Limits.**

* The submitted vector is a rank, not a calibrated probability. KS is invariant
  under any increasing transform, so the ranking is the whole model, but the
  values must not be read as default rates.
* The public calibration rests on three observations. Three points fix a mean
  far better than a spread, so the spread is floored at 0.0040 rather than
  taken from the sample.
* `customer_profiles` has no as-of date, so profile fields are used as given.
  Every other source is truncated at the decision time, verified in section 2.
* `features.py` does not apply a positive `days_ago` filter in three of its main
  blocks. That is safe only because the provided tables carry no post-cutoff
  records, which section 2 re-checks on every run.
* No external data, no AutoML, no automated feature engineering. Monotone
  constraint directions come from one explicitly defined statistic computed
  inside each training fold.
""")

NB.parent.mkdir(parents=True, exist_ok=True)
import uuid
NB.write_text(json.dumps({
    "cells": [{**c, "id": uuid.uuid4().hex[:8],
               "source": c["source"].splitlines(keepends=True)} for c in CELLS],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5}, indent=1))
print(f"wrote {NB} with {len(CELLS)} cells")
