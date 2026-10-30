"""v86: the committee notebook, rebuilt to stand on its own.

The v79-v84 notebook only ran inside this repository: it imported `features.py`,
read feature caches from `agas/shared/cache/` and stored predictions from
`agas/v66/output/`. Given only the competition data it failed in its first cell.
It also printed a training command with `--lgb-seeds 0,1,2,3,4`, while the run
behind the stored predictions used `0,1` (train_log_v66.txt), so its retrain
switch would not have reproduced the submission.

This builder writes every piece of code into the notebook itself:

  * the three feature modules are written to disk with `%%writefile`, from the
    exact source files (`features.py`, `agas/shared/feat_extra.py`,
    `agas/v65/training/feat_v65.py`); the only edit is how `feat_extra` finds the
    v65 blocks;
  * the trainer body is lines 134-285 of `agas/v66/training/train_submission_v66.py`
    copied verbatim into a function;
  * data is found automatically (repository `data/`, `/kaggle/input`, or
    `DSC_DATA`); stored predictions are used when present, otherwise the
    notebook rebuilds features and retrains from the raw CSVs.

Run:  python3 agas/v86/training/make_notebook_v86.py
then execute with  agas/v86/training/run_notebook_v86.py
"""

import ast
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "agas/v86/notebook/DSC0152_kali ini beneran deh_Notebook.ipynb"

FEATURES_SRC = (ROOT / "features.py").read_text()
V65_SRC = (ROOT / "agas/v65/training/feat_v65.py").read_text()
EXTRA_SRC = (ROOT / "agas/shared/feat_extra.py").read_text()
TRAIN_LINES = (ROOT / "agas/v66/training/train_submission_v66.py").read_text().splitlines()

# feat_extra finds feat_v65 by walking up to agas/v65/; in the notebook it sits beside it
tree = ast.parse(EXTRA_SRC)
loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_v65_blocks")
loader_src = ast.get_source_segment(EXTRA_SRC, loader)
assert EXTRA_SRC.count(loader_src) == 1
EXTRA_SRC = EXTRA_SRC.replace(loader_src + "\n\n\n_v65 = _load_v65_blocks()",
                              "import fe_v65 as _v65  # the v65 blocks, written by the cell above")
assert "_load_v65_blocks" not in EXTRA_SRC

assert TRAIN_LINES[133] == "    app = ft.load_apps()", TRAIN_LINES[133]
assert TRAIN_LINES[284] == "        seed_oof[fs] = per_seed", TRAIN_LINES[284]
TRAIN_BODY = "\n".join(TRAIN_LINES[133:285])

TRAINER_SRC = '''"""Seven-leg trainer of the v66 production run.

The body of `train_legs` from `app = ft.load_apps()` to `seed_oof[fs] = per_seed`
is lines 134-285 of agas/v66/training/train_submission_v66.py, copied verbatim.
Only the shell around it changed: it takes the feature frame as an argument
instead of reading cache files, and it returns the predictions instead of
writing them.
"""

import time
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer

NOISE_EXACT = {
    "region", "employment_type", "housing_status", "age", "employment_tenure_months",
    "dependents", "tenure_to_age", "income_per_dependent", "loan_purpose",
    "decision_lag_h", "app_hour", "app_dow",
}
NOISE_PREFIX = ("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_")
ROUND2_PREFIX = ("r2_", "b2_", "a2_", "e2_")

LEGS = ["fac_full", "lgb_full", "lgb_lean", "lgbx_full", "cat_full", "xgb_full", "linear"]


def ks_score(y, p):
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def split_frame(f):
    """Train/test design matrices, as load_frames() in the v66 script."""
    tr = f[f.target.notna()].reset_index(drop=True)
    te = f[f.target.isna()].reset_index(drop=True)
    ids = ["application_id", "customer_id", "target"]
    X, Xte = tr.drop(columns=ids), te.drop(columns=ids)
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    return X, Xte, tr.application_id.values, te.application_id.values


def train_legs(f, ft, folds=10, fold_seeds="42,7,13,21,5,101", lgb_seeds="0,1",
               lr=0.01, l2=5.0, log=print):
    args = SimpleNamespace(folds=folds, fold_seeds=fold_seeds, lgb_seeds=lgb_seeds,
                           lr=lr, l2=l2)
    X, Xte, train_ids, test_ids = split_frame(f)
''' + TRAIN_BODY + '''

    return {"train_ids": train_ids, "test_ids": test_ids,
            **{f"oof_{k}": v for k, v in oof_all.items()},
            **{f"test_{k}": v for k, v in test_all.items()}}
'''
ast.parse(TRAINER_SRC)

LEDGER = [
    ("r=0.75  tm=0.30  core=2:1  ref=lgb_lean   SUBMITTED", 0.56520),
    ("all three per-axis quadratic peaks at once", 0.56481),
    ("r=1.00", 0.56464),
    ("r=0.50 (v66_04)", 0.56424),
    ("r=0.50 on v70 features", 0.56419),
    ("r=0.50, ten fold seeds", 0.56411),
    ("core=1:1", 0.56400),
    ("trees4 + fac_full (five trees)", 0.56315),
    ("core=3:1", 0.56294),
    ("second core slot = cat_full", 0.56257),
    ("three legs retrained, +0.0008 AUC each", 0.56159),
    ("tm=0.20", 0.56158),
    ("MLP-heavy blend", 0.56111),
    ("tm=0.40", 0.56084),
    ("80% MLP mix", 0.56073),
    ("MLP + v70 pair", 0.56045),
    ("anchor + 5% GRU sequence", 0.55968),
    ("reference leg = lgb_full", 0.55934),
    ("single raw leg", 0.55801),
]

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md("""
# DSC LOGIKA UI 2026 — Credit Default Prediction

| | |
|---|---|
| **Participant number** | DSC0152 |
| **Team** | kali ini beneran deh |
| **Members** | Agastya Tristan Dhaniswara · Dimaz Satya Ardendi · Jan Agra Adyuta Harnowo |
| **Metric** | Kolmogorov–Smirnov statistic |
| **Public score** | **0.56520** |

**Task.** Rank 30,000 test applications by the probability that the loan reaches
90+ days past due within 12 months of origination.

**Submitted model.** Six gradient-boosting legs and one spline logistic
regression, trained with 10-fold cross-validation repeated over six fold
partitions, then blended in rank space and extrapolated away from the
lean-column LightGBM leg:

```
trees4 = mean of rank(lgb_full, lgb_lean, cat_full, xgb_full)
core   = (2 * trees4 + lgbx_full) / 3
tm30   = 0.7 * core + 0.3 * linear
final  = rank01(tm30 + 0.75 * (tm30 - lgb_lean))
```

### How to run

Run all cells from top to bottom (*Run All*). Nothing has to be configured.

| mode | when | what happens | time |
|---|---|---|---|
| **Fast path** (default) | the stored fold predictions of the production run exist (`agas/v66/output/preds_v66.npz`) | rebuilds the submitted vector from them and asserts it reproduces the submitted CSV exactly (zero rank changes, same SHA-256) | about 30 s |
| **Full path** | those predictions are absent, or `RUN_TRAINING = True` | builds every feature from the raw CSVs, retrains all seven legs, then blends | about 6.5 h on a 10-core CPU |

The **data folder** is found automatically: `data/` next to or above the
notebook, any folder under `/kaggle/input` that holds `train_applications.csv`,
or a path given in the `DSC_DATA` environment variable.

### Inputs and outputs

| | |
|---|---|
| **Input** | the eight competition CSVs, nothing else |
| **Output** | `submission_DSC0152.csv` in the work folder (`application_id,default_probability`, 30,000 rows) |
| **Work folder** | `agas/v86/notebook/work/` inside the project, otherwise `./dsc0152_work/` |
| **Code** | all of it is in this notebook; sections 4 and 6 write the feature and training modules to the work folder with `%%writefile` and import them |

### Environment used for the submission

Python 3.13.11 · numpy 2.5.1 · pandas 3.0.5 · scipy 1.18.0 · scikit-learn 1.9.0 ·
LightGBM 4.7.0 · XGBoost 3.3.0 · CatBoost 1.2.10 · matplotlib 3.11.1 · CPU only
(Apple M4, 10 cores).

### Contents

| section | what it does |
|---|---|
| 1. Setup | libraries, switches, data location, run mode |
| 2. Data | reads the tables and prints their basic facts |
| 3. Cutoff audit | proves no record is dated after `decision_datetime` |
| 4. Feature code | writes the three hand-written feature modules |
| 5. Feature frame | assembles the 598-column frame and the three column sets |
| 6. Models | the seven-leg trainer, or the stored predictions of its run |
| 7. Blend and submission | rank blend, out-of-fold scores, submission file, KS plot |
| 8. How the recipe was chosen | why every setting of the blend is the one it is |
| 9. Submission validation | format checks and the byte-identity check |
| 10. Final summary | one table with the result of the run |
| 11. Limits | what this notebook does not claim |
""")

md("""
## 1. Setup

### 1.1 Libraries and switches

`RUN_TRAINING` is the only switch. Leave it `False`: the notebook falls back to
the full path on its own when the stored predictions are missing.
`SUBMITTED_SHA256` is the fingerprint of the CSV that was submitted to Kaggle and
scored 0.56520; section 9 compares the rebuilt file against it.
""")

code('''
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

RUN_TRAINING = False   # True rebuilds features and retrains all seven legs from the raw CSVs
R_FINAL = 0.75         # extrapolation ratio of the submitted vector
SUBMITTED_SHA256 = "35260db3730a3eec7bc2cce9fb3b540d047c64d5d980e39ab19b07ebf07c733b"
''')

md("""
### 1.2 Locate the data and choose the run mode

Searches for the data folder, creates the work folder, and decides between the
fast and the full path. The printed versions are the ones this run used.
""")

code('''
def find_data():
    """Folder holding the competition CSVs."""
    here = Path.cwd().absolute()
    candidates = [Path(os.environ["DSC_DATA"])] if "DSC_DATA" in os.environ else []
    candidates += [p / "data" for p in [here, *here.parents]]
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        candidates += [p.parent for p in kaggle.rglob("train_applications.csv")]
    for c in candidates:
        if (c / "train_applications.csv").exists():
            return c.absolute()
    raise FileNotFoundError("train_applications.csv not found; set DSC_DATA to the data folder")


DATA = find_data()
PROJECT = DATA.parent                        # the project folder when run inside it
STORED = PROJECT / "agas/v66/output/preds_v66.npz"
REFERENCE_CSV = PROJECT / "agas/v76/output/submission_v76_02_r0.75.csv"
IN_PROJECT = (PROJECT / "agas").is_dir()
WORK = PROJECT / "agas/v86/notebook/work" if IN_PROJECT else Path.cwd() / "dsc0152_work"
(WORK / "modules").mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK / "modules"))

if not RUN_TRAINING and not STORED.exists():
    print("stored predictions not found: switching to the full path (features + training)")
    RUN_TRAINING = True

import sklearn, lightgbm, xgboost, catboost
print(f"data folder   : {DATA}")
print(f"work folder   : {WORK}")
print(f"mode          : {'FULL (features + training)' if RUN_TRAINING else 'FAST (stored predictions)'}")
print(f"python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | "
      f"sklearn {sklearn.__version__}")
print(f"lightgbm {lightgbm.__version__} | xgboost {xgboost.__version__} | catboost {catboost.__version__}")
''')

md("""
## 2. Data

The eight provided CSVs and nothing else. No external data is read anywhere in
this notebook.

| table | grain | role |
|---|---|---|
| `train_applications` / `test_applications` | one row per application | the rows to score; `target` = reached 90+ DPD within 12 months |
| `customer_profiles` | one row per customer | demographics and declared income |
| `previous_applications` | many per customer | earlier loan applications and their outcomes |
| `repayment_history` | many per customer | instalment due dates, payments and days past due |
| `account_behavior` | one per customer-month | balances, inflows and outflows |
| `bureau_observations` | many per customer | credit-bureau snapshots with an availability date |
| `ecommerce_transactions` | many per customer | e-commerce purchases |

### 2.1 Load the applications and print basic facts

Counts every table, loads the two application tables, and checks the facts the
model relies on: the default rate, that no customer appears in both train and
test, and that every loan has a 12-month term.
""")

code('''
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
print(f"\\ntrain {len(train):,} rows | test {len(test):,} rows "
      f"| positive rate {train.target.mean():.6f}")
print(f"customers appearing in both sets: "
      f"{len(set(train.customer_id) & set(test.customer_id))}")
print(f"term_months values   : {sorted(int(v) for v in train.term_months.unique())}")
print(f"interest rate values : {sorted(float(v) for v in train.annual_interest_rate.unique())}")
''')

md("""
## 3. Cutoff audit

Every feature may use only information available at the selected application's
`decision_datetime`. Two things are checked: whether any provided table holds a
record dated after that cutoff, and whether the bureau observation attached to
the application was actually available at decision time.

### 3.1 Count records dated after the cutoff

Joins every event table to the decision time of its customer's application and
counts events that happen later. An account month counts as known only once it
has closed. The cell stops the notebook if any count is above zero.
""")

code('''
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
print(f"\\ntotal records dated after the cutoff: {post}")
assert post == 0, "a source holds post-cutoff records; window features need a positive-time filter"
print("PASS: the provided tables are already truncated at decision_datetime")
del pa, rh, ab, ec, bu, sel
''')

md("""
## 4. Feature code

Three hand-written modules build one row per application. Every historical
aggregate is computed from `days_ago = decision_datetime - event` and windowed
over that quantity, so nothing dated after the decision can enter a feature.

| module | source in the project | content |
|---|---|---|
| `fe_base.py` | `features.py` | application terms, customer profile, previous applications, repayment history, account behavior, bureau observations, e-commerce behavior, ratio and trend interactions |
| `fe_v65.py` | `agas/v65/training/feat_v65.py` | data-quality (missingness and coverage) and cash-flow stress blocks |
| `fe_extra.py` | `agas/shared/feat_extra.py` | second-round blocks: exponentially weighted repayment and bureau history, within-grade z-scores, account and e-commerce extras, cross-source consistency, interactions |

The cells below write each module exactly as it is in the project; the only edit
is that `fe_extra.py` imports the v65 blocks from `fe_v65.py` instead of searching
for the project folder. Running them only writes files; nothing is computed yet.

### 4.1 `fe_base.py` — base features (311 columns)

Entry point `build()`. One block per source table, each returning customer-level
aggregates over time windows measured back from the decision date (from 7 days
up to 800 days), followed by debt-to-income and cash-flow ratios that combine blocks.
`factor_inputs` / `factor_features` at the end build fold-safe factor encodings,
used by the trainer in section 6.
""")

code("%%writefile modules/fe_base.py\n" + FEATURES_SRC)

md("""
### 4.2 `fe_v65.py` — data quality and stress blocks

Missingness and coverage across sources (`dataquality`), bureau direction
(`burdirection`, written but not part of the final frame), and affordability
measured against observed cash flow rather than declared income (`stress`).
""")

code("%%writefile modules/fe_v65.py\n" + V65_SRC)

md("""
### 4.3 `fe_extra.py` — second-round blocks

Exponentially weighted history of repayment and bureau observations,
within-grade z-scores (`gradenorm`), account and e-commerce extras,
cross-source consistency checks, and a small block of hand-picked interactions.
`build(names)` returns the requested blocks keyed by `application_id`.
""")

code("%%writefile modules/fe_extra.py\n" + EXTRA_SRC)

md("""
## 5. Feature frame

`fe_base.build()` gives 311 columns (308 features plus `application_id`,
`customer_id` and `target`). Thirteen second-round blocks and the interaction
block add 290 more, in the order the production cache was built, for 598
features in total. Three column sets are then used by the legs:

| set | rule | used by |
|---|---|---|
| `full` | every column | `lgb_full`, `lgbx_full`, `cat_full`, `xgb_full` |
| `lean` | drops demographics and share-type columns with no fold-stable gain | `lgb_lean`, `linear` |
| `fac` | drops the second-order rounding family (`r2_`, `b2_`, `a2_`, `e2_`) | `fac_full` |

### 5.1 Import the modules and define the assembly

Points every module at the data folder and defines `build_feature_frame()`,
which reproduces exactly what `features.py` plus `agas/shared/build_extra.py`
produced for the production run. Column order matters: LightGBM samples
columns by position, so a reordered frame would train different trees.
""")

code('''
import fe_base, fe_extra, fe_v65
for module in (fe_base, fe_extra, fe_v65):
    module.D = f"{DATA}/"          # every module reads the CSVs through this prefix

EXTRA_BLOCKS = ["bureau", "repayment", "previous", "account", "ecommerce", "gradenorm",
                "rxtra", "bxtra", "axtra", "ecommerce2", "consistency", "dataquality",
                "stress", "interactions"]


def build_feature_frame():
    """Exactly what features.py + agas/shared/build_extra.py produced for the run."""
    base = fe_base.build()
    app = fe_base.load_apps()
    parts = [fe_extra.build([n], app=app, base=base) for n in EXTRA_BLOCKS]
    parts.append(fe_extra.with_interactions(parts[0], base))
    extra = parts[0]
    for p in parts[1:]:
        extra = extra.merge(p, on="application_id", how="outer")
    add = [c for c in extra.columns if c != "application_id" and c not in base.columns]
    return base.merge(extra[["application_id"] + add], on="application_id", how="left")
''')

md("""
### 5.2 Build (full path) or load (fast path) the frame, and derive the column sets

* **Full path:** runs `build_feature_frame()` on the raw CSVs (several minutes).
* **Fast path:** reads the frame the same code produced earlier, from the
  project cache, only to report its size; the stored predictions do not need it.
  Outside the project the fast path does not exist, so this branch never runs
  there.
""")

code('''
CACHE = PROJECT / "agas/shared/cache"
frame = None
if RUN_TRAINING:
    t0 = time.time()
    frame = build_feature_frame()
    print(f"feature frame built from the raw CSVs in {(time.time() - t0) / 60:.1f} min")
elif (CACHE / "features_cache.pkl").exists() and (CACHE / "extra_cache.pkl").exists():
    base = pd.read_pickle(CACHE / "features_cache.pkl")
    extra = pd.read_pickle(CACHE / "extra_cache.pkl")
    add = [c for c in extra.columns if c != "application_id" and c not in base.columns]
    frame = base.merge(extra[["application_id"] + add], on="application_id", how="left")
    print("feature frame read from the project cache (same code, built earlier)")

if frame is not None:
    NOISE_EXACT = {"region", "employment_type", "housing_status", "age",
                   "employment_tenure_months", "dependents", "tenure_to_age",
                   "income_per_dependent", "loan_purpose", "decision_lag_h",
                   "app_hour", "app_dow"}
    cols = [c for c in frame.columns if c not in ("application_id", "customer_id", "target")]
    lean = [c for c in cols if c not in NOISE_EXACT
            and not c.startswith(("ec_share_", "ec_catqty_", "cs_shift", "buh_", "gn_"))]
    fac = [c for c in cols if not c.startswith(("r2_", "b2_", "a2_", "e2_"))]
    print(f"rows x columns    : {frame.shape[0]:,} x {len(cols)}")
    print(f"full / lean / fac : {len(cols)} / {len(lean)} / {len(fac)}")
    print(f"labelled rows     : {int(frame.target.notna().sum()):,} "
          f"| positives {int(frame.target.sum()):,}")
''')

md("""
## 6. Models

Seven legs, each trained with 10-fold stratified cross-validation repeated over
six fold partitions (fold seeds 42, 7, 13, 21, 5, 101). The three plain LightGBM
legs average two booster seeds (0, 1) inside every fold. Nothing is ever fitted
on the test set: test predictions are the mean of the fold models.

| leg | model | columns |
|---|---|---|
| `lgb_full` | LightGBM, 15 leaves, lr 0.01, `lambda_l2` 5, early stopping 300 | `full` |
| `lgb_lean` | same parameters | `lean` |
| `fac_full` | same parameters | `fac` |
| `lgbx_full` | LightGBM, 31 leaves, lr 0.012, `lambda_l2` 12, seed 99 | `full` |
| `cat_full` | CatBoost, depth 6, lr 0.03, `l2_leaf_reg` 5, early stopping 300 | `full` |
| `xgb_full` | XGBoost, depth 4, lr 0.01, `reg_lambda` 20, early stopping 300 | `full` |
| `linear` | quantile transform → spline (4 knots) → logistic regression, C 0.03 | `lean`, numeric only |

The trainer also builds fold-safe factor encodings from each fold's training
labels (`fe_base.factor_inputs` / `factor_features`). They are kept so the code
is the production code verbatim, but none of the seven column lists selects
them, so they do not enter any leg.

### 6.1 `trainer.py` — the seven-leg trainer

`train_legs(frame, fe_base)` loops over the six fold partitions; inside each it
trains every leg on 10 folds and accumulates its out-of-fold predictions (for
the 70,000 training rows) and its averaged test predictions. It returns both for
all seven legs. The loop body is the production script copied line for line.
""")

code("%%writefile modules/trainer.py\n" + TRAINER_SRC)

md("""
### 6.2 Train (full path) or load the stored predictions (fast path)

The full path prints one line per leg and fold partition with its out-of-fold
AUC and KS, and saves the result to `preds_retrained.npz` in the work folder.
The fast path loads the predictions saved by the production run.
""")

code('''
import trainer

if RUN_TRAINING:
    t0 = time.time()
    preds = trainer.train_legs(frame, fe_base,
                               log=lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True))
    np.savez_compressed(WORK / "preds_retrained.npz", **preds)
    print(f"training done in {(time.time() - t0) / 3600:.2f} h")
else:
    z = np.load(STORED, allow_pickle=True)
    preds = {k: z[k] for k in z.files}
    print(f"stored predictions of the production run: {STORED.relative_to(PROJECT)}")
    print("produced by agas/v66/training/train_submission_v66.py --folds 10 "
          "--fold-seeds 42,7,13,21,5,101 --lgb-seeds 0,1 --out-prefix v66")
''')

md("""
## 7. Blend and submission

KS depends only on the ordering of scores, so every leg is converted to ranks
before blending. The submitted vector is rebuilt from the fold predictions and,
when the submitted CSV is available, checked against it: ranks must match
exactly, Spearman must be 1.0, and no rank position may differ.

### 7.1 Blend functions

* `rank01` maps scores to ranks in (0, 1].
* `ks_stat` is the competition metric: the largest gap between the cumulative
  score distributions of defaulted and non-defaulted loans.
* `chain` is the submitted recipe, with every setting exposed as a parameter so
  section 8 can refer to the axes by name:
  `r` = extrapolation ratio, `tm` = linear-leg weight, `cw` = weight of `trees4`
  against `lgbx_full`, `ref` = the leg extrapolated away from.
""")

code('''
def rank01(v):
    return rankdata(v) / len(v)


def ks_stat(score, y):
    order = np.argsort(score, kind="mergesort")
    yy = y[order]
    return float(np.abs(np.cumsum(yy) / yy.sum() - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def chain(d, r=R_FINAL, tm=0.30, cw=2.0, ref="lgb_lean"):
    trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
    core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
    tm30 = (1 - tm) * core + tm * d["linear"]
    return rank01(tm30 + r * (tm30 - d[ref]))
''')

md("""
### 7.2 Out-of-fold performance of every leg and of the blend

Scores each leg and the submitted blend on the out-of-fold predictions of all
70,000 training applications. These numbers describe the models; section 8
explains why they were not used to pick the blend settings.
""")

code('''
train_ids, test_ids = preds["train_ids"], preds["test_ids"]
y = train.set_index("application_id").loc[train_ids].target.to_numpy().astype(int)
ro = {k[4:]: rank01(v) for k, v in preds.items() if k.startswith("oof_")}
rt = {k[5:]: rank01(v) for k, v in preds.items() if k.startswith("test_")}
oof_final, test_final = chain(ro), chain(rt)

perf = pd.DataFrame({k: (roc_auc_score(y, v), ks_stat(v, y)) for k, v in ro.items()},
                    index=["OOF AUC", "OOF KS"]).T.sort_values("OOF AUC", ascending=False)
perf.loc["SUBMITTED blend"] = (roc_auc_score(y, oof_final), ks_stat(oof_final, y))
print(perf.round(6).to_string())
''')

md("""
### 7.3 Write the submission and compare it with the submitted file

Writes `submission_DSC0152.csv`. Inside the project the submitted CSV is
available, so the rebuilt ranking is compared position by position; in the
fast path any difference stops the notebook.
""")

code('''
OUT_CSV = WORK / "submission_DSC0152.csv"
pd.DataFrame({"application_id": test_ids, "default_probability": test_final}).to_csv(OUT_CSV, index=False)
print(f"\\nwrote {OUT_CSV}")

if REFERENCE_CSV.exists():
    ref = pd.read_csv(REFERENCE_CSV).set_index("application_id").loc[test_ids].default_probability.to_numpy()
    changed = int((rankdata(test_final) != rankdata(ref)).sum())
    print(f"\\nrebuilt against {REFERENCE_CSV.name}")
    print(f"  max |difference|       : {np.abs(test_final - ref).max():.3e}")
    print(f"  Spearman               : {spearmanr(test_final, ref).statistic:.10f}")
    print(f"  rank positions changed : {changed}")
    if not RUN_TRAINING:
        assert changed == 0, "the rebuilt vector does not reproduce the submitted file"
''')

md("""
### 7.4 What the KS statistic measures, on the submitted blend

The KS statistic is the largest vertical gap between the score distributions of
loans that defaulted and loans that did not. The plot shows both cumulative
distributions for the submitted blend on the out-of-fold predictions of all
70,000 training applications, and marks the gap.
""")

code('''
order = np.sort(oof_final)
cdf_pos = np.searchsorted(np.sort(oof_final[y == 1]), order, side="right") / (y == 1).sum()
cdf_neg = np.searchsorted(np.sort(oof_final[y == 0]), order, side="right") / (y == 0).sum()
i = int(np.argmax(cdf_neg - cdf_pos))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(order, cdf_neg, color="#2a78d6", lw=2, label="no serious delinquency")
ax.plot(order, cdf_pos, color="#eb6834", lw=2, label="reached 90+ DPD")
ax.vlines(order[i], cdf_pos[i], cdf_neg[i], color="#555555", lw=1.5, linestyles="--")
ax.annotate(f"KS = {cdf_neg[i] - cdf_pos[i]:.4f}", (order[i], (cdf_pos[i] + cdf_neg[i]) / 2),
            xytext=(10, 0), textcoords="offset points", va="center", color="#222222")
ax.set_xlabel("blended score (rank, 0 = safest)")
ax.set_ylabel("cumulative share of loans")
ax.set_title("Out-of-fold score distributions of the submitted blend", loc="left")
ax.grid(color="#e6e6e6", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.show()
''')

md("""
## 8. How the recipe was chosen

The blend structure comes from v61 and was kept because it is the only chain
with repeated public results behind it. What changed, and what this notebook
submits, is the **extrapolation ratio**: `0.75` instead of the `0.5` used from
v61 through v76.

**Why out-of-fold scores were not the selection key.** v61 and v66 both swept
the ratio and both picked `0.5` by out-of-fold AUC. On this axis out-of-fold
points the wrong way: `r = 0.75` is −0.000350 AUC and −0.000920 KS against
`r = 0.5`, and +0.00102 on public. The `r = 0.75` vector had been built four
versions before it was ever submitted, ranked 7th of 9 by an out-of-fold score.

**The ratio axis, mapped.**

| r | public |
|---:|---:|
| 0.50 | 0.56424, 0.56419, 0.56411 (three independent trainings) |
| **0.75** | **0.56520** |
| 1.00 | 0.56464 |

The three `r = 0.50` values come from different feature sets and fold-seed
counts and agree to 0.00013, which is what makes the +0.00102 at `r = 0.75`
meaningful rather than noise.

**What the extrapolation does.** It is a weighted subtraction, not an
amplification:

```
tm30 + r * (tm30 - lgb_lean)  ==  (1 + r) * tm30 - r * lgb_lean
```

so the ratio sets how hard the lean-column leg is subtracted. It amplifies the
residual between the blend and `lgb_lean`, and public rewards that. Why that
residual carries signal is not established.

**Every other setting was moved and lost.** Each axis of the chain was measured
on public on both sides of the submitted value where possible:

| axis | moved to | public | change |
|---|---|---:|---:|
| ratio `r` | 0.50 / 1.00 | 0.56424 / 0.56464 | −0.00096 / −0.00056 |
| linear weight `tm` | 0.20 / 0.40 | 0.56158 / 0.56084 | −0.00362 / −0.00436 |
| core weight | 1:1 / 3:1 | 0.56400 / 0.56294 | −0.00120 / −0.00226 |
| reference leg | `lgb_full` | 0.55934 | −0.00586 |
| `trees4` membership | add `fac_full` | 0.56315 | −0.00205 |
| second core slot | `cat_full` instead of `lgbx_full` | 0.56257 | −0.00263 |
| all three fitted peaks at once | r 0.786, tm 0.295, core 1.85 | 0.56481 | −0.00039 |
| leg quality | three legs retrained, +0.0008 AUC each | 0.56159 | −0.00361 |

Two last probes on the final day — the fitted `r` peak alone, and `cat_full`
removed from `core` — also did not exceed 0.56520, so the recipe above was
frozen. The leg-quality row deserves a plain statement: an 8.7-hour run raised
three legs by +0.00075 to +0.00083 AUC, verified as a controlled A/B, and the
resulting chain scored 0.0036 **lower**. Better legs did not make this chain
better on public.

### 8.1 Public leaderboard ledger

Every scored submission of this chain and its relatives, best first.
""")

code('''
LEDGER = [
''' + "".join(f"    ({s!r}, {v:.5f}),\n" for s, v in LEDGER) + ''']
led = pd.DataFrame(LEDGER, columns=["submission", "public"])
led["vs submitted"] = (led.public - 0.56520).round(5)
print(led.to_string(index=False))
print(f"\\n{len(led)} scored submissions; the submitted recipe is the maximum.")
''')

md("""
## 9. Submission validation

The file must have exactly the columns `application_id,default_probability`,
30,000 rows, the exact test id set, unique ids, and finite probabilities in
[0, 1]. In the fast path the file must also be byte-identical to the submitted
one, checked by SHA-256.

### 9.1 Run the checks

Each rule is printed as PASS or FAIL; any FAIL stops the notebook.
""")

code('''
sub = pd.read_csv(OUT_CSV)
test_id_set = set(test.application_id)
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
digest = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
if not RUN_TRAINING:
    checks["byte-identical to the submitted file (SHA-256)"] = digest == SUBMITTED_SHA256
for k, v in checks.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
assert all(checks.values()), "submission failed validation"

print(f"\\nrows {len(sub):,} | min {sub.default_probability.min():.6f} "
      f"| max {sub.default_probability.max():.6f}")
print(f"sha256 rebuilt   : {digest}")
print(f"sha256 submitted : {SUBMITTED_SHA256}")
print("\\nSTATUS: PASS")
''')

md("""
## 10. Final summary

### 10.1 Result of this run

One table with what this run produced: the mode it ran in, the out-of-fold
quality of the submitted blend, and the file it wrote.
""")

code('''
summary = pd.Series({
    "team": "DSC0152 — kali ini beneran deh",
    "run mode": "full (features + training)" if RUN_TRAINING else "fast (stored predictions)",
    "legs blended": len(ro),
    "out-of-fold AUC of the blend": f"{roc_auc_score(y, oof_final):.6f}",
    "out-of-fold KS of the blend": f"{ks_stat(oof_final, y):.6f}",
    "public leaderboard KS": "0.56520",
    "submission file": OUT_CSV.name,
    "rows": f"{len(sub):,}",
    "identical to the submitted file": "yes" if digest == SUBMITTED_SHA256 else "no (retrained)",
})
print(summary.to_string())
''')

md("""
## 11. Limits

* The submitted values are ranks, not calibrated probabilities. KS is invariant
  under any increasing transform, so the ranking is the whole model, but the
  numbers must not be read as default rates.
* `customer_profiles` carries no as-of date, so profile fields are used as
  given. Every other source is truncated at the decision time, which section 3
  re-verifies on every run.
* `fe_base.py` does not apply a positive `days_ago` filter in three of its main
  blocks. That is safe only because the provided tables carry no post-cutoff
  records, which is exactly what section 3 checks.
* The recipe was selected against the public leaderboard, so it carries public
  selection risk: the private split is an independent sample and the measured
  +0.00102 need not reproduce there.
* The full path retrains with fixed seeds and deterministic LightGBM. On a
  different CPU or library version the boosters can still differ in the last
  bits, so a retrain is expected to match the submitted ranking closely but not
  necessarily byte for byte; the fast path is the exact reproduction.
* No external data, no AutoML and no automated feature engineering. Every
  feature in sections 4–5 is hand written, and the one tuned blend parameter was
  chosen from an explicitly written grid.
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
