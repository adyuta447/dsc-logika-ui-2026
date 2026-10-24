"""Create the v72 reproducibility notebook.

The notebook is artifact-only by default (``RUN_TRAINING=False``).  It reloads
the sequence tensor, runs a point-in-time audit against the raw tables, and
recomputes the OOF gate from the stored predictions.  Heavy training stays in
``build_sequences_v72.py`` and ``train_seq_v72.py``.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebook"
OUT.mkdir(parents=True, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = [
    md("""
# v72 — sequence representation gate

**Status:** executed, negative result. **Anchor:** `agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv` (public 0.56424).

v71 `core71` scored 0.56111 on public even though its OOF KS was +0.004 above the
anchor. v72 therefore tested a genuinely different representation: the last 24
calendar months of account cash flow, repayment, e-commerce, bureau reports, and
previous applications as a sequence. Every event is cut at `decision_datetime`.

The notebook is reproducible with `RUN_TRAINING=False`. It reads the stored
tensor, predictions, and summaries; it does not retrain the GRU.
"""),
    md("## 0. Setup"),
    code("""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

RUN_TRAINING = False
ROOT = Path.cwd()
while not (ROOT / "agas/shared/paths.py").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)

OUT = ROOT / "agas/v72/output"
print("RUN_TRAINING:", RUN_TRAINING)
print("ROOT:", ROOT)
print("output exists:", OUT.exists())
"""),
    md("""
## 1. Data representation

`sequences_v72.npz` has shape `(100000, 24, 56)` and five source masks. The
channels are grouped as account (10), repayment (13), e-commerce (16), bureau
(8), and previous applications (9). A month with no event for a source has mask
0, so the model can distinguish "zero" from "not observed".
"""),
    code("""
seq = np.load(OUT / "sequences_v72.npz", allow_pickle=True)
X, M = seq["X"], seq["M"]
channels = seq["channels"].tolist()
print("X:", X.shape, "M:", M.shape, "dtype:", X.dtype)
print("apps:", len(seq["application_id"]), "train:", int(np.isfinite(seq["y"]).sum()),
      "test:", int((~np.isfinite(seq["y"])).sum()))
print("channels:", len(channels))
print(channels)
for i, name in enumerate(["account", "repayment", "ecommerce", "bureau", "previous"]):
    print(f"  {name:10s} mask mean={M[:, :, i].mean():.3f}")
"""),
    md("""
## 2. Point-in-time audit

The sequence builder keeps only events at or before the decision. This cell
rechecks the raw sources directly. All four "future" counts must be zero.
"""),
    code("""
dt = ["application_datetime", "decision_datetime"]
tr = pd.read_csv(ROOT / "data/train_applications.csv", parse_dates=dt)
te = pd.read_csv(ROOT / "data/test_applications.csv", parse_dates=dt)
cur = pd.concat([
    tr[["application_id", "customer_id", "decision_datetime"]],
    te[["application_id", "customer_id", "decision_datetime"]],
], ignore_index=True)
print("target prevalence:", round(tr.target.mean(), 6))

rh = pd.read_csv(ROOT / "data/repayment_history.csv", parse_dates=["due_date"])
rh = rh.merge(cur[["customer_id", "decision_datetime"]], on="customer_id", how="left")
print("repayment due after decision:", int((rh.due_date > rh.decision_datetime).sum()))

ab = pd.read_csv(ROOT / "data/account_behavior.csv", parse_dates=["month"])
ab = ab.merge(cur[["customer_id", "decision_datetime"]], on="customer_id", how="left")
print("account month after decision:", int((ab.month > ab.decision_datetime).sum()))

ec = pd.read_csv(ROOT / "data/ecommerce_transactions.csv",
                 usecols=["customer_id", "transaction_datetime"],
                 parse_dates=["transaction_datetime"])
ec = ec.merge(cur[["customer_id", "decision_datetime"]], on="customer_id", how="left")
print("ecommerce transaction after decision:", int((ec.transaction_datetime > ec.decision_datetime).sum()))

bu = pd.read_csv(ROOT / "data/bureau_observations.csv",
                 usecols=["customer_id", "available_datetime"],
                 parse_dates=["available_datetime"])
bu = bu.merge(cur[["customer_id", "decision_datetime"]], on="customer_id", how="left")
print("bureau report available after decision:", int((bu.available_datetime > bu.decision_datetime).sum()))

pa = pd.read_csv(ROOT / "data/previous_applications.csv", usecols=["application_id"])
print("current application ids found in previous_applications:",
      len(set(tr.application_id) & set(pa.application_id)))
"""),
    md("""
## 3. Sequence legs

Both runs use the same 5-fold stratified split (fold seed 42, model seed 0).
`sequence + static` appends the 770 numeric v70 columns, so it is the fair
comparison against the existing static family. The pure sequence leg is the
decorrelation probe.
"""),
    code("""
legs = {}
for name, fname in [("sequence", "preds_seq_v72_pure.npz"),
                    ("sequence+static", "preds_seq_v72_static.npz")]:
    z = np.load(OUT / fname, allow_pickle=True)
    y = z["y"].astype(int)
    p = z["oof_42"]
    legs[name] = (y, p)
    print(f"{name:16s} AUC={roc_auc_score(y, p):.6f} KS={ks_2samp(p[y == 1], p[y == 0]).statistic:.6f}")
"""),
    md("""
## 4. Blend gate against the anchor

The anchor is rebuilt from the stored v66 legs using exactly the v66_04 chain.
A candidate must add KS, not only AUC. The table uses rank blends so the scale
of each leg is irrelevant.
"""),
    code("""
z66 = np.load(ROOT / "agas/v66/output/preds_v66.npz", allow_pickle=True)
train_ids = z66["train_ids"]
y = pd.read_csv(ROOT / "data/train_applications.csv",
                usecols=["application_id", "target"]
                ).set_index("application_id").loc[train_ids].target.values.astype(int)
rank01 = lambda v: rankdata(v) / len(v)
ro = {k[4:]: rank01(z66[k]) for k in z66.files if k.startswith("oof_")}
trees = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
core = (2 * trees + ro["lgbx_full"]) / 3
tm30 = 0.7 * core + 0.3 * ro["linear"]
anchor = rank01(tm30 + 0.5 * (tm30 - ro["lgb_lean"]))
print(f"anchor AUC={roc_auc_score(y, anchor):.6f} KS={ks_2samp(anchor[y == 1], anchor[y == 0]).statistic:.6f}")

for name, (_, p) in legs.items():
    print()
    print(name)
    for w in (0.0, 0.02, 0.05, 0.10, 0.15, 0.25):
        b = rank01((1 - w) * anchor + w * rank01(p))
        print(f"  w={w:0.2f}  AUC={roc_auc_score(y, b):.6f}  "
              f"KS={ks_2samp(b[y == 1], b[y == 0]).statistic:.6f}  "
              f"rho={spearmanr(b, anchor).statistic:.6f}")
"""),
    md("""
## 5. Candidate files and decision

Only the anchor is recommended. The two sequence blends are diagnostics: they
are valid CSVs, but their OOF gain is inside the noise floor and their KS is not
higher than the anchor. The bootstrap column is `dKS` against the anchor on
2,000 stratified 30,000-row resamples.
"""),
    code("""
summary = json.loads((OUT / "summary_v72_candidates.json").read_text())
print("decision:", summary["decision"])
print()
print(f"{'file':34s} {'w':>4s} {'AUC':>9s} {'KS':>9s} {'rho':>8s} valid rec")
for r in summary["candidates"]:
    d = pd.read_csv(OUT / r["file"])
    valid = (len(d) == 30000 and d.application_id.is_unique
             and np.isfinite(d.default_probability).all()
             and d.default_probability.between(0, 1).all())
    print(f"{r['file']:34s} {r['weight']:4.2f} {r['oof_auc']:9.6f} "
          f"{r['oof_ks']:9.6f} {r['spearman_anchor']:8.6f} {valid} {r['recommended']}")
print()
for name, b in summary["bootstrap_dks_vs_anchor"].items():
    print(f"{name}: dKS={b['mean_dks']:+.6f}, P(dKS>0)={b['p_gt0']:.3f}")
"""),
    md("""
## 6. Conclusion

- The sequence is a real but weak signal: pure OOF AUC 0.7833, rank correlation
  only 0.76 to the anchor.
- Adding the static block raises it to 0.8415, still below the static MLP
  (0.8484), so the sequence branch is not a better static model.
- Blend weights that buy a small AUC increase do not buy KS. The 5% pure
  sequence blend has P(dKS>0) = 0.676, which is not evidence.
- **No v72 candidate is promoted.** Keep `v66_04` as the public anchor and
  move to the v73 freeze/verification plan in `agas/plan.md`.

Artifacts:

- `agas/v72/training/build_sequences_v72.py`
- `agas/v72/training/train_seq_v72.py`
- `agas/v72/training/make_submission_v72.py`
- `agas/v72/output/summary_v72_candidates.json`
- `agas/v72/output/submission_v72_01_anchor.csv`
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
path = OUT / "notebook_v72.ipynb"
nbf.write(nb, path)
print(f"wrote {path} ({len(cells)} cells)")
