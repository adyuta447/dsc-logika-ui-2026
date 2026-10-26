"""Fase 2: best leg per slot, then the chain on top of it.

The production run was stopped after one fold seed of five. The reason is a
design fault in the trainer this file's sibling copied from v66: that script
caches one concatenated feature frame per fold inside a fold seed, roughly
0.5 GB each, so twelve folds need about 5.8 GB on top of `X` and `Xte`. Fold
seed 42 completed in 143 minutes; fold seed 7 then began swapping and its
`lgb_lean` leg alone took 98 minutes against 27 for the same leg one seed
earlier. With swap at 15.2 of 16.4 GB the remaining four seeds would have
taken fifteen hours or more with a real chance of being killed, so the run was
stopped and its first checkpoint kept.

That checkpoint is not a consolation prize. Comparing it against v66's fully
bagged legs:

    leg          v66 (6 fold seeds)   v76 (1 fold seed)
    lgb_full          0.846682            0.847231
    lgb_lean          0.846721            0.847220
    fac_full          0.846662            0.847159
    lgbx_full         0.846550            0.845799
    cat_full          0.846311            0.845588
    xgb_full          0.846756            0.846184
    linear            0.842771            0.842330

The three legs that received `extra_trees` beat their v66 counterparts while
carrying a sixth of the fold-seed bagging. The four that did not receive it are
worse by almost exactly what one fold seed instead of six costs. So neither run
dominates: v76 holds the better version of three slots and v66 the better
version of four.

The chain is a rank-space combination of seven independently trained legs, so
each slot can be filled from whichever run trained it better. That is what
`hybrid` below does. `pure` fills every slot from v76 and is built only so the
comparison is on the record rather than asserted.

Nothing here is retrained; both runs' stored predictions are re-assembled.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGAS = ROOT / "agas"
OUT = HERE.parent / "output"

EXTRA_TREES_LEGS = ("lgb_full", "lgb_lean", "fac_full")
LEGS = ("fac_full", "lgb_full", "lgb_lean", "lgbx_full", "cat_full",
        "xgb_full", "linear")


def rank01(v):
    return rankdata(v) / len(v)


def ks(score, y):
    o = np.argsort(score, kind="mergesort")
    yy = y[o]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z66 = np.load(AGAS / "v66/output/preds_v66.npz", allow_pickle=True)
    z76 = np.load(OUT / "preds_v76legs_partial.npz", allow_pickle=True)

    tr_ids, te_ids = pd.Index(z66["train_ids"]), pd.Index(z66["test_ids"])
    si = pd.Index(z76["train_ids"]).get_indexer(tr_ids)
    ti = pd.Index(z76["test_ids"]).get_indexer(te_ids)
    assert (si >= 0).all() and (ti >= 0).all(), "id mismatch between v66 and v76"
    assert (si == np.arange(len(si))).all(), "v76 train id order differs from v66"

    y = (pd.read_csv(ROOT / "data/train_applications.csv",
                     usecols=["application_id", "target"])
         .set_index("application_id").loc[tr_ids].target.values.astype(int))

    o66 = {k: rank01(z66[f"oof_{k}"]) for k in LEGS}
    t66 = {k: rank01(z66[f"test_{k}"]) for k in LEGS}
    o76 = {k: rank01(z76[f"oof_{k}"][si]) for k in LEGS}
    t76 = {k: rank01(z76[f"test_{k}"][ti]) for k in LEGS}

    print(f"{'leg':12s} {'v66 AUC':>10} {'v76 AUC':>10} {'delta':>10} "
          f"{'extra_trees':>12} {'pick':>6}")
    picks, leg_rows = {}, []
    for k in LEGS:
        a66, a76 = roc_auc_score(y, o66[k]), roc_auc_score(y, o76[k])
        pick = "v76" if a76 > a66 else "v66"
        picks[k] = pick
        leg_rows.append({"leg": k, "v66_auc": round(float(a66), 6),
                         "v76_auc": round(float(a76), 6),
                         "delta": round(float(a76 - a66), 6),
                         "extra_trees": k in EXTRA_TREES_LEGS, "pick": pick})
        print(f"{k:12s} {a66:10.6f} {a76:10.6f} {a76 - a66:+10.6f} "
              f"{str(k in EXTRA_TREES_LEGS):>12} {pick:>6}")

    # every leg that got extra_trees should have been picked from v76 and every
    # leg that did not should have been picked from v66; if that ever stops
    # holding, the explanation above is wrong and the run needs re-reading
    expected = {k: ("v76" if k in EXTRA_TREES_LEGS else "v66") for k in LEGS}
    if picks != expected:
        print(f"\nnote: pick pattern departs from the extra_trees split: {picks}")

    def chain(d, tm=0.30, r=0.50, ref="lgb_lean", cw=2.0):
        trees4 = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
        core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)
        base = (1 - tm) * core + tm * d["linear"]
        return rank01(base + r * (base - d[ref]))

    sets = {
        "anchor_v66": (o66, t66),
        "hybrid": ({k: (o76 if picks[k] == "v76" else o66)[k] for k in LEGS},
                   {k: (t76 if picks[k] == "v76" else t66)[k] for k in LEGS}),
        "pure_v76": (o76, t76),
    }

    a_o = chain(sets["anchor_v66"][0])
    a_t = chain(sets["anchor_v66"][1])
    a_auc, a_ks = float(roc_auc_score(y, a_o)), ks(a_o, y)

    rows, vecs = [], {}
    for name, (od, td) in sets.items():
        vo, vt = chain(od), chain(td)
        vecs[name] = vt
        rho = float(spearmanr(vt, a_t).statistic)
        rows.append({"name": name,
                     "oof_auc": round(float(roc_auc_score(y, vo)), 6),
                     "oof_ks": round(ks(vo, y), 6),
                     "d_auc": round(float(roc_auc_score(y, vo)) - a_auc, 6),
                     "d_ks": round(ks(vo, y) - a_ks, 6),
                     "spearman_vs_anchor": round(rho, 6),
                     "legs_from_v76": (list(LEGS) if name == "pure_v76"
                                       else [k for k in LEGS if picks[k] == "v76"]
                                       if name == "hybrid" else [])})

    print(f"\n{'chain':12s} {'OOF AUC':>10} {'OOF KS':>10} {'dAUC':>10} "
          f"{'dKS':>10} {'rho':>9}")
    for r in rows:
        print(f"{r['name']:12s} {r['oof_auc']:10.6f} {r['oof_ks']:10.6f} "
              f"{r['d_auc']:+10.6f} {r['d_ks']:+10.6f} "
              f"{r['spearman_vs_anchor']:9.6f}")

    test_set = set(pd.read_csv(ROOT / "data/test_applications.csv",
                               usecols=["application_id"]).application_id)
    for f in sorted(OUT.glob("submission_v76legs_*.csv")):
        f.unlink()
    for r in rows:
        if r["name"] == "anchor_v66":
            continue
        p = OUT / f"submission_v76legs_{r['name']}.csv"
        frame = pd.DataFrame({"application_id": te_ids.values,
                              "default_probability": rank01(vecs[r["name"]])})
        frame.to_csv(p, index=False)
        ok = (list(frame.columns) == ["application_id", "default_probability"]
              and len(frame) == 30000 and frame.application_id.is_unique
              and set(frame.application_id) == test_set
              and not frame.default_probability.isna().any()
              and np.isfinite(frame.default_probability).all()
              and frame.default_probability.between(0, 1).all())
        assert ok, f"invalid csv {p}"
        r["file"] = p.name
        print(f"  wrote {p.name}")

    (OUT / "summary_v76legs.json").write_text(json.dumps(
        {"source_v76": "preds_v76legs_partial.npz (12 folds x 1 fold seed x "
                       "5 booster seeds, extra_trees on 3 legs)",
         "source_v66": "preds_v66.npz (10 folds x 6 fold seeds x 5 booster seeds)",
         "extra_trees_legs": list(EXTRA_TREES_LEGS),
         "legs": leg_rows, "picks": picks,
         "anchor_oof_auc": a_auc, "anchor_oof_ks": a_ks,
         "chains": rows}, indent=2))
    print(f"\nsaved {OUT / 'summary_v76legs.json'}")


if __name__ == "__main__":
    main()
