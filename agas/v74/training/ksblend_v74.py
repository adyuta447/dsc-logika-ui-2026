"""Fase 2: select a blend by KS, with honest out-of-selection validation.

v57-v73 chose candidates by OOF AUC and reported KS. AUC is the integral of the
ROC; the competition metric is one point on it - the threshold where the local
default rate equals the base rate. Two blends with the same AUC can differ in
KS, so the selection key has never matched the metric.

The reason earlier versions avoided a KS key is sound: KS is a maximum
statistic, so its argmax over a few hundred vectors is mostly noise. This
script keeps the KS key and removes the noise instead:

  1. the objective is the *bootstrap mean* KS over 400 resamples, not raw KS;
  2. weights are fitted on one stratified half of the OOF and scored on the
     other half, both directions, so the number reported is an
     out-of-selection gain, not the gain that was optimised for;
  3. the search is a greedy forward pass over whole legs with a minimum
     improvement, not a free fit of 13 weights.

Leg pool: the seven v66 legs that build the anchor, plus v70 `extra`/`stiff7`,
the v71 MLP/MLP-wide/CatBoost legs, and the v72 pure sequence leg. Every leg is
an out-of-fold vector already on disk, so this phase trains nothing.
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
OUT = HERE.parent / "output"
AGAS = ROOT / "agas"

N_PUBLIC = 3000          # implied by the calibration in fase 0
BOOT = 400
GATE_GAIN = 0.002
GATE_PROB = 0.80
MIN_STEP = 0.0005        # a forward step must buy at least this much
MAX_ADD = 3


def rank01(v):
    return rankdata(v) / len(v)


def ks(score, y):
    o = np.argsort(score, kind="mergesort")
    yy = y[o]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def legs():
    """Every available out-of-fold leg, aligned to the v66 id order."""
    z66 = np.load(AGAS / "v66/output/preds_v66.npz", allow_pickle=True)
    tr_ids = pd.Index(z66["train_ids"])
    te_ids = pd.Index(z66["test_ids"])
    o, t = {}, {}
    for k in z66.files:
        if k.startswith("oof_"):
            o[k[4:]] = rank01(z66[k])
            t[k[4:]] = rank01(z66["test_" + k[4:]])

    def add(name, path, okey, tkey, seeds=None):
        z = np.load(AGAS / path, allow_pickle=True)
        si = pd.Index(z["train_ids"]).get_indexer(tr_ids)
        ti = pd.Index(z["test_ids"]).get_indexer(te_ids)
        assert (si >= 0).all() and (ti >= 0).all(), f"id mismatch in {path}"
        if seeds:
            ov = np.mean([z[okey % s] for s in seeds], axis=0)
            tv = np.mean([z[tkey % s] for s in seeds], axis=0)
        else:
            ov, tv = z[okey], z[tkey]
        o[name] = rank01(ov[si])
        t[name] = rank01(tv[ti])

    add("v70_extra", "v70/output/preds_v70.npz", "oof_extra", "test_extra")
    add("v70_stiff7", "v70/output/preds_v70.npz", "oof_stiff7", "test_stiff7")
    add("mlp", "v71/output/preds_mlp_v71.npz", "oof_%d", "test_%d", (42, 7, 13))
    add("mlp_wide", "v71/output/preds_mlp_wide_v71.npz", "oof_%d", "test_%d", (42, 7, 13))
    add("cat71", "v71/output/preds_cat_v71.npz", "oof_%d", "test_%d", (42, 7, 13))
    add("seq", "v72/output/preds_seq_v72_pure.npz", "oof_42", "test_42")

    y = (pd.read_csv(ROOT / "data/train_applications.csv",
                     usecols=["application_id", "target"])
         .set_index("application_id").loc[tr_ids].target.values.astype(int))
    return o, t, y, tr_ids.values, te_ids.values


def anchor_chain(o, t):
    """The v66_04 recipe: ext0.5 of tm30 away from lgb_lean."""
    def build(d):
        trees = (d["lgb_full"] + d["lgb_lean"] + d["cat_full"] + d["xgb_full"]) / 4
        core = (trees * 2 + d["lgbx_full"]) / 3
        tm30 = 0.7 * core + 0.3 * d["linear"]
        return rank01(tm30 + 0.5 * (tm30 - d["lgb_lean"]))
    return build(o), build(t)


def boot_idx(y, n, draws, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    npos = int(round(n * y.mean()))
    return [np.concatenate([rng.choice(pos, npos, replace=False),
                            rng.choice(neg, n - npos, replace=False)])
            for _ in range(draws)]


def mean_ks(score, y, idxs):
    return float(np.mean([ks(score[i], y[i]) for i in idxs]))


def greedy(o, y, base, pool, idxs, rng):
    """Forward-select legs into `base` by bootstrap-mean KS."""
    cur = base.copy()
    best = mean_ks(rank01(cur), y, idxs)
    chosen = []
    for _ in range(MAX_ADD):
        trials = []
        for name in pool:
            if name in [c[0] for c in chosen]:
                continue
            for w in (0.05, 0.10, 0.20, 0.30):
                cand = rank01((1 - w) * cur + w * o[name])
                trials.append((mean_ks(cand, y, idxs), name, w))
        trials.sort(reverse=True)
        if not trials or trials[0][0] < best + MIN_STEP:
            break
        best, name, w = trials[0]
        cur = rank01((1 - w) * cur + w * o[name])
        chosen.append((name, w))
    return chosen, best


def apply_chain(vec, o, chosen):
    cur = vec.copy()
    for name, w in chosen:
        cur = rank01((1 - w) * cur + w * o[name])
    return cur


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    o, t, y, tr_ids, te_ids = legs()
    print(f"legs: {sorted(o)}")
    a_o, a_t = anchor_chain(o, t)
    print(f"anchor OOF auc={roc_auc_score(y, a_o):.6f} ks={ks(a_o, y):.6f}")

    pool = ["v70_extra", "v70_stiff7", "mlp", "mlp_wide", "cat71", "seq",
            "fac_full", "lgb_full", "cat_full", "xgb_full", "linear"]
    rng = np.random.default_rng(0)

    # honest validation: fit the blend on one stratified half, score on the other
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng.shuffle(pos), rng.shuffle(neg)
    halves = [np.concatenate([pos[::2], neg[::2]]),
              np.concatenate([pos[1::2], neg[1::2]])]
    oos = []
    for fit_i, (hf, hs) in enumerate([(0, 1), (1, 0)]):
        fi, si = halves[hf], halves[hs]
        o_fit = {k: rank01(v[fi]) for k, v in o.items()}
        idxs = boot_idx(y[fi], min(N_PUBLIC, len(fi) // 2), BOOT, rng)
        chosen, fit_score = greedy(o_fit, y[fi], rank01(a_o[fi]), pool, idxs, rng)
        o_sc = {k: rank01(v[si]) for k, v in o.items()}
        got = apply_chain(rank01(a_o[si]), o_sc, chosen)
        gain = ks(got, y[si]) - ks(rank01(a_o[si]), y[si])
        oos.append({"fit_half": hf, "chosen": chosen,
                    "fit_mean_ks": round(fit_score, 6),
                    "score_half_ks_anchor": round(ks(rank01(a_o[si]), y[si]), 6),
                    "score_half_ks_blend": round(ks(got, y[si]), 6),
                    "oos_gain": round(gain, 6)})
        print(f"  half {hf}->{hs}: chose {chosen} | out-of-selection KS gain {gain:+.6f}")

    oos_gain = float(np.mean([r["oos_gain"] for r in oos]))
    print(f"\nmean out-of-selection KS gain {oos_gain:+.6f} (gate {GATE_GAIN})")

    # full-data fit, reported with bootstrap probability against the anchor
    idxs_full = boot_idx(y, N_PUBLIC, 1000, rng)
    chosen_full, _ = greedy(o, y, a_o, pool, boot_idx(y, N_PUBLIC, BOOT, rng), rng)
    blend_o = apply_chain(a_o, o, chosen_full)
    blend_t = apply_chain(a_t, t, chosen_full)
    d = np.array([ks(blend_o[i], y[i]) - ks(a_o[i], y[i]) for i in idxs_full])
    prob = float((d > 0).mean())
    print(f"full fit chose {chosen_full}")
    print(f"  OOF  auc {roc_auc_score(y, a_o):.6f} -> {roc_auc_score(y, blend_o):.6f}")
    print(f"  OOF  ks  {ks(a_o, y):.6f} -> {ks(blend_o, y):.6f}")
    print(f"  bootstrap dKS at n={N_PUBLIC}: mean {d.mean():+.6f} "
          f"sd {d.std(ddof=1):.6f}  P(>anchor)={prob:.3f}")
    rho = float(spearmanr(blend_t, a_t).statistic)
    print(f"  Spearman vs anchor on test: {rho:.6f}")

    verdict = ("PASS" if oos_gain >= GATE_GAIN and prob >= GATE_PROB else "FAIL")
    print(f"\ngate: oos_gain>={GATE_GAIN} and P>={GATE_PROB} -> {verdict}")

    np.savez_compressed(OUT / "ksblend_v74.npz", train_ids=tr_ids, test_ids=te_ids,
                        y=y, anchor_oof=a_o, anchor_test=a_t,
                        blend_oof=blend_o, blend_test=blend_t,
                        **{f"oof_{k}": v for k, v in o.items()},
                        **{f"test_{k}": v for k, v in t.items()})
    (OUT / "ksblend_v74.json").write_text(json.dumps(
        {"n_public_assumed": N_PUBLIC, "gate_gain": GATE_GAIN, "gate_prob": GATE_PROB,
         "out_of_selection": oos, "mean_oos_gain": oos_gain,
         "full_fit_chosen": chosen_full, "prob_beat_anchor": prob,
         "anchor_oof_ks": ks(a_o, y), "blend_oof_ks": ks(blend_o, y),
         "anchor_oof_auc": float(roc_auc_score(y, a_o)),
         "blend_oof_auc": float(roc_auc_score(y, blend_o)),
         "spearman_vs_anchor": rho, "verdict": verdict}, indent=2))
    print(f"saved {OUT / 'ksblend_v74.json'}")


if __name__ == "__main__":
    main()
