"""Build v70 candidates from the stored v66, v69 and v70 predictions.

The v66 legs are reused exactly, so the object that twice produced the best
public score (`tm30` extrapolated away from `lgb_lean`) keeps its pivot. What
changes is the core: the v70 legs enter the tree average, and the extrapolation
ratios are searched around the same plateau.

Selection follows v64/v66: AUC is the key, KS is reported with the OOF sampling
spread subtracted, and vectors that rank almost identically are deduplicated.
The one candidate the run exists to produce is force-kept:

    ext0.5 | tm30 (v66 + v70 legs) | lgb_lean

which is the proven public direction with a stronger core.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"repo root wrong: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import paths  # noqa: E402

ROOT = paths.ROOT
OUT = HERE.parent / "output"
SIGMA_OOF, SIGMA_PUB = 0.0036, 0.0053
TAU = 0.9995
FLOOR = 0.0025
PRIMARY = "ext0.5|tm30v70|lgb_lean"


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in z.files:
        if k.startswith("oof_"):
            out[k[4:]] = (z[k], z["test_" + k[4:]])
    return out, z


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    legs66, z66 = load_npz(ROOT / "agas/v66/output/preds_v66.npz")
    legs69, _ = load_npz(ROOT / "agas/v69/output/preds_v69.npz")
    legs70, z70 = load_npz(OUT / "preds_v70.npz")
    train_ids, test_ids = z70["train_ids"], z70["test_ids"]
    y = z70["y"].astype(int)

    legs = {}
    for name, (o, t) in legs66.items():
        legs[name] = (o, t)
    for name, (o, t) in legs69.items():
        legs["x69_" + name] = (o, t)
    for name, (o, t) in legs70.items():
        legs["v70_" + name] = (o, t)
    ro = {k: rank01(v[0]) for k, v in legs.items()}
    rt = {k: rank01(v[1]) for k, v in legs.items()}
    print("legs:", ", ".join(f"{k}={roc_auc_score(y, v[0]):.6f}" for k, v in legs.items()))

    v70_names = [k for k in legs if k.startswith("v70_")]
    assert v70_names, "no v70 legs in preds_v70.npz"
    best_v70 = max(v70_names, key=lambda k: roc_auc_score(y, legs[k][0]))
    print(f"best v70 leg: {best_v70}")

    trees4_o = (ro["lgb_full"] + ro["lgb_lean"] + ro["cat_full"] + ro["xgb_full"]) / 4
    trees4_t = (rt["lgb_full"] + rt["lgb_lean"] + rt["cat_full"] + rt["xgb_full"]) / 4
    core_o = (2 * trees4_o + ro["lgbx_full"]) / 3
    core_t = (2 * trees4_t + rt["lgbx_full"]) / 3
    tm30_o = 0.7 * core_o + 0.3 * ro["linear"]
    tm30_t = 0.7 * core_t + 0.3 * rt["linear"]

    # one extended core per v70 leg, and one that uses every new leg
    news = v70_names + [k for k in ("x69_extra", "x69_stiff7") if k in ro]

    def extend(members):
        return extend_w(members, len(members) / (4 + len(members)))

    def extend_w(members, w):
        m_o = sum(ro[k] for k in members) / len(members)
        m_t = sum(rt[k] for k in members) / len(members)
        tr_o = (1 - w) * trees4_o + w * m_o
        tr_t = (1 - w) * trees4_t + w * m_t
        c_o = (2 * tr_o + ro["lgbx_full"]) / 3
        c_t = (2 * tr_t + rt["lgbx_full"]) / 3
        return 0.7 * c_o + 0.3 * ro["linear"], 0.7 * c_t + 0.3 * rt["linear"]

    cores = {"tm30v66": (tm30_o, tm30_t)}
    for k in v70_names:
        cores["tm30|" + k] = extend([k])
        for w in (0.3, 0.5, 0.7):
            cores[f"tm30w{int(w * 10)}|{k}"] = extend_w([k], w)
    cores["tm30v70"] = extend(v70_names)
    for w in (0.3, 0.5, 0.7):
        cores[f"tm30w{int(w * 10)}v70"] = extend_w(v70_names, w)
    if news:
        cores["tm30all"] = extend(news)

    def ext(bo, bt, pivot, r):
        po, pt = ro[pivot], rt[pivot]
        return rank01(bo + r * (bo - po)), rank01(bt + r * (bt - pt))

    cands = {}
    for k in legs:
        cands[f"leg|{k}"] = legs[k]
    cands["trees4"] = (trees4_o, trees4_t)
    cands["tm30v66"] = (tm30_o, tm30_t)
    for cn, (co, ct) in cores.items():
        if cn == "tm30v66":
            continue
        cands[cn] = (co, ct)
        for pivot in ["lgb_lean"] + ([best_v70] if best_v70 else []) + (
                ["x69_extra"] if "x69_extra" in ro else []):
            for r in (0.25, 0.5, 0.75, 1.0):
                cands[f"ext{r}|{cn}|{pivot}"] = ext(co, ct, pivot, r)
        # ratio plateau: averaging neighbouring ratios in rank space
        acc_o = np.zeros(len(y))
        acc_t = np.zeros(len(rt["linear"]))
        for r in (0.25, 0.5, 0.75):
            a, b = ext(co, ct, "lgb_lean", r)
            acc_o += a
            acc_t += b
        cands[f"plateau|{cn}|lgb_lean"] = (rank01(acc_o), rank01(acc_t))

    scored = []
    for name, (o, t) in cands.items():
        scored.append({"name": name, "oof": o, "test": t,
                       "auc": float(roc_auc_score(y, o)),
                       "ks": ks(y, o)})
    scored.sort(key=lambda r: -r["auc"])
    best = scored[0]["auc"]
    eligible = [r for r in scored if r["auc"] >= best - FLOOR]
    print(f"pool {len(scored)}, eligible {len(eligible)} at AUC floor {FLOOR} below {best:.6f}")

    anchor = pd.read_csv(ROOT / "agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv")
    anchor_rank = rank01(anchor.set_index("application_id").loc[test_ids]
                         .default_probability.values)

    keep_names = [PRIMARY, f"leg|{best_v70}"]
    kept = []
    for nm in keep_names:
        rec = next(r for r in scored if r["name"] == nm)
        if all(rec is not k for k in kept):
            kept.append(rec)
    for r in eligible:
        if any(r is k for k in kept):
            continue
        if all(spearmanr(r["test"], k["test"]).statistic < TAU for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: -r["auc"])
    print(f"kept {len(kept)} after dedupe at {TAU}")

    for f in OUT.glob("submission_v70_*.csv"):
        f.unlink()

    rows = []
    for i, k in enumerate(kept, 1):
        p = OUT / f"submission_v70_{i:02d}_{k['name'].replace('|', '_')}.csv"
        d = pd.DataFrame({"application_id": test_ids,
                          "default_probability": rank01(k["test"])})
        d.to_csv(p, index=False)
        ok = (len(d) == 30000 and d.application_id.is_unique
              and d.application_id.isin(test_ids).all()
              and not d.default_probability.isna().any()
              and d.default_probability.between(0, 1).all())
        rows.append({"slot": i, "name": k["name"], "auc": round(k["auc"], 6),
                     "ks": round(k["ks"], 6),
                     "ks_minus_sigma": round(k["ks"] - SIGMA_OOF, 6),
                     "spearman_v66_04": round(
                         float(spearmanr(k["test"], anchor_rank).statistic), 6),
                     "file": p.name, "valid": bool(ok)})

    print(f"\n=== candidates ranked by AUC (KS minus OOF sampling sd) ===")
    print(f"{'slot':>4} {'name':38s} {'auc':>9} {'ks':>9} {'ks-sd':>9} {'vs v66_04':>10}")
    for r in rows:
        print(f"{r['slot']:>4} {r['name']:38s} {r['auc']:9.6f} {r['ks']:9.6f} "
              f"{r['ks_minus_sigma']:9.6f} {r['spearman_v66_04']:10.6f}")

    summary = {"sigma_oof": SIGMA_OOF, "sigma_public": SIGMA_PUB,
               "dedupe_threshold": TAU, "auc_floor_gap": FLOOR, "best_auc": best,
               "best_v70_leg": best_v70, "primary": PRIMARY,
               "pool_size": len(scored), "eligible": len(eligible),
               "kept": len(kept), "candidates": rows}
    (OUT / "summary_v70_candidates.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'summary_v70_candidates.json'}")


if __name__ == "__main__":
    main()
