"""Build and select v71 submissions.

The v71 story is not the historical panel (it did not add signal to the current
model).  The story is model diversity: the LightGBM legs from v70, a CatBoost leg
and a tabular MLP.  On the 15-fold OOF the MLP alone reaches ~0.8484 AUC, and a
rank blend with the v70 legs reaches ~0.8501.  That is a real gain over the
0.8484 LGBM-only ceiling.

Selection uses AUC plus a stratified public-proxy bootstrap.  The bootstrap is
not a claim about the true public score; it only asks whether a candidate beats
the v66_04 anchor on 30k-row subsets of the OOF distribution.  The anchor chain
is still force-kept, because it twice produced the best public result.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert (ROOT / "agas/shared/paths.py").exists(), f"wrong root: {ROOT}"
sys.path.insert(0, str(ROOT / "agas/shared"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = HERE.parent / "output"
SIGMA_OOF, SIGMA_PUB = 0.0036, 0.0053
TAU = 0.9995
FLOOR = 0.0020


def rank01(v):
    return rankdata(v) / len(v)


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in z.files:
        if k.startswith("oof_") and ("test_" + k[4:]) in z.files:
            out[k[4:]] = (z[k], z["test_" + k[4:]])
    return out, z


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def bootstrap_proxy(y, anchor, cand, n_boot=600, n_sub=30000, seed=2026):
    """Paired stratified bootstrap of KS(cand) - KS(anchor) on public-size subsets."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    # keep the OOF prevalence; the real public prevalence is unknown, so this
    # measures rank separation rather than absolute LB level
    n_pos = int(round(n_sub * y.mean()))
    n_neg = n_sub - n_pos
    dks, dauc = [], []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, n_pos, replace=False),
            rng.choice(neg, n_neg, replace=False),
        ])
        yy = y[idx]
        dks.append(ks(yy, cand[idx]) - ks(yy, anchor[idx]))
        dauc.append(roc_auc_score(yy, cand[idx]) - roc_auc_score(yy, anchor[idx]))
    dks = np.asarray(dks)
    dauc = np.asarray(dauc)
    return {
        "mean_dks": float(dks.mean()),
        "sd_dks": float(dks.std()),
        "p_ks_better": float((dks > 0).mean()),
        "mean_dauc": float(dauc.mean()),
        "sd_dauc": float(dauc.std()),
        "p_auc_better": float((dauc > 0).mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    legs66, z66 = load_npz(ROOT / "agas/v66/output/preds_v66.npz")
    legs70, z70 = load_npz(ROOT / "agas/v70/output/preds_v70.npz")
    legsmlp, zmlp = load_npz(OUT / "preds_mlp_v71.npz")
    legscat, zcat = load_npz(OUT / "preds_cat_v71.npz")
    legswide, zwide = (load_npz(OUT / "preds_mlp_wide_v71.npz")
                       if (OUT / "preds_mlp_wide_v71.npz").exists() else ({}, None))
    legsxgb, zxgb = (load_npz(OUT / "preds_xgb_v71.npz")
                     if (OUT / "preds_xgb_v71.npz").exists() else ({}, None))
    train_ids, test_ids = z70["train_ids"], z70["test_ids"]
    y = z70["y"].astype(int)
    assert np.array_equal(zmlp["train_ids"], train_ids)
    assert np.array_equal(zcat["train_ids"], train_ids)
    assert np.array_equal(z66["train_ids"], train_ids)
    assert np.array_equal(zmlp["test_ids"], test_ids)
    assert np.array_equal(zcat["test_ids"], test_ids)
    assert np.array_equal(z66["test_ids"], test_ids)
    if zwide is not None:
        assert np.array_equal(zwide["train_ids"], train_ids)
        assert np.array_equal(zwide["test_ids"], test_ids)
    if zxgb is not None:
        assert np.array_equal(zxgb["train_ids"], train_ids)
        assert np.array_equal(zxgb["test_ids"], test_ids)

    R = {}
    for k, (o, t) in legs66.items():
        R["v66_" + k] = (rank01(o), rank01(t))
    for k, (o, t) in legs70.items():
        R["v70_" + k] = (rank01(o), rank01(t))
    for k, (o, t) in legsmlp.items():
        R["mlp_" + k] = (rank01(o), rank01(t))
    for k, (o, t) in legscat.items():
        R["cat_" + k] = (rank01(o), rank01(t))
    for k, (o, t) in legswide.items():
        R["mlpw_" + k] = (rank01(o), rank01(t))
    for k, (o, t) in legsxgb.items():
        R["xgb_" + k] = (rank01(o), rank01(t))

    print("legs:")
    for k, (o, _) in sorted(R.items(), key=lambda kv: -roc_auc_score(y, kv[1][0])):
        print(f"  {k:24s} auc={roc_auc_score(y, o):.6f} ks={ks(y, o):.6f}")

    def avg_seeds(prefix):
        ks_ = [k for k in R if k.startswith(prefix)]
        return (np.mean([R[k][0] for k in ks_], axis=0),
                np.mean([R[k][1] for k in ks_], axis=0))

    mlp_base_o, mlp_base_t = avg_seeds("mlp_")
    wide_o = wide_t = None
    if any(k.startswith("mlpw_") for k in R):
        wide_o, wide_t = avg_seeds("mlpw_")
    cat_o, cat_t = avg_seeds("cat_")
    v70_o = np.mean([R["v70_extra"][0], R["v70_stiff7"][0]], axis=0)
    v70_t = np.mean([R["v70_extra"][1], R["v70_stiff7"][1]], axis=0)
    linear_o, linear_t = R["v66_linear"]
    if any(k.startswith("xgb_") for k in R):
        xgb_seeds = [k for k in R if k.startswith("xgb_")]
        xgb_o = np.mean([R[k][0] for k in xgb_seeds], axis=0)
        xgb_t = np.mean([R[k][1] for k in xgb_seeds], axis=0)
    else:
        xgb_o = xgb_t = None

    # v66 anchor, reconstructed and checked against the CSV
    trees4_o = np.mean([R["v66_lgb_full"][0], R["v66_lgb_lean"][0],
                        R["v66_cat_full"][0], R["v66_xgb_full"][0]], axis=0)
    trees4_t = np.mean([R["v66_lgb_full"][1], R["v66_lgb_lean"][1],
                        R["v66_cat_full"][1], R["v66_xgb_full"][1]], axis=0)
    core66_o = (2 * trees4_o + R["v66_lgbx_full"][0]) / 3
    core66_t = (2 * trees4_t + R["v66_lgbx_full"][1]) / 3
    tm30_o = 0.7 * core66_o + 0.3 * linear_o
    tm30_t = 0.7 * core66_t + 0.3 * linear_t
    anchor_o = rank01(tm30_o + 0.5 * (tm30_o - R["v66_lgb_lean"][0]))
    anchor_t = rank01(tm30_t + 0.5 * (tm30_t - R["v66_lgb_lean"][1]))
    anchor_csv = pd.read_csv(ROOT / "agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv")
    anchor_csv = anchor_csv.set_index("application_id").loc[test_ids].default_probability.values
    print(f"anchor reconstruction spearman vs CSV: {spearmanr(anchor_t, anchor_csv).statistic:.6f}")

    # Weight search for the new core.  The wide MLP is a separate optional leg:
    # its 15-fold OOF was worse than the original MLP, so it is allowed to take
    # only a small fraction of the MLP weight.  AUC is primary, KS is a
    # tie-break.
    wgrid = np.arange(0.0, 0.66, 0.05)
    wide_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] if wide_o is not None else [0.0]
    best = None
    for ww in wide_fracs:
        mix_o = ((1 - ww) * mlp_base_o + ww * wide_o if wide_o is not None else mlp_base_o)
        for wm in wgrid:
            for wc in wgrid:
                for wx in (wgrid if xgb_o is not None else [0.0]):
                    wv = 1.0 - wm - wc - wx
                    if wv < 0.15:
                        continue
                    o = (wm * mix_o + wc * cat_o + wv * v70_o
                         + (wx * xgb_o if xgb_o is not None else 0))
                    auc = roc_auc_score(y, o)
                    kv = ks(y, o)
                    score = auc + 0.10 * kv
                    if best is None or score > best[0]:
                        best = (score, ww, wm, wc, wx, wv, auc, kv)
    _, ww, wm, wc, wx, wv, core_auc, core_ks = best
    mix_o = (1 - ww) * mlp_base_o + ww * wide_o if wide_o is not None else mlp_base_o
    mix_t = (1 - ww) * mlp_base_t + ww * wide_t if wide_o is not None else mlp_base_t
    print(f"best core: mlp={wm:.2f} (wide_frac={ww:.2f}) cat={wc:.2f} "
          f"xgb={wx:.2f} v70={wv:.2f} auc={core_auc:.6f} ks={core_ks:.6f}")
    core_o = wm * mix_o + wc * cat_o + wv * v70_o + (wx * xgb_o if xgb_o is not None else 0)
    core_t = wm * mix_t + wc * cat_t + wv * v70_t + (wx * xgb_t if xgb_t is not None else 0)

    # one KS-oriented vector, but only among weights whose AUC is close to the
    # AUC-optimal core.  This is a separate candidate, not the main selection.
    ks_best = None
    for ww in wide_fracs:
        mix2_o = ((1 - ww) * mlp_base_o + ww * wide_o if wide_o is not None else mlp_base_o)
        for wm2 in wgrid:
            for wc2 in wgrid:
                for wx2 in (wgrid if xgb_o is not None else [0.0]):
                    wv2 = 1.0 - wm2 - wc2 - wx2
                    if wv2 < 0.15:
                        continue
                    o2 = (wm2 * mix2_o + wc2 * cat_o + wv2 * v70_o
                          + (wx2 * xgb_o if xgb_o is not None else 0))
                    auc2 = roc_auc_score(y, o2)
                    if auc2 < core_auc - 0.0005:
                        continue
                    kv2 = ks(y, o2)
                    if ks_best is None or kv2 > ks_best[0]:
                        ks_best = (kv2, ww, wm2, wc2, wx2, wv2, auc2)
    _, ww_ks, wm_ks, wc_ks, wx_ks, wv_ks, auc_ks = ks_best
    mix_ks_o = ((1 - ww_ks) * mlp_base_o + ww_ks * wide_o
                if wide_o is not None else mlp_base_o)
    mix_ks_t = ((1 - ww_ks) * mlp_base_t + ww_ks * wide_t
                if wide_o is not None else mlp_base_t)
    core_ks_o = (wm_ks * mix_ks_o + wc_ks * cat_o + wv_ks * v70_o
                 + (wx_ks * xgb_o if xgb_o is not None else 0))
    core_ks_t = (wm_ks * mix_ks_t + wc_ks * cat_t + wv_ks * v70_t
                 + (wx_ks * xgb_t if xgb_t is not None else 0))
    print(f"best KS-constrained: mlp={wm_ks:.2f} (wide_frac={ww_ks:.2f}) "
          f"cat={wc_ks:.2f} xgb={wx_ks:.2f} v70={wv_ks:.2f} auc={auc_ks:.6f} "
          f"ks={ks(y, core_ks_o):.6f}")

    cands = {}
    cands["leg|mlp_base"] = (mlp_base_o, mlp_base_t)
    cands["leg|mlp_mix"] = (mix_o, mix_t)
    if wide_o is not None:
        cands["leg|mlp_wide"] = (wide_o, wide_t)
    cands["leg|cat_avg"] = (cat_o, cat_t)
    cands["leg|v70_avg"] = (v70_o, v70_t)
    if xgb_o is not None:
        cands["leg|xgb_avg"] = (xgb_o, xgb_t)
    cands["anchor_v66_04"] = (anchor_o, anchor_t)
    cands["core71"] = (core_o, core_t)
    cands["core71_ks"] = (rank01(core_ks_o), rank01(core_ks_t))
    # keep the proven tm30/extrapolation grammar, now with the new core
    tm30n_o = 0.7 * core_o + 0.3 * linear_o
    tm30n_t = 0.7 * core_t + 0.3 * linear_t
    cands["tm30_71"] = (tm30n_o, tm30n_t)
    for r in (0.25, 0.5, 0.75, 1.0):
        eo = rank01(tm30n_o + r * (tm30n_o - R["v66_lgb_lean"][0]))
        et = rank01(tm30n_t + r * (tm30n_t - R["v66_lgb_lean"][1]))
        cands[f"ext{r}|tm30_71|lgb_lean"] = (eo, et)
    acc_o = np.zeros(len(y))
    acc_t = np.zeros(len(test_ids))
    for r in (0.25, 0.5, 0.75):
        eo = rank01(tm30n_o + r * (tm30n_o - R["v66_lgb_lean"][0]))
        et = rank01(tm30n_t + r * (tm30n_t - R["v66_lgb_lean"][1]))
        acc_o += eo
        acc_t += et
    cands["plateau|tm30_71|lgb_lean"] = (rank01(acc_o), rank01(acc_t))
    # conservative blends toward the public anchor
    for w in (0.25, 0.50, 0.75):
        cands[f"anchor{w:.2f}|core71"] = (rank01((1 - w) * anchor_o + w * core_o),
                                          rank01((1 - w) * anchor_t + w * core_t))
    cands["anchor0.50|mlp_base"] = (rank01(0.5 * anchor_o + 0.5 * mlp_base_o),
                                    rank01(0.5 * anchor_t + 0.5 * mlp_base_t))
    cands["anchor0.50|cat_avg"] = (rank01(0.5 * anchor_o + 0.5 * cat_o),
                                   rank01(0.5 * anchor_t + 0.5 * cat_t))
    # The panel score is deliberately not a blend leg.  It added no AUC in the
    # 5-fold screen (0.847202 vs 0.847219) and a 1% blend only moved OOF KS by
    # noise.  Keeping it here would make the candidate list look more diverse
    # than the evidence supports.

    # logistic stack on rank legs; cross-fitted, so its OOF is honest
    stack_keys = [k for k in R if k.startswith(("v66_", "v70_", "mlp_", "cat_"))]
    # aggregate fold seeds first to avoid giving one family more votes
    stack_legs = {
        "v66_trees4": (trees4_o, trees4_t),
        "v66_lgbx": R["v66_lgbx_full"],
        "v66_linear": (linear_o, linear_t),
        "v70_extra": R["v70_extra"],
        "v70_stiff7": R["v70_stiff7"],
        "mlp_base": (mlp_base_o, mlp_base_t),
        "cat_avg": (cat_o, cat_t),
    }
    if wide_o is not None:
        stack_legs["mlp_wide"] = (wide_o, wide_t)
    if xgb_o is not None:
        stack_legs["xgb_avg"] = (xgb_o, xgb_t)
    keys = list(stack_legs)
    Xs = np.column_stack([stack_legs[k][0] for k in keys])
    Xt = np.column_stack([stack_legs[k][1] for k in keys])
    stack_o = np.zeros(len(y))
    stack_t = np.zeros(len(test_ids))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=2026).split(Xs, y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        lr.fit(Xs[a], y[a])
        stack_o[b] = lr.predict_proba(Xs[b])[:, 1]
        stack_t += lr.predict_proba(Xt)[:, 1] / 5
    cands["stack_lr"] = (rank01(stack_o), rank01(stack_t))

    scored = []
    for name, (o, t) in cands.items():
        scored.append({"name": name, "oof": o, "test": t,
                       "auc": float(roc_auc_score(y, o)),
                       "ks": ks(y, o)})
    scored.sort(key=lambda r: -r["auc"])
    print("\npool:")
    for r in scored:
        print(f"  {r['name']:36s} auc={r['auc']:.6f} ks={r['ks']:.6f}")

    print("\nbootstrap public proxy (600 x 30k, paired vs anchor):")
    for r in scored:
        r["proxy"] = bootstrap_proxy(y, anchor_o, r["oof"])
        p = r["proxy"]
        print(f"  {r['name']:36s} dKS={p['mean_dks']:+.6f} "
              f"P(ks better)={p['p_ks_better']:.3f} dAUC={p['mean_dauc']:+.6f}")

    best_auc = max(r["auc"] for r in scored)
    eligible = [r for r in scored if r["auc"] >= best_auc - FLOOR]
    force = [
        "anchor_v66_04",
        "core71",
        "core71_ks",
        "stack_lr",
        "leg|mlp_base",
        "anchor0.50|core71",
        "ext0.5|tm30_71|lgb_lean",
    ]
    kept = []
    for nm in force:
        rec = next(r for r in scored if r["name"] == nm)
        if rec not in kept:
            kept.append(rec)
    # then pick diverse, bootstrap-positive candidates
    eligible.sort(key=lambda r: (-r["proxy"]["mean_dks"], -r["auc"]))
    for r in eligible:
        if r in kept:
            continue
        if r["proxy"]["p_ks_better"] < 0.55:
            continue
        if all(spearmanr(r["test"], k["test"]).statistic < TAU for k in kept):
            kept.append(r)
    kept.sort(key=lambda r: -r["auc"])

    for f in OUT.glob("submission_v71_*.csv"):
        f.unlink()
    rows = []
    for i, k in enumerate(kept, 1):
        d = pd.DataFrame({"application_id": test_ids,
                          "default_probability": rank01(k["test"])})
        p = OUT / f"submission_v71_{i:02d}_{safe_name(k['name'])}.csv"
        d.to_csv(p, index=False)
        ok = (len(d) == 30000 and d.application_id.is_unique
              and d.application_id.isin(test_ids).all()
              and not d.default_probability.isna().any()
              and d.default_probability.between(0, 1).all())
        rows.append({
            "slot": i, "name": k["name"], "auc": round(k["auc"], 6),
            "ks": round(k["ks"], 6),
            "ks_minus_sigma": round(k["ks"] - SIGMA_OOF, 6),
            "proxy": {kk: round(vv, 6) if isinstance(vv, float) else vv
                      for kk, vv in k["proxy"].items()},
            "spearman_v66_04": round(float(spearmanr(k["test"], anchor_t).statistic), 6),
            "file": p.name, "valid": bool(ok),
        })

    print(f"\n=== kept {len(rows)} candidates ===")
    for r in rows:
        print(f"{r['slot']:02d} {r['name']:36s} auc={r['auc']:.6f} "
              f"ks={r['ks']:.6f} Pks={r['proxy']['p_ks_better']:.3f} "
              f"rho={r['spearman_v66_04']:.5f}")
    summary = {
        "sigma_oof": SIGMA_OOF, "sigma_public": SIGMA_PUB,
        "dedupe_threshold": TAU, "auc_floor_gap": FLOOR,
        "best_auc": best_auc,
        "core_weights": {"mlp_mix": float(wm), "mlp_base": float(wm * (1 - ww)),
                          "mlp_wide": float(wm * ww), "cat": float(wc),
                          "xgb": float(wx), "v70": float(wv)},
        "core_ks_weights": {"mlp_mix": float(wm_ks),
                             "mlp_base": float(wm_ks * (1 - ww_ks)),
                             "mlp_wide": float(wm_ks * ww_ks),
                             "cat": float(wc_ks), "xgb": float(wx_ks),
                             "v70": float(wv_ks)},
        "core_ks_auc": float(auc_ks),
        "core_auc": core_auc, "core_ks": core_ks,
        "pool_size": len(scored), "eligible": len(eligible),
        "kept": len(kept), "bootstrap": "600 x 30000 stratified OOF subsets",
        "candidates": rows,
    }
    (OUT / "summary_v71_candidates.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'summary_v71_candidates.json'}")


if __name__ == "__main__":
    main()
