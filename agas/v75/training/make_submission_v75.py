"""Fase C: build v75 candidates and price each one's chance of reaching 0.57.

The target is public 0.57 from an anchor of 0.56424, so +0.00576. Two ways to
get there, and the measurements say which is available.

**A true quality gain of +0.00576 KS.** Eighteen versions span 0.008 OOF KS in
total, the fase A learning curve says removing all remaining variance is worth
about +0.001 to +0.002 AUC, and v74 showed OOF gains of +0.004 arriving at
public as -0.003. Nothing supports a reliable +0.006.

**A favourable public draw.** This was the design this script started from, on
the model that a candidate's public score is the anchor's plus the part of the
sampling noise it does not share, `sd_differential = sd_public * sqrt(2(1-rho))`,
so buying decorrelation buys a chance. That model is **falsified** by the three
public results that can be mapped exactly onto candidate classes:

| candidate | rho | sd predicted | predicted | actual | residual |
|---|---:|---:|---:|---:|---:|
| v72 seq05 | 0.99950 | 0.00063 | 0.56424 | 0.55968 | -7.2 sd |
| v70_01 raw leg | 0.98988 | 0.00285 | 0.56403 | 0.55801 | -2.1 sd |
| v71 core71 | 0.96500 | 0.00530 | 0.56424 | 0.56111 | -0.6 sd |

All three are below prediction, and the residual in sigma units gets *worse* as
rho gets *higher* - the opposite of what a shared-noise model implies. The
actual deviations are all about -0.003 to -0.006 and barely depend on rho.

The explanation is selection, not noise shape. 0.56424 is not a draw from this
family, it is the **maximum** of the draws already taken, so it carries the
positive noise of that maximum. The mean of the family is the mean of the three
measured non-anchor results, about **0.5596**. A new candidate's expected public
is that mean, not the anchor's score, and 0.57 is three to five sigma above it.

So this script keeps writing a decorrelated portfolio, because that is still
the only lever, but it prices it against the measured family mean instead of
against the anchor. The resulting probabilities are small, and they are the
honest ones.

The expected true difference is scored **asymmetrically**: a candidate is
credited with none of its OOF KS gain and charged all of its OOF KS loss.

    E[public] = anchor_public + min(0, d_ks_oof)

Both halves of that rule come from evidence. Gains are not credited because
every measured OOF gain so far arrived at public as a loss - v71 gained +0.0041
KS out of fold and lost 0.0031 on public, v70_06 gained +0.0015 and lost
0.0038. Losses are charged in full because the no-transfer finding was measured
over differences of 0.004 or less, and it would be indefensible to extend it to
a candidate sitting 0.015 AUC below the anchor: a model that much worse is
worse everywhere. An earlier version of this script applied zero transfer in
both directions and overstated the chance of reaching 0.57 for the
sequence-heavy tickets by more than an order of magnitude.

The same quality deficit is also charged to private, which is why a degraded
ticket cannot win on total score even when its public draw is lucky: the 0.45
public weight gains at most the lucky draw while the 0.45 private weight pays
the full deficit.

This is a free option **only because the final Kaggle pick is made after the
public scores are visible**: a ticket that lands below the anchor is simply not
selected, while private is an independent draw of roughly equal true quality.
The downside is truncated; that is what makes buying variance rational here.

A candidate is only worth submitting if its true quality is not worse, so every
ticket must still clear a quality floor. Diversity is bought from legs measured
to be as good as the ones they replace, never from weaker models - that is the
mistake v67-v73 made five times.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGAS = ROOT / "agas"
OUT = HERE.parent / "output"

N_PUBLIC, BOOT = 3000, 1000
PUBLIC_ANCHOR = 0.56424
TARGET_PUBLIC = 0.57
QUALITY_FLOOR = -0.0010      # a ticket may not be worse than this in OOF AUC

# public results of candidates that can be mapped exactly onto a candidate
# class here: (name, spearman vs anchor, OOF KS delta, actual public)
CALIBRATION = [
    ("v72_seq05", 0.99950, +0.000131, 0.55968),
    ("v70_01_leg_extra", 0.98988, -0.000210, 0.55801),
    ("v71_core71", 0.96500, +0.004121, 0.56111),
]
# gains are not credited, losses are charged in full: see the module docstring
GAIN_TRANSFER, LOSS_TRANSFER = 0.0, 1.0


def rank01(v):
    return rankdata(v) / len(v)


def ks(score, y):
    o = np.argsort(score, kind="mergesort")
    yy = y[o]
    return float(np.abs(np.cumsum(yy) / yy.sum()
                        - np.cumsum(1 - yy) / (1 - yy).sum()).max())


def main():
    z66 = np.load(AGAS / "v66/output/preds_v66.npz", allow_pickle=True)
    tr_ids, te_ids = pd.Index(z66["train_ids"]), pd.Index(z66["test_ids"])
    o = {k[4:]: rank01(z66[k]) for k in z66.files if k.startswith("oof_")}
    t = {k[4:]: rank01(z66["test_" + k[4:]]) for k in z66.files if k.startswith("oof_")}

    new = []
    p75 = OUT / "preds_v75.npz"
    if p75.exists():
        z75 = np.load(p75, allow_pickle=True)
        si = pd.Index(z75["train_ids"]).get_indexer(tr_ids)
        ti = pd.Index(z75["test_ids"]).get_indexer(te_ids)
        assert (si >= 0).all() and (ti >= 0).all(), "id mismatch v66 vs v75"
        for k in [k[4:] for k in z75.files if k.startswith("oof_")]:
            o["v75_" + k] = rank01(z75["oof_" + k][si])
            t["v75_" + k] = rank01(z75["test_" + k][ti])
            new.append("v75_" + k)
    # legs from earlier versions that are already aligned and measured
    for name, path, ok, tk, seeds in [
        ("v70_extra", "v70/output/preds_v70.npz", "oof_extra", "test_extra", None),
        ("v70_stiff7", "v70/output/preds_v70.npz", "oof_stiff7", "test_stiff7", None),
        ("mlp", "v71/output/preds_mlp_v71.npz", "oof_%d", "test_%d", (42, 7, 13)),
        ("cat71", "v71/output/preds_cat_v71.npz", "oof_%d", "test_%d", (42, 7, 13)),
        ("seq", "v72/output/preds_seq_v72_pure.npz", "oof_42", "test_42", None)]:
        z = np.load(AGAS / path, allow_pickle=True)
        si = pd.Index(z["train_ids"]).get_indexer(tr_ids)
        ti = pd.Index(z["test_ids"]).get_indexer(te_ids)
        ov = (np.mean([z[ok % s] for s in seeds], axis=0) if seeds else z[ok])
        tv = (np.mean([z[tk % s] for s in seeds], axis=0) if seeds else z[tk])
        o[name] = rank01(ov[si])
        t[name] = rank01(tv[ti])
        new.append(name)

    y = (pd.read_csv(ROOT / "data/train_applications.csv",
                     usecols=["application_id", "target"])
         .set_index("application_id").loc[tr_ids].target.values.astype(int))

    def chain(d, lean_key="lgb_lean", extra=None, ref=None):
        members = [d["lgb_full"], d[lean_key], d["cat_full"], d["xgb_full"]]
        if extra is not None:
            members.append(d[extra])
        trees = sum(members) / len(members)
        core = (trees * 2 + d["lgbx_full"]) / 3
        tm30 = 0.7 * core + 0.3 * d["linear"]
        return rank01(tm30 + 0.5 * (tm30 - d[ref or lean_key]))

    a_o, a_t = chain(o), chain(t)
    a_auc = float(roc_auc_score(y, a_o))
    print(f"anchor auc={a_auc:.6f} ks={ks(a_o, y):.6f}\n")
    print("available legs, bagged out of fold")
    for k in new:
        print(f"  {k:14s} auc={roc_auc_score(y, o[k]):.6f} ks={ks(o[k], y):.6f} "
              f"rho_vs_lgb_lean={spearmanr(t[k], t['lgb_lean']).statistic:.5f}")

    cands = {"anchor": (a_o, a_t)}
    # the raw legs themselves: the most decorrelated thing available that can
    # still clear the quality floor, since a leg is not a blend of the anchor
    for k in new:
        cands[f"leg_{k}"] = (o[k], t[k])
    for k in new:
        cands[f"swap_{k}"] = (chain(o, lean_key=k), chain(t, lean_key=k))
        cands[f"add_{k}"] = (chain(o, extra=k), chain(t, extra=k))
        for w in (0.20, 0.35, 0.50, 0.65, 0.80):
            cands[f"mix{int(w*100)}_{k}"] = (rank01((1 - w) * a_o + w * o[k]),
                                             rank01((1 - w) * a_t + w * t[k]))
    # Deliberately decorrelated multi-leg tickets. Under the asymmetric transfer
    # rule, E[public] is capped at the anchor's score, so the only way to raise
    # the chance of reaching the target is to raise sd_differential while
    # keeping the OOF KS difference at or above zero. These push the weight on
    # the non-anchor legs much further than a 50/50 mix does, because a 50/50
    # mix against a seven-leg blend still sits at rho 0.994.
    strong = sorted(new, key=lambda k: -roc_auc_score(y, o[k]))[:3]
    tag = "_".join(k.replace("v75_", "") for k in strong)
    for wa in (0.40, 0.25, 0.10):
        wl = (1 - wa) / len(strong)
        cands[f"wide{int(wa*100)}_{tag}"] = (
            rank01(wa * a_o + wl * sum(o[k] for k in strong)),
            rank01(wa * a_t + wl * sum(t[k] for k in strong)))
    # the two strongest decorrelated legs on their own, equally weighted
    if len(strong) >= 2:
        cands[f"pair_{strong[0]}_{strong[1]}".replace("v75_", "")] = (
            rank01(0.5 * o[strong[0]] + 0.5 * o[strong[1]]),
            rank01(0.5 * t[strong[0]] + 0.5 * t[strong[1]]))

    rng = np.random.default_rng(11)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    npos = int(round(N_PUBLIC * y.mean()))
    idxs = [np.concatenate([rng.choice(pos, npos, replace=False),
                            rng.choice(neg, N_PUBLIC - npos, replace=False)])
            for _ in range(BOOT)]
    a_boot = np.array([ks(a_o[i], y[i]) for i in idxs])
    sd_pub = float(a_boot.std(ddof=1))

    actuals = np.array([c[3] for c in CALIBRATION])
    FAMILY_MEAN = float(actuals.mean())
    # n = 3, so the observed spread is itself uncertain; it is floored at a
    # value wide enough not to pretend the three points pin it down
    FAMILY_SD = max(float(actuals.std(ddof=1)), 0.0040)
    print(f"calibration on {len(CALIBRATION)} measured public results: "
          f"family mean {FAMILY_MEAN:.5f}, spread {FAMILY_SD:.5f} "
          f"(anchor {PUBLIC_ANCHOR:.5f} is the maximum of the draws taken)")

    rng2 = np.random.default_rng(3)
    p2, n2 = pos.copy(), neg.copy()
    rng2.shuffle(p2)
    rng2.shuffle(n2)
    halves = [np.concatenate([p2[::2], n2[::2]]), np.concatenate([p2[1::2], n2[1::2]])]

    rows = []
    for name, (vo, vt) in cands.items():
        auc = float(roc_auc_score(y, vo))
        rho = float(spearmanr(vt, a_t).statistic)
        # the shared-noise formula underestimated every measured case, so the
        # spread used is the wider of it and the observed family spread
        sd_d = max(sd_pub * float(np.sqrt(max(2 * (1 - rho), 0.0))), FAMILY_SD)
        d_ks_boot = np.array([ks(vo[i], y[i]) for i in idxs]) - a_boot
        half = [ks(rank01(vo[h]), y[h]) - ks(rank01(a_o[h]), y[h]) for h in halves]
        d_ks = ks(vo, y) - ks(a_o, y)
        # priced against the measured family mean, not against the anchor
        mu = (PUBLIC_ANCHOR if name == "anchor" else FAMILY_MEAN) + (
            (GAIN_TRANSFER if d_ks > 0 else LOSS_TRANSFER) * d_ks)
        if name == "anchor":
            # the anchor's public score is measured, not drawn: resubmitting it
            # returns 0.56424 exactly, so it carries no probability at all
            p57 = float(PUBLIC_ANCHOR >= TARGET_PUBLIC)
        else:
            p57 = (0.0 if sd_d <= 0
                   else float(norm.sf((TARGET_PUBLIC - mu) / sd_d)))
        rows.append({
            "name": name, "oof_auc": round(auc, 6), "oof_ks": round(ks(vo, y), 6),
            "d_auc": round(auc - a_auc, 6), "d_ks": round(d_ks, 6),
            "expected_public": round(mu, 5),
            "half0_d_ks": round(half[0], 6), "half1_d_ks": round(half[1], 6),
            "spearman_vs_anchor": round(rho, 6), "sd_differential": round(sd_d, 5),
            "boot_p_beat_anchor": round(float((d_ks_boot > 0).mean()), 3),
            "p_reach_0_57": round(p57, 4),
            "quality_ok": bool(auc - a_auc >= QUALITY_FLOOR)})

    # tickets: pass the quality floor, ranked by the chance of reaching 0.57
    tickets = [r for r in rows if r["name"] != "anchor" and r["quality_ok"]]
    # equal expected public is the common case once gains are not credited, so
    # the tie is broken towards the ticket that carries the most fresh variance
    # once gains are not credited, every candidate with d_ks >= 0 is priced
    # identically, so the remaining tie-breaks are the only evidence left:
    # first the best out-of-fold KS, then the most fresh variance
    tickets.sort(key=lambda r: (-r["p_reach_0_57"], -r["d_ks"], -r["sd_differential"]))

    print(f"\npublic noise sd at n={N_PUBLIC}: {sd_pub:.5f}")
    print(f"target {TARGET_PUBLIC} is {(TARGET_PUBLIC - PUBLIC_ANCHOR):.5f} above the anchor\n")
    print(f"{'name':26s} {'auc':>9} {'dAUC':>9} {'dKS':>9} {'rho':>8} "
          f"{'sd_dif':>7} {'E[pub]':>8} {'P>=0.57':>8} {'ok':>3}")
    for r in sorted(rows, key=lambda r: (-r["p_reach_0_57"], -r["d_ks"],
                                        -r["sd_differential"])):
        print(f"{r['name']:26s} {r['oof_auc']:9.6f} {r['d_auc']:+9.6f} "
              f"{r['d_ks']:+9.6f} {r['spearman_vs_anchor']:8.5f} "
              f"{r['sd_differential']:7.5f} {r['expected_public']:8.5f} "
              f"{r['p_reach_0_57']:8.4f} {'y' if r['quality_ok'] else 'n':>3}")

    # portfolio: chance at least one ticket lands at or above the target.
    # independent independent-parts is an upper bound, since tickets are also
    # correlated with each other, not just with the anchor.
    print("\nchance at least one ticket reaches 0.57 (independence upper bound):")
    port = {}
    for k in (1, 2, 3, 4, 5):
        sel = tickets[:k]
        if not sel:
            continue
        pr = 1.0 - float(np.prod([1 - r["p_reach_0_57"] for r in sel]))
        port[k] = round(pr, 4)
        print(f"  best {k} ticket(s): {pr:.3f}   ({', '.join(r['name'] for r in sel)})")

    for f in sorted(OUT.glob("submission_v75_*.csv")):
        f.unlink()
    order = [r for r in rows if r["name"] == "anchor"] + tickets[:5]
    for i, r in enumerate(order, 1):
        vt = cands[r["name"]][1]
        p = OUT / f"submission_v75_{i:02d}_{r['name']}.csv"
        frame = pd.DataFrame({"application_id": te_ids.values,
                              "default_probability": rank01(vt)})
        frame.to_csv(p, index=False)
        ok = (len(frame) == 30000 and frame.application_id.is_unique
              and set(frame.application_id) == set(te_ids.values)
              and not frame.default_probability.isna().any()
              and frame.default_probability.between(0, 1).all())
        assert ok, f"invalid csv {p}"
        r["file"] = p.name
        print(f"  wrote {p.name}")

    (OUT / "summary_v75_candidates.json").write_text(json.dumps(
        {"public_anchor": PUBLIC_ANCHOR, "target_public": TARGET_PUBLIC,
         "n_public_assumed": N_PUBLIC, "public_noise_sd": sd_pub,
         "gain_transfer": GAIN_TRANSFER, "loss_transfer": LOSS_TRANSFER,
         "quality_floor_auc": QUALITY_FLOOR,
         "anchor_oof_auc": a_auc, "anchor_oof_ks": ks(a_o, y),
         "calibration": [{"name": c[0], "rho": c[1], "d_ks_oof": c[2],
                          "actual_public": c[3]} for c in CALIBRATION],
         "family_mean_public": FAMILY_MEAN, "family_sd_public": FAMILY_SD,
         "portfolio_p_reach_target": port,
         "submit_order": [r["name"] for r in order],
         "candidates": sorted(rows, key=lambda r: (-r["p_reach_0_57"], -r["d_ks"],
                                                   -r["sd_differential"]))}, indent=2))
    print(f"\nsaved {OUT / 'summary_v75_candidates.json'}")


if __name__ == "__main__":
    main()
