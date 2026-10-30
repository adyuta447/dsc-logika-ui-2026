"""Point the committee notebook at a final pick on the slot-2 axis, run it, verify it.

    second = (1 - A_MIX) * lgbx_full + A_MIX * SECOND_LEG     A_MIX = 0 is v76_02

Default arguments rebuild the current final (v76_02_r0.75) and must change no
result, which is what proves the patch before it is needed. Examples:

    python3 agas/v84/training/switch_notebook_v84.py
    python3 agas/v84/training/switch_notebook_v84.py --leg cat_full --a 1 \
        --csv agas/v83/output/submission_v83_slot2_cat_full.csv --public 0.56600

Every patch asserts it hit exactly once, the executed notebook must print zero
rank changes and a byte-identical CSV, and only then is the root copy replaced
(the previous one is kept in agas/OtherTries/).
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[3]
COMMITTEE = ROOT / "DSC0152_kali ini beneran deh_Notebook.ipynb"
OUT_NB = ROOT / "agas/v84/notebook/notebook_v84_final.ipynb"
BASE = ROOT / "agas/v84/notebook/committee_base_v83.ipynb"  # pristine, so patches stay re-runnable
OLD_FINAL = 'FINAL_CSV = V76 / "output/submission_v76_02_r0.75.csv"'
OLD_ROW = '("r=0.75  ref=lgb_lean tm=0.30 core=2:1   SUBMITTED", 0.56520),'


def patch(nb, i, old, new):
    src = "".join(nb.cells[i].source)
    assert src.count(old) == 1, f"cell {i}: expected one {old!r}"
    nb.cells[i].source = src.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", default="lgbx_full")
    ap.add_argument("--a", type=float, default=0.0)
    ap.add_argument("--csv", default="agas/v76/output/submission_v76_02_r0.75.csv")
    ap.add_argument("--public", type=float, default=None,
                    help="public score of --csv; required when it is not v76_02")
    args = ap.parse_args()
    switching = args.a != 0.0
    assert (ROOT / args.csv).exists(), args.csv
    assert not switching or args.public, "--public is required for a new final"

    nb = nbformat.read(BASE, as_version=4)
    patch(nb, 1, OLD_FINAL,
          f'FINAL_CSV = ROOT / "{args.csv}"\n'
          f'SECOND_LEG = "{args.leg}"   # occupant mixed into the second core slot\n'
          f"A_MIX = {args.a}            # 0 keeps lgbx_full alone (the v76_02 recipe)")
    patch(nb, 11, '    core = (trees4 * cw + d["lgbx_full"]) / (cw + 1.0)',
          '    d = {**d, "catxgb": (d["cat_full"] + d["xgb_full"]) / 2}\n'
          '    second = (1 - A_MIX) * d["lgbx_full"] + A_MIX * d[SECOND_LEG]\n'
          '    core = (trees4 * cw + second) / (cw + 1.0)')
    if switching:
        p = f"{args.public:.5f}"
        patch(nb, 0, "core   = (2 * trees4 + lgbx_full) / 3",
              f"second = {1 - args.a:g} * lgbx_full + {args.a:g} * {args.leg}\n"
              "core   = (2 * trees4 + second) / 3")
        patch(nb, 0, "Public score **0.56520**.", f"Public score **{p}**.")
        patch(nb, 12, "The blend structure comes from v61",
              f"**Update v84 — the second core slot.** `core` gives its second slot "
              f"1/3 of the weight, twice any single `trees4` leg, and from v61 to v83 "
              f"only its weight was tuned, never its occupant. Moving it from "
              f"`lgbx_full` toward `{args.leg}` (`A_MIX = {args.a:g}`) scored "
              f"**{p}** on public against 0.56520 for `A_MIX = 0`; everything below "
              f"is unchanged and still describes the rest of the chain.\n\n"
              "The blend structure comes from v61")
        patch(nb, 13, OLD_ROW,
              f'("r=0.75 slot2 {args.leg} a={args.a:g}  SUBMITTED", {p}),\n'
              '    ("r=0.75  ref=lgb_lean tm=0.30 core=2:1 (v76_02)", 0.56520),')
        patch(nb, 13, "(led.public - 0.56520)", f"(led.public - {p})")
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None

    t0 = time.time()
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(ROOT)}})
    text = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code"
                     for o in c.outputs)
    for must in ("rank positions changed : 0",
                 "byte-identical to the submitted file: True", "STATUS: PASS"):
        assert must in text, f"executed notebook is missing: {must}"
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT_NB)
    backup = ROOT / f"agas/OtherTries/DSC0152_Notebook_before_v84_{time.strftime('%H%M%S')}.ipynb"
    shutil.copy2(COMMITTEE, backup)
    shutil.copy2(OUT_NB, COMMITTEE)
    print(json.dumps({"final_csv": args.csv, "leg": args.leg, "a": args.a,
                      "run_s": round(time.time() - t0, 1), "notebook": str(OUT_NB),
                      "backup": str(backup), "status": "PASS"}, indent=1))


if __name__ == "__main__":
    main()
