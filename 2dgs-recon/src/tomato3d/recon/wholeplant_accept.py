"""A4 acceptance, exactly as declared in CONTRACT.md: the first run in order R0..R4 where every clause holds on the NEW
held-out set. A clause holds when the MEDIAN over the 4 views meets the threshold AND no single view is more than 0.05
below it ("no single view carries the median"). The gap clause has no per-view guard (it is already an aggregate).

usage: python -m tomato3d.recon.wholeplant_accept --out <run dir> --k <kernel out dir>
Writes <run>/acceptance.json and prints the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tomato3d.recon.wholeplant_contract import A4, RUNS

GUARD = 0.05
CLAUSES = (("iou_no_band", A4["band_excluded_iou_min"], "band-excluded IoU"),
           ("iou", A4["raw_iou_min"], "raw IoU"),
           ("edge_f2", A4["edge_f2_min"], "edge F at 2 px"),
           ("stem_coverage", A4["stem_coverage_min"], "stem coverage"))


def judge(s: dict) -> dict:
    h = s["heldout_new"]; per = h["per_view"]; out = {}
    for key, thr, name in CLAUSES:
        med = h[key]; vals = {v: m[key] for v, m in per.items()}
        if med is None:
            out[key] = dict(name=name, threshold=thr, median=None, passes=False, reason="not measured in this run")
            continue
        low = {v: x for v, x in vals.items() if x is not None and x < thr - GUARD}
        out[key] = dict(name=name, threshold=thr, median=med, median_ok=med >= thr,
                        per_view_floor=round(thr - GUARD, 4), views_below_floor={v[9:18]: round(x, 4) for v, x in low.items()},
                        passes=bool(med >= thr and not low))
    g = s.get("gap_iou")
    out["gap_iou"] = dict(name="held-out minus training IoU", threshold=A4["gap_min"], median=g, passes=bool(g is not None and g >= A4["gap_min"]))
    return dict(clauses=out, accepted=all(c["passes"] for c in out.values()))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--k", required=True)
    a = ap.parse_args(argv); rc = Path(a.out)
    runs = json.load(open(Path(a.k) / "results.json"))["runs"]
    order = [r[0] for r in RUNS]
    res, accepted = {}, None
    for rid in order:
        if rid not in runs:
            continue
        j = judge(runs[rid]); res[rid] = j
        if j["accepted"] and accepted is None:
            accepted = rid
    rep = dict(rule="first run in R0..R4 order where every clause holds; median over the 4 new held-out views and no view "
                    f"more than {GUARD} below the threshold", thresholds=A4, guard=GUARD, accepted_run=accepted, runs=res,
               not_run=dict(R1="A1 board refinement was rejected (corner RMS 5.16 px > 0.8); the refinement moved cameras by at "
                               "most 0.38 mm / 0.011 deg, so R1 would be R0",
                            R4="only if the GPU budget allows after R0-R3"))
    json.dump(rep, open(rc / "acceptance.json", "w"), indent=1)
    for rid, j in res.items():
        print(f"{rid}: {'ACCEPTED' if j['accepted'] else 'not accepted'}")
        for k, c in j["clauses"].items():
            if c["median"] is None:
                print(f"   {c['name']:<28} not measured -> fails"); continue
            mark = "pass" if c["passes"] else "FAIL"
            extra = f"  views below {c['per_view_floor']}: {c['views_below_floor']}" if c.get("views_below_floor") else ""
            print(f"   {c['name']:<28} median {c['median']:.3f} vs {c['threshold']:<5} {mark}{extra}")
    print("\naccepted run:", accepted)


if __name__ == "__main__":
    main()
