"""Gather every human answer about a leaflet mask of the KEPT runs and resolve it to (photo, run, mask id).

Sources (human answers are read, never rewritten):
  scoring page 01 (B = run 02 vs raw M1): B credits only (the raw-M1 masks are not a kept proposal set);
  scoring page 02 (B = run 02, H = H2(M1) run 05);
  scoring page 03 (B = H2(M1) run 05, H = H2(M2a) run 06 - comparison only, not adopted: skipped);
  scoring page 04 (B = H2(M1) run 05, H = run 07 filter - comparison only: skipped);
  M2 dev labelling page (H2(M1) run 05, development photos; key = photo + h2_id);
  M1-improve pages (neg label + re-ask; origin M1 = run 05; origins M2a / M2b not adopted: skipped). The re-ask answer
  replaces the neg answer for the same item (the run's own labels.py rule: re-ask > neg).
Answer keys: 1 whole leaflet, 2 mostly one, 3 merged, 4 fragment, 5 not a leaflet (+ sub-class), 6 can't tell.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RUNS = {"run02": Path("build/p1_leafseg_20261019_02"), "run05": Path("build/p1_leafseg_20261020_05/H2")}
HELD_OUT = ("tray_top_07", "tray_top_08", "tray_top_09", "dwt_104")
PAGES = {
    "scoring_01": dict(dir="build/p1_tray_scoring_20261020_01", methods={"B": "run02"}),
    "scoring_02": dict(dir="build/p1_tray_scoring_20261020_02", methods={"B": "run02", "H": "run05"}),
    "scoring_03": dict(dir="build/p1_tray_scoring_20261020_03", methods={"B": "run05"}),
    "scoring_04": dict(dir="build/p1_tray_scoring_20261021_04", methods={"B": "run05"}),
}
M2_DEV = Path("build/p1_leafinst_m2_20261020_01"); M1_IMPROVE = Path("build/p1_m1_improve_20261021_01")


def _answers(folder: Path, prefix: str) -> dict:
    files = sorted((folder / "human").glob(prefix + "*.json"))
    files = [f for f in files if not f.name.endswith("_summary.json")]
    assert files, f"no {prefix}*.json in {folder / 'human'}"
    return json.load(open(files[-1], encoding="utf-8")), files[-1].name


def _mask_id(run: str, photo: str, index: int) -> str:
    """Stable mask id = the proposal's own id at that index (H001… for run 05, L-ids for run 02)."""
    d = _proposals(run, photo); return d["instances"][index]["id"]


_cache: dict = {}


def _proposals(run: str, photo: str) -> dict:
    key = (run, photo)
    if key not in _cache:
        _cache[key] = json.load(open(RUNS[run] / "proposals" / f"{photo}_leafseg.json", encoding="utf-8"))
    return _cache[key]


def gather() -> list[dict]:
    """One record per (source page, item) that maps to a kept-run mask."""
    out = []
    for page, cfg in PAGES.items():
        d = Path(cfg["dir"]); key = json.load(open(d / "key.json", encoding="utf-8"))["key"]; ans, fname = _answers(d, "answers_")
        for item, k in key.items():
            if k.get("kind") != "mask" or item not in ans:
                continue
            a = ans[item]
            for c in k["credit"]:
                run = cfg["methods"].get(c["method"])
                if run is None:
                    continue
                mid = _mask_id(run, k["photo"], c["index"])
                out.append(dict(page=page, answers_file=fname, item=item, photo=k["photo"], run=run, mask_id=mid, index=c["index"], answer=a.get("answer"),
                                sub=a.get("sub"), occluded_by_human=a.get("occluded"), updated=a.get("updated"), reused_from=a.get("reused_from"),
                                split="heldout_photo" if k["photo"] in HELD_OUT else "development_photo"))
    key = json.load(open(M2_DEV / "key.json", encoding="utf-8")); key = key.get("key", key); ans, fname = _answers(M2_DEV, "dev_answers_")
    for item, k in key.items():
        if k.get("kind") != "mask" or item not in ans:
            continue
        a = ans[item]
        out.append(dict(page="m2_dev", answers_file=fname, item=item, photo=k["photo"], run="run05", mask_id=k["h2_id"], index=None, answer=a.get("answer"),
                        sub=a.get("sub"), occluded_by_human=a.get("occluded"), updated=a.get("updated"), reused_from=None,
                        split="heldout_photo" if k["photo"] in HELD_OUT else "development_photo"))
    neg_key = json.load(open(M1_IMPROVE / "key.json", encoding="utf-8")); neg_key = neg_key.get("key", neg_key)
    neg, nf = _answers(M1_IMPROVE, "neg_answers_"); fix, ff = _answers(M1_IMPROVE, "fix_answers_")
    fix_key = json.load(open(M1_IMPROVE / "fix_key.json", encoding="utf-8")); fix_key = fix_key.get("key", fix_key)
    for item, k in neg_key.items():
        if k.get("origin") != "M1":
            continue
        a = fix.get(item) or neg.get(item)
        if not a:
            continue
        out.append(dict(page="m1_improve_" + ("reask" if item in fix else "neg"), answers_file=ff if item in fix else nf, item=item, photo=k["photo"], run="run05",
                        mask_id=k["h2_id"], index=None, answer=a.get("answer"), sub=a.get("sub"), occluded_by_human=a.get("occluded"), updated=a.get("updated"),
                        reused_from=None, split="heldout_photo" if k["photo"] in HELD_OUT else "development_photo"))
    return out


def resolve(records: list[dict]) -> tuple[dict, dict]:
    """Group by (run, photo, mask_id). Returns (agreed, disagreed): agreed = masks whose every answer is the same non-null
    key; disagreed = masks with conflicting answers (kept out of the library)."""
    by = defaultdict(list)
    for r in records:
        if r["answer"] is None:
            continue
        by[(r["run"], r["photo"], r["mask_id"])].append(r)
    agreed, disagreed = {}, {}
    for k, rs in by.items():
        keys = {r["answer"] for r in rs}
        (agreed if len(keys) == 1 else disagreed)[k] = dict(run=k[0], photo=k[1], mask_id=k[2], answers=[r["answer"] for r in rs], pages=[r["page"] for r in rs],
                                                            times=[r["updated"] for r in rs], split=rs[0]["split"], n=len(rs),
                                                            occluded_by_human=any(bool(r.get("occluded_by_human")) for r in rs))
    return agreed, disagreed
