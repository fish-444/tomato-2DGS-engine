"""Store for skeleton_v1 (contract build/p1_skeleton_layer_20261101_01/CONTRACT.md, sections 2-3): the skeleton of one plant and date is always RE-DERIVED from two append-only logs,
    base clicks  build/p1_all_3dfirst_20261027_01/user_base_click.jsonl   (page 8835 `mode=base`; `add` / `undo`, with `line` since the new editor)
    editor log   data/skeletons/<plant>/<date>/clicks.jsonl               (extend / move_point / delete_point / mark_lost / merge / heldout_pick / end_line / start / done)
and saved as data/skeletons/<plant>/<date>/skeleton_v1.json (previous versions kept). Nothing is edited by hand; logs are never rewritten.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import skeleton_io as SIO

HELDOUT = dict(n=10, min_dist_mm=5.0, thirds=3)


def paths(root, plant_id, capture_date):
    d = SIO.skeleton_dir(root, plant_id, capture_date); return d, d / "clicks.jsonl", d / "skeleton_v1.json"


def append_edit(root, plant_id, capture_date, rec):
    d, log, _ = paths(root, plant_id, capture_date); d.mkdir(parents=True, exist_ok=True)
    rec = dict(rec, at=rec.get("at") or time.strftime("%Y-%m-%dT%H:%M:%S"), t=rec.get("t") or time.time(), provenance=rec.get("provenance", "user_skeleton_edit"))
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_edits(log):
    return [json.loads(l) for l in Path(log).read_text(encoding="utf-8").splitlines() if l.strip()] if Path(log).exists() else []


def rebuild(root, base_store, plant_id, capture_date, reconstruction_id, surfels_xyz, pot_xy=None, save=True):
    """base clicks -> leaves, then the editor log replayed, then the validator; saved as a new version. Returns (skeleton, flags, skipped_records)."""
    sk = SIO.from_click_log(base_store, plant_id, capture_date, reconstruction_id); d, log, _ = paths(root, plant_id, capture_date)
    edits = read_edits(log); legacy = any("legacy_split_dz10mm" in l.get("flags", []) for l in sk["leaves"]); stale = []
    cut = max(SIO.V["legacy_resplit_at"] if legacy else "", SIO.last_reset_at(base_store, plant_id))      # leaf numbers changed (10 mm re-split or a leaf reset): older leaf-naming edits are skipped
    if cut:
        keys = ("leaf_uid", "keep_uid", "absorb_uid")
        stale = [e for e in edits if e.get("at", "") < cut and any(e.get(k) for k in keys)]; edits = [e for e in edits if e not in stale]
    skipped = SIO.apply_edits(sk, edits) + [dict(record=e, reason="made_under_the_20mm_split_numbering") for e in stale]; flags = SIO.validate(sk, surfels_xyz, pot_xy=pot_xy); sk["minutes"] = minutes(sk, base_store, plant_id)
    if sk.get("axes"):                                                         # Amendment A6: the axis fit is derived data, recomputed on every rebuild
        from tomato3d.organs import skeleton_axis as SA
        sk["axis_summary"] = SA.fit_all(sk, surfels_xyz, root=SA.root_for(plant_id)); flags += SIO.validate_axes(sk)
    main = next((a for a in sk.get("axes", []) if a["type"] == "main" and a["status"] == "active" and a.get("fit")), None)          # dz10 Part A: display-only review aids
    SIO.review_aids(sk, (np.asarray(main["fit"]["curve_mm"]) / 1000.0) if main else None, SIO.plant_axis_xy(surfels_xyz, pot_xy)); sk["skelver"] = SIO.skeleton_version(sk)
    if save:
        SIO.save(sk, root)
    return sk, flags, skipped


def minutes(sk, base_store, plant_id):
    """Clicking time per individual: first to last base click (alive or undone) plus editor records; `done` records close a sitting."""
    ts = [json.loads(l)["t"] for l in Path(base_store).read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("individual") == plant_id and "t" in json.loads(l)]
    ts += [x["t"] for x in sk.get("timing", []) if x.get("t")]
    ts += [time.mktime(time.strptime(h["at"], "%Y-%m-%dT%H:%M:%S")) for h in sk.get("heldout", []) if h.get("at")]
    if not ts:
        return dict(minutes=None, note="no timestamps")
    ts = sorted(ts); gaps = np.diff(ts); active = float(sum(g for g in gaps if g <= 600))                  # pauses longer than 10 min are not counted as clicking time
    return dict(minutes=round(active / 60, 1), span_minutes=round((ts[-1] - ts[0]) / 60, 1), clicks_and_edits=len(ts), rule="sum of gaps <= 10 min between consecutive records")


def heldout_candidates(sk, surfels_xyz, seed=None, n=HELDOUT["n"]):
    """n random surfels, stratified by height thirds, excluding surfels within 5 mm of any skeleton point. Deterministic per (plant, date) unless a seed is given."""
    X = np.asarray(surfels_xyz, float); pts = [p for l in sk["leaves"] if l["status"] in ("active", "new", "candidate_lost") for p in l["polyline_1mm_xyz"]]
    ok = np.ones(len(X), bool)
    if pts:
        ok &= cKDTree(np.asarray(pts, float)).query(X)[0] > HELDOUT["min_dist_mm"] / 1000
    rng = np.random.default_rng(seed if seed is not None else int.from_bytes((sk["plant_id"] + sk["capture_date"]).encode(), "little") % (2 ** 32))
    z = X[:, 2]; edges = np.percentile(z[ok], [100 / 3, 200 / 3]); out = []
    for k in range(HELDOUT["thirds"]):
        lo = -np.inf if k == 0 else edges[k - 1]; hi = np.inf if k == 2 else edges[k]; idx = np.nonzero(ok & (z > lo) & (z <= hi))[0]; want = n // 3 + (1 if k < n % 3 else 0)
        if len(idx):
            out += rng.choice(idx, min(want, len(idx)), replace=False).tolist()
    picked = {h["surfel"] for h in sk.get("heldout", [])}
    return [dict(surfel=int(i), point_mm=(X[i] * 1000).round(2).tolist(), done=(int(i) in picked)) for i in out]
