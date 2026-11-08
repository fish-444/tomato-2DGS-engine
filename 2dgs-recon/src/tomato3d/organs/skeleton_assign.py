"""User skeleton clicks -> compound-leaf labels (contract build/p1_skeleton_assign_20261101_01/CONTRACT.md).

Branches = the user's clicks in order, one line ("다음 잎") = one compound leaf's skeleton (legacy clicks without a line: cut at |dz| > 10 mm). 3D-first groups are assigned whole by the median distance
of their surfels to each branch (best <= 25 mm and margin >= 3 mm, else contested); ungrouped surfels one by one with the same rule. Light: distances only.

usage: python -m tomato3d.organs.skeleton_assign --id n_04
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/p1_all_3dfirst_20261027_01"
STORE_BASE = OUT / "user_base_click.jsonl"
R = dict(dz_mm=10.0, step_mm=1.0, max_mm=25.0, margin_mm=3.0)
LABELS = ["user_skeleton_seeds", "no_reference", "conditional_on_unvalidated_scale", "R3-1b-style_unaccepted"]


def clicks(iid, store=None):
    alive = []
    st = Path(store or STORE_BASE)
    for l in (st.read_text(encoding="utf-8").splitlines() if st.exists() else []):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("individual") != iid:
            continue
        if r["action"] == "add":
            alive.append(r)
        elif r["action"] == "undo" and alive:
            alive.pop()
        elif r["action"] == "reset_leaves":
            alive = []
    return alive


def branches(cl):
    """Contract A1 (2026-11-01 23:45): clicks that carry a `line` (the new editor: "다음 잎" ends a leaf) are one compound leaf per line, never cut by distance.
    Clicks without a `line` (the legacy log of 2026-11-01 00:02-00:41) keep the editor's rule: a vertical step |dz| > 10 mm between consecutive clicks starts a new leaf."""
    lines = {}
    for r in cl:
        key = ("L", int(r["line"])) if "line" in r else ("old", 0)
        lines.setdefault(key, []).append(np.asarray(r["point_mm"], float))
    out = []
    for key in sorted(lines, key=lambda k: (k[0] != "old", k[1])):
        if key[0] == "L":
            out.append(np.array(lines[key])); continue
        cur = []
        for q in lines[key]:
            if cur and abs(q[2] - cur[-1][2]) > R["dz_mm"]:
                out.append(np.array(cur)); cur = []
            cur.append(q)
        if cur:
            out.append(np.array(cur))
    return out


def densify(poly):
    if len(poly) == 1:
        return poly
    pts = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / R["step_mm"])))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(pts)


def decide(dist):
    """dist: (m, K) -> label (m,) [1..K or 0], status (m,) [0 none / 1 assigned / 2 contested]."""
    if dist.shape[1] == 0:
        return np.zeros(len(dist), int), np.zeros(len(dist), int)
    o = np.argsort(dist, 1); best = dist[np.arange(len(dist)), o[:, 0]]
    second = dist[np.arange(len(dist)), o[:, 1]] if dist.shape[1] > 1 else np.full(len(dist), np.inf)
    ok = (best <= R["max_mm"]) & (second - best >= R["margin_mm"]); far = best > R["max_mm"]
    lab = np.where(ok, o[:, 0] + 1, 0); st = np.where(ok, 1, np.where(far, 0, 2)); return lab, st


def assign(iid, store=None):
    z = np.load(OUT / iid / "merge" / "groups.npz"); xyz = z["xyz"].astype(float) * 1000.0; grp = z["group"].astype(int); n = len(xyz)
    br = branches(clicks(iid, store))
    if not br:
        return None
    D = np.stack([cKDTree(densify(b)).query(xyz)[0] for b in br], 1)
    lab = np.zeros(n, int); st = np.zeros(n, int)
    gids = np.unique(grp[grp >= 0])
    if len(gids):
        order = np.argsort(grp); gs = grp[order]; starts = np.searchsorted(gs, gids); ends = np.searchsorted(gs, gids, side="right")
        med = np.stack([np.median(D[order[a:b]], 0) for a, b in zip(starts, ends)])
        gl, gst = decide(med)
        for g, l_, s_, a, b in zip(gids, gl, gst, starts, ends):
            lab[order[a:b]] = l_; st[order[a:b]] = s_
    rest = grp < 0; l2, s2 = decide(D[rest]); lab[rest] = l2; st[rest] = s2
    d = OUT / iid / "skeleton"; d.mkdir(parents=True, exist_ok=True); stamp = time.strftime("%Y%m%dT%H%M%S")
    np.savez_compressed(d / f"labels_skeleton_{stamp}.npz", rows=z["rows"], label=lab.astype(np.int16), status=st.astype(np.int8), branches=np.array([json.dumps(b.round(3).tolist()) for b in br]))
    summ = dict(labels=LABELS, rule=R, individual=iid, clicks=sum(len(b) for b in br), branches=len(br), points_per_branch=[len(b) for b in br],
                surfels_per_branch={str(k + 1): int((lab == k + 1).sum()) for k in range(len(br))}, assigned_share=round(float((st == 1).mean()), 4), contested_share=round(float((st == 2).mean()), 4),
                none_share=round(float((st == 0).mean()), 4), file=f"labels_skeleton_{stamp}.npz", at=stamp)
    json.dump(summ, open(d / f"summary_{stamp}.json", "w", encoding="utf-8"), indent=1); json.dump(summ, open(d / "latest.json", "w", encoding="utf-8"), indent=1)
    return summ


def latest(iid):
    f = OUT / iid / "skeleton" / "latest.json"
    if not f.exists():
        return None, None
    s = json.load(open(f, encoding="utf-8")); z = np.load(OUT / iid / "skeleton" / s["file"]); return s, z


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); a = ap.parse_args(argv); print(json.dumps(assign(a.id), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
