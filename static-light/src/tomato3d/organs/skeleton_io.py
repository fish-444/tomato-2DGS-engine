"""skeleton_v1: the persistent description of compound-leaf identity (contract build/p1_skeleton_layer_20261101_01/CONTRACT.md, schema doc SKELETON_V1.md).

One polyline per compound leaf (base -> tip) in the board frame (metres, z up), stored apart from surfel labels. Labels are always re-derived from it, never edited by hand.
Pure numpy/scipy: schema, validator, conversions (click log, frozen rachis traces), Wilson interval and grade. No I/O of other sessions' files except what the callers pass in.

usage (library): load / save / validate / from_click_log / from_traces / to_traces
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = "skeleton_v1"
SCHEMA_AXES = "skeleton_v1.1"                    # written once a skeleton has an axis (Amendment A6); v1 files without axes are unchanged
AXIS_V = dict(unattached_mm=12.0, observed_gap_mm=15.0)
AXIS_SOURCES = ("user_click", "pot_centre_auto", "leaf_base_anchor", "fspm_fill", "observed_photo_click", "observed_3d_click")
LABELS = ["user_skeleton_seeds", "conditional_on_unvalidated_scale", "after_checkpoint_v1"]
V = dict(shared_base_mm=3.0, off_surface_mm=4.0, gap_warn_mm=20.0, resample_mm=1.0, legacy_split_dz_mm=10.0, legacy_resplit_at="2026-11-01T19:15:00")    # user 2026-11-01 19:10: legacy lines cut at |dz| > 10 mm
GRADE = dict(a_min=0.8, a_n=8, b_min=0.6)
EDITOR_VERSION = "skeleton_editor_1"


# ---------------------------------------------------------------- basic geometry
def resample(poly, step_m=V["resample_mm"] / 1000.0):
    """Linear resampling of a polyline (n, 3) at step_m; the first and the last input point are always kept."""
    p = np.asarray(poly, float)
    if len(p) < 2:
        return p.copy()
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


def path_length(poly):
    p = np.asarray(poly, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def leaf_uid(plant_id, k):
    return f"{plant_id}-L{k:02d}"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- construct
def new_skeleton(plant_id, capture_date, reconstruction_id, frame="board_m_zup", scale_status="conditional_on_unvalidated_scale", transform_to_pot=None):
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    return dict(schema=SCHEMA, plant_id=plant_id, capture_date=capture_date, reconstruction_id=reconstruction_id, frame=dict(name=frame, transform_to_pot=transform_to_pot),
                scale_status=scale_status, labels=list(LABELS), leaves=[], stem_display=dict(display_only=True, polyline_xyz=[]), heldout=[], editor_version=EDITOR_VERSION, created=now, edited=now)


def add_leaf(sk, polyline_xyz, source="user_click", status="active", click_log_lines=None, flags=None, uid=None, attrs=None):
    """Append one leaf; uid assigned in creation order and never reused (also after `lost` / `merged_into`)."""
    p = np.asarray(polyline_xyz, float)
    used = {int(l["leaf_uid"].rsplit("-L", 1)[1]) for l in sk["leaves"]}
    uid = uid or leaf_uid(sk["plant_id"], (max(used) + 1) if used else 1)
    leaf = dict(leaf_uid=uid, rank_hint=0, base_xyz=p[0].tolist(), polyline_xyz=p.tolist(), polyline_1mm_xyz=resample(p).round(6).tolist(), source=source, status=status,
                n_clicks=int(len(p)), click_log_lines=list(click_log_lines or []), flags=list(flags or []), attrs=dict(attrs or {}))
    sk["leaves"].append(leaf); update_ranks(sk); sk["edited"] = time.strftime("%Y-%m-%dT%H:%M:%S"); return leaf


def update_ranks(sk):
    """rank_hint = order of base height among the leaves that are not lost / merged (recomputed each time)."""
    act = [l for l in sk["leaves"] if l["status"] in ("active", "new", "candidate_lost")]
    for r, l in enumerate(sorted(act, key=lambda l: l["base_xyz"][2]), 1):
        l["rank_hint"] = r
    for l in sk["leaves"]:
        if l not in act:
            l["rank_hint"] = 0


def refresh_leaf(leaf):
    p = np.asarray(leaf["polyline_xyz"], float); leaf["base_xyz"] = p[0].tolist(); leaf["polyline_1mm_xyz"] = resample(p).round(6).tolist(); leaf["n_clicks"] = int(len(p))


# ---------------------------------------------------------------- validator
def plant_axis_xy(surfels_xyz, pot_xy=None):
    return np.asarray(pot_xy, float) if pot_xy is not None else np.asarray(surfels_xyz, float)[:, :2].mean(0)


def validate(sk, surfels_xyz, pot_xy=None):
    """Flags (never fixes): base_tip_order, shared_base (< 3 mm), off_surface (> 4 mm from every surfel), gap_over_20mm (warning). Writes them into each leaf and returns the list."""
    S = np.asarray(surfels_xyz, float); tree = cKDTree(S); axis = plant_axis_xy(S, pot_xy); flags = []
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new", "candidate_lost")]
    for l in sk["leaves"]:
        l["flags"] = [f for f in l.get("flags", []) if f in ("legacy_split_20mm", "legacy_split_dz10mm", "check_identity")]
    for l in live:
        P = np.asarray(l["polyline_xyz"], float)
        if len(P) >= 2 and np.linalg.norm(P[0, :2] - axis) > np.linalg.norm(P[-1, :2] - axis):
            l["flags"].append("base_tip_order"); flags.append((l["leaf_uid"], "base_tip_order", "base is farther from the plant axis than the tip"))
        d = tree.query(P)[0] * 1000.0
        if (d > V["off_surface_mm"]).any():
            l["flags"].append("off_surface"); flags.append((l["leaf_uid"], "off_surface", f"{int((d > V['off_surface_mm']).sum())} points > {V['off_surface_mm']} mm from every surfel (max {d.max():.1f} mm)"))
        if len(P) >= 2:
            st = np.linalg.norm(np.diff(P, axis=0), axis=1) * 1000.0
            if (st > V["gap_warn_mm"]).any():
                l["flags"].append("gap_over_20mm"); flags.append((l["leaf_uid"], "gap_over_20mm", f"largest step {st.max():.1f} mm: is this still one leaf?"))
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            if np.linalg.norm(np.asarray(live[i]["base_xyz"]) - np.asarray(live[j]["base_xyz"])) * 1000.0 < V["shared_base_mm"]:
                for l in (live[i], live[j]):
                    l["flags"].append("shared_base")
                flags.append((live[i]["leaf_uid"], "shared_base", f"with {live[j]['leaf_uid']} (< {V['shared_base_mm']} mm)"))
    return flags


# ---------------------------------------------------------------- file
def skeleton_dir(root, plant_id, capture_date):
    return Path(root) / "data" / "skeletons" / plant_id / capture_date


def save(sk, root):
    d = skeleton_dir(root, sk["plant_id"], sk["capture_date"]); d.mkdir(parents=True, exist_ok=True); f = d / "skeleton_v1.json"
    if f.exists():                                                           # previous versions are kept
        f.replace(d / f"skeleton_v1.{time.strftime('%Y%m%dT%H%M%S')}.json")
    sk["edited"] = time.strftime("%Y-%m-%dT%H:%M:%S"); f.write_text(json.dumps(sk, ensure_ascii=False, indent=1), encoding="utf-8"); return f


def load(path):
    sk = json.loads(Path(path).read_text(encoding="utf-8")); assert sk.get("schema") in (SCHEMA, SCHEMA_AXES), "not a skeleton_v1 / v1.1 file"; sk.setdefault("axes", []); return sk


# ---------------------------------------------------------------- conversions
def alive_clicks(store, plant_id):
    """Alive clicks of one plant from the append-only base-click log (add / undo), in click order, with their 1-based log line numbers."""
    alive = []
    for ln, l in enumerate(Path(store).read_text(encoding="utf-8").splitlines(), 1):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("individual") != plant_id:
            continue
        if r["action"] == "add":
            alive.append((ln, r))
        elif r["action"] == "undo" and alive:
            alive.pop()
        elif r["action"] == "reset_leaves":                                  # leaf skeleton restarted (main axis in the editor log is kept)
            alive = []
    return alive


def last_reset_at(store, plant_id):
    """Time of the plant's latest `reset_leaves` record ('' if none)."""
    at = ""
    for l in Path(store).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("individual") == plant_id and r.get("action") == "reset_leaves":
                at = max(at, r.get("at", ""))
    return at


def from_click_log(store, plant_id, capture_date, reconstruction_id, legacy_split_dz_mm=V["legacy_split_dz_mm"]):
    """The legacy log (clicks without a `line` field, 2026-11-01 00:02-00:41): cut where the vertical step |dz| between consecutive clicks exceeds 10 mm (user 2026-11-01 19:10,
    replaces the earlier 20 mm 3D-step rule); each piece = one compound-leaf axis, flagged `legacy_split_dz10mm` until the user re-checks it in the editor."""
    cl = alive_clicks(store, plant_id); sk = new_skeleton(plant_id, capture_date, reconstruction_id); cur, lns, legacy = [], [], False
    def flush():
        if cur:
            add_leaf(sk, np.array(cur) / 1000.0, source="user_click", click_log_lines=lns[:], flags=(["legacy_split_dz10mm"] if legacy else []))
    last_line = None
    for ln, r in cl:
        q = np.asarray(r["point_mm"], float); has_line = "line" in r; line = int(r.get("line") or 1)
        split = (line != last_line) if has_line else (bool(cur) and abs(q[2] - cur[-1][2]) > legacy_split_dz_mm)       # new editor: the user ends lines; legacy log: vertical step > 10 mm
        if cur and (split or (has_line != (not legacy))):
            flush(); cur, lns = [], []
        cur.append(q); lns.append(ln); last_line = line; legacy = not has_line
    flush(); return sk


def from_traces(traces, plant_id, capture_date, reconstruction_id, source="simulated_from_human_traces", flip_keys=()):
    """Frozen-format traces (original order) [{key, leaf, pts (m)}] -> skeleton: ONE leaf per trace, uid in creation order, points kept exactly.
    The pipeline leaf number each trace carries (`leaf`, 0 = unmapped) and its index / key go to `attrs`, so the export reproduces the frozen arrays exactly."""
    sk = new_skeleton(plant_id, capture_date, reconstruction_id)
    for k, t in enumerate(traces):
        P = np.asarray(t["pts"], float); fl = t.get("key") in set(flip_keys)             # traces stored tip-first are flipped to base -> tip; the export flips them back
        add_leaf(sk, P[::-1] if fl else P, source=source, attrs=dict(trace_index=k, trace_key=t.get("key"), pipeline_leaf=int(t["leaf"]), flipped_from_trace=bool(fl)))
    return sk


def to_traces(sk):
    """Skeleton -> frozen trace format in the original trace order (attrs.trace_index): `leaf` = attrs.pipeline_leaf if present else the number in the leaf_uid,
    `pts` = the stored polyline in metres (not resampled). Leaves with pipeline leaf <= 0 are returned too (the frozen bundle code skips them itself)."""
    out = []
    for l in sk["leaves"]:
        if l["status"] in ("active", "new", "candidate_lost"):
            a = l.get("attrs", {}); P = np.asarray(l["polyline_xyz"], float)
            out.append(dict(index=a.get("trace_index", len(out)), key=a.get("trace_key"), leaf=int(a["pipeline_leaf"]) if "pipeline_leaf" in a else int(l["leaf_uid"].rsplit("-L", 1)[1]),
                            leaf_uid=l["leaf_uid"], pts=(P[::-1] if a.get("flipped_from_trace") else P)))
    return sorted(out, key=lambda t: t["index"])


# ---------------------------------------------------------------- editor operations (each one is an append-only log record; the skeleton is re-derived by replaying them)
def _leaf(sk, uid):
    for l in sk["leaves"]:
        if l["leaf_uid"] == uid:
            return l
    raise KeyError(uid)


def extend_leaf(sk, uid, point_xyz, log_line=None):
    l = _leaf(sk, uid); l["polyline_xyz"].append([float(v) for v in point_xyz]); l["click_log_lines"].append(log_line); refresh_leaf(l); l["source"] = "propagated_edited" if l["source"].startswith("propagated") else l["source"]; return l


def move_point(sk, uid, index, point_xyz):
    l = _leaf(sk, uid); l["polyline_xyz"][int(index)] = [float(v) for v in point_xyz]; refresh_leaf(l); l["source"] = "propagated_edited" if l["source"].startswith("propagated") else l["source"]; return l


def delete_point(sk, uid, index=-1):
    l = _leaf(sk, uid)
    if len(l["polyline_xyz"]) <= 1:
        l["status"] = "lost"; update_ranks(sk); return l              # deleting the last remaining point = the leaf is withdrawn
    l["polyline_xyz"].pop(int(index)); refresh_leaf(l); return l


def mark_lost(sk, uid):
    l = _leaf(sk, uid); l["status"] = "lost"; update_ranks(sk); return l


def merge_leaves(sk, keep_uid, absorb_uid):
    """absorb's points are appended to keep (keep base stays the base); absorb gets status merged_into:<keep>."""
    k, a = _leaf(sk, keep_uid), _leaf(sk, absorb_uid); k["polyline_xyz"] += a["polyline_xyz"]; k["click_log_lines"] += a["click_log_lines"]; refresh_leaf(k)
    a["status"] = f"merged_into:{keep_uid}"; update_ranks(sk); return k


def split_leaf(sk, uid, index):
    """Points [index:] of uid become a new leaf (new uid, never reused); uid keeps [:index]. index must leave >= 1 point on each side."""
    l = _leaf(sk, uid); P = l["polyline_xyz"]; i = int(index)
    if not 1 <= i < len(P):
        raise KeyError(f"split index {i} outside 1..{len(P) - 1}")
    tail, lines = P[i:], l["click_log_lines"][i:]; l["polyline_xyz"] = P[:i]; l["click_log_lines"] = l["click_log_lines"][:i]; refresh_leaf(l)
    new = add_leaf(sk, np.asarray(tail, float), source=l["source"], click_log_lines=lines, flags=[f for f in l.get("flags", []) if f.startswith("legacy_split")], attrs=dict(split_from=uid))
    update_ranks(sk); return new


def flip_direction(sk, uid):
    l = _leaf(sk, uid); l["polyline_xyz"] = l["polyline_xyz"][::-1]; l["click_log_lines"] = l["click_log_lines"][::-1]; refresh_leaf(l)
    l.setdefault("attrs", {})["direction"] = "observed"; l["attrs"]["direction_flipped"] = not l["attrs"].get("direction_flipped", False); update_ranks(sk); return l


def confirm_direction(sk, uid):
    l = _leaf(sk, uid); l.setdefault("attrs", {})["direction"] = "observed"; return l


def review_aids(sk, ref_curve_xyz=None, ref_xy=None, cut_mm=25.0):
    """Display-only aids (dz10 Part A contract section 1): suggested cuts (3D step > cut_mm) and a base -> tip proposal (base = end nearer the main-axis curve, else nearer ref_xy).
    Nothing is applied; only user records (split / flip_direction / confirm_direction) change a line."""
    from scipy.spatial import cKDTree as _T
    tr = _T(np.asarray(ref_curve_xyz, float)) if ref_curve_xyz is not None and len(ref_curve_xyz) else None
    for l in sk["leaves"]:
        a = l.setdefault("attrs", {}); a.setdefault("direction", "direction_auto"); P = np.asarray(l["polyline_xyz"], float)
        a["suggested_cuts"] = [int(i + 1) for i in np.nonzero(np.linalg.norm(np.diff(P, axis=0), axis=1) * 1000.0 > cut_mm)[0]] if len(P) >= 2 else []
        if len(P) < 2:
            a["direction_auto"] = None; continue
        if tr is not None:
            d0, d1 = tr.query(P[0])[0], tr.query(P[-1])[0]; a["direction_ref"] = "main_axis"
        else:
            d0, d1 = np.linalg.norm(P[0, :2] - ref_xy), np.linalg.norm(P[-1, :2] - ref_xy); a["direction_ref"] = "pot_or_plant_centre_xy"
        a["direction_auto"] = "flip" if d1 < d0 else "keep"


def skeleton_version(sk):
    """skelver = first 12 hex of sha256 over the live leaves (uid, status, polyline) in uid order; held-out picks, timing and axes excluded."""
    live = sorted((l["leaf_uid"], l["status"], [[round(float(v), 7) for v in q] for q in l["polyline_xyz"]]) for l in sk["leaves"] if l["status"] in ("active", "new", "candidate_lost"))
    return hashlib.sha256(json.dumps(live).encode()).hexdigest()[:12]


def heldout_pick(sk, surfel, leaf_uid, point_xyz=None):
    sk["heldout"].append(dict(surfel=int(surfel), leaf_uid=(leaf_uid if leaf_uid not in (None, "", "모름", "unknown") else None), point_xyz=([float(v) for v in point_xyz] if point_xyz is not None else None), at=time.strftime("%Y-%m-%dT%H:%M:%S")))


# ---------------------------------------------------------------- axes (skeleton_v1.1, Amendment A6): the user's stem clicks; the fit lives in skeleton_axis.py
def axis_uid(plant_id, k):
    return f"{plant_id}-A{k:02d}"


def add_axis(sk, type_="main", parent_axis_uid=None):
    """New axis, uid in creation order and never reused. A main axis may exist once (a second `main` start continues the existing one)."""
    sk.setdefault("axes", [])
    if type_ == "main":
        for a in sk["axes"]:
            if a["type"] == "main" and a["status"] == "active":
                return a
    used = [int(a["axis_uid"].rsplit("-A", 1)[1]) for a in sk["axes"]]
    a = dict(axis_uid=axis_uid(sk["plant_id"], (max(used) + 1) if used else 1), type=type_, parent_axis_uid=parent_axis_uid, attach_s_mm=None, status="active", points=[], fit=None, segments=[], nodes=[], flags=[])
    sk["axes"].append(a); sk["schema"] = SCHEMA_AXES; return a


def _axis(sk, uid):
    for a in sk.get("axes", []):
        if a["axis_uid"] == uid:
            return a
    raise KeyError(uid)


def axis_add_point(sk, uid, point_xyz, source="user_click", log_line=None):
    assert source in AXIS_SOURCES, source
    a = _axis(sk, uid); a["points"].append(dict(xyz=[float(v) for v in point_xyz], source=source, log_line=log_line)); return a


def axis_move_point(sk, uid, index, point_xyz):
    a = _axis(sk, uid); a["points"][int(index)]["xyz"] = [float(v) for v in point_xyz]; return a


def axis_delete_point(sk, uid, index=-1):
    a = _axis(sk, uid)
    if a["points"]:
        a["points"].pop(int(index))
    if not a["points"]:
        a["status"] = "withdrawn"
    return a


def validate_axes(sk):
    """Flags (never fixes): axis_not_bottom_to_top (first user point above the last), unattached leaves (> 12 mm), node indices not increasing with attach_s. Returns the list."""
    flags = []
    for a in sk.get("axes", []):
        if a["status"] != "active":
            continue
        a["flags"] = [f for f in a.get("flags", []) if f not in ("axis_not_bottom_to_top", "node_order")]
        P = [p["xyz"] for p in a["points"] if p["source"] in ("user_click", "observed_photo_click", "observed_3d_click")]
        if len(P) >= 2 and P[0][2] > P[-1][2]:
            a["flags"].append("axis_not_bottom_to_top"); flags.append((a["axis_uid"], "axis_not_bottom_to_top", "first click is higher than the last: draw from the base upward"))
        nd = [n for n in a.get("nodes", []) if n.get("leaf_uid")]
        if any(b["node_index"] <= c["node_index"] or b["s_mm"] < c["s_mm"] for c, b in zip(nd, nd[1:])):
            a["flags"].append("node_order"); flags.append((a["axis_uid"], "node_order", "node indices must increase with attach_s"))
    for l in sk["leaves"]:
        if l["status"] in ("active", "new", "candidate_lost") and sk.get("axes") and l.get("attach_axis_uid") is None and l.get("attach_distance_mm") is not None:
            flags.append((l["leaf_uid"], "unattached", f"base {l['attach_distance_mm']:.1f} mm from the axis (> {AXIS_V['unattached_mm']:g} mm)"))
    return flags


def apply_edits(sk, records):
    """Replay editor records (dicts from clicks.jsonl) in order. Unknown uids are skipped and listed. Returns the skipped records."""
    skipped = []
    for r in records:
        a = r.get("action")
        try:
            if a == "extend":
                extend_leaf(sk, r["leaf_uid"], np.asarray(r["point_mm"], float) / 1000.0, r.get("log_line"))
            elif a == "move_point":
                move_point(sk, r["leaf_uid"], r["index"], np.asarray(r["point_mm"], float) / 1000.0)
            elif a == "delete_point":
                delete_point(sk, r["leaf_uid"], r.get("index", -1))
            elif a == "mark_lost":
                mark_lost(sk, r["leaf_uid"])
            elif a == "merge":
                merge_leaves(sk, r["keep_uid"], r["absorb_uid"])
            elif a == "split":
                split_leaf(sk, r["leaf_uid"], r["index"])
            elif a == "flip_direction":
                flip_direction(sk, r["leaf_uid"])
            elif a == "confirm_direction":
                confirm_direction(sk, r["leaf_uid"])
            elif a == "heldout_pick":
                heldout_pick(sk, r["surfel"], r.get("leaf_uid"), (np.asarray(r["point_mm"], float) / 1000.0) if r.get("point_mm") else None)
            elif a == "axis_start":
                ax = add_axis(sk, r.get("type", "main"), r.get("parent_axis_uid")); sk["_current_axis"] = ax["axis_uid"]
            elif a == "axis_add":
                uid = r.get("axis_uid") or sk.get("_current_axis")
                if uid is None:
                    ax = add_axis(sk, "main"); uid = ax["axis_uid"]; sk["_current_axis"] = uid
                axis_add_point(sk, uid, np.asarray(r["point_mm"], float) / 1000.0, r.get("source", "user_click"), r.get("log_line"))
            elif a == "axis_move_point":
                axis_move_point(sk, r["axis_uid"], r["index"], np.asarray(r["point_mm"], float) / 1000.0)
            elif a == "axis_delete_point":
                axis_delete_point(sk, r["axis_uid"], r.get("index", -1))
            elif a in ("end_line", "done", "start"):
                sk.setdefault("timing", []).append(dict(action=a, at=r.get("at"), t=r.get("t")))
            else:
                skipped.append(r)
        except (KeyError, IndexError):                                       # a record pointing at a point / leaf that no longer exists is skipped, never fatal
            skipped.append(r)
    sk.pop("_current_axis", None); return skipped


# ---------------------------------------------------------------- held-out agreement
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def grade(agree, n):
    """A: agreement >= 0.8 with n >= 8; B: 0.6-0.8; else C (n < 8 can never be A)."""
    if n == 0:
        return "C"
    a = agree / n
    if a >= GRADE["a_min"] and n >= GRADE["a_n"]:
        return "A"
    return "B" if a >= GRADE["b_min"] else "C"


def heldout_agreement(picks, route_label_of_surfel):
    """picks: [{surfel, leaf_uid or None ('모름')}]; route_label_of_surfel(surfel) -> leaf_uid or None. Contested / none count as disagreement; '모름' picks are excluded."""
    use = [p for p in picks if p.get("leaf_uid")]
    k = sum(1 for p in use if route_label_of_surfel(p["surfel"]) == p["leaf_uid"]); n = len(use)
    return dict(n=n, agree=k, rate=(round(k / n, 4) if n else None), wilson95=wilson(k, n), grade=grade(k, n), unknown_picks=len(picks) - n)
