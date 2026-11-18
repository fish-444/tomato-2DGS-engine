"""microtom_recon one-zip v0.2.1 — stage 7: T-Prox compound-leaf labels on the reconstructed surfels.

What runs here is the frozen maths only: `pc_absorb.py` (unchanged file, weight set A) and the rachis step `pc_rachis.py`.
What does NOT run here, and is read from `--tprox-inputs` instead: the 3D-first groups, the sheet geometry and the stem-like /
rachis-like axis classification of this run's surfels. They are made on the project PC from this run's own `surfels_full.ply`
(they need candidate masks and project data that cannot be shipped), so stage 7 is the second half of a round trip.

The skeleton comes from a person (`--skeleton skeleton_v1.json`). No skeleton is invented here.
Nothing is validated: the labels are `tprox_unverified_until_heldout`; the `stem` flag is the attached axis classification, unscored.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE / "microtom_recon"

SET = "A"                                                   # frozen: weight set A (geometry only)
SEED_MIN = 50                                               # frozen: leaves with fewer seed surfels are listed, not removed
ADAPT = dict(rachis_mm=3.0, blade_mm=8.0, blade_margin_mm=5.0)   # frozen seed rule (skeleton_r8pc.ADAPT)
U_MM = 22.72                                                # chain_r6.U_MM, carried in the bundle as reach_mm
OPAQUE = 0.5
INPUT_FILES = ("groups.npz", "geometry.npz", "geometry.json", "axis_labels.npz")
META_FILE = "TPROX_INPUTS_META.json"
FRAME = "board_m_zup"
FRAME_NEAR_MM, FRAME_MIN_SHARE = 10.0, 0.5                  # packaging check only: is the skeleton in this run's frame at all
LABELS = ["route_tprox", "tprox_unverified_until_heldout", "conditional_on_unvalidated_scale", "user_skeleton_seeds", "frozen_code_reuse",
          "stem_from_axis_classifier_unscored"]
HOWTO = ("7단계(T-Prox 복엽 라벨)는 실행하지 않았습니다. 실행하려면 사람이 만든 스켈레톤과, 이 복원의 surfels_full.ply 로 PC 에서 만든 "
         "그룹·축 분류 데이터셋이 필요합니다: --skeleton <skeleton_v1.json> --tprox-inputs <폴더>  (README_KO.md 4절)")


class Skip(Exception):
    """Stage 7 is not run; stages 1-6 stay as they are. The message is Korean and says what is missing."""


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------------
# The two functions below are copied verbatim from tomato3d/organs/skeleton_r8pc.py (seeds, traces_arrays); the build script checks
# that their text is identical to the source. That module itself is not shipped: the rest of it drives Kaggle from the project PC.
def seeds(xyz_m, axis_kind, group, leaf_polys, rachis_any=False):
    """stage1 seeds: rachis-like surfels within 3 mm of one leaf's 1 mm polyline(s); 3D-first groups (whole) with median distance <= 8 mm to one leaf and >= 13 mm to every other.
    leaf_polys[k] = list of polylines (m) of leaf k + 1 (a leaf traced twice has two)."""
    K = len(leaf_polys); n = len(xyz_m); st = np.zeros(n, np.int16)
    if K == 0:
        return st, dict(rachis_seeds=0, blade_groups=0, blade_seeds=0)
    D = np.stack([cKDTree(np.vstack([SIO.resample(np.asarray(p, float)) for p in polys])).query(xyz_m)[0] for polys in leaf_polys], 1) * 1000.0
    o = np.argsort(D, 1); best = D[np.arange(n), o[:, 0]]; second = D[np.arange(n), o[:, 1]] if K > 1 else np.full(n, np.inf)
    rach = ((axis_kind == 2) | rachis_any) & (best <= ADAPT["rachis_mm"]) & (second > ADAPT["rachis_mm"]); st[rach] = o[rach, 0] + 1      # T-prox (A5): rachis_any drops the axis-label condition
    nb = 0; ng = 0
    for gid in np.unique(group[group >= 0]):
        m = (group == gid) & (axis_kind == 0)
        if not m.any():
            continue
        med = np.median(D[m], 0); oo = np.argsort(med)
        if med[oo[0]] <= ADAPT["blade_mm"] and (K == 1 or med[oo[1]] >= ADAPT["blade_mm"] + ADAPT["blade_margin_mm"]):
            st[m] = oo[0] + 1; nb += int(m.sum()); ng += 1
    return st, dict(rachis_seeds=int(rach.sum()), blade_groups=ng, blade_seeds=nb)


def traces_arrays(sk):
    pts, lf, ids = [], [], []
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new") and len(l["polyline_xyz"]) >= 2]      # same filter and order as leaf_map / seeds (a 1-point line used to shift the numbering)
    for k, l in enumerate(live):
        P = np.asarray(l["polyline_xyz"], float)
        d = SIO.resample(P, 0.001); pts.append(d); lf.append(np.full(len(d), k + 1)); ids.append(np.full(len(d), k))
    uids = [l["leaf_uid"] for l in live]
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int), uids
    return np.vstack(pts), np.concatenate(lf), np.concatenate(ids), uids
# ---------------------------------------------------------------------------------------------------------------------------------

def leaf_lines(sk):
    """v0.2.1: skeleton lines that record the same leaf (skeleton_io.to_traces: attrs.pipeline_leaf, else the number in the uid) are ONE leaf, as in the project's own
    bundle code for a leaf traced with several lines. Leaves are numbered in the order their first line appears in the file, so a skeleton with one line per leaf
    gives exactly traces_arrays(sk). Lines with a recorded leaf number <= 0 are unmapped and skipped, as the project's bundle code does."""
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new") and len(l["polyline_xyz"]) >= 2]
    try:
        ident = {t["leaf_uid"]: int(t["leaf"]) for t in SIO.to_traces(sk)}
    except (KeyError, ValueError, IndexError):                                # no leaf number can be read: every line is its own leaf (the v0.2 rule)
        ident = {l["leaf_uid"]: k + 1 for k, l in enumerate(live)}
    used = [l for l in live if ident[l["leaf_uid"]] > 0]; order = []
    for l in used:
        if ident[l["leaf_uid"]] not in order:
            order.append(ident[l["leaf_uid"]])
    polys = [[np.asarray(l["polyline_xyz"], float) for l in used if ident[l["leaf_uid"]] == L] for L in order]
    names = ["+".join(l["leaf_uid"] for l in used if ident[l["leaf_uid"]] == L) for L in order]; pts, lf, ids = [], [], []
    for k, l in enumerate(used):
        d = SIO.resample(np.asarray(l["polyline_xyz"], float), 0.001); pts.append(d); lf.append(np.full(len(d), order.index(ident[l["leaf_uid"]]) + 1)); ids.append(np.full(len(d), k))
    if not pts:
        return [], [], [], np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int)
    tp, tl, tid = np.vstack(pts), np.concatenate(lf), np.concatenate(ids)
    if len(order) == len(live):                                                # self-check: without merged lines this is the v0.2 / generic-branch result
        a = traces_arrays(sk); assert np.allclose(tp, a[0]) and np.array_equal(tl, a[1]) and np.array_equal(tid, a[2])
    return [int(x) for x in order], polys, names, tp, tl, tid


def _imports():
    """numpy / scipy and the copied modules are imported when stage 7 starts, not when run_recon.py is loaded."""
    global np, cKDTree, SIO, sio
    import numpy as np                                                                    # noqa: PLC0415
    from scipy.spatial import cKDTree                                                     # noqa: PLC0415
    if str(PKG) not in sys.path:
        sys.path.insert(0, str(PKG))
    from tomato3d.fuse3d import surfels_io as sio                                         # noqa: PLC0415
    from tomato3d.organs import skeleton_io as SIO                                        # noqa: PLC0415


def run_frozen(d: Path, log) -> dict:
    """d/kaggle_data/r8pc_bundle.npz -> d/kaggle_out (pc_absorb.py, unchanged, own process) -> d/R8_pc (set A) -> d/R8_pc_rachis (pc_rachis).
    Same order and same file layout as skeleton_tprox.download_set + skeleton_r8pc.step_rachis; no Kaggle kernel in between."""
    _imports()
    pa = PKG / "tomato3d" / "organs" / "pc_absorb.py"; got = d / "kaggle_out"; got.mkdir(parents=True, exist_ok=True)
    if not (got / "results.json").exists() or "seconds" not in json.load(open(got / "results.json")):
        log(f"tprox: pc_absorb.py (frozen, sha256 {sha(pa)[:16]}) start")
        with open(d / "pc_absorb.log", "w", encoding="utf-8") as lf:
            r = subprocess.run([sys.executable, str(pa), "--bundle", str(d / "kaggle_data" / "r8pc_bundle.npz"), "--out", str(got)], stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f"pc_absorb.py exit {r.returncode}; see {d / 'pc_absorb.log'}")
    res = json.load(open(got / "results.json")); chosen = res["choice"]["chosen"]
    f = got / ("R8_pc" if chosen == SET else f"set_{SET}") / "labels.npz"; z = np.load(f); assert str(z["set_name"]) == SET, (str(z["set_name"]), f)
    (d / "R8_pc").mkdir(exist_ok=True); shutil.copy2(f, d / "R8_pc" / "labels.npz"); shutil.copy2(got / "results.json", d / "R8_pc" / "results.json")
    from tomato3d.organs import pc_rachis as PR                                           # noqa: PLC0415
    PR.D0 = d; PR.OUT = d / "R8_pc_rachis"; PR.main()
    return dict(in_kernel_choice=chosen, in_kernel_gate=res["choice"]["gate_pass"], used_set=SET, seconds=res.get("seconds"), pc_absorb_sha256=sha(pa))


def provenance(r8, zr):
    """A1 classes: 1 seed (R8_pc stage1), 2 absorbed (R8_pc rounds), 3 rachis_step (rachis-file trace_seed / rounds), 0 none.  (skeleton_tprox.provenance)"""
    c = np.zeros(len(zr["label"]), np.int8); s8 = r8["source"]; sr = zr["source"]
    c[s8 == 1] = 1; c[(s8 >= 2) & (s8 <= 6)] = 2; c[(sr >= 2) & (sr <= 7)] = 3; c[zr["label"] <= 0] = 0; return c


def check_inputs(a, out: Path):
    """Everything that must hold before the frozen code is started. Raises Skip with what to do."""
    if a.skeleton == "auto":
        raise Skip("--skeleton auto: v0.2 에는 줄기 모델이 들어 있지 않아 자동 주축 제안을 쓸 수 없습니다(학습된 줄기 모델 없음). 사람이 만든 skeleton_v1.json 경로를 주세요.")
    if not a.skeleton:
        raise Skip(HOWTO)
    skp = Path(a.skeleton)
    if not skp.is_file():
        raise Skip(f"--skeleton: 파일이 없습니다: {skp}")
    if not a.tprox_inputs:
        raise Skip("--tprox-inputs 가 없습니다. " + HOWTO)
    tin = Path(a.tprox_inputs); hit = sorted(tin.rglob(META_FILE)) if tin.is_dir() else []
    if not hit:
        raise Skip(f"--tprox-inputs: {META_FILE} 을 찾지 못했습니다: {tin}\n그룹·축 분류 데이터셋을 Input 에 추가했는지 확인하세요. 다른 곳에서 내려받지 않습니다.")
    tin = hit[0].parent; miss = [f for f in INPUT_FILES if not (tin / f).exists()]
    if miss:
        raise Skip(f"--tprox-inputs: 파일이 빠졌습니다: {', '.join(miss)} ({tin})")
    return skp, tin


def stage_tprox(a, out: Path, log) -> dict:
    _imports()
    skp, tin = check_inputs(a, out); meta_in = json.load(open(tin / META_FILE, encoding="utf-8"))
    full = out / "surfels_full.ply"; s_sha = sha(full)
    if meta_in.get("surfels_full_sha256") != s_sha:
        raise Skip("그룹·축 분류 데이터셋이 이 복원의 것이 아닙니다(surfels_full.ply 의 sha256 이 다름).\n"
                   f"  이 복원: {s_sha}\n  데이터셋: {meta_in.get('surfels_full_sha256')}\n이 실행의 surfels_full.ply 로 PC 에서 그룹을 다시 만들어 붙이세요.")
    sf = sio.read_surfels_full(full); rows = np.nonzero(sf["opacity"] >= OPAQUE)[0]
    g = np.load(tin / "groups.npz"); geo = np.load(tin / "geometry.npz"); gj = json.load(open(tin / "geometry.json", encoding="utf-8")); ax = np.load(tin / "axis_labels.npz")
    if not (np.array_equal(g["rows"], rows) and np.array_equal(ax["rows"], rows) and np.allclose(g["xyz"], sf["xyz"][rows], atol=1e-6)):
        raise Skip("그룹·축 분류의 서펠 행이 이 복원과 맞지 않습니다(rows 또는 xyz 불일치). 이 실행의 surfels_full.ply 로 다시 만들어야 합니다.")
    # ---- skeleton: frame check, then the live leaf lines
    sk = SIO.load(skp); fr = sk.get("frame") or {}
    if fr.get("name") != FRAME or fr.get("transform_to_pot") is not None:
        raise Skip(f"스켈레톤의 좌표계가 이 복원과 다릅니다: frame={fr}. 필요한 것은 '{FRAME}'(보드 기준 m, z 위쪽, 변환 없음)입니다.")
    leaf_ids, polys, uids, tp, tl, tid = leaf_lines(sk)
    if not uids:
        raise Skip("스켈레톤에 쓸 수 있는 복엽 선(2점 이상, active/new)이 없습니다. 주축만 있는 스켈레톤으로는 복엽 라벨을 만들 수 없습니다.")
    xyz = g["xyz"].astype(float); near = cKDTree(xyz).query(tp)[0] * 1000.0; share = float((near <= FRAME_NEAR_MM).mean())
    if share < FRAME_MIN_SHARE:
        raise Skip(f"스켈레톤이 이 복원의 서펠과 겹치지 않습니다: 선 위의 점 중 서펠에서 {FRAME_NEAR_MM:.0f} mm 안에 있는 것이 {share:.0%} 뿐입니다(최소 {FRAME_MIN_SHARE:.0%}).\n"
                   "다른 복원이나 다른 좌표계(예: 화분 윗면 기준, mm 단위)에서 만든 스켈레톤으로 보입니다.")
    # ---- bundle, frozen layout (skeleton_r8pc.step_bundle, generic branch, mode "prox")
    d = out / "tprox"; (d / "kaggle_data").mkdir(parents=True, exist_ok=True); n = len(rows)
    st, sinfo = seeds(xyz, ax["axis_kind"], g["group"].astype(int), polys, rachis_any=True); ts = float(gj["t_sheet_mm"])
    N = sf["normal"][rows].astype(float); N[geo["b"]] = geo["normal"]; sidx = np.zeros(n, np.int16); nsh = np.ones(n, np.int16); sidx[geo["b"]] = geo["index"]; nsh[geo["b"]] = geo["n_sheets"]
    Z = dict(xyz=g["xyz"].astype(np.float64), normal=N, scale=sf["scale"][rows][:, :2].astype(np.float64), tu=sf["tu"][rows], tv=sf["tv"][rows], rgb=sf["rgb"][rows].astype(np.float32), opacity=sf["opacity"][rows].astype(np.float32),
             axis_kind=ax["axis_kind"].astype(np.int8), sheet_index=sidx, n_sheets=nsh, stage1=st, r7=np.zeros(n, np.int16), trace_pts=tp, trace_leaf=tl.astype(np.int64), trace_id=tid.astype(np.int64),
             t_sheet_mm=np.float64(ts), offset_mm=np.float64(max(ts, 1.0)), reach_mm=np.float64(U_MM), rows=rows)
    np.savez_compressed(d / "kaggle_data" / "r8pc_bundle.npz", **Z)
    leaf_map = {k + 1: u for k, u in enumerate(uids)}
    bmeta = dict(mode="prox", adapter=ADAPT, seeds=sinfo, stage1_labelled=int((st > 0).sum()), leaves=len(leaf_map), leaf_map={str(k): v for k, v in leaf_map.items()},
                 skeleton_sha256=sha(skp), surfels_full_sha256=s_sha, tprox_inputs={f: sha(tin / f) for f in INPUT_FILES}, tprox_inputs_meta=meta_in)
    json.dump(bmeta, open(d / "bundle_meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"tprox: bundle built, {n} surfels, {len(uids)} leaf lines, seeds {sinfo}, skeleton-in-frame share {share:.2f}")
    if int((st > 0).sum()) == 0:
        raise Skip("스켈레톤 선 근처에서 씨앗 서펠을 하나도 찾지 못했습니다(선에서 3 mm 안의 서펠, 8 mm 안의 그룹이 없음). 선이 잎 위를 지나는지 확인하세요.")
    try:
        kinfo = run_frozen(d, log)
    except (RuntimeError, AssertionError, KeyError) as ex:
        raise Skip(f"고정 코드(pc_absorb.py / pc_rachis.py)가 이 묶음에서 멈췄습니다: {ex}\n"
                   "고정 코드는 파일럿 개체에 맞춰져 있어 잎 배치에 따라 실행되지 않을 수 있습니다. 1–6단계 결과는 그대로입니다. tprox/pc_absorb.log 를 함께 전달해 주세요.")
    # ---- labels (skeleton_tprox.step_post without the distance-route comparison)
    z = np.load(d / "R8_pc_rachis" / "labels_rachis.npz"); r8 = np.load(d / "R8_pc" / "labels.npz"); assert np.array_equal(z["rows"], rows) and np.array_equal(r8["rows"], rows)
    lab = z["label"].astype(int); p1 = r8["p1"]; p2 = r8["p2"]; prov = provenance(r8, z); stem = ax["axis_kind"].astype(np.int8) == 1
    tags = LABELS + ["skeleton_human"] + list(meta_in.get("labels", [])) + [t for t in sk.get("labels", []) if t not in LABELS]      # a file path was given: the skeleton is the person's (prompt A.3)
    np.savez_compressed(out / "labels_tprox.npz", rows=rows, label=lab.astype(np.int16), status=(lab > 0).astype(np.int8), leaf_uids=np.array(uids), p1=p1, p2=p2,
                        margin=np.where(prov == 2, p1 - p2, np.nan).astype(np.float32), provenance=prov, provenance_names=np.array(["0:none", "1:seed", "2:absorbed", "3:rachis_step"]),
                        stem=stem, labels=np.array(tags))
    assigned = lab > 0; seed_n = {leaf_map[k]: int((st == k).sum()) for k in leaf_map}
    meta = dict(package="microtom_recon_onezip v0.2.1 stage 7", labels=tags, validation="none: assigned share is not accuracy", kernel=kinfo, leaves=len(uids), leaf_uids=uids, leaf_ids_recorded=leaf_ids, skeleton_lines_used=int(len(set(tid.tolist()))),
                surfels=int(n), assigned=int(assigned.sum()), assigned_share=round(float(assigned.mean()), 4), unassigned_share=round(float((~assigned).mean()), 4),
                provenance={nm: int((prov == i).sum()) for i, nm in enumerate(("none", "seed", "absorbed", "rachis_step"))}, seeds=sinfo, seed_surfels_per_leaf=seed_n,
                leaves_seed_below_min=[u for u, c in seed_n.items() if c < SEED_MIN], seed_min=SEED_MIN,
                surfels_per_leaf={leaf_map[k]: int((lab == k).sum()) for k in leaf_map},
                stem=dict(label="stem_from_axis_classifier_unscored", surfels=int(stem.sum()), share=round(float(stem.mean()), 4), stem_also_assigned_to_a_leaf=int((stem & assigned).sum())),
                skeleton=dict(file=skp.name, sha256=sha(skp), frame=fr, in_frame_share=round(share, 4), leaves_total=len(sk["leaves"]), leaves_used=len(uids)),
                inputs=dict(surfels_full_sha256=s_sha, tprox_inputs={f: sha(tin / f) for f in INPUT_FILES}, tprox_inputs_meta=meta_in),
                frozen=dict(set=SET, seed_rule=ADAPT, seed_min=SEED_MIN, reach_mm=U_MM, note="gamma 1.0, p1 >= 0.6, margin >= 0.25, 5 rounds, edge 3 mm are inside pc_absorb.py / pc_rachis.py, unchanged"))
    json.dump(meta, open(out / "tprox_meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    # ---- a point file a person can open: leaf number and stem flag per surfel
    rgb = sf["rgb"][rows].astype(np.uint8); rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("leaf", "<i2"), ("stem", "u1")])
    for kk, c in zip("xyz", g["xyz"].T): rec[kk] = c
    for kk, c in zip("rgb", rgb.T): rec[kk] = c
    rec["leaf"] = lab; rec["stem"] = stem
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment board metres, z up; leaf = index into leaf_uids of tprox_meta.json (0 = none); tprox_unverified_until_heldout; "
           f"stem = stem_from_axis_classifier_unscored\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
           "property short leaf\nproperty uchar stem\nend_header\n")
    (out / "points_tprox.ply").write_bytes(hdr.encode() + rec.tobytes())
    L = ["# 7단계 — T-Prox 복엽 라벨", "", "검증되지 않은 결과입니다. 아래 비율은 **배정 비율**이며 정확도가 아닙니다 (`tprox_unverified_until_heldout`).", "",
         "| 항목 | 값 |", "|---|---|", f"| 서펠 | {n:,}개 |", f"| 복엽 (스켈레톤) | {len(uids)}개 — 선 {len(set(tid.tolist()))}개 사용 / 전체 {len(sk['leaves'])}개 (같은 잎의 선은 한 잎으로 묶음) |",
         f"| 배정된 서펠 | {int(assigned.sum()):,}개 ({assigned.mean():.1%}) |", f"| 씨앗 | 엽축 {sinfo['rachis_seeds']:,} + 잎몸 {sinfo['blade_seeds']:,} (그룹 {sinfo['blade_groups']}개) |",
         f"| 씨앗이 {SEED_MIN}개 미만인 잎 | {', '.join(meta['leaves_seed_below_min']) or '없음'} |",
         f"| 줄기 표시 | {int(stem.sum()):,}개 ({stem.mean():.1%}) — `stem_from_axis_classifier_unscored`, 정확도 미측정 |",
         f"| 그룹 출처 | {', '.join(meta_in.get('labels', [])) or '기록 없음'} |", "",
         "파일: `labels_tprox.npz`(서펠별 잎 번호·출처·줄기 표시), `tprox_meta.json`, `points_tprox.ply`(leaf, stem 속성 포함), `tprox/`(묶음과 고정 코드 출력).", ""]
    (out / "REPORT_TPROX_KO.md").write_text("\n".join(L), encoding="utf-8")
    log(f"tprox: leaves {len(uids)}, assigned share {assigned.mean():.4f} (not accuracy), stem-flagged {int(stem.sum())} [stem_from_axis_classifier_unscored]")
    return meta


DONE = ("labels_tprox.npz", "tprox_meta.json", "points_tprox.ply", "REPORT_TPROX_KO.md")


def attached_recon(a, out: Path, log) -> bool:
    """Second half of the round trip: the groups fit only the reconstruction they were made from, so when `--tprox-inputs` carries
    that reconstruction (surfels_full.ply, sha256 in the meta file) it is used and stages 1-6 are not run again."""
    if (out / "surfels_full.ply").exists():
        return not a.photos                                                   # a finished run folder given again without photos: stage 7 only
    if not a.tprox_inputs or not Path(a.tprox_inputs).is_dir():
        return False
    hit = sorted(Path(a.tprox_inputs).rglob(META_FILE))
    if not hit or not (hit[0].parent / "surfels_full.ply").exists():
        return False
    src = hit[0].parent / "surfels_full.ply"; want = json.load(open(hit[0], encoding="utf-8")).get("surfels_full_sha256")
    if sha(src) != want:
        log(f"--tprox-inputs: 붙어 온 surfels_full.ply 의 sha256 이 {META_FILE} 과 다릅니다. 이 복원은 쓰지 않습니다."); return False
    shutil.copy2(src, out / "surfels_full.ply")
    log(f"--tprox-inputs 에 복원(surfels_full.ply, sha256 {want[:16]})이 붙어 있습니다. 그룹은 이 복원에만 맞으므로 1–6단계는 다시 실행하지 않고 7단계만 실행합니다.")
    return True


def run(a, out: Path, log) -> bool:
    """Called by run_recon.py after stage 6. Returns True when the stage-7 files exist. A Skip never fails stages 1-6."""
    if all((out / f).exists() for f in DONE):
        log("stage 7 tprox: skip (output exists)"); return True
    t = time.time()
    try:
        check_inputs(a, out); log("stage 7 tprox: start"); stage_tprox(a, out, log)
    except Skip as ex:
        log("stage 7 tprox: " + str(ex)); return False
    log(f"stage 7 tprox: done {time.time() - t:.0f} s"); return True
