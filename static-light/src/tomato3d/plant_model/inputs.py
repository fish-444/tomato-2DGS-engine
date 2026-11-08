"""Input locations and loaders (contract section 1). Every path is read-only."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tomato3d.fuse3d import surfels_io as sio

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "p1_plant_model_20261026_01"

R31B_SURFELS = ROOT / "build/p1_recon_r3-1_contam_20261022_01/outputs/R3-1b_surfels_full.ply"
CLEANUP = ROOT / "build/p1_vp_cleanup_20261026_01"
TRACE_DIR = ROOT / "build/p1_compound_2dvote_20261025_01/traces_export/rachis_traces"
TRACE_ORIENT = ROOT / "build/p1_compound_2dvote_20261025_01/traces_oriented.json"
COUNT_CSV = ROOT / "build/p1_compound_userlabel_20261025_01/map_v3_charuco/user_labelled/leaves.csv"
POT_SPEC = ROOT / "data/records/pots/pot_spec.csv"
ALL_OUT = ROOT / "build/p1_recon_all_20261025_01/outputs"
ALL_PHOTOS = ROOT / "build/p1_all_compound_20261026_01"
R31B_POSES = ROOT / "build/p1_charuco_a3_20261021_01/poses.json"
R31B_UNDIST = ROOT / "build/p1_charuco_a3_20261021_01/undist_3024"
R31B_MASKS = ROOT / "build/p1_recon_r3-1_20261022_01/masks_b"
R31B_A5 = ROOT / "build/p1_recon_r3-1_contam_20261022_01/outputs/recon_meta.json"

NINE = ("n_01", "n_02", "n_03", "n_04", "n_05", "n_06", "t2_02", "t2_04", "t2_05")
OPACITY_MIN = 0.5                     # light_organs.OPACITY_MIN
# human-stated trace -> label map (userlabel.HUMAN_TRACE_MAP, userlabel CONTRACT section 9); copied, not imported, so the
# provenance is visible here; test_plant_model checks it equals the source.
HUMAN_TRACE_MAP = {"leaf3_mvnnjt1z": 7, "leaf5_mvnnmmh3": 6, "leaf6_mvnnnds1": 2}
SCALE_TAG = "conditional_on_unvalidated_scale"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_commented_csv(p: Path) -> tuple[list[str], list[dict]]:
    lines = Path(p).read_text(encoding="utf-8").splitlines()
    comments = [l for l in lines if l.startswith("#")]
    return comments, list(csv.DictReader([l for l in lines if not l.startswith("#")]))


@dataclass
class Plant:
    pid: str
    status: str
    surfels_path: Path
    xyz: np.ndarray                    # board metres
    normal: np.ndarray
    tu: np.ndarray
    tv: np.ndarray
    scale: np.ndarray                  # s0, s1 (m)
    rgb: np.ndarray
    rows: np.ndarray                   # index into the surfel file
    axis_kind: np.ndarray | None = None   # 0 blade, 1 stem-like, 2 rachis-like (R3-1b only)
    leaf: np.ndarray | None = None        # leaf id (R3-1b only), -1 none
    photos: dict = field(default_factory=dict)


def _subset(sf: dict, rows: np.ndarray) -> dict:
    return {k: (v[rows] if isinstance(v, np.ndarray) and len(v) == len(sf["xyz"]) else v) for k, v in sf.items()}


def load_r31b() -> Plant:
    sf = sio.read_surfels_full(R31B_SURFELS)
    z = np.load(CLEANUP / "surfel_clusters.npz")
    rows = z["rows"].astype(int); s = _subset(sf, rows)
    _, ids = read_commented_csv(CLEANUP / "surfel_leaf_ids.csv")
    by_row = {int(r["row"]): int(r["leaf_id"]) for r in ids}
    leaf = np.array([by_row.get(int(r), -1) for r in rows], int)
    if not np.array_equal(leaf, z["leaf"].astype(int)):
        raise ValueError("surfel_leaf_ids.csv and surfel_clusters.npz disagree")
    return Plant("R3-1b", "R3-1b_unaccepted", R31B_SURFELS, s["xyz"], s["normal"], s["tu"], s["tv"], s["scale"], s["rgb"], rows,
                 z["axis_kind"].astype(int), leaf,
                 photos=_r31b_photos())


def _r31b_photos() -> dict:
    poses = json.load(open(R31B_POSES, encoding="utf-8"))
    views = [v for v in sorted(poses["views"]) if (R31B_UNDIST / v).exists() and (R31B_MASKS / (Path(v).stem + ".png")).exists()]
    return dict(poses=poses, poses_path=R31B_POSES, views=views, undist=R31B_UNDIST, masks=R31B_MASKS, mask_plant_values=(1, 2), kind="undist")


def load_other(pid: str) -> Plant:
    p = next((ALL_OUT / pid).glob("*_surfels_full.ply"))
    sf = sio.read_surfels_full(p)
    rows = np.nonzero(sf["opacity"] >= OPACITY_MIN)[0]; s = _subset(sf, rows)
    return Plant(pid, "R3-1b-style_unaccepted", p, s["xyz"], s["normal"], s["tu"], s["tv"], s["scale"], s["rgb"], rows, photos=photo_source(pid))


def stage2_poses_path(pid: str) -> Path:
    if pid.startswith("t2_"):
        return ROOT / "build/p1_recon_tomato2_20261023_01/stage2_d7" / pid / "poses.json"
    if pid == "n_06":
        return ROOT / "build/p1_recon_all_20261025_01/stage2_new_n06/n_06/poses.json"
    return ROOT / "build/p1_recon_all_20261025_01/stage2_new" / pid / "poses.json"


def heic_paths(pid: str) -> dict[str, Path]:
    """View name (IMG_xxxx.jpg) -> original HEIC, from the capture census (one folder per individual)."""
    if pid.startswith("t2_"):
        census, top = ROOT / "build/p1_recon_tomato2_20261023_01/census.csv", ROOT / "토마토2"
    else:
        census, top = ROOT / "build/p1_recon_all_20261025_01/census_new/census.csv", ROOT / "새 폴더"
    with open(census, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["individual"] == pid]
    return {Path(r["file"]).stem + ".jpg": top / r["folder"] / r["file"] for r in rows}


def photo_source(pid: str) -> dict:
    """Accepted views with an original photo and a K3a plant mask. Images are decoded from HEIC and undistorted with the
    stage-2 intrinsics (the same operation as kernel_all_points) at load time; nothing is written."""
    pz = json.load(open(stage2_poses_path(pid), encoding="utf-8")); acc = pz["accepted"]; v0 = pz["views"][acc[0]]
    heic = heic_paths(pid); masks = ROOT / "build/p1_recon_all_20261025_01/k3a" / pid / "masks"
    views = [v for v in acc if v in heic and heic[v].exists() and (masks / (Path(v).stem + ".png")).exists()]
    poses = dict(full_size=v0["size"], K_full=v0["K"], dist=v0["dist"], undistorted_res=v0["size"][0], K_undist=v0["K"],
                 views={v: dict(R_cw=pz["views"][v]["R_cw"], t_cw=pz["views"][v]["t_cw"]) for v in acc})
    return dict(poses=poses, poses_path=stage2_poses_path(pid), views=views, heic=heic, masks=masks, mask_plant_values=(1,), kind="heic")


def load_photo(photos: dict, view: str) -> np.ndarray:
    """RGB at the full undistorted resolution for one view."""
    from PIL import Image
    if photos.get("kind") == "heic":
        import cv2
        import pillow_heif
        pillow_heif.register_heif_opener()
        im = np.asarray(Image.open(photos["heic"][view]).convert("RGB"))
        K = np.asarray(photos["poses"]["K_full"], float); d = np.asarray(photos["poses"]["dist"], float)
        return cv2.undistort(im, K, d, None, K)
    return np.asarray(Image.open(Path(photos["undist"]) / view).convert("RGB"))


def load_poses(plant: Plant) -> dict:
    return plant.photos["poses"]


def load_traces() -> list[dict]:
    """The 10 user rachis traces, oriented base -> tip (base = end nearer the pot-centre vertical, traces_oriented.json)."""
    ori = json.load(open(TRACE_ORIENT, encoding="utf-8"))
    flipped = set(ori["flipped"])
    out = []
    for t in ori["traces"]:
        d = json.load(open(TRACE_DIR / f"{t['key']}.json", encoding="utf-8"))
        pts = np.asarray(d["points"], float)
        if t["key"] in flipped:
            pts = pts[::-1]
        out.append(dict(key=t["key"], trace_number=int(d["leaf"]), points_m=pts, split=t["split"], tie=t["key"] in ori["ties"],
                        uncertain=d.get("uncertain")))
    return out


def load_leaves_csv() -> list[dict]:
    return read_commented_csv(CLEANUP / "leaves.csv")[1]


def load_counts() -> dict[int, int]:
    return {int(r["leaf"]): int(r["count_2d_estimate"]) for r in read_commented_csv(COUNT_CSV)[1]}


def load_pot_spec() -> dict:
    with open(POT_SPEC, encoding="utf-8-sig") as f:
        r = next(csv.DictReader(f))
    return dict(pot_type=r["pot_type"], plant_ids=r["plant_ids"], inner_edge_mm=float(r["pot_rim_inner_edge_mm"]),
                medium_depth_mm=float(r["medium_surface_depth_below_rim_mm"]), source_kind=r["source_kind"], note=r["note"])


def input_manifest() -> list[dict]:
    files = [R31B_SURFELS, CLEANUP / "surfel_clusters.npz", CLEANUP / "surfel_leaf_ids.csv", CLEANUP / "leaves.csv", TRACE_ORIENT, COUNT_CSV,
             POT_SPEC, R31B_POSES, R31B_A5] + sorted(TRACE_DIR.glob("*.json"))
    for pid in NINE:
        files.append(next((ALL_OUT / pid).glob("*_surfels_full.ply")))
        files.append(stage2_poses_path(pid))
    return [dict(path=str(p.relative_to(ROOT)).replace("\\", "/"), sha256=sha256(p), bytes=p.stat().st_size) for p in files]
