"""Link a real top photo to its capture record, and the camera distance from its rim (X1/X2 of
``handoff/CLAUDE_CODE_CLI_PROMPT_TOPVIEW_REAL_LINK_2026-10-12.md``).

**Link (X1).** Rim records (``data/records/pots/rim_points.csv``) are keyed by ``plant_id`` and the sha256 of the RAW
HEIC, and their pixel coordinates live on that HEIC's decoded grid. A measurement may carry mm only when the photo it
was made on is shown to share that grid. The only photos accepted are the P1 package's decoded top copies
(``build/roboflow_upload_20261009/top_level_views/<plant>_top.jpg``) after the same refusals the rim-point tool
makes: the raw file still has the package's sha256, the HEIC primary size (``ispe``) equals the JPG size, and
neither carries a rotation (HEIC ``irot``, JPG EXIF orientation). Any other file - e.g. a Roboflow download of the
same photo - is refused: equal size is not the same grid, and re-encoded pixels have no hash link to the record.

**Camera distance (X2).** The rim homography (image px -> rim-plane mm) and a pinhole focal length give the pose of
the rim plane in the camera: ``K^-1 H^-1 ~ [r1 r2 t]``. Stored per photo: depth of the pot centre along the optical
axis ``d`` (mm), camera height above the rim plane, tilt of the optical axis from the rim normal, and the two-axis
self-check ``|r1| / |r2|`` (1 for an exact pinhole). The principal point is taken at the image centre because the
3024 x 3024 photo is a crop of the 4032 x 3024 frame at an unknown offset, and the focal length has two candidate
bases (35 mm-equivalent by frame width or by diagonal) - so every pose is ``approximate``. The P1 session owns the
pose table (Y2/Y4 of its prompt); when it is published it supersedes this one.

Bias of the rim-plane scale for a leaf ``h`` above the rim near the pot centre (OVERestimate, never corrected
here): length factor ``d / (d - h cos(tilt))``, area factor its square.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

UNAVAILABLE = "unavailable"
DEFAULT_IMAGES = Path("build/roboflow_upload_20261009/top_level_views")
POSE_STATUS = "approximate"
FILM_35MM_WIDTH_MM, FILM_35MM_HEIGHT_MM = 36.0, 24.0   # the 35 mm film frame that defines "35 mm-equivalent"


class GridRefusal(ValueError):
    """The photo cannot be shown to share the pixel grid of its capture record - no mm for it."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def top_captures(package: Path) -> list[dict[str, str]]:
    from tomato3d.io.tables import read_table

    rows = [r for r in read_table(Path(package) / "captures.csv") if r["view"] == "top" and r["evidence"] == "real_photo"]
    return sorted(rows, key=lambda r: r["source_plant_id"])


def link_top_capture(capture: dict[str, str], project_root: Path, images: Path = DEFAULT_IMAGES) -> dict[str, Any]:
    """The checked link (plant_id, raw sha256, decoded copy) of one real top capture, or :class:`GridRefusal`."""
    from PIL import Image

    from tomato3d.cli.rim_points_tool import heic_geometry

    plant = capture["source_plant_id"]
    raw = Path(project_root) / capture["image_path"]
    jpg = Path(project_root) / images / f"{plant}_top.jpg"
    problems = []
    if not raw.exists():
        problems.append(f"raw photo missing: {capture['image_path']}")
    if not jpg.exists():
        problems.append(f"decoded copy missing: {(Path(images) / jpg.name).as_posix()}")
    if problems:
        raise GridRefusal(f"{capture['capture_id']}: " + "; ".join(problems))
    raw_sha = sha256_file(raw)
    if raw_sha != capture["image_sha256"]:
        raise GridRefusal(f"{capture['capture_id']}: raw photo sha256 {raw_sha[:12]} != captures.csv "
                          f"{capture['image_sha256'][:12]} (changed since the package)")
    size, rots = heic_geometry(raw)
    with Image.open(jpg) as im:
        jsize, orient = im.size, im.getexif().get(0x0112)
    if size != jsize:
        problems.append(f"HEIC primary size {size} != decoded copy size {jsize} - different pixel grids")
    if any(rots) or orient not in (None, 1):
        problems.append(f"rotation present (HEIC irot {rots}, JPG EXIF orientation {orient})")
    if problems:
        raise GridRefusal(f"{capture['capture_id']}: " + "; ".join(problems))
    return dict(plant_id=plant, capture_id=capture["capture_id"], image_sha256=raw_sha,
                raw_path=capture["image_path"], image_path=(Path(images) / jpg.name).as_posix(),
                image_file_sha256=sha256_file(jpg), size_px=list(jsize),
                grid_check="raw sha256 == captures.csv; HEIC ispe == JPG size; no irot / EXIF rotation "
                           "(pixel content not compared: no HEIC decoder in the main environment)")


def refuse_unlinked(image_path: Path, captures: list[dict[str, str]], project_root: Path,
                    images: Path = DEFAULT_IMAGES) -> GridRefusal:
    """Why a photo that is not a package copy gets no mm (returned, not raised, so a caller can report it)."""
    sha = sha256_file(image_path)
    known = {c["image_sha256"] for c in captures}
    copies = {sha256_file(Path(project_root) / images / f"{c['source_plant_id']}_top.jpg") for c in captures
              if (Path(project_root) / images / f"{c['source_plant_id']}_top.jpg").exists()}
    if sha in known or sha in copies:
        return GridRefusal(f"{Path(image_path).name}: is a package photo - link it through its capture instead")
    return GridRefusal(f"{Path(image_path).name} (sha256 {sha[:12]}) is neither a raw capture nor a checked package "
                       "copy: its pixel grid cannot be shown identical to the rim record's, so no plant_id and no mm")


# --------------------------------------------------------------------------- focal length and pose
def focal_px_bases(f35_mm: float, frame_w: int, frame_h: int) -> dict[str, float]:
    """Pixel focal length of the full sensor frame from the 35 mm-equivalent focal, on two bases."""
    return dict(
        frame_width=f35_mm / FILM_35MM_WIDTH_MM * frame_w,
        frame_diagonal=f35_mm / math.hypot(FILM_35MM_WIDTH_MM, FILM_35MM_HEIGHT_MM) * math.hypot(frame_w, frame_h))


def pose_from_rim(h_img_to_mm: np.ndarray, f_px: float, principal_px: tuple[float, float]) -> dict[str, float]:
    """Rim-plane pose in the camera from the homography (image px -> rim-plane mm) and a focal length."""
    g = np.linalg.inv(np.asarray(h_img_to_mm, float))          # rim-plane mm -> image px
    k_inv = np.linalg.inv(np.array([[f_px, 0.0, principal_px[0]], [0.0, f_px, principal_px[1]], [0.0, 0.0, 1.0]]))
    m = k_inv @ g
    if m[2, 2] < 0:                                             # the pot centre is in front of the camera
        m = -m
    na, nb = np.linalg.norm(m[:, 0]), np.linalg.norm(m[:, 1])
    lam = 2.0 / (na + nb)
    r1, r2, t = lam * m[:, 0], lam * m[:, 1], lam * m[:, 2]
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)
    cos_tilt = abs(float(r3[2]))
    return dict(pot_centre_depth_mm=float(t[2]), pot_centre_distance_mm=float(np.linalg.norm(t)),
                camera_height_above_rim_mm=abs(float(r3 @ t)), tilt_from_vertical_deg=math.degrees(math.acos(
                    min(1.0, cos_tilt))), cos_tilt=cos_tilt, axis_ratio_self_check=float(na / nb),
                axis_orthogonality_cos=float(m[:, 0] @ m[:, 1] / (na * nb)))


def bias_factor(depth_mm: float, cos_tilt: float, leaf_height_mm: float) -> float:
    """Rim-plane length OVERestimate for a leaf ``leaf_height_mm`` above the rim (illustration only)."""
    return depth_mm / (depth_mm - leaf_height_mm * cos_tilt)


def read_focal(raw_heic: Path, heic_python: str) -> dict[str, Any]:
    """EXIF focal (mm, 35 mm-equivalent) and the full frame size, read from the RAW photo by an interpreter that has
    ``pillow_heif`` (the tomato_synth env; the main environment cannot open HEIC)."""
    import json
    import subprocess

    code = ("import sys,json\nfrom PIL import Image\nimport pillow_heif\npillow_heif.register_heif_opener()\n"
            "im=Image.open(sys.argv[1]);ex=im.getexif();s=ex.get_ifd(0x8769)\n"
            "print(json.dumps(dict(focal_mm=float(s.get(0x920A) or 0) or None,f35_mm=s.get(0xA405),"
            "frame_w=s.get(0xA002),frame_h=s.get(0xA003),stored=list(im.size),model=ex.get(0x0110))))")
    out = subprocess.run([heic_python, "-I", "-c", code, str(raw_heic)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def photo_pose(link: dict[str, Any], scale: Any, exif: dict[str, Any]) -> dict[str, Any]:
    """Both focal bases' poses for one linked photo with a solved rim scale; ``unavailable`` otherwise."""
    if scale.status != "solved":
        return dict(status=UNAVAILABLE, reason=f"rim scale {scale.status}: {scale.reason}")
    if not exif.get("f35_mm") or not exif.get("frame_w") or not exif.get("frame_h"):
        return dict(status=UNAVAILABLE, reason="EXIF 35 mm-equivalent focal or frame size missing")
    w, h = link["size_px"]
    centre = (w / 2.0, h / 2.0)
    bases = focal_px_bases(float(exif["f35_mm"]), int(exif["frame_w"]), int(exif["frame_h"]))
    out = dict(status=POSE_STATUS, principal_point_px=list(centre),
               principal_point_note=f"image centre assumed: the stored {w}x{h} photo is a crop of the "
                                    f"{exif['frame_w']}x{exif['frame_h']} frame at an unknown offset",
               exif=dict(focal_mm=exif.get("focal_mm"), f35_mm=exif["f35_mm"], frame_px=[exif["frame_w"],
                                                                                         exif["frame_h"]]),
               bases={})
    for name, f in bases.items():
        out["bases"][name] = dict(f_px=round(f, 2), **{k: round(v, 4) for k, v in
                                                      pose_from_rim(np.asarray(scale.homography), f, centre).items()})
    return out
