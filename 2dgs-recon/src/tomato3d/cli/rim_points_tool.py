"""Build the rim-point clicking page for the real top or side photos (P1 T6 / side rim pose human record helper).

    python -X utf8 -m tomato3d.cli.rim_points_tool --package build/p1_state_<YYYYMMDD>_<nn> --out build/rim_click_<YYYYMMDD>_<nn> [--views top|side]

The page (open ``index.html`` in a browser) shows each real photo of the chosen view; a person clicks points on the
flat outer rim edges (and optionally ruler ticks) and saves ``rim_points.csv`` (top) or ``rim_points_side.csv`` (side)
in exactly the columns of :data:`tomato3d.state.potscale.RIM_POINTS_COLUMNS`, for ``data/records/pots/``. Each side
records WHICH line its points lie on (``edge``: outer / inner top edge, or the medium junction 10 mm lower) and a
hidden side can be recorded as ``side_occluded`` (2026-10-13). The page's
numbers are a preview only (``rim_core.js`` mirrors ``potscale.py``); the record's values come from the P1 run.

Side views (about 25 deg from above): the rim is a flattened square; the upright ruler's ticks are recorded, but a
single photo cannot test a vertical scale with them (depth-length ambiguity, see ``tomato3d.state.camera``).

The photos shown are the decoded JPG copies of the raw HEIC files. Their pixel grid must be the HEIC's: the builder
refuses a photo whose raw file no longer has the package's sha256, whose HEIC primary size (``ispe``) differs from
the JPG size, or whose HEIC (``irot``) or JPG (EXIF) carries a rotation. Pixel content itself cannot be compared here
(no HEIC decoder in this environment) - stated in the page.

Exit codes: 0 written; 2 input error (missing/changed photo, size or rotation mismatch, existing output folder).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image

from tomato3d.io.tables import read_table
from tomato3d.state.camera import heic_geometry
from tomato3d.state.potscale import RIM_POINTS_COLUMNS

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
VIEWS = ("top", "side")
DEFAULT_IMAGES = {"top": Path("build/roboflow_upload_20261009/top_level_views"),
                  "side": Path("build/roboflow_upload_20261009/side_views")}
OUTPUT_NAME = {"top": "rim_points.csv", "side": "rim_points_side.csv"}
EXIT_OK, EXIT_INPUT = 0, 2


class ToolInputError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jpg_name(capture: dict, view: str) -> str:
    """``dwt_101_top.jpg`` / ``dwt_101_side_090.jpg`` (from the raw ``side_090.heic``)."""
    if view == "top":
        return f"{capture['source_plant_id']}_top.jpg"
    return f"{capture['source_plant_id']}_{Path(capture['image_path']).stem}.jpg"


def manifest(package: Path, images: Path, out: Path, view: str = "top", stem: str | None = None) -> dict:
    if view not in VIEWS:
        raise ToolInputError(f"--views {view!r} not in {VIEWS}")
    caps = [r for r in read_table(Path(package) / "captures.csv") if r["view"] == view and r["evidence"] == "real_photo"
            and (stem is None or Path(r["image_path"]).stem == stem)]
    if not caps:
        raise ToolInputError(f"{package}: no real {view} captures" + (f" with photo {stem}" if stem else ""))
    photos, problems = [], []
    for c in sorted(caps, key=lambda r: (r["plant_instance_id"], r["capture_id"])):
        raw = PROJECT_ROOT / c["image_path"]
        jpg = Path(images) / jpg_name(c, view)
        if not raw.exists() or not jpg.exists():
            problems.append(f"{c['capture_id']}: missing {raw if not raw.exists() else jpg}")
            continue
        if _sha(raw) != c["image_sha256"]:
            problems.append(f"{c['capture_id']}: raw photo changed since the package (sha256)")
            continue
        size, rots = heic_geometry(raw)
        with Image.open(jpg) as im:
            jsize, orient = im.size, im.getexif().get(0x0112)
        if size != jsize:
            problems.append(f"{c['capture_id']}: HEIC size {size} != JPG size {jsize} - different pixel grids")
        if any(rots) or orient not in (None, 1):
            problems.append(f"{c['capture_id']}: rotation present (HEIC irot {rots}, JPG EXIF {orient})")
        photos.append(dict(plant_id=c["source_plant_id"], capture_id=c["capture_id"], image_sha256=c["image_sha256"],
                           label=f"{c['source_plant_id']} {Path(c['image_path']).stem}",
                           jpg=Path(os.path.relpath(jpg.resolve(), out.resolve())).as_posix(), jpg_sha256=_sha(jpg),
                           size_px=list(jsize)))
    if problems:
        raise ToolInputError("; ".join(problems))
    kind = "위" if view == "top" else "옆(약 25°)"
    prov = (f"사진 {len(photos)}장: {Path(package).as_posix()}/captures.csv의 실사진 {kind} 촬영. 화면의 사진은 원본 HEIC를 "
            "디코딩한 JPG 사본이며, HEIC 크기(ispe)와 같고 회전이 없음을 확인함. 픽셀 내용은 이 환경에 HEIC 디코더가 "
            "없어 대조하지 못함. 기록의 image_sha256은 원본 HEIC 값.")
    name = OUTPUT_NAME[view] if stem is None or view == "top" else f"rim_points_{stem}.csv"
    return dict(schema="rim-points-tool-2", view=view, output_name=name, package=Path(package).as_posix(),
                columns=list(RIM_POINTS_COLUMNS), photos=photos, provenance=prov)


def _prefill(path: Path | None) -> dict[str, list[dict]]:
    """Automatic proposals (``tomato3d.state.rim_auto``) per photo sha256, for the page to pre-fill."""
    if path is None:
        return {}
    from tomato3d.state.potscale import read_rim_points

    out: dict[str, list[dict]] = {}
    for (_, sha), pts in read_rim_points(Path(path)).items():
        out[sha] = [dict(kind=p.kind, side=p.side, x=p.xy[0], y=p.xy[1], edge=p.edge) for p in pts
                    if p.kind == "rim_edge"] + [dict(kind="side_occluded", side=p.side) for p in pts
                                                if p.kind == "side_occluded"]
    return out


def build(package: Path, out: Path, images: Path | None = None, view: str = "top", stem: str | None = None,
          prefill: Path | None = None) -> Path:
    out = Path(out)
    if out.exists():
        raise ToolInputError(f"{out} exists; outputs are never overwritten")
    doc = manifest(package, images or DEFAULT_IMAGES.get(view, DEFAULT_IMAGES["top"]), out, view, stem)
    pre = _prefill(prefill)
    for ph in doc["photos"]:
        ph["prefill"] = [q for q in pre.get(ph["image_sha256"], []) if q["kind"] == "rim_edge"]
        ph["prefill_occluded"] = [q["side"] for q in pre.get(ph["image_sha256"], []) if q["kind"] == "side_occluded"]
    doc["prefill_source"] = Path(prefill).as_posix() if prefill else None
    page = (HERE / "rim_points_tool.html").read_text(encoding="utf-8")
    core = (HERE / "rim_core.js").read_text(encoding="utf-8")
    # Self-contained page: the photos (original JPG bytes, so the same pixel grid) and the preview code are
    # embedded, so it also works where neighbouring files are not loaded (e.g. a page opened as a saved snapshot).
    embedded = dict(doc, photos=[
        dict(ph, jpg="data:image/jpeg;base64," + base64.b64encode((out / ph["jpg"]).resolve().read_bytes()).decode("ascii"))
        for ph in doc["photos"]])
    data = json.dumps(embedded, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace('<script src="rim_core.js"></script>', "<script>\n" + core + "\n</script>")
    out.mkdir(parents=True)
    (out / "index.html").write_text(page.replace("__MANIFEST__", data), encoding="utf-8")
    (out / "rim_core.js").write_text((HERE / "rim_core.js").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return out / "index.html"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rim_points_tool", description=__doc__.splitlines()[0])
    ap.add_argument("--package", type=Path, required=True, help="a real P1 package (its captures.csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", choices=VIEWS, default="top", help="top (default) or side photos")
    ap.add_argument("--images", type=Path, default=None, help="folder of the JPG copies (default per view)")
    ap.add_argument("--prefill", type=Path, default=None,
                    help="CSV of automatic proposals (tomato3d.state.rim_auto) shown as hollow squares to review")
    ap.add_argument("--stem", default=None, help="only photos with this raw name, e.g. side_090 (smaller pages; the "
                                                 "page then saves rim_points_side_090.csv)")
    args = ap.parse_args(argv)
    try:
        page = build(args.package, args.out, args.images, args.views, args.stem, args.prefill)
    except ToolInputError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_INPUT
    print(f"wrote {page}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
