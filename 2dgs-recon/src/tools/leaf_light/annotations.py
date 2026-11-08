"""CVAT image XML -> explicit CSV points and leaf contours; no automatic identities."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from .common import StudyError, file_hash, output_dir, read_json, write_json, write_rows

POINT_COLUMNS = ("image_file", "organ_id", "leaflet_id", "point_id", "x_px", "y_px",
                 "boundary_index", "boundary_count")
CONTOUR_COLUMNS = ("image_file", "organ_id", "leaflet_id", "vertex_index", "x_px", "y_px")


def import_cvat(xml_path: Path, capture_path: Path, out: Path, variant_id: str | None = None) -> None:
    content = xml_path.read_text(encoding="utf-8-sig")
    if "<!DOCTYPE" in content or "<!ENTITY" in content:
        raise StudyError("XML entities are not accepted")
    capture = read_json(capture_path)
    names = {f["image_file"] for f in capture["frames"]}
    if variant_id is not None:
        matches = [v for v in capture["variants"] if v["variant_id"] == variant_id]
        if len(matches) != 1:
            raise StudyError("Unknown annotation variant")
        names = set(matches[0]["image_files"])
    points, contours, seen = [], [], set()
    for image in ET.fromstring(content).findall("image"):
        name = image.attrib["name"]
        if name not in names:
            matches = [n for n in names if Path(n).name == name]
            if len(matches) != 1:
                raise StudyError(f"CVAT image cannot be uniquely mapped: {name}")
            name = matches[0]
        for shape in image:
            if shape.tag not in ("points", "polygon") or shape.attrib.get("occluded", "0") == "1":
                continue
            attrs = {node.attrib["name"]: node.text for node in shape.findall("attribute")}
            if not attrs.get("organ_id") or not attrs.get("leaflet_id"):
                raise StudyError("Every shape needs organ_id and leaflet_id attributes")
            xy = [tuple(map(float, pair.split(","))) for pair in shape.attrib["points"].split(";")]
            common = dict(image_file=name, organ_id=attrs["organ_id"], leaflet_id=attrs["leaflet_id"])
            if shape.tag == "points":
                if len(xy) != 1 or any(k not in attrs for k in ("point_id", "boundary_index", "boundary_count")):
                    raise StudyError("Each point shape needs one point and point_id/boundary_index/boundary_count")
                key = (name, attrs["organ_id"], attrs["leaflet_id"], attrs["point_id"])
                if key in seen:
                    raise StudyError(f"Duplicate keypoint {key}")
                seen.add(key)
                points.append(dict(common, point_id=attrs["point_id"], x_px=xy[0][0], y_px=xy[0][1],
                                   boundary_index=int(attrs["boundary_index"]), boundary_count=int(attrs["boundary_count"])))
            else:
                if len(xy) < 3:
                    raise StudyError("A contour requires at least three vertices")
                key = (name, attrs["organ_id"], attrs["leaflet_id"], "polygon")
                if key in seen:
                    raise StudyError(f"Duplicate contour {key}")
                seen.add(key)
                contours.extend(dict(common, vertex_index=i, x_px=x, y_px=y) for i, (x, y) in enumerate(xy))
    folder = output_dir(out)
    write_rows(folder / "points.csv", points, POINT_COLUMNS)
    write_rows(folder / "contours.csv", contours, CONTOUR_COLUMNS)
    write_json(folder / "provenance.json", {"cvat_sha256": file_hash(xml_path),
               "variant_id": variant_id, "allowed_image_files": sorted(names),
               "capture_sha256": file_hash(capture_path),
               "notice": "Use only visible observations. Manual cross-view identities must not depend on excluded views."})
