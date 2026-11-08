"""Prepare a private local editing session; copy verified images, never launch it.

python -m tools.prepare_leaf_edit_session --capture build/.../capture.board_evidence.json
  --annotations build/.../manual_candidates.review03.json --out build/.../edit_session
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote

import numpy as np

from tomato3d.config import get_settings
from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json


BOOTSTRAP_JS = r"""'use strict';
// Only the prepared loopback session is loaded. No model, upload or cloud service runs.
async function loadEditSession({api, config, fetchJson, storage}) {
  const sessionKey = 'leafPointAnnotator:' + config.session_id;
  let hadSavedSession = false;
  try { hadSavedSession = storage.getItem(sessionKey) !== null; } catch (_) { /* App displays its storage warning. */ }
  await api.loadFromUrls(config.image_urls, [config.capture_url]);
  api.ui.variant = '__all__';
  const containsWork = () => Object.values(api.buildBackup().variants || {}).some(state =>
    Object.keys(state.leaflets || {}).length || Object.values(state.boxes || {}).some(list => list.length));
  const current = api.buildBackup();
  if (current.session_id !== config.session_id || current.plant_id !== config.plant_id)
    throw new Error('Prepared session identity differs from the loaded capture');
  if (hadSavedSession || containsWork()) return {status:'preserved', restored_seed:false};
  const seed = await fetchJson(config.seed_url);
  // Check again after fetch so newly entered work cannot be overwritten while loading.
  let savedDuringLoad = false;
  try { savedDuringLoad = storage.getItem(sessionKey) !== null; } catch (_) {}
  if (savedDuringLoad || containsWork()) return {status:'preserved', restored_seed:false};
  if (seed.session_id !== config.session_id || seed.plant_id !== config.plant_id)
    throw new Error('Prepared seed identity mismatch');
  await api.restoreBackup(seed);
  const first = Object.values(api.vstate().leaflets || {})[0];
  if (first && api.selectLeaf) api.selectLeaf(first.organ_id, first.leaflet_id);
  return {status:'seeded', restored_seed:true};
}
if (typeof module !== 'undefined' && module.exports) module.exports = {loadEditSession};
if (typeof window !== 'undefined') window.__leafEditSessionBootstrap = {loadEditSession};
if (typeof document !== 'undefined') {
  const button = document.getElementById('loadPreparedSession');
  const status = document.getElementById('preparedSessionStatus');
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      if (location.protocol === 'file:') throw new Error('먼저 아래 안내대로 로컬 서버를 실행한 뒤 http://127.0.0.1 주소로 여세요.');
      const config = JSON.parse(document.getElementById('preparedSessionConfig').textContent);
      status.textContent = '사진과 보정 정보를 이 컴퓨터에서 불러오는 중…';
      const result = await loadEditSession({api:window.__leafAnnotator, config, storage:localStorage,
        fetchJson:async url => { const response = await fetch(url); if (!response.ok) throw Error('로컬 파일을 불러오지 못했습니다'); return response.json(); }});
      status.textContent = result.restored_seed ? '사진과 기존 후보 라벨을 불러왔습니다. 수정 전후 작업 백업을 저장하세요.' :
        '사진을 불러왔습니다. 이 브라우저에 있던 라벨 작업은 그대로 유지했습니다.';
    } catch (error) { status.textContent = '불러오기 실패: ' + error.message; }
    finally { button.disabled = false; }
  });
}
"""

CLOUD_GUARD = "  if (!window.claude || typeof window.claude.use !== 'function') return;"
LOCAL_GUARD = "  if (true) return; // LOCAL_ONLY_PREPARED_SESSION: cloud initialization is disabled."
CSP = "default-src 'self' data: blob:; connect-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'"


def basename(value: str) -> str:
    return re.split(r"[\\/]", value)[-1]


def verify_path(value: str, expected_hash: str, base: Path) -> Path:
    path = Path(value)
    path = (path if path.is_absolute() else base / path).resolve()
    if not expected_hash or file_hash(path) != expected_hash:
        raise StudyError(f"Source hash mismatch: {path.name}")
    return path


def physical_point_leaf(candidate: dict, names: set[str], intrinsics: dict, provenance: dict) -> dict:
    """Preserve explicitly named sparse landmarks without inventing a polygon ring."""
    if candidate.get("geometry_status") != "unmeshed_physical_point_candidates":
        raise StudyError("Expected explicitly typed unmeshed physical point candidates")
    count = candidate.get("boundary_count")
    if type(count) is not int or not 3 <= count <= 64:
        raise StudyError("Physical point boundary_count must be an integer from 3 to 64")
    pts, semantics, observed_keys = {}, {}, set()
    excluded_ids = set(candidate.get("excluded_original_point_ids", []))
    for observation in candidate.get("observations", []):
        if observation["image_file"] not in names:
            raise StudyError("Unknown physical point observation image")
        key = basename(observation["image_file"]).lower()
        if key in observed_keys:
            raise StudyError("Duplicate physical point observation")
        observed_keys.add(key)
        size = np.asarray(observation["label_image_size_px"], dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise StudyError("Invalid physical point label image size")
        seen_points = set()
        for point in observation.get("points", []):
            pid = point.get("point_id", "")
            if not isinstance(pid, str) or not re.fullmatch(r"[bi]\d+", pid) or pid in seen_points or pid in excluded_ids:
                raise StudyError("Invalid, duplicate or excluded physical point ID")
            if pid.startswith("b") and int(pid[1:]) >= count:
                raise StudyError("Physical point ID exceeds boundary_count")
            seen_points.add(pid)
            xy_px = np.asarray(point["xy_px"], dtype=float)
            if xy_px.shape != (2,) or not np.isfinite(xy_px).all() or np.any(xy_px < 0) or np.any(xy_px >= size):
                raise StudyError("Physical point outside label image")
            name = point.get("semantic_name")
            if name is not None:
                if pid in semantics and semantics[pid] != name:
                    raise StudyError("Physical point semantic name differs between views")
                semantics[pid] = deepcopy(name)
            original_px = xy_px * np.array([intrinsics["image_width_px"], intrinsics["image_height_px"]]) / size
            pts.setdefault(pid, {})[key] = original_px.tolist()
    if not pts:
        raise StudyError("Physical point candidate has no observed landmarks")
    editable = {key: deepcopy(value) for key, value in candidate.items() if key != "observations"}
    editable.update(pts=pts, contours={}, boundary_count=count, point_semantic_names=semantics,
                    outline_completeness="partial_observed_patch", geometry_status="unmeshed_physical_point_candidates",
                    validated_metric=False, annotation_origin="assistant_manual_candidate",
                    source_annotations_sha256=provenance["annotations_sha256"],
                    original_candidate_observations=deepcopy(candidate["observations"]),
                    original_physical_point_candidate=deepcopy(candidate))
    return editable


def build_seed(capture: dict, annotation: dict, provenance: dict, include_partial: bool = False) -> tuple[dict, dict]:
    """Convert physical-point candidates; optionally retain explicitly partial patches."""
    if capture.get("schema_version") != "diagnostic-preview-capture-1" or annotation.get("schema_version") != "manual-leaf-preview-1":
        raise StudyError("Expected diagnostic capture and manual preview annotations")
    intrinsics = capture["intrinsics"]
    names = {frame["image_file"] for frame in capture["frames"]}
    keys = [basename(name).lower() for name in names]
    if len(names) != len(capture["frames"]) or len(set(keys)) != len(keys):
        raise StudyError("Duplicate photo basename")
    leaves, excluded, included_partial, seen = {}, [], [], set()
    excluded_two_d = deepcopy(annotation.get("two_d_only_candidates", []))
    for leaf in annotation["leaflets"]:
        identity = leaf["organ_id"] + "|" + leaf["leaflet_id"]
        if identity in seen or not leaf["organ_id"] or not leaf["leaflet_id"] or "|" in leaf["organ_id"] or "|" in leaf["leaflet_id"]:
            raise StudyError("Duplicate or invalid leaf identity")
        seen.add(identity)
        completeness = str(leaf.get("outline_completeness", ""))
        partial = "partial" in completeness or leaf.get("surface_extent") == "visible_fragment_only"
        # Independent 2D polygon indices are never physical landmark identities,
        # even if a malformed source puts them in the leaflets collection.
        two_d_only = leaf.get("geometry_status") == "2d_only" or leaf.get("correspondence_status") in {
            "not_point_correspondences", "not_a_landmark"
        } or (leaf.get("surface_extent") == "visible_fragment_only" and completeness != "partial_observed_patch")
        if two_d_only:
            excluded_two_d.append(deepcopy(leaf))
            continue
        if partial and (not include_partial or completeness != "partial_observed_patch"):
            reason = "Partial physical-point editing not enabled" if not include_partial else "Unsupported partial outline type; retain outside point editor"
            excluded.append(dict(leaflet_id=leaf["leaflet_id"], reason=reason, source=deepcopy(leaf)))
            continue
        if partial:
            included_partial.append(leaf["leaflet_id"])
        pts, observed_keys, counts = {}, set(), set()
        for observation in leaf["observations"]:
            if observation["image_file"] not in names:
                raise StudyError("Unknown observation image")
            key = basename(observation["image_file"]).lower()
            if key in observed_keys:
                raise StudyError("Duplicate leaf observation")
            observed_keys.add(key)
            label_size = np.asarray(observation["label_image_size_px"], dtype=float)
            polygon = np.asarray(observation["boundary_xy_px"], dtype=float)
            if label_size.shape != (2,) or not np.isfinite(label_size).all() or np.any(label_size <= 0):
                raise StudyError("Invalid label image size")
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
                raise StudyError("Invalid candidate polygon")
            if np.any(polygon < 0) or np.any(polygon >= label_size):
                raise StudyError("Candidate polygon outside image")
            original_px = polygon * np.array([intrinsics["image_width_px"], intrinsics["image_height_px"]]) / label_size
            counts.add(len(original_px))
            for index, xy_px in enumerate(original_px):
                pts.setdefault(f"b{index}", {})[key] = xy_px.tolist()
        if len(counts) != 1:
            raise StudyError("Inconsistent boundary counts or no observations")
        # The editor must retain assistant provenance and uncertainty on every backup.
        editable = {key: deepcopy(value) for key, value in leaf.items() if key != "observations"}
        editable.update(pts=pts, contours={}, boundary_count=next(iter(counts)),
                        validated_metric=False, annotation_origin="assistant_manual_candidate",
                        source_annotations_sha256=provenance["annotations_sha256"],
                        original_candidate_observations=deepcopy(leaf["observations"]))
        leaves[identity] = editable
    included_physical, excluded_physical = [], []
    for candidate in annotation.get("physical_point_candidates", []):
        identity = candidate["organ_id"] + "|" + candidate["leaflet_id"]
        if identity in seen or not candidate["organ_id"] or not candidate["leaflet_id"] or "|" in candidate["organ_id"] or "|" in candidate["leaflet_id"]:
            raise StudyError("Duplicate or invalid physical point identity")
        seen.add(identity)
        if not include_partial:
            excluded_physical.append(deepcopy(candidate))
            continue
        leaves[identity] = physical_point_leaf(candidate, names, intrinsics, provenance)
        included_physical.append(candidate["leaflet_id"])
        included_partial.append(candidate["leaflet_id"])
    seed_metadata = dict(provenance, human_review=annotation.get("human_review", "pending"),
                         annotation_source=annotation.get("annotation_source"), notice=annotation.get("notice"),
                         include_partial=include_partial, included_partial_leaflet_ids=included_partial,
                         included_physical_point_leaflet_ids=included_physical,
                         excluded_physical_point_candidates=excluded_physical,
                         excluded_partial_candidates=excluded,
                         two_d_only_candidates=excluded_two_d,
                         unresolved_observations=deepcopy(annotation.get("unresolved_observations", [])),
                         validated_metric=False)
    seed = dict(plant_id=capture["plant_id"], session_id=capture["session_id"], capture=deepcopy(capture),
                variants={"__all__": dict(leaflets=leaves, boxes={}, seed_provenance=seed_metadata)})
    report = dict(imported_leaflet_ids=[leaf["leaflet_id"] for leaf in leaves.values()],
                  imported_leaflet_count=len(leaves), include_partial=include_partial,
                  included_partial_leaflet_ids=included_partial, included_partial_count=len(included_partial),
                  included_physical_point_leaflet_ids=included_physical,
                  excluded_physical_point_candidates=excluded_physical,
                  excluded_partial_candidates=excluded, excluded_two_d_only_candidates=excluded_two_d)
    return seed, report


def local_html(source: str, config: dict, out: Path) -> str:
    """Fail closed if the expected cloud initialization hook changes upstream."""
    if source.count(CLOUD_GUARD) != 1 or source.count("<header>") != 1 or source.count("</html>") != 1:
        raise StudyError("Unexpected annotator source; local-only guard injection needs review")
    if "async loadFromUrls(imageUrls, poseUrls)" not in source or "buildBackup, restoreBackup" not in source:
        raise StudyError("Required local annotator API is unavailable")
    result = source.replace(CLOUD_GUARD, LOCAL_GUARD)
    result = re.sub(r'<link\b[^>]*href="https://fonts\.googleapis\.com[^>]*>\s*', "", result)
    result = result.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n<meta http-equiv="Content-Security-Policy" content="' + CSP + '">', 1)
    server_command = f'python -m http.server 8821 --bind 127.0.0.1 --directory "{out.resolve()}"'
    from html import escape
    partial_notice = (f'이 중 {config.get("partial_leaflet_count", 0)}개 부분 잎의 대응점도 수정할 수 있습니다. '
                      '부분 표면은 완전한 잎 윤곽으로 취급하지 않습니다. 독립적인 2D 잎 조각의 꼭짓점은 편집 대응점에 넣지 않았습니다.'
                      if config.get("include_partial", False) else
                      '부분 윤곽과 2D 잎 조각은 초기 편집 대상에서 제외했습니다.')
    banner = '<section style="padding:8px;border:1px solid #735438;margin:6px 0">' + \
        f'<button id="loadPreparedSession" class="primary">기존 {config["photo_count"]}장 사진·후보 라벨 불러오기</button> ' + \
        '<span id="preparedSessionStatus">이 컴퓨터에서만 작업합니다. 기존 수정 내용이 있으면 초기 라벨로 덮어쓰지 않습니다.</span>' + \
        f'<div class="small">{config["leaflet_count"]}개 소엽 후보를 수정할 수 있습니다. {partial_notice} 모두 사람 검토 전 후보이며 깊이는 미검증입니다.</div>' + \
        '<details><summary>여는 방법</summary><code>' + escape(server_command) + '</code>' + \
        '<p>명령 실행 후 <a href="http://127.0.0.1:8821/index.html">로컬 편집 화면</a>을 여세요. 자동 실행하거나 업로드하지 않습니다.</p></details></section>\n'
    result = result.replace("<header>", banner + "<header>", 1)
    payload = json.dumps(config, ensure_ascii=False).replace("<", "\\u003c")
    result = result.replace("</html>", '<script id="preparedSessionConfig" type="application/json">' + payload +
                            '</script>\n<script src="bootstrap.js"></script>\n</html>')
    if LOCAL_GUARD not in result or CSP not in result or "fonts.googleapis.com" in result:
        raise StudyError("Local-only source guard verification failed")
    return result


def prepare(capture_path: Path, annotations_path: Path, out: Path, include_partial: bool = False) -> dict:
    capture_path, annotations_path = capture_path.resolve(), annotations_path.resolve()
    capture, annotation = read_json(capture_path), read_json(annotations_path)
    calibration_path = verify_path(capture["intrinsics_path"], capture["intrinsics_sha256"], capture_path.parent)
    if read_json(calibration_path) != capture["intrinsics"]:
        raise StudyError("Embedded intrinsics differ from the hashed calibration")
    board_path = get_settings().board_spec_path.resolve()
    board_hash = file_hash(board_path)
    if capture.get("board_spec_sha256") != board_hash or read_json(board_path)["variant_id"] != capture["board_variant_id"]:
        raise StudyError("Enriched capture board specification mismatch")
    paths = {}
    for frame in capture["frames"]:
        name = basename(frame["image_file"])
        if name in paths or Path(name).suffix.lower() != ".png":
            raise StudyError("Expected unique working PNG photographs")
        paths[name] = verify_path(frame["image_path"], frame["sha256"], capture_path.parent)
        if frame.get("board_evidence", {}).get("source_image_sha256") != frame["sha256"]:
            raise StudyError("Board evidence photo hash mismatch")
    source_html_path = Path(__file__).with_name("leaf_point_annotator.html")
    source_html = source_html_path.read_text(encoding="utf-8")
    source_html_hash = file_hash(source_html_path)
    provenance = dict(capture_source_path=str(capture_path), capture_source_sha256=file_hash(capture_path),
                      annotations_source_path=str(annotations_path), annotations_sha256=file_hash(annotations_path),
                      annotator_source_path=str(source_html_path), annotator_source_sha256=source_html_hash,
                      calibration_sha256=capture["intrinsics_sha256"], board_spec_sha256=board_hash)
    copied_capture = deepcopy(capture)
    copied_capture.update(intrinsics_path="intrinsics.json", board_spec_path="board_spec.snapshot.json", validated_metric=False)
    for frame in copied_capture["frames"]:
        frame["image_path"] = "images/" + basename(frame["image_file"])
    seed, report = build_seed(copied_capture, annotation, provenance, include_partial=include_partial)
    config = dict(image_urls=["images/" + quote(basename(frame["image_file"])) for frame in copied_capture["frames"]],
                  capture_url="capture.json", seed_url="seed.backup.json", plant_id=capture["plant_id"],
                  session_id=re.sub(r"[^A-Za-z0-9_\-.~:@+]", "_", capture["session_id"] or "none")[:90],
                  photo_count=len(paths), leaflet_count=report["imported_leaflet_count"],
                  include_partial=include_partial, partial_leaflet_count=report["included_partial_count"])
    seed["session_id"] = config["session_id"]
    html = local_html(source_html, config, out)
    folder = output_dir(out)
    (folder / "images").mkdir()
    expected_hashes = {basename(frame["image_file"]): frame["sha256"] for frame in capture["frames"]}
    for name, source in paths.items():
        shutil.copyfile(source, folder / "images" / name)
        if file_hash(folder / "images" / name) != expected_hashes[name]:
            raise StudyError("Copied photograph hash mismatch")
    shutil.copyfile(calibration_path, folder / "intrinsics.json")
    shutil.copyfile(board_path, folder / "board_spec.snapshot.json")
    shutil.copyfile(annotations_path, folder / "annotations.source.json")
    if (file_hash(folder / "intrinsics.json") != capture["intrinsics_sha256"] or
            file_hash(folder / "board_spec.snapshot.json") != board_hash or
            file_hash(folder / "annotations.source.json") != provenance["annotations_sha256"]):
        raise StudyError("Copied input hash mismatch")
    write_json(folder / "capture.json", copied_capture)
    write_json(folder / "seed.backup.json", seed)
    (folder / "bootstrap.js").write_text(BOOTSTRAP_JS, encoding="utf-8")
    (folder / "index.html").write_text(html, encoding="utf-8")
    report.update(schema_version="prepared-local-leaf-edit-session-1", **provenance,
                  photo_count=len(paths), local_only=True, launched=False, uploaded=False, deep_learning_run=False,
                  validated_metric=False, source_photos_copied_without_reencoding=True,
                  copied_photo_sha256={name: file_hash(folder / "images" / name) for name in paths},
                  seed_sha256=file_hash(folder / "seed.backup.json"), capture_sha256=file_hash(folder / "capture.json"),
                  html_sha256=file_hash(folder / "index.html"),
                  startup_policy="Explicit button; load photos/capture first; never restore seed over existing leaf/box work or any persisted session record",
                  serve_command=f'python -m http.server 8821 --bind 127.0.0.1 --directory "{folder}"')
    write_json(folder / "session_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "annotations", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--include-partial", action="store_true", help="Include partial physical-point candidates; independent 2D polygons remain excluded.")
    args = parser.parse_args()
    result = prepare(args.capture, args.annotations, args.out, include_partial=args.include_partial)
    print(f"Prepared {result['photo_count']} photos and {result['imported_leaflet_count']} leaf candidates; not launched")
    print(result["serve_command"])


if __name__ == "__main__":
    main()
