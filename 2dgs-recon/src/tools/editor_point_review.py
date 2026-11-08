"""Render saved editor observations on their source photos without inventing contours.

The caller supplies actual saved observations, not projected positions. Output stays
under an existing build directory. This module does not estimate depth or review labels.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.leaf_light import common
from tools.leaf_light.reconstruction import project


_BOUNDARY_COLOR = "#ffb347"
_INTERIOR_COLOR = "#ff69dc"
_BOX_USED_COLOR = "#4fd1ff"
_BOX_EXCLUDED_COLOR = "#b9c4bd"
_VISIBILITY = {'full': '전체 보임', 'partial': '일부 가려짐', 'unknown': '보임 정도 모름'}
_REVIEW = {'confirmed': '연결 확인', 'unconfirmed': '연결 미확인'}
_EXCLUSION = {
    'invalid_bbox_geometry': '박스 좌표 오류',
    'bbox_image_size_differs_from_calibrated_original': '원본 해상도가 아닌 박스',
    'camera_pose_unavailable': '촬영 위치 미산출',
    'visibility_partial_not_a_full_silhouette': '일부 가려짐 · 전체 범위 제한에 쓰지 않음',
    'visibility_unknown_not_a_full_silhouette': '보임 정도 모름 · 범위 제한에 쓰지 않음',
    'identity_link_unconfirmed': '연결 미확인 · 형상 계산에 쓰지 않음',
}
_DISPLAY = {
    'boundary': '가장자리', 'interior': '잎맥·내부',
    'experimental_unvalidated': '미검증 시험값',
    'fewer_than_two_solved_views': '대응 사진 부족',
    'context_physical_correspondence_unresolved': '같은 점인지 재검토 필요',
    'unchanged_seed_points_superseded_by_context': '최근 검토에서 바뀐 이전 점',
    'camera_pose_unavailable': '촬영 위치 미산출',
    'triangulated_estimate': '삼각측량 추정값', 'image_observation_only': '사진 관찰만',
    'True': '예', 'False': '아니요 · 불일치 (점은 보존)',
}
_COLUMNS = (
    ("organ_id", "잎 묶음 ID"), ("leaflet_id", "소엽 ID"),
    ("point_id", "대응점 ID"), ("point_kind", "종류"), ("point_semantic_name", "점 의미"),
    ("x_px", "x_px"), ("y_px", "y_px"),
    ("z_board_mm", "보드 위 높이 (mm)"), ("camera_axis_z_mm", "카메라 방향 깊이 (mm)"),
    ("depth_status", "깊이 상태"), ("estimate_kind", "값 종류"),
    ("inside_feasible_region", "소엽 가능 공간 안"), ("observation_inside_leaflet_bbox", "이 사진 소엽 박스 안"),
    ("detected_marker_ids", "검출 marker IDs"),
    ("detected_corner_ids", "검출 ChArUco corner IDs"),
    ("context_depth_restriction", "깊이 제한 / 표시 상태"),
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _text(value: Any) -> str:
    if value is None or value == "":
        return "unavailable"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value) if value else "none_detected"
    if isinstance(value, float):
        return f"{value:.3f}" if math.isfinite(value) else "unavailable"
    return str(value)


def _key(value: Any) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _saved_xy(row: dict) -> tuple[float, float] | None:
    # Do not substitute projections when an observed coordinate is unavailable.
    kind = str(row.get("observation_kind", "")).casefold()
    if kind.startswith("project") or kind.startswith("reproject"):
        return None
    if "observed_x_px" in row or "observed_y_px" in row:
        x_px, y_px = _number(row.get("observed_x_px")), _number(row.get("observed_y_px"))
    else:
        x_px, y_px = _number(row.get("x_px")), _number(row.get("y_px"))
    return (x_px, y_px) if x_px is not None and y_px is not None else None


def _point_kind(row: dict) -> str:
    explicit = row.get("point_kind")
    if explicit in ("boundary", "interior"):
        return str(explicit)
    index = _number(row.get("boundary_index"))
    if index is not None:
        return "boundary" if index >= 0 else "interior"
    return "interior" if str(row.get("point_id", "")).startswith("i") else "boundary"


def point_scene_payload(capture: dict, rows: list[dict]) -> list[dict]:
    """Keep distinct physical landmarks, with no surface or missing-point inference."""
    photo_indices = {_key(f['image_file']): i for i, f in enumerate(capture.get('frames', []))}
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row.get(k, '')) for k in ('organ_id', 'leaflet_id', 'point_id')), []).append(row)
    points = []
    for identity, observations in grouped.items():
        coordinates = [tuple(_number(r.get(k)) for k in ('x_board_mm', 'y_board_mm', 'z_board_mm')) for r in observations]
        if not coordinates or any(None in xyz for xyz in coordinates):
            continue
        # Conflicting coordinates for one identity must not be averaged into a new point.
        if any(xyz != coordinates[0] for xyz in coordinates[1:]):
            continue
        photos = sorted({photo_indices[_key(r.get('image_file'))] for r in observations
                         if _key(r.get('image_file')) in photo_indices and _saved_xy(r) is not None})
        if not photos:
            continue
        points.append(dict(organ_id=identity[0], leaflet_id=identity[1], point_id=identity[2],
                           xyz_mm=coordinates[0], photo_indices=photos, validated_metric=False,
                           viewer_leaflet_id=observations[0].get('viewer_leaflet_id', identity[1]),
                           point_semantic_name=observations[0].get('point_semantic_name'),
                           inside_feasible_region=observations[0].get('inside_feasible_region'),
                           estimate_kind='triangulated_estimate'))
    return points


def leaflet_scene_payload(capture: dict, leaflets: list[dict], bbox_rows: list[dict], depth_rows: list[dict],
                          surfaces: dict | None = None) -> list[dict]:
    """Leaflet-centric view data: bbox extents stay separate from point extents."""
    frames = capture.get('frames', [])
    photo_indices = {_key(f['image_file']): i for i, f in enumerate(frames)}
    result = []
    for leaf in leaflets:
        vid = leaf['viewer_leaflet_id']
        boxes = [dict(photo_index=photo_indices.get(_key(r['image_file'])), image_file=r['image_file'],
                      visibility=r['visibility'], identity_review=r['identity_review'],
                      used_for_extent=r['used_for_extent'], exclusion_reason=r['exclusion_reason'],
                      rect_px=[r['x_min_px'], r['y_min_px'], r['x_max_px'], r['y_max_px']],
                      passes_pose_gate=r.get('passes_pose_gate'))
                 for r in bbox_rows if r.get('viewer_leaflet_id') == vid]
        point_photos = {photo_indices[_key(r['image_file'])] for r in depth_rows
                        if r.get('viewer_leaflet_id') == vid and _key(r.get('image_file')) in photo_indices}
        photos = sorted(point_photos | {b['photo_index'] for b in boxes if b['photo_index'] is not None})
        result.append(dict(
            viewer_leaflet_id=vid, organ_id=leaf['organ_id'], leaflet_id=leaf['leaflet_id'],
            geometry_status=leaf['geometry_status'], cuboid_status=leaf['cuboid_status'],
            cuboid_detail=leaf['cuboid_detail'], cuboid_min_mm=leaf['cuboid_min_mm'], cuboid_max_mm=leaf['cuboid_max_mm'],
            feasible_vertices_mm=leaf['feasible_vertices_mm'], feasible_edges=leaf['feasible_edges'],
            feasible_fraction_of_cuboid=leaf['feasible_fraction_of_cuboid'],
            point_extent_status=leaf['point_extent_status'], point_extent_min_mm=leaf['point_extent_min_mm'],
            point_extent_max_mm=leaf['point_extent_max_mm'], triangulated_point_ids=leaf['triangulated_point_ids'],
            image_point_ids=leaf['image_point_ids'], points_outside_feasible_region=leaf['points_outside_feasible_region'],
            boxes=boxes, photo_indices=photos,
            photos=[dict(index=i, image_file=frames[i]['image_file'], pose_status=frames[i].get('pose_status'),
                         passes_pose_gate=frames[i].get('passes_pose_gate'),
                         detected_marker_ids=_text(_detected_ids(frames[i], [], 'detected_marker_ids')))
                    for i in photos],
            any_accepted_pose=leaf['any_accepted_pose'], validated_metric=False, suitable_for_ppfd=False,
            attached_surface=next((dict(status=s['status'], retained_faces=s.get('retained_faces'),
                                        source_images=s.get('source_images', []),
                                        landmark_point_ids=[l.get('point_id') for l in s.get('physical_landmarks', [])])
                                   for s in (surfaces or {}).get('surfaces', []) if s.get('viewer_leaflet_id') == vid), None)))
    return result



_SURFACE_COLOR = (53, 224, 122)
_REPROJECTION_COLOR = "#ffffff"


def _pose(frame: dict) -> bool:
    return frame.get('pose_status') == 'solved' and frame.get('rvec_rad') is not None and frame.get('tvec_mm') is not None


def _project(points_mm, frame: dict, intrinsics: dict):
    """Existing projection model; distortion included. Board mm -> original pixels."""
    return project(np.asarray(points_mm, dtype=float).reshape(-1, 3), frame, intrinsics)


def _draw_surface(image: Image.Image, surface: dict, frame: dict, intrinsics: dict, stroke_px: int) -> int:
    """Reproject an interpolated patch as a footprint outline, distinct from observations."""
    vertices = np.asarray(surface['vertices_mm'], dtype=float)
    faces = np.asarray(surface['faces'], dtype=int)
    if not len(vertices) or not len(faces):
        return 0
    uv = _project(vertices, frame, intrinsics)
    if not np.isfinite(uv).all():
        return 0
    canvas = np.array(image)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    polygons = np.round(uv[faces] * 16).astype(np.int32)
    cv2.fillPoly(mask, polygons, 1, lineType=cv2.LINE_8, shift=4)
    if not mask.any():
        return 0
    inside = mask.astype(bool)
    canvas[inside] = (canvas[inside] * 0.75 + np.asarray(_SURFACE_COLOR) * 0.25).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, _SURFACE_COLOR, max(2, stroke_px))
    image.paste(Image.fromarray(canvas))
    return int(len(faces))


def surface_scene_payload(surfaces: dict) -> dict:
    """Keep vertices/faces with their status so the viewer can toggle and flag stale geometry."""
    return dict(status=surfaces.get('status', 'not_attached'), stale_note=surfaces.get('stale_note'),
                patch_dir=surfaces.get('patch_dir'), model=(surfaces.get('model') or {}).get('model_name'),
                surfaces=[dict(viewer_leaflet_id=s.get('viewer_leaflet_id'), organ_id=s.get('organ_id'),
                               leaflet_id=s.get('leaflet_id'), vertices_mm=s['vertices_mm'], faces=s['faces'],
                               status=s['status'], retained_faces=s.get('retained_faces'),
                               source_images=s.get('source_images', []),
                               landmark_point_ids=[l.get('point_id') for l in s.get('physical_landmarks', [])],
                               geometry_kind='interpolated_surface', validated_metric=False)
                          for s in surfaces.get('surfaces', [])],
                unavailable=surfaces.get('unavailable', []))


def _detected_ids(frame: dict, rows: list[dict], field: str) -> Any:
    evidence = frame.get("board_evidence") or {}
    evidence_hash = evidence.get("source_image_sha256")
    frame_hash = frame.get("sha256") or frame.get("image_sha256")
    # Use actual detector evidence only; never configured board marker ranges.
    if field in evidence and evidence_hash and frame_hash and evidence_hash == frame_hash:
        return evidence[field]
    for row in rows:
        if field in row:
            return row[field]
    return None


def _font(size_px: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def _box_table(rows: list[dict]) -> str:
    if not rows:
        return ''
    def reason(value: Any) -> str:
        return ' / '.join(_EXCLUSION.get(part, part) for part in str(value).split(';')) if value else '가능 공간 계산에 사용'
    def coords(row: dict) -> str:
        values = [_number(row.get(k)) for k in ('x_min_px', 'y_min_px', 'x_max_px', 'y_max_px')]
        return 'unavailable' if None in values else '{:.1f}, {:.1f} ~ {:.1f}, {:.1f}'.format(*values)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(value)}</td>' for value in (
        _text(r.get('viewer_leaflet_id')), _text(r.get('annotation_id')), coords(r),
        _VISIBILITY.get(r.get('visibility'), _text(r.get('visibility'))),
        _REVIEW.get(r.get('identity_review'), _text(r.get('identity_review'))),
        reason(r.get('exclusion_reason')))) + '</tr>' for r in rows)
    head = ''.join(f'<th>{title}</th>' for title in ('소엽', '박스 ID', '원본 픽셀 가장자리 좌표', '보임 정도', '연결 검토', '가능 공간 사용'))
    return f'<h3>이 사진의 소엽 박스</h3><div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _table(rows: list[dict]) -> str:
    head = "".join(f"<th>{escape(title)}</th>" for _, title in _COLUMNS)
    def display(row: dict, key: str) -> str:
        value = _text(row.get(key))
        if value == 'unavailable':
            return '미산출'
        if key in ('x_px', 'y_px', 'z_board_mm', 'camera_axis_z_mm'):
            numeric = _number(row.get(key))
            return f'{numeric:.1f}' if numeric is not None else '미산출'
        if key in ('inside_feasible_region', 'observation_inside_leaflet_bbox') and value == 'unavailable':
            return '해당 없음'
        return _DISPLAY.get(value, value) if key in ('point_kind', 'depth_status', 'context_depth_restriction', 'estimate_kind',
                                                     'inside_feasible_region', 'observation_inside_leaflet_bbox') else value
    body = "".join("<tr>" + "".join(f"<td>{escape(display(row, key))}</td>" for key, _ in _COLUMNS) + "</tr>" for row in rows)
    if not rows:
        body = f'<tr><td colspan="{len(_COLUMNS)}">이 사진에 저장된 관찰점이 없습니다.</td></tr>'
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


#: Pose provenance shown per photograph. An inferred camera is never displayed as a board measurement.
POSE_SOURCE_LABELS = {
    'charuco_board_pnp': '측정된 ChArUco 보드 위치',
    'leaflet_landmark_pnp': '소엽 랜드마크로 추정한 위치 (진단용·측정 아님)',
    'stale_candidate': '추정 후보 있으나 입력이 바뀜 · 다시 계산 필요',
    'candidate_not_eligible': '추정 후보 있으나 진단 기준 미달 · 사용 안 함',
    'unavailable': '촬영 위치 없음',
}


def pose_source_badge(frame: dict, candidate: dict | None) -> tuple[str, str]:
    """Return (source key, display text) for one photograph, keeping measurement and inference apart."""
    if frame.get('pose_status') == 'solved':
        key = 'charuco_board_pnp'
    elif candidate is None:
        key = 'unavailable'
    elif candidate.get('freshness') != 'current':
        key = 'stale_candidate'
    elif candidate.get('used_in_hybrid'):
        key = 'leaflet_landmark_pnp'
    else:
        key = 'candidate_not_eligible'
    detail = ''
    if candidate is not None:
        reasons = candidate.get('stale_reasons') or []
        detail = (f" · 후보 {_text(candidate.get('candidate_id'))} · 계산 {_text(candidate.get('calculation_status'))}"
                  f" · 품질 {_text(candidate.get('quality_status'))} · 검증 {_text(candidate.get('validation'))}"
                  f" · 적합 잔차 {_text(candidate.get('fit_rms_px'))} px · 검증 잔차 {_text(candidate.get('holdout_rms_px'))} px"
                  + (f" · 오래됨: {', '.join(reasons)}" if reasons else ''))
    return key, POSE_SOURCE_LABELS[key] + detail


def hybrid_scene_payload(hybrid: dict) -> list[dict]:
    """One entry per physical point that the hybrid branch places in 3D, with its baseline value."""
    result: dict[tuple[str, str], dict] = {}
    for row in hybrid.get('rows', []):
        xyz = [_number(row.get(k)) for k in ('x_board_mm', 'y_board_mm', 'z_board_mm')]
        if None in xyz:
            continue
        key = (str(row.get('viewer_leaflet_id')), str(row.get('point_id')))
        baseline = [_number(row.get(f'baseline_{axis}_board_mm')) for axis in 'xyz']
        result.setdefault(key, dict(
            viewer_leaflet_id=key[0], leaflet_id=str(row.get('leaflet_id')), point_id=key[1],
            xyz_mm=xyz, baseline_xyz_mm=None if None in baseline else baseline,
            shift_from_baseline_mm=_number(row.get('shift_from_baseline_mm')),
            uses_inferred_camera=str(row.get('uses_inferred_camera')) in ('True', 'true'),
            validated_metric=False, suitable_for_ppfd=False))
    return list(result.values())


ROUTE_LABELS = {'global': '전체 사진', 'vertical': '수직 경로 01/03 → 2X → 1X',
                'circumferential': '둘레 경로 11 → 12 → … → 18 → 11'}
FINDING_LABELS = {
    'fewer_reviewed_leaflets_than_recorded': '기록보다 검토된 소엽이 적음 (검토 미완료 또는 기록 재확인 대상)',
    'more_reviewed_leaflets_than_recorded': '기록보다 검토된 소엽이 많음 (개수·연결 재확인 대상)',
    'growth_leaflet_count_not_recorded': '생육 기록의 소엽 수가 미입력(null)',
    'parent_not_in_growth_record': '생육 기록에 없는 복엽을 라벨이 가리킴',
    'compound_leaf_entry_count_differs': '복엽 항목 수가 compound_leaf_count 와 다름',
    'contradictory_coordinates_for_one_observation': '한 사진의 같은 물리점에 서로 다른 좌표가 저장됨 (둘 다 보존)',
    'leaflet_membership_unresolved': '이 복엽을 가리키는 소속 후보가 미해결 (개수에 넣지 않음)',
    'leaflet_membership_not_reviewed': '복엽 이름은 있으나 소속 확정 기록이 없는 소엽이 있음',
}


def reference_section(references: dict) -> str:
    """Count/association review and route summary. Findings only: nothing here changes a label."""
    if not references:
        return ''
    growth = references.get('growth_review', {})
    rows = ''.join(
        f"<tr><td>{escape(_text(entry.get('compound_leaf_id')))}</td>"
        f"<td>{escape(_text(entry.get('recorded_leaflet_count')))}</td>"
        f"<td>{escape(_text(entry.get('reviewed_leaflet_count')))}</td>"
        f"<td>{escape(', '.join(entry.get('unconfirmed_leaflet_ids') or []) or '없음')}</td>"
        f"<td>{escape(', '.join(entry.get('unresolved_candidate_leaflets') or []) or '없음')}</td>"
        f"<td>{escape(FINDING_LABELS.get(entry.get('finding'), entry.get('finding') or '일치'))}</td></tr>"
        for entry in growth.get('compound_leaves', []))
    routes = ''.join(
        f"<tr><td>{escape(ROUTE_LABELS.get(route, route))}</td><td>{data['images']}</td>"
        f"<td>{data['unique_physical_points']}</td><td>{data['unique_observations']}</td>"
        f"<td>{data['duplicate_observations_collapsed']}</td>"
        f"<td>{data['points_with_two_or_more_eligible_views']}</td></tr>"
        for route, data in (references.get('routes') or {}).items())
    other = [f for f in references.get('findings', [])
             if f.get('code') not in ('fewer_reviewed_leaflets_than_recorded',
                                      'more_reviewed_leaflets_than_recorded', 'growth_leaflet_count_not_recorded')]
    notes = ''.join(f"<li>{escape(FINDING_LABELS.get(f['code'], f['code']))} · {escape(_text(f.get('reference')))}</li>"
                    for f in other) or '<li>추가 지적 사항 없음</li>'
    return (f'<section><h2>JSON 참조 검토 · 개수와 경로</h2>'
            f'<p class="notice">생육 기록의 개수는 검토 기준일 뿐 좌표가 아닙니다. 불일치는 사진을 다시 보고 판단할 지적 사항이며, '
            f'라벨을 자동으로 합치거나 지우거나 만들어 내지 않습니다. 경로는 같은 관찰을 돌아보는 방법일 뿐이라 '
            f'여러 경로로 닿은 관찰도 한 번만 셉니다.</p>'
            f'<p>요청 파일: {escape(_text(references.get("request_path")))} · 상태 '
            f'{escape(_text(references.get("status")))} · 모드 {escape(_text(references.get("reconstruction_mode")))}</p>'
            f'<div class="tablewrap"><table><thead><tr><th>복엽</th><th>기록된 소엽 수</th><th>검토된 소엽 수</th>'
            f'<th>연결 미확인 라벨</th><th>소속 미해결 후보</th><th>지적</th></tr></thead><tbody>'
            f'{rows or "<tr><td colspan=6>생육 기록 항목이 없습니다.</td></tr>"}</tbody></table></div>'
            f'<div class="tablewrap"><table><thead><tr><th>경로</th><th>사진</th><th>고유 물리점</th>'
            f'<th>고유 관찰</th><th>경로 중복 제거</th><th>2뷰 이상 가능</th></tr></thead><tbody>'
            f'{routes}</tbody></table></div><ul>{notes}</ul></section>')


IMPORTED_CLASSES = ('compoundleaves', 'leaflet', 'sub-stem', 'main-stem', 'growth-shoot')


def imported_section(imported: dict | None) -> str:
    """Imported detector/human 2D labels per photo. They are observations only, never geometry."""
    if not imported or not imported.get('photos'):
        return ''
    rows = ''.join(
        f"<tr><td>{escape(_text(p['image_file']))}</td><td>{escape(_text(p['view_code']))}</td>"
        f"<td>{escape(_text(p['pose_status']))}{' · 1px 통과' if p.get('passes_pose_gate') else ''}</td>"
        f"<td>{escape(_text(p.get('n_markers_detected')))}</td>"
        + ''.join(f"<td>{p['by_class'].get(c, 0)}</td>" for c in IMPORTED_CLASSES)
        + f"<td>{p['adopted_leaflets']}</td><td>{escape(p['state'])}</td></tr>"
        for p in imported['photos'])
    heads = ''.join(f'<th>{escape(c)}</th>' for c in IMPORTED_CLASSES)
    return (f'<section><h2>가져온 2D 라벨 · 기하 아님</h2>'
            f'<p class="notice">사진마다 사람이 그린 Roboflow 라벨을 원본 픽셀로 옮긴 것입니다. 소엽·복엽 영역은 직사각형이며 잎 윤곽이 아닙니다. '
            f'사진 사이 같은 소엽 연결과 대응점이 없으면 3D 점·범위가 만들어지지 않습니다. '
            f'포즈 상태는 ChArUco 풀이 결과이며 1px 기준 통과 여부를 따로 표시합니다.</p>'
            f'<div class="tablewrap"><table><thead><tr><th>사진</th><th>뷰</th><th>포즈</th><th>검출 마커 수</th>{heads}'
            f'<th>소엽 라벨로 가져옴</th><th>상태</th></tr></thead><tbody>{rows}</tbody></table></div>'
            f'<p>합계: 라벨 {imported["total"]}개 · 소엽 라벨로 가져온 것 {imported["adopted_total"]}개 · '
            f'확인된 소엽 동일성 {imported["confirmed_identities"]}개</p></section>')


def write_point_review(capture: dict, depth_rows: list[dict], out: Path,
                       leaflets: list[dict] | None = None, bbox_rows: list[dict] | None = None,
                       accepted_regions: list[dict] | None = None, surfaces: dict | None = None,
                       hybrid: dict | None = None, references: dict | None = None,
                       imported: dict | None = None) -> None:
    """Write ``out/labels/index.html`` and one full-resolution labeled JPEG per frame.

    Required frame fields are ``image_file`` and ``image_path``. Point rows use
    ``x_px/y_px`` (or observed_x_px/observed_y_px), organ/leaflet/point IDs and
    point_kind. Numerical depth and detected ID fields are displayed unchanged;
    absent values stay unavailable. All capture frames are listed, even without
    observations or solved poses. Optional accepted regions are saved 2D outlines,
    not projected geometry or correspondence points. Draft proposals are excluded.
    Saved leaflet bboxes (``bbox_rows``) are drawn as saved; ``leaflets`` adds the
    leaflet-centric 3D selector with separate bbox and point extents.
    """
    leaflets, bbox_rows = leaflets or [], bbox_rows or []
    accepted_regions = accepted_regions or []
    surfaces = surfaces or {}
    hybrid = hybrid or {}
    references = references or {}
    hybrid_points = hybrid_scene_payload(hybrid)
    side_candidates = {_key(c.get('image_file')): c for c in hybrid.get('side', {}).get('candidates', [])}
    drawable_surfaces = [s for s in surfaces.get('surfaces', []) if s.get('status') == 'current_for_this_point_snapshot']
    reconstructed = {}
    for row in depth_rows:
        xyz = [_number(row.get(k)) for k in ('x_board_mm', 'y_board_mm', 'z_board_mm')]
        if None not in xyz:
            reconstructed.setdefault((str(row.get('viewer_leaflet_id')), str(row.get('point_id'))), xyz)
    out = Path(out).resolve()
    if not out.is_dir() or not out.is_relative_to(common.BUILD.resolve()) or out == common.BUILD.resolve():
        raise ValueError("out must be an existing generated directory inside build")
    labels = (out / "labels").resolve()
    if not labels.is_relative_to(out):
        raise ValueError("labels directory must stay inside the output directory")
    labels.mkdir(exist_ok=True)
    frames = capture.get("frames") or []
    keys = [_key(frame.get("image_file")) for frame in frames]
    if len(keys) != len(set(keys)):
        raise ValueError("capture has ambiguous duplicate image basenames")
    by_image: dict[str, list[dict]] = {key: [] for key in keys}
    unmatched: list[dict] = []
    for row in depth_rows:
        target = by_image.get(_key(row.get("image_file")))
        if target is None:
            unmatched.append(dict(row))
        else:
            target.append(dict(row))

    sections: list[str] = []
    options: list[str] = []
    for index, (frame, key) in enumerate(zip(frames, keys)):
        source_rows = by_image[key]
        image_file = str(frame.get("image_file", "unavailable"))
        marker_ids = _detected_ids(frame, source_rows, "detected_marker_ids")
        corner_ids = _detected_ids(frame, source_rows, "detected_corner_ids")
        rendered_rows: list[dict] = []
        image_error = ""
        image: Image.Image | None = None
        try:
            if frame.get('sha256') and common.file_hash(Path(frame['image_path'])) != frame['sha256']:
                raise ValueError('Source photograph hash mismatch')
            with Image.open(Path(frame["image_path"])) as original:
                image = original.convert("RGB")
        except (KeyError, OSError, ValueError) as error:
            image_error = f"원본 사진을 열 수 없습니다: {error}"
        drawn = 0
        photo_boxes = [r for r in bbox_rows if _key(r.get('image_file')) == key]
        photo_regions = [r for r in accepted_regions if _key(r.get('image_file')) == key]
        region_labels = []
        if image is not None:
            draw = ImageDraw.Draw(image)
            radius_px = max(5, round(min(image.size) / 250))
            stroke_px = max(2, radius_px // 4)
            font = _font(max(15, round(min(image.size) / 105)))
            for region in photo_regions:
                for ring in region.get('rings', []):
                    points = ring.get('points_px', [])
                    if len(points) < 3 or any(len(p) != 2 or any(_number(v) is None for v in p)
                                              or not (0 <= p[0] < image.width and 0 <= p[1] < image.height) for p in points):
                        raise ValueError('Accepted region has invalid original-pixel coordinates')
                    xy = [tuple(p) for p in points]
                    draw.line(xy + [xy[0]], fill='#45e5df', width=stroke_px * 2)
                region_labels.append(escape(str(region.get('leaflet_id', 'unavailable'))))
        for source_row in source_rows:
            row = dict(source_row)
            xy_px = _saved_xy(source_row)
            row["x_px"] = xy_px[0] if xy_px else None
            row["y_px"] = xy_px[1] if xy_px else None
            row["point_kind"] = _point_kind(source_row)
            row.setdefault("depth_status", "unavailable")
            row.setdefault("detected_marker_ids", marker_ids)
            row.setdefault("detected_corner_ids", corner_ids)
            if xy_px is None:
                row["context_depth_restriction"] = _text(row.get("context_depth_restriction")) + " · 실제 관찰 좌표 없음: 표시 제외"
            elif image is not None:
                x_px, y_px = xy_px
                if not (0 <= x_px < image.width and 0 <= y_px < image.height):
                    row["context_depth_restriction"] = _text(row.get("context_depth_restriction")) + " · 원본 이미지 범위 밖: 표시 제외"
                else:
                    color = _INTERIOR_COLOR if row["point_kind"] == "interior" else _BOUNDARY_COLOR
                    bounds = (x_px-radius_px, y_px-radius_px, x_px+radius_px, y_px+radius_px)
                    if row["point_kind"] == "interior":
                        draw.ellipse(bounds, outline=color, width=stroke_px * 2)
                    else:
                        draw.ellipse(bounds, fill=color, outline="black", width=stroke_px)
                    visible_id = row.get('leaflet_id')
                    if not visible_id or visible_id == 'unavailable':
                        visible_id = row.get('organ_id')
                    label = _text(visible_id) + '/' + _text(row.get('point_id'))
                    # Display full identifiers in the table; bound their photo overlay width.
                    label = label if len(label) <= 80 else label[:77] + "..."
                    text_box = draw.textbbox((0, 0), label, font=font, stroke_width=stroke_px)
                    text_x = max(0, min(x_px + radius_px + 4, image.width - (text_box[2] - text_box[0]) - 4))
                    text_y = max(0, min(y_px - radius_px - (text_box[3] - text_box[1]) - 4, image.height - (text_box[3] - text_box[1]) - 4))
                    draw.text((text_x, text_y), label, font=font, fill=color, stroke_width=stroke_px, stroke_fill="black")
                    drawn += 1
            rendered_rows.append(row)
        if image is not None:
            for box in photo_boxes:
                values = [_number(box.get(k)) for k in ('x_min_px', 'y_min_px', 'x_max_px', 'y_max_px')]
                if None in values:
                    continue
                x0, y0, x1, y1 = values
                color = _BOX_USED_COLOR if box.get('used_for_extent') else _BOX_EXCLUDED_COLOR
                # Stored pixel-edge coordinates; PIL outlines whole pixels.
                draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=color, width=stroke_px)
                label = (f"{_text(box.get('viewer_leaflet_id'))} 박스 · {_VISIBILITY.get(box.get('visibility'), '?')}"
                         f" · {_REVIEW.get(box.get('identity_review'), '?')}")
                text_box = draw.textbbox((0, 0), label, font=font, stroke_width=stroke_px)
                draw.text((x0 + 4, max(0, y0 - (text_box[3] - text_box[1]) - 6)), label, font=font, fill=color,
                          stroke_width=stroke_px, stroke_fill='black')
        reprojected_points, reprojected_faces = 0, 0
        if image is not None and _pose(frame) and capture.get('intrinsics'):
            for surface in drawable_surfaces:
                reprojected_faces += _draw_surface(image, surface, frame, capture['intrinsics'], stroke_px)
            draw = ImageDraw.Draw(image)
            for (_, point_id), xyz in sorted(reconstructed.items()):
                uv = _project([xyz], frame, capture['intrinsics'])[0]
                if not np.isfinite(uv).all() or not (0 <= uv[0] < image.width and 0 <= uv[1] < image.height):
                    continue
                arm = radius_px + 2
                draw.line((uv[0]-arm, uv[1], uv[0]+arm, uv[1]), fill=_REPROJECTION_COLOR, width=stroke_px)
                draw.line((uv[0], uv[1]-arm, uv[0], uv[1]+arm), fill=_REPROJECTION_COLOR, width=stroke_px)
                reprojected_points += 1
        photo_name = f"photo_{index:04d}.jpg"
        if image is not None:
            image.save(labels / photo_name, "JPEG", quality=92, subsampling=0)
            image.close()
            photograph = f'<a href="{photo_name}" target="_blank" rel="noopener"><img src="{photo_name}" loading="lazy" alt="{escape(image_file, quote=True)} 저장된 관찰점"></a>'
        else:
            photograph = f'<p class="notice">{escape(image_error)}</p>'
        pose = '시험 위치 있음' if frame.get('pose_status') == 'solved' else '위치 미산출'
        gate = '통과' if frame.get('passes_pose_gate') is True else '미통과'
        source_key, source_text = pose_source_badge(frame, side_candidates.get(key))
        badge = (f'<p class="posesrc {source_key}">촬영 위치 출처: {escape(source_text)}</p>')
        options.append(f'<option value="{index}">{escape(image_file)} · 관찰점 {len(source_rows)}</option>')
        reprojection = ''
        if reprojected_points or reprojected_faces:
            reprojection = (f'<p class="reproj">재투영(비교용): 복원된 점 {reprojected_points}개 흰 십자 · '
                            f'보간 부분 면 {reprojected_faces}개 삼각형의 초록 발자국. 저장된 관찰이 아니라 계산 결과를 이 사진에 다시 투영한 것입니다.</p>')
        elif drawable_surfaces or reconstructed:
            reprojection = '<p class="reproj">이 사진은 촬영 위치가 없어 재투영을 표시하지 않았습니다.</p>'
        sections.append(f'<section class="photo" id="photo-{index}"{" hidden" if index else ""}><h2>{escape(image_file)}</h2>'
                        f'<p>원본 픽셀에 표시한 관찰점 {drawn}개 / 저장된 행 {len(source_rows)}개 · 포즈 {pose} · 포즈 기준 통과 {gate}</p>'
                        f'{badge}'
                        f'<p>실제 검출 marker IDs: {escape(_text(marker_ids))}<br>실제 검출 corner IDs: {escape(_text(corner_ids))}'
                        f'<br>검출 ID는 이 사진에서 실제로 검출된 보드 표식입니다. 추정 위치를 쓴 사진에는 보드 검출이 없습니다.</p>'
                        f'<p>수락한 2D 윤곽 {len(region_labels)}개 (청록색): {", ".join(region_labels) or "없음"} · 대응점/3D 표면 아님</p>'
                        f'{reprojection}{photograph}{_box_table(photo_boxes)}{_table(rendered_rows)}</section>')
    reference_html = reference_section(references) + imported_section(imported)
    unmatched_html = ""
    if unmatched:
        unmatched_html = f'<section><h2>사진과 연결되지 않은 관찰 {len(unmatched)}행</h2><p>파일 이름을 확인하세요. 사진 좌표를 다른 사진에 임의로 적용하지 않았습니다.</p>{_table(unmatched)}</section>'
    scene_data = json.dumps(point_scene_payload(capture, depth_rows), ensure_ascii=True).replace('<', '\\u003c')
    leaflet_data = json.dumps(leaflet_scene_payload(capture, leaflets, bbox_rows, depth_rows, surfaces),
                              ensure_ascii=True).replace('<', '\\u003c')
    surface_data = json.dumps(surface_scene_payload(surfaces), ensure_ascii=True).replace('<', '\\u003c')
    hybrid_data = json.dumps(dict(points=hybrid_points, summary=hybrid.get('summary', {}),
                                  inferred_images=hybrid.get('side', {}).get('included_images', [])),
                             ensure_ascii=True).replace('<', '\\u003c')
    scene = (Path(__file__).with_name('editor_point_scene.html').read_text(encoding='utf-8')
             .replace('__POINT_DATA__', scene_data).replace('__LEAFLET_DATA__', leaflet_data)
             .replace('__SURFACE_DATA__', surface_data).replace('__HYBRID_DATA__', hybrid_data))
    document = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>저장한 점 · 사진 확인</title><style>
body{margin:0 auto;max-width:1500px;padding:20px;background:#102018;color:#ecf3ee;font:15px/1.5 system-ui,sans-serif}
h1{font-size:24px}h2{font-size:19px}.notice{padding:12px;border:1px solid #c3a64b;background:#302f18}
nav{display:flex;align-items:center;gap:8px;flex-wrap:wrap;position:sticky;top:0;background:#102018;padding:10px 0;z-index:1}
button,select{font:inherit;padding:8px;color:inherit;background:#25392e;border:1px solid #749781;border-radius:5px}
img{display:block;width:100%;max-height:70vh;object-fit:contain;background:#08110b;border:1px solid #385441}
.tablewrap{overflow:auto;margin:16px 0}table{border-collapse:collapse;min-width:1100px;width:100%;font-size:13px}th,td{padding:8px;border:1px solid #41634c;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#243e2e}
[hidden]{display:none!important}.boundary{color:#ffb347}.interior{color:#ff69dc}a{color:#9cd4b0}
.toggles{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:10px 0;padding:8px;border:1px solid #41634c;border-radius:6px}
.toggles label{display:inline-flex;gap:5px;align-items:center}
.reproj{margin:6px 0;padding:6px 10px;border-left:3px solid #35e07a;background:#16291f}
.posesrc{margin:6px 0;padding:6px 10px;border-left:3px solid #749781;background:#1a2a21}
.posesrc.leaflet_landmark_pnp{border-left-color:#4fd1ff;background:#16262e}
.posesrc.stale_candidate,.posesrc.candidate_not_eligible{border-left-color:#c3a64b;background:#302f18}
</style><h1>저장한 라벨 점과 깊이 확인</h1>
<p class="notice">깊이는 미검증 시험값이며 실제 길이·깊이의 정확도를 보장하지 않습니다. 사진에는 저장된 관찰점·박스와 수락한 2D 윤곽(청록색)을 표시합니다. 수락한 윤곽도 3D 대응점이나 검증된 잎 표면을 뜻하지 않습니다.</p>
<p><span class="boundary">● 가장자리 점</span> · <span class="interior">○ 잎맥/내부 점</span> · 사진을 누르면 원본 해상도의 표시 사진을 엽니다.<br>보드 위 높이와 카메라 광축 방향 깊이는 서로 다른 값입니다. 계산되지 않은 값은 미산출로 표시합니다.</p>
""" + scene + """<nav><button id="previous" type="button">이전 사진</button><label>사진 <select id="photoSelect">""" + "".join(options) + """</select></label><button id="next" type="button">다음 사진</button><span id="position"></span></nav>
""" + reference_html + ("".join(sections) if sections else '<p>촬영 목록에 사진이 없습니다.</p>') + unmatched_html + """
<script>
'use strict';const selector=document.getElementById('photoSelect'),photos=[...document.querySelectorAll('.photo')];
function show(index){if(!photos.length)return;index=(index+photos.length)%photos.length;photos.forEach((el,i)=>el.hidden=i!==index);selector.value=String(index);document.getElementById('position').textContent=(index+1)+' / '+photos.length;}
selector.addEventListener('change',()=>show(Number(selector.value)));
document.getElementById('previous').addEventListener('click',()=>show(Number(selector.value)-1));
document.getElementById('next').addEventListener('click',()=>show(Number(selector.value)+1));
if(!photos.length){selector.disabled=true;document.getElementById('previous').disabled=true;document.getElementById('next').disabled=true;}else show(0);
</script></html>"""
    (labels / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--depth-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Existing generated directory inside build.")
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    with args.depth_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    write_point_review(capture, rows, args.out)
    print(args.out.resolve() / "labels" / "index.html")


if __name__ == "__main__":
    main()
