"""Export accepted leaflet boxes/regions from an editor backup for future model training.

python -m tools.export_leaflet_training --session build/.../edit_session07
  --backup build/.../editor_runs/run_.../editor.backup.json --out build/.../training_export01
  [--eval-plants DT-002 DT-005]

Only saved leaflet labels are exported: every ``leaflets[*].bboxes[photo]`` and, when
present, the accepted 2D region in ``regions[photo]``. AI proposals that are proposed,
edited-but-not-accepted or rejected are never exported. Original photographs are copied
byte-for-byte with verified hashes; masks are rasterized from accepted rings at original
resolution (even-odd, holes kept). Splits are by plant: one plant cannot be split into a
held-out evaluation, so a single-plant export is labelled as such, never randomly split
by photograph. Nothing here trains, runs or uploads a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image

from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json, write_rows
from tools.prepare_leaf_edit_session import basename, verify_path
from tools.rebuild_leaf_edit_session import compact_capture, current_state, view_status

COLUMNS = ('split_group_plant_id', 'plant_id', 'session_id', 'variant_id', 'image_file', 'image_sha256', 'exported_image',
           'instance_id', 'organ_id', 'leaflet_id', 'annotation_id', 'x_min_px', 'y_min_px', 'x_max_px', 'y_max_px',
           'image_width_px', 'image_height_px', 'has_mask', 'mask_file', 'mask_area_px', 'visibility', 'identity_review',
           'label_origin', 'proposal_id', 'proposal_model', 'proposal_model_is_mock', 'user_edited_proposal',
           'leaf_review_status', 'human_review')


def region_mask(rings: list[dict], width: int, height: int) -> np.ndarray:
    """Even-odd rasterization of pixel-centre rings, so holes stay empty."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ring in rings:
        layer = np.zeros_like(mask)
        points = np.round(np.asarray(ring['points_px'], dtype=float) * 16).astype(np.int32)
        cv2.fillPoly(layer, [points], 1, lineType=cv2.LINE_8, shift=4)
        mask ^= layer
    return mask


def plant_split(plant_ids: set[str], eval_plants: list[str]) -> dict:
    """Assign whole plants to train/eval. Photos of one plant never cross the split."""
    plants = sorted(plant_ids)
    unknown = sorted(set(eval_plants) - set(plants))
    if unknown:
        raise StudyError('Evaluation plants not present in the export: ' + ', '.join(unknown))
    base = dict(rule='group_by_plant_id', plants=plants,
                notice='Photographs of one plant are correlated; split only by plant. One plant does not establish generalization.')
    if len(plants) < 2:
        return dict(base, status='insufficient_plants_for_held_out_split', train_plants=[], eval_plants=[])
    if not eval_plants:
        return dict(base, status='eval_plants_not_selected', train_plants=[], eval_plants=[])
    if set(eval_plants) == set(plants):
        raise StudyError('At least one plant must remain for training')
    return dict(base, status='plant_level_split', eval_plants=sorted(eval_plants),
                train_plants=[p for p in plants if p not in eval_plants])


def export(session: Path, backup_path: Path, out: Path, eval_plants: list[str] | None = None) -> dict:
    session, backup_path = session.resolve(), backup_path.resolve()
    report = read_json(session / 'session_report.json')
    capture, seed = read_json(session / 'capture.json'), read_json(session / 'seed.backup.json')
    backup = read_json(backup_path)
    if file_hash(session / 'capture.json') != report['capture_sha256']:
        raise StudyError('Prepared capture changed')
    if (backup.get('plant_id'), backup.get('session_id')) != (seed['plant_id'], seed['session_id']):
        raise StudyError('Backup plant/session identity mismatch')
    if compact_capture(backup.get('capture') or {}) != compact_capture(capture):
        raise StudyError('Backup capture differs from the prepared session')
    size = (capture['intrinsics']['image_width_px'], capture['intrinsics']['image_height_px'])
    frames = {basename(f['image_file']).lower(): f for f in capture['frames']}
    state = current_state(backup)
    rows, skipped, used_photos = [], [], {}
    for key, leaf in sorted(state.get('leaflets', {}).items()):
        for image_key, box in sorted((leaf.get('bboxes') or {}).items()):
            frame = frames.get(image_key)
            coords = [box.get(k) for k in ('x_min_px', 'y_min_px', 'x_max_px', 'y_max_px')]
            if frame is None or (box.get('image_width_px'), box.get('image_height_px')) != size or \
                    any(type(v) not in (int, float) for v in coords) or coords[0] < 0 or coords[1] < 0 or \
                    coords[2] > size[0] or coords[3] > size[1] or coords[2] - coords[0] < 1 or coords[3] - coords[1] < 1:
                skipped.append(dict(editor_identity=key, image_key=image_key, reason='invalid_or_non_original_bbox'))
                continue
            visibility, review = view_status(leaf, image_key)
            region = (leaf.get('regions') or {}).get(image_key)
            stem = Path(frame['image_file']).stem
            instance = re.sub(r'[^A-Za-z0-9_.-]', '_', f"{stem}__{leaf['organ_id']}__{leaf['leaflet_id']}")
            mask_file, area = None, None
            if region:
                mask = region_mask(region['rings'], *size)
                area = int(mask.sum())
                if area:
                    mask_file = f'masks/{instance}.png'
            model = (region or {}).get('model') or {}
            rows.append(dict(split_group_plant_id=backup['plant_id'], plant_id=backup['plant_id'], session_id=backup['session_id'],
                             variant_id='__all__', image_file=frame['image_file'], image_sha256=frame['sha256'],
                             exported_image='images/' + basename(frame['image_file']), instance_id=instance,
                             organ_id=leaf['organ_id'], leaflet_id=leaf['leaflet_id'], annotation_id=box.get('annotation_id'),
                             x_min_px=coords[0], y_min_px=coords[1], x_max_px=coords[2], y_max_px=coords[3],
                             image_width_px=size[0], image_height_px=size[1], has_mask=mask_file is not None,
                             mask_file=mask_file, mask_area_px=area, visibility=visibility, identity_review=review,
                             label_origin=box.get('region_source', 'manual_bbox'), proposal_id=box.get('proposal_id'),
                             proposal_model=model.get('model_name'), proposal_model_is_mock=model.get('is_mock'),
                             user_edited_proposal=(region or {}).get('user_edited'),
                             leaf_review_status=leaf.get('review_status'), human_review='saved_label_by_user'))
            if mask_file:
                rows[-1]['_mask'] = mask
            used_photos[image_key] = frame
    proposals = Counter(p.get('state') for p in (state.get('proposals') or {}).values())
    split = plant_split({r['plant_id'] for r in rows} or {backup['plant_id']}, eval_plants or [])
    folder = output_dir(out)
    (folder / 'images').mkdir()
    (folder / 'masks').mkdir()
    for image_key, frame in sorted(used_photos.items()):
        source = verify_path(frame['image_path'], frame['sha256'], session)
        target = folder / 'images' / basename(frame['image_file'])
        shutil.copyfile(source, target)
        if file_hash(target) != frame['sha256']:
            raise StudyError('Copied photograph hash mismatch')
    for row in rows:
        mask = row.pop('_mask', None)
        if mask is not None:
            Image.fromarray(mask * 255).save(folder / row['mask_file'])
    write_rows(folder / 'leaflet_instances.csv', rows, COLUMNS)
    write_json(folder / 'splits.json', split)
    manifest = dict(schema_version='leaflet-training-export-1', plant_id=backup['plant_id'], session_id=backup['session_id'],
                    backup_path=str(backup_path), backup_sha256=file_hash(backup_path),
                    prepared_capture_sha256=report['capture_sha256'], calibration_sha256=capture.get('intrinsics_sha256'),
                    exported_instances=len(rows), instances_with_mask=sum(r['has_mask'] for r in rows),
                    exported_photos=len(used_photos), skipped_labels=skipped,
                    excluded_ai_proposals_by_state={k: v for k, v in proposals.items() if k != 'accepted'},
                    accepted_ai_proposals=proposals.get('accepted', 0),
                    included='saved leaflet bboxes and accepted 2D regions only',
                    excluded='proposed/edited-unaccepted/rejected AI drafts; point correspondences; flower/bud/truss boxes',
                    mask_note='Masks rasterize accepted outline rings (approxPolyDP tolerance at proposal time); even-odd keeps holes.',
                    identity_note='identity_review and visibility are exported as recorded; unconfirmed links are not ground-truth identity.',
                    validated_metric=False, deep_learning_run=False, uploaded=False, split=split)
    write_json(folder / 'export_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('session', 'backup', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--eval-plants', nargs='*', default=[])
    args = parser.parse_args()
    result = export(args.session, args.backup, args.out, args.eval_plants)
    print({k: result[k] for k in ('exported_instances', 'instances_with_mask', 'exported_photos')}, result['split']['status'])


if __name__ == '__main__':
    main()
