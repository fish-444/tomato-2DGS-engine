"""Import reviewed cross-view leaflet identities into an editor backup.

Human decisions made on the cross-view review pages (threshold-decision-log-1 documents, one JSON
file per decision) are turned into leaflet identities with the editor's own adopt/status code, so
the existing identity model and Document 06 fusion read them. Nothing is inferred beyond the saved
decisions:

* a "same" decision links two masks; a "duplicate" decision says the left-hand masks are one leaf;
* "different" decisions are constraints: a group that would join two masks declared different is
  reported as a conflict and not imported;
* a mask judged to merge several leaves ("merged") is never adopted;
* visibility (full/partial) is set only from explicit human 'vis' answers, never inferred;
* compound membership is not assigned.

Where one leaf has several masks in one photo (duplicate masks), only the mask with a direct "same"
decision to another photo is adopted; otherwise that photo is skipped and reported. The editor keeps
one box per leaf per photo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tools.leaf_light.common import StudyError, file_hash, read_json, write_json
from tools.leaflet_mask_batch import Batch

NODE_HELPER = Path(__file__).with_name('annotator_adopt_identities.cjs')
SCHEMA = 'cross-view-identity-import-1'
LINK_KINDS = ('match', 'pair', 'ext', 'trip')


def load_decisions(paths: list[Path]) -> list[dict]:
    docs = []
    for root in paths:
        files = sorted(Path(root).glob('*.json')) if Path(root).is_dir() else [Path(root)]
        for f in files:
            raw = read_json(f)
            # one decision per file (review-page database export) or a saved review file holding a list
            for doc in (raw['decisions'] if isinstance(raw, dict) and isinstance(raw.get('decisions'), list) else [raw]):
                if doc.get('schema_version') != 'threshold-decision-log-1' or 'decision_detail' not in doc:
                    raise StudyError(f'Not a cross-view decision document: {f}')
                doc['_file'], doc['_sha256'] = str(f), file_hash(f)
                docs.append(doc)
    # the same question answered in several files: the latest answer counts
    latest = {}
    for d in docs:
        if d['object_id'] not in latest or str(d.get('decided_at') or '') >= str(latest[d['object_id']].get('decided_at') or ''):
            latest[d['object_id']] = d
    return list(latest.values())


def visibility_decisions(docs: list[dict], plant: str, batch) -> list[dict]:
    """Human visibility answers (kind 'vis'): one entry per mask, the latest answer wins."""
    out = {}
    for d in sorted((d for d in docs if d['decision_detail']['kind'] == 'vis' and d.get('plant_id') == plant),
                    key=lambda d: str(d.get('decided_at') or '')):
        for mask_id, value in (d.get('visibility') or {}).items():
            if value in ('full', 'partial'):
                item = batch.item(mask_id)
                out[mask_id] = dict(key=item['image_file'].lower(), source_annotation_id=item['source_annotation_id'],
                                    value=value, decision=d['object_id'])
    return list(out.values())


def group_decisions(docs: list[dict], plant: str) -> dict:
    """Union-find over human decisions. Returns groups, conflicts, excluded masks and direct links."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    different, merged, direct, used_docs = [], set(), defaultdict(set), defaultdict(list)
    for d in docs:
        if d.get('plant_id') != plant:
            raise StudyError(f"Decision {d.get('object_id')} belongs to {d.get('plant_id')}, not {plant}")
        kind, verdict = d['decision_detail']['kind'], d['decision']
        panels = [m['mask_ids'] for m in d['decision_detail']['masks']]
        ids = [i for p in panels for i in p]
        if kind in LINK_KINDS and verdict == 'same':
            for i in ids[1:]:
                union(ids[0], i)
            for i in ids:
                direct[i] |= set(ids) - {i}
        elif kind in LINK_KINDS and verdict == 'different':
            different.append((d['object_id'], ids))
        elif kind == 'dup' and verdict == 'duplicate':
            for i in panels[0][1:]:
                union(panels[0][0], i)
        elif kind == 'merge' and verdict == 'merged':
            merged.add(panels[0][0])
        else:
            continue
        for i in ids:
            used_docs[i].append(d['object_id'])
    groups = defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    conflicts = []
    for object_id, ids in different:
        roots = {find(i) for i in ids if i in parent}
        if len(roots) == 1 and len([i for i in ids if i in parent]) == len(ids):
            conflicts.append(dict(decision=object_id, masks=ids, root=roots.pop()))
    return dict(groups=[sorted(g) for g in groups.values()], conflicts=conflicts, merged=sorted(merged),
                direct={k: sorted(v) for k, v in direct.items()}, used_docs=used_docs, root=find)


def plan_groups(decided: dict, batch: Batch) -> tuple[list[dict], list[dict]]:
    """Resolve masks to (photo, source annotation id), one per photo, and drop conflicting groups."""
    conflict_roots = {c['root'] for c in decided['conflicts']}
    plans, skipped = [], []
    for n, members in enumerate(decided['groups']):
        gid = f'xv_{n + 1:03d}'
        if decided['root'](members[0]) in conflict_roots:
            skipped.append(dict(group_id=gid, masks=members, reason='conflicts_with_a_different_decision'))
            continue
        by_photo = defaultdict(list)
        for m in members:
            if m in decided['merged']:
                skipped.append(dict(group_id=gid, masks=[m], reason='mask_judged_to_merge_leaves'))
                continue
            item = batch.item(m)
            by_photo[item['image_file']].append((m, item['source_annotation_id']))
        observations = []
        for photo, entries in sorted(by_photo.items()):
            if len(entries) > 1:
                linked = [e for e in entries if set(decided['direct'].get(e[0], ())) - {x for x, _ in entries}]
                if len(linked) != 1:
                    skipped.append(dict(group_id=gid, masks=[e[0] for e in entries], reason='several_masks_in_one_photo_without_one_direct_link'))
                    continue
                skipped.extend(dict(group_id=gid, masks=[e[0]], reason='duplicate_mask_not_adopted') for e in entries if e is not linked[0])
                entries = linked
            observations.append(dict(key=photo.lower(), image_file=photo, mask_id=entries[0][0], source_annotation_id=entries[0][1]))
        if len(observations) < 2:
            skipped.append(dict(group_id=gid, masks=members, reason='fewer_than_two_photos'))
            continue
        plans.append(dict(group_id=gid, observations=observations,
                          provenance=dict(decision_ids=sorted({x for m in members for x in decided['used_docs'].get(m, [])}),
                                          mask_ids=members, method='human cross-view review page decisions',
                                          visibility_note='set separately, only from human vis answers')))
    return plans, skipped


def import_identities(backup_path: Path, decisions: list[Path], batch_root: Path, out: Path) -> dict:
    backup_path, out = Path(backup_path).resolve(), Path(out).resolve()
    if out.exists():
        raise StudyError(f'Output exists; choose a new directory: {out}')
    node = shutil.which('node')
    if not node:
        raise StudyError('Node.js is required to apply identities with the editor code')
    backup = read_json(backup_path)
    docs = load_decisions([Path(p) for p in decisions])
    batch = Batch(Path(batch_root))
    decided = group_decisions(docs, backup['plant_id'])
    plans, skipped = plan_groups(decided, batch)
    capture = backup['capture']
    frames = [dict(key=f['image_file'].lower(), image_file=f['image_file'],
                   width=capture['intrinsics']['image_width_px'], height=capture['intrinsics']['image_height_px'])
              for f in capture['frames']]
    visibility = visibility_decisions(docs, backup['plant_id'], batch)
    job = dict(backup=backup, frames=frames, visibility=visibility, groups=[dict(group_id=p['group_id'], provenance=p['provenance'],
               observations=[dict(key=o['key'], source_annotation_id=o['source_annotation_id']) for o in p['observations']])
               for p in plans])
    run = subprocess.run([node, str(NODE_HELPER)], input=json.dumps(job), capture_output=True, encoding='utf-8', timeout=120)
    if run.returncode != 0:
        raise StudyError('Editor identity import failed: ' + run.stderr[-2000:])
    result = json.loads(run.stdout)
    new = result['backup']
    new['identity_import'] = dict(tool='tools.cross_view_identity_import', schema=SCHEMA, base_backup=str(backup_path),
                                  base_backup_sha256=file_hash(backup_path), decisions=[dict(file=d['_file'], sha256=d['_sha256']) for d in docs],
                                  imported_at=datetime.now(timezone.utc).isoformat())
    out.mkdir(parents=True)
    write_json(out / 'identities.backup.json', new)
    report = dict(schema=SCHEMA, plant_id=backup['plant_id'], decisions_read=len(docs), groups_planned=len(plans),
                  applied=result['applied'], skipped=skipped + result['skipped'], conflicts=decided['conflicts'],
                  visibility_applied=result.get('visibility_applied', []), visibility_skipped=result.get('visibility_skipped', []),
                  plans=plans,
                  compound_membership='not assigned', validated_metric=False, suitable_for_ppfd=False)
    write_json(out / 'identity_import_report.json', report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--backup', required=True, type=Path)
    ap.add_argument('--decisions', required=True, nargs='+', type=Path, help='decision JSON files or folders')
    ap.add_argument('--batch', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    r = import_identities(a.backup, a.decisions, a.batch, a.out)
    print(json.dumps(dict(applied=len(r['applied']), skipped=len(r['skipped']), conflicts=len(r['conflicts']),
                          visibility_applied=len(r['visibility_applied'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
