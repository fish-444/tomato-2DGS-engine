"""Resolve new outline defaults without reinterpreting frozen native requests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from tools.vtc_growth_io import ROOT, read_json, require, sha

DEFAULT_SOURCE = 'user_decision_2026-10-15_21:02'
F4_SOURCE = 'user_decision_f4_legacy_exception_2026-10-15'
F4_WORK_ORDER = ROOT / 'build/f4_runs_20261015_01'


def resolve(folder: Path, out: Path, *, replay_prepared: Path | None = None) -> dict:
    """Return an effective raw request and its resolution evidence; never write inputs.

    Only an adjacent or explicitly supplied hash-bound prepared record establishes
    frozen replay. An unlabelled detached copy is a fresh request, not inferred old
    input. The caller must check every expected replay input hash before preparing.
    """
    original = read_json(folder / 'request.json')
    request = deepcopy(original)
    hooks = request.get('engine_hooks')
    require(hooks is None or isinstance(hooks, dict), 'engine_hooks must be an object')
    explicit = isinstance(hooks, dict) and 'leaflet_outline' in hooks
    if explicit:
        require(type(hooks['leaflet_outline']) is bool, 'leaflet_outline must be boolean')
    adjacent = folder.parent / 'prepared.json'
    if replay_prepared is None and folder.name == 'inputs' and adjacent.is_file():
        replay_prepared = adjacent
    expected = None
    evidence = {}
    if replay_prepared is not None:
        prepared = read_json(replay_prepared)
        require(prepared.get('mode') == 'vtc_development_v3', 'unsupported replay preparation')
        expected = prepared.get('frozen_hashes')
        require(isinstance(expected, dict) and expected.get('request.json') == sha(folder / 'request.json'),
                'replay request hash mismatch; refuse new-default fallback')
        effective = hooks['leaflet_outline'] if explicit else False
        previous = prepared.get('outline_policy')
        if previous is not None:
            require(previous['effective_hooks']['leaflet_outline'] == effective,
                    'frozen outline policy disagrees with request')
        require(('leaflet_outline' in prepared) == effective, 'replay outline metadata inconsistent')
        source = 'frozen_request' if explicit else 'legacy_frozen_request'
        evidence = dict(replay_prepared=str(replay_prepared.resolve()),
                        replay_prepared_sha256=sha(replay_prepared),
                        origin_default_source=prepared.get('default_source', 'legacy_explicit_or_implicit_off'))
    else:
        # Freeze the user's explicit exception in new generated inputs, without
        # changing the running driver or any existing F4 request/result.
        f4 = out.resolve().is_relative_to(F4_WORK_ORDER.resolve())
        if f4:
            require(not explicit or hooks['leaflet_outline'] is False,
                    'F4 work order requires leaflet_outline=false')
        effective = hooks['leaflet_outline'] if explicit else not f4
        source = 'explicit_request' if explicit else F4_SOURCE if f4 else DEFAULT_SOURCE
        if not explicit:
            if hooks is None:
                hooks = dict(schema='fsp-engine-hooks-1')
                request['engine_hooks'] = hooks
            hooks['leaflet_outline'] = effective
    if out.resolve().is_relative_to(F4_WORK_ORDER.resolve()):
        require(effective is False, 'F4 work order cannot prepare an outline-enabled replay')
    metadata = dict(default_source=source, input_role='frozen_replay' if expected is not None else 'fresh_request',
                    effective_hooks=dict(leaflet_outline=effective),
                    default_applied=source in (DEFAULT_SOURCE,F4_SOURCE),
                    source_request_sha256=sha(folder / 'request.json'), **evidence)
    return dict(request=request, rewrite_request=request != original, metadata=metadata,
                expected_frozen_hashes=expected)


def verify_replay(policy: dict, source_hashes: dict) -> None:
    """Validate all replay dependencies, not just the request document."""
    expected = policy['expected_frozen_hashes']
    require(expected is None or expected == source_hashes,
            'replay frozen input hashes changed; refuse new-default fallback')


def shape_registry(request: dict, registry: dict) -> dict:
    """Add policy provenance to the exported registry, never native parameters."""
    from tools.fsp_leaflet_outline import active, LIBRARY_SHA256, EXCLUDED
    result = deepcopy(registry)
    if active(request):
        result['LEAFLET_SHAPE_SOURCE'] = dict(value='library_v2_outline', unit='categorical',
            source_kind='user_decision', source_ref=DEFAULT_SOURCE,
            cultivar='Micro-Tom', uncertainty='software policy; physical accuracy not established',
            scope='leaflets without stored shapes; non-blade organs unchanged',
            library_sha256=LIBRARY_SHA256, candidates=194, excluded=sorted(EXCLUDED),
            evidence=['build/p2_d3_preserving_20261015_09/results/RESULTS.md',
                      'build/p2_d3_diamond_20261015_02/results/RESULTS.md'])
    return result


def main() -> None:
    """Read-only policy and input audit for an intended preparation target."""
    import argparse
    import json
    from tools.vtc_development_io import load_package
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True,help='intended native output path; audit does not write it')
    parser.add_argument('--replay-prepared',type=Path)
    args=parser.parse_args()
    chosen=resolve(args.input,args.out,replay_prepared=args.replay_prepared)
    package=load_package(args.input,request_override=chosen['request'])
    verify_replay(chosen,package['source_hashes'])
    print(json.dumps(dict(status='validated',rewrite_request=chosen['rewrite_request'],
                         outline_policy=chosen['metadata']),indent=2))


if __name__=='__main__':
    main()
