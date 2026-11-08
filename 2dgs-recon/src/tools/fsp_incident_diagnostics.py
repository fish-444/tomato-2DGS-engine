"""Default-off, pre-growth incident-light observer using the reviewed B1 collector."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import struct

from tools.vtc_growth_io import require, sha

SWITCH = 'incident_diagnostics'
OUTPUTS = ('incident_by_interval.csv', 'incident_counts.csv')
FIELDS = ('interval', 'plant_instance_id', 'organ_id', 'kind', 'area_m2',
          'incident_upper_ppfd_umol_m2_s', 'incident_lower_ppfd_umol_m2_s',
          'incident_two_sided_ppfd_umol_m2_s', 'intercepted_umol_s', 'absorbed_umol_s')
COUNT_FIELDS = ('interval', 'observed_organs', 'zero_absorptance_organs')
NATIVE = Path(__file__).with_name('fsp_incident_diagnostics_native.rgg')


def active(request: dict) -> bool:
    return request.get('engine_hooks', {}).get(SWITCH, False) is True


def validate(request: dict) -> None:
    hooks = request.get('engine_hooks', {})
    require(type(hooks.get(SWITCH, False)) is bool, 'incident_diagnostics must be boolean')
    if active(request):
        require(not request.get('native_boundary'), 'incident_diagnostics cannot observe the empirical/native_boundary path')


def _float32(value: float) -> float:
    return struct.unpack('f', struct.pack('f', float(value)))[0]


def absorptance(request: dict, kind: str) -> float:
    """Actual constant Phong coefficient after the existing float32 casts."""
    ext = request.get('extensions')
    if not ext:
        return 1.
    material = ext['materials'][ext['leaf_material' if kind == 'Leaflet' else 'nonleaf_material']]
    return max(0., 1.-_float32(material['reflectance'])-_float32(material['transmittance']))


def transform_native(template: str, request: dict) -> str:
    if not active(request):
        return template
    validate(request)
    from tools.fsp_stage_parameters import active as stage_active
    from tools.p2_light_revision_native import library_source
    collector_hash = hashlib.sha256(library_source().encode('utf-8')).hexdigest()
    code = '// B1 collector source SHA-256: '+collector_hash+'\n'+NATIVE.read_text(encoding='utf-8')
    code = code.replace('@INCIDENT_LEAF_ALPHA@', repr(absorptance(request, 'Leaflet')))
    code = code.replace('@INCIDENT_NONLEAF_ALPHA@', repr(absorptance(request, 'Internode')))
    code = code.replace('@INCIDENT_KIND@', '(o.s1Fruit?"Fruit":kinds[o.kind])' if stage_active(request) else 'kinds[o.kind]')

    def replace(old: str, new: str) -> None:
        nonlocal template
        require(template.count(old) == 1, 'incident hook insertion point changed: '+old[:80])
        template = template.replace(old, new)

    replace('@BOUNDARY_CODE@', '@BOUNDARY_CODE@\n'+code)
    replace('setShader(organShader(o))', 'setShader(new Revision.ObservedShader(organShader(o)))')
    replace('else {lm.setSeed(seed);lm.compute();extensionLight();',
            'else {lm.setSeed(seed);lm.compute(new Revision.FaceSpectrum());extensionLight();incidentMeasure();')
    replace('attachGraph();if(!observationMode)updateGeometry();commitGeometry();extensionStart();boundaryStart();exportState(0);',
            'incidentStart();attachGraph();if(!observationMode)updateGeometry();commitGeometry();extensionStart();boundaryStart();exportState(0);')
    replace('names=boundaryFiles(names,boundary);',
            'incidentOutput.flush();incidentCounts.flush();names=incidentFiles(names);\n    names=boundaryFiles(names,boundary);')
    replace('if(integrationMode)proposals.close();',
            'incidentOutput.close();incidentCounts.close();\n    if(integrationMode)proposals.close();')
    replace('names=boundaryFiles(names,n);', 'names=incidentFiles(names);\n    names=boundaryFiles(names,n);')
    return template


def prepare_helper(working: Path, compiler_jar: Path | None = None) -> dict:
    """Compile a project-local B1 helper with an explicit, recorded ECJ dependency.

    No download, installation, cache deletion or historical-binary substitution.
    Class files and compiler logs remain beside the bound source and helper jar.
    """
    from tools.p2_light_revision_native import compile_helper, library_source
    compiler = compiler_jar or os.environ.get('FSP_INCIDENT_ECJ_JAR')
    require(bool(compiler), 'incident helper requires compiler_jar or FSP_INCIDENT_ECJ_JAR')
    compiler = Path(compiler).resolve()
    require(compiler.is_file(), 'incident ECJ compiler unavailable')
    folder = Path(working)/'incident_helper'
    require(not folder.exists(), 'incident helper output already exists; preserve and use a fresh preparation')
    source_hash = hashlib.sha256(library_source().encode('utf-8')).hexdigest()
    jar = compile_helper(folder, compiler)
    source = folder/'Revision.java'
    require(hashlib.sha256(source.read_text(encoding='utf-8').encode('utf-8')).hexdigest() == source_hash,
            'incident collector source changed during preparation')
    paths = (jar, source)
    return dict(schema='fsp-incident-diagnostics-1', helper_jar=jar.relative_to(working).as_posix(),
                auxiliary_hashes={p.relative_to(working).as_posix(): sha(p) for p in paths},
                collector_source_sha256=source_hash,
                compiler=dict(path=str(compiler), sha256=sha(compiler)),
                collector='tools.p2_light_revision_native.library_source / B1 ObservedShader + FaceSpectrum',
                area_basis='actual placed float32 triangle mesh, one side, before interval growth',
                photon_conversion='photonPerJoule once on each native power',
                zero_absorptance='incident fields unavailable; absorbed flux retained; interval counts exported',
                zero_mesh_area='PPFD unavailable; observed flux retained',
                output_files=list(OUTPUTS))


def converted_quantities(area_m2: float, upper_power: float, lower_power: float,
                         absorbed_power: float, photon_per_joule: float,
                         alpha: float = 1.) -> dict:
    """Independent arithmetic reference, also usable for the static B1 unit check."""
    values = (area_m2, upper_power, lower_power, absorbed_power, photon_per_joule, alpha)
    require(all(math.isfinite(v) and v >= 0 for v in values) and photon_per_joule > 0,
            'invalid incident conversion input')
    upper = upper_power*photon_per_joule
    lower = lower_power*photon_per_joule
    total = upper+lower
    ppfd = alpha > 0 and area_m2 > 0
    return dict(incident_upper_ppfd_umol_m2_s=upper/area_m2 if ppfd else 'unavailable',
                incident_lower_ppfd_umol_m2_s=lower/area_m2 if ppfd else 'unavailable',
                incident_two_sided_ppfd_umol_m2_s=total/area_m2 if ppfd else 'unavailable',
                intercepted_umol_s=total if alpha > 0 else 'unavailable',
                absorbed_umol_s=absorbed_power*photon_per_joule)


def _mesh_areas(mesh_rows) -> dict:
    """Bounded-memory triangle summation of exported float32 transport vertices."""
    areas = {}
    key = None
    points = []
    vertex = 0
    for row in mesh_rows:
        current = (int(row['state']), row['organ_id'])
        if current != key:
            require(not points, 'incident consumed mesh incomplete triangle')
            require(current not in areas, 'incident consumed mesh noncontiguous organ')
            key = current
            areas[key] = 0.
            vertex = 0
        require(int(row['vertex']) == vertex, 'incident consumed mesh vertex ordering')
        vertex += 1
        point = tuple(_float32(float(row[k])) for k in ('x_m', 'y_m', 'z_m'))
        require(all(math.isfinite(v) for v in point), 'incident consumed mesh nonfinite')
        points.append(point)
        if len(points) == 3:
            a, b, c = points
            u = tuple(b[i]-a[i] for i in range(3)); v = tuple(c[i]-a[i] for i in range(3))
            x = u[1]*v[2]-u[2]*v[1]; y = u[2]*v[0]-u[0]*v[2]; z = u[0]*v[1]-u[1]*v[0]
            areas[key] += math.sqrt(x*x+y*y+z*z)/2
            points = []
    require(not points, 'incident consumed mesh incomplete triangle')
    return areas


def validate_outputs(pkg: dict, data: dict, n: int) -> dict:
    request = pkg['request']
    if not active(request):
        require(not set(OUTPUTS).intersection(data), 'disabled incident output present')
        return dict(enabled=False)
    validate(request)
    require(set(OUTPUTS) <= set(data), 'incident output missing')
    areas = _mesh_areas(data['consumed.csv'])
    states = {(int(s['day']), s['organ_id']): s for s in data['states.csv']}
    flux = {(int(s['interval']), s['organ_id']): s for s in data['flux.csv']}
    observed = {}; totals = {i: [0, 0] for i in range(n)}

    def number(value, label: str) -> float:
        try:
            value = float(value)
        except (ValueError, TypeError):
            require(False, 'incident '+label+' must be numeric')
        require(math.isfinite(value) and value >= 0, 'incident '+label+' invalid')
        return value

    def close(a: float, b: float, label: str, rtol: float = 1e-12) -> None:
        require(abs(a-b) <= 1e-12+rtol*max(abs(a), abs(b)), 'incident '+label+' mismatch')

    for row in data[OUTPUTS[0]]:
        require(tuple(row) == FIELDS, 'incident table fields changed')
        i = int(row['interval']); oid = row['organ_id']; key = (i, oid)
        require(key not in observed and i in totals and key in areas and key in states,
                'incident duplicate or unexpected organ/interval')
        observed[key] = row
        state = states[key]
        require(row['plant_instance_id'] == state['plant_instance_id'] and row['kind'] == state['class'],
                'incident plant/kind is not the pre-growth state')
        area = number(row['area_m2'], 'area_m2')
        require((area == 0) == (areas[key] == 0)
                and math.isclose(area, areas[key], rel_tol=1e-12, abs_tol=1e-18),
                'incident transport mesh area mismatch')
        alpha = absorptance(request, row['kind'])
        absorbed = number(row['absorbed_umol_s'], 'absorbed_umol_s')
        totals[i][0] += 1
        if alpha <= 0:
            totals[i][1] += 1
            require(all(row[k] == 'unavailable' for k in FIELDS[5:9]), 'zero-absorptance incident must be unavailable')
            close(absorbed, 0., 'zero-absorptance energy')
        else:
            intercepted = number(row['intercepted_umol_s'], 'intercepted_umol_s')
            if area == 0:
                require(all(row[k] == 'unavailable' for k in FIELDS[5:8]), 'zero-area PPFD must be unavailable')
            else:
                upper, lower, total = (number(row[k], k) for k in FIELDS[5:8])
                close(upper+lower, total, 'side sum')
                close(total*area, intercepted, 'area times PPFD')
            close(intercepted*alpha, absorbed, 'absorptance identity', rtol=3e-7)
        if row['kind'] == 'Leaflet':
            require(key in flux, 'incident leaflet flux missing')
            close(absorbed, number(flux[key]['absorbed_umol_s'], 'growth absorbed flux'), 'growth flux unchanged')
    require(set(observed) == set(areas), 'incident transport surface coverage')
    counts = {}
    for row in data[OUTPUTS[1]]:
        require(tuple(row) == COUNT_FIELDS, 'incident count fields changed')
        i = int(row['interval'])
        require(i in totals and i not in counts, 'incident count interval coverage')
        counts[i] = [int(row['observed_organs']), int(row['zero_absorptance_organs'])]
        require(counts[i] == totals[i], 'incident counts mismatch')
    require(set(counts) == set(totals), 'incident count interval missing')
    return dict(enabled=True, observed_organ_intervals=len(observed),
                zero_absorptance_organ_intervals=sum(v[1] for v in totals.values()),
                area_basis='actual pre-growth transport mesh')
