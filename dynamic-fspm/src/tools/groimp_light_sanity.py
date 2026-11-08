"""Phase 1 of the XL light-simulation plan: one lamp, flat receivers, three calculations that must agree (2026-09-29).

    python -X utf8 -m tools.groimp_light_sanity --out build/groimp_light_sanity_<date> [--rays 20000000]

Plan: handoff/CLAUDE_XL_LIGHT_SIMULATION_ADOPTION_2026-09-29.md (Codex) and the answers in
handoff/CLAUDE_XL_LIGHT_REVIEW_ANSWERS_2026-09-29.md. A point lamp of flux P hangs above a row of thin black boxes
lying on the floor; each box receives light at a different distance and angle. The same scene is computed three ways:

* analytic: I = P / (4 pi) per steradian (isotropic point), E = I cos(beta) / d^2, integrated over every face of every box;
* python: tomato3d.light.raytrace.direct_light, the project's own direct-light code, on points sampled over the faces;
* groimp: GroIMP 2.2.1 LightModel (CPU Monte-Carlo raytracer) on the generated RGG, run twice to show the sampling spread.

Units: P is a photon flux in umol s-1, so every receiver value is an absorbed photon rate (umol s-1) and divided by the
top-face area an incident PPFD (umol m-2 s-1) for the top face. Black boxes absorb everything that arrives, so absorbed
equals incident. Coordinates: metres, +Z up, lamp at (0, 0, H); GroIMP Box(length, width, height) extends length along
local +Z from its base, so a box is a plate of thickness `length` whose top face is at z = length.

Nothing here touches the plant models, the existing XL exports or GroIMP projects of other tools.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from tools.leaf_light.common import StudyError, file_hash, output_dir, write_json

CORE = Path(r'C:\Program Files\GroIMP 2.2.1\core.jar')
LAMP_HEIGHT_M = 0.5
FLUX = 1000.0  # umol s-1, an arbitrary test value
PLATE_M, THICK_M = 0.05, 0.0002
XS_M = (0.0, 0.2, 0.4, 0.6, 0.8)

RGG = """// Phase-1 optical sanity scene (tools.groimp_light_sanity). Generated; do not edit.
// metres, +Z up; point lamp flux {flux} (read as umol s-1); black plates absorb all incident light.
import de.grogra.ray.physics.Spectrum;
module Receiver(int rid) extends Box({thick}f, {plate}f, {plate}f);
module Lamp extends LightNode;
LightModel lm = new LightModel({rays}, 1);

protected void init()
[
    Axiom ==> [ Translate(0, 0, {height}f) Lamp().(setLight(new PointLight().(setPower({flux}f)))) ]
{receivers};
]

public void measure()
{{
    for (int run = 1; run <= 2; run++) {{
        lm.compute();
        for (Receiver r : (* Receiver *)) {{
            Spectrum s = lm.getAbsorbedPower(r);
            println("RESULT\\t" + run + "\\t" + r.rid + "\\t" + s.integrate());
        }}
    }}
    println("SANITY_DONE");
}}
"""


def faces(x: float, n: int = 40) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """(points, normal, area per point) for the top face and the four side faces of the plate centred at (x, 0)."""
    h, t = PLATE_M / 2, THICK_M
    g = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(g, g, indexing='ij')
    out = []
    top = np.stack([x - h + u.ravel() * PLATE_M, -h + v.ravel() * PLATE_M, np.full(n * n, t)], 1)
    out.append((top, np.array([0.0, 0.0, 1.0]), PLATE_M * PLATE_M / (n * n)))
    zs = t * (np.arange(4) + 0.5) / 4
    for normal, fixed in (((1, 0, 0), x + h), ((-1, 0, 0), x - h), ((0, 1, 0), h), ((0, -1, 0), -h)):
        along = g * PLATE_M - h
        a, z = np.meshgrid(along, zs, indexing='ij')
        if normal[0]:
            pts = np.stack([np.full(a.size, fixed), a.ravel(), z.ravel()], 1)
        else:
            pts = np.stack([x + a.ravel(), np.full(a.size, fixed), z.ravel()], 1)
        out.append((pts, np.array(normal, float), PLATE_M * t / a.size))
    return out


def analytic(x: float) -> dict:
    lamp = np.array([0.0, 0.0, LAMP_HEIGHT_M])
    total, top = 0.0, 0.0
    for k, (pts, normal, dA) in enumerate(faces(x, 200)):
        d = lamp - pts
        r2 = (d * d).sum(1)
        cosb = np.clip(d @ normal / np.sqrt(r2), 0, None)
        rate = float((FLUX / (4 * math.pi) * cosb / r2 * dA).sum())
        total += rate
        top += rate if k == 0 else 0.0
    return dict(absorbed_umol_s=total, top_umol_s=top)


def python_module(x: float) -> dict:
    """The project's direct-light code (tomato3d.light.raytrace) on the same faces, points in mm."""
    from types import SimpleNamespace
    from tomato3d.io.schemas import LightSimulationSettings
    from tomato3d.light.raytrace import Occluders, Receivers, direct_light
    from tomato3d.light.scene import Emitters
    em = Emitters(fixture_id='point', positions_grow_mm=np.array([[0.0, 0.0, LAMP_HEIGHT_M * 1000]]),
                  aim=np.array([0.0, 0.0, -1.0]), axial_intensity_umol_s_sr=FLUX / (4 * math.pi),
                  cosine_exponent=0.0, cos_cutoff=-1.0)  # isotropic over the whole sphere, like GroIMP's PointLight
    state = SimpleNamespace(emitters=em, fixture_id='point')
    settings = LightSimulationSettings(triangle_subdivisions=1, ray_offset_mm=0.01, max_ray_triangle_pairs_per_chunk=1_000_000)
    total, top = 0.0, 0.0
    for k, (pts, normal, dA) in enumerate(faces(x, 200)):
        rec = Receivers(points_grow_mm=pts * 1000, unit_normals=np.tile(normal, (len(pts), 1)), two_sided=False,
                        own_triangle=np.full(len(pts), -1))
        ppfd = direct_light(rec, (state,), Occluders.none(), None, settings).ppfd_umol_m2_s[0]
        rate = float((ppfd * dA).sum())
        total += rate
        top += rate if k == 0 else 0.0
    return dict(absorbed_umol_s=total, top_umol_s=top)


def run_groimp(project: Path, core: Path = CORE, java: str = 'java', timeout_s: int = 1800) -> tuple[dict, str]:
    from tools.groimp_static_verify import cli_path  # read-only reuse of the existing CLI path helper
    log_path = project / 'groimp.log'
    with log_path.open('wb') as stream:
        p = subprocess.Popen([java, '-Dfile.encoding=UTF-8', '-jar', str(core), '--headless', '-a', 'cli'],
                             stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        end = time.monotonic() + timeout_s

        def wait(pred):
            while time.monotonic() < end:
                text = log_path.read_text(encoding='utf-8', errors='replace')
                if pred(text):
                    return text
                if p.poll() is not None:
                    break
                time.sleep(0.25)
            raise StudyError(f'GroIMP did not finish; see {log_path}')
        try:
            wait(lambda s: '[base]>>' in s)
            p.stdin.write(f'$cd {cli_path(project)}\n$open Model.rgg\n'); p.stdin.flush()
            wait(lambda s: '[Model]>>' in s)
            time.sleep(2)
            p.stdin.write('measure\n'); p.stdin.flush()
            text = wait(lambda s: 'SANITY_DONE' in s or 'Command not found' in s)
        finally:
            if p.poll() is None:
                p.stdin.close(); p.kill(); p.wait()
    if 'SANITY_DONE' not in text:
        raise StudyError(f'measure() did not run (RGG not compiled?); see {log_path}')
    runs: dict = {}
    for run, rid, value in re.findall(r'RESULT\t(\d+)\t(\d+)\t([0-9.eE+-]+)', text):
        runs.setdefault(int(rid), {})[int(run)] = float(value)
    version = (re.findall(r'V (\d+\.\d+\.\d+)', text) or ['unknown'])[0]
    return runs, version


def build(out: Path, rays: int = 20_000_000, core: Path = CORE, java: str = 'java') -> dict:
    out = output_dir(Path(out))
    receivers = '\n'.join(f'              [ Translate({x}f, 0, 0) Receiver({i}).(setShader(BLACK)) ]' for i, x in enumerate(XS_M))
    rgg = RGG.format(flux=FLUX, thick=THICK_M, plate=PLATE_M, rays=rays, height=LAMP_HEIGHT_M, receivers=receivers)
    (out / 'Model.rgg').write_text(rgg, encoding='utf-8')
    started = time.time()
    g, version = run_groimp(out, core, java)
    seconds = round(time.time() - started, 1)
    rows, worst = [], 0.0
    # GroIMP 2.2.1 repeats identical numbers for repeated compute() calls (fixed random stream, seen 09-29), so the two
    # runs are not independent samples: the Monte-Carlo sigma is that of one run
    deterministic = all(g[i][1] == g[i][2] for i in range(len(XS_M)))
    top_area = PLATE_M * PLATE_M
    for i, x in enumerate(XS_M):
        a, py = analytic(x), python_module(x)
        g1, g2 = g[i][1], g[i][2]
        gm = (g1 + g2) / 2
        hits = rays * a['absorbed_umol_s'] / FLUX  # expected rays absorbed by this box
        sigma = 1 / math.sqrt(hits) if hits > 0 else float('inf')
        dev = (gm - a['absorbed_umol_s']) / a['absorbed_umol_s']
        sigma = sigma if deterministic else sigma / math.sqrt(2)
        worst = max(worst, abs(dev) / sigma)
        rows.append(dict(receiver=i, x_m=x, incidence_deg=round(math.degrees(math.atan2(x, LAMP_HEIGHT_M)), 1),
                         distance_m=round(math.hypot(x, LAMP_HEIGHT_M), 4),
                         analytic_umol_s=a['absorbed_umol_s'], python_umol_s=py['absorbed_umol_s'],
                         groimp_run1_umol_s=g1, groimp_run2_umol_s=g2,
                         top_face_ppfd_analytic_umol_m2_s=a['top_umol_s'] / top_area,
                         top_face_ppfd_python_umol_m2_s=py['top_umol_s'] / top_area,
                         python_vs_analytic=(py['absorbed_umol_s'] - a['absorbed_umol_s']) / a['absorbed_umol_s'],
                         groimp_vs_analytic=dev, groimp_run_spread=(g1 - g2) / gm,
                         expected_mc_sigma=sigma))
    with open(out / 'sanity_results.csv', 'w', encoding='utf-8') as fh:
        fh.write(','.join(rows[0]) + '\n')
        for r in rows:
            fh.write(','.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in r.values()) + '\n')
    passed = dict(python_within_0_5pct=all(abs(r['python_vs_analytic']) < 0.005 for r in rows),
                  groimp_within_3_sigma=worst <= 3.0)
    manifest = dict(schema='groimp-light-sanity-1', status='passed' if all(passed.values()) else 'failed', checks=passed,
                    worst_groimp_deviation_in_sigma=round(worst, 2), groimp_version=version, core_sha256=file_hash(core),
                    backend='GroIMP LightModel (CPU Monte-Carlo raytracer), depth 1', rays_per_run=rays, runs=2,
                    groimp_seconds=seconds, groimp_repeat_identical=deterministic, model_sha256=file_hash(out / 'Model.rgg'),
                    units=dict(lamp='photon flux umol s-1 (PointLight power read as umol s-1)',
                               receiver='absorbed photon rate umol s-1; top-face PPFD umol m-2 s-1'),
                    frame='metres, +Z up, lamp at (0, 0, H); Box(length) extends along local +Z',
                    scene=dict(lamp_height_m=LAMP_HEIGHT_M, flux_umol_s=FLUX, plate_m=PLATE_M, thickness_m=THICK_M, xs_m=XS_M),
                    assumptions=['PointLight is isotropic over 4 pi sr', 'black shader absorbs 100 %, no reflection',
                                 'side faces included in every calculation'],
                    validated_metric=False, suitable_for_ppfd=False,
                    note='software sanity check of units, axes and sidedness only; says nothing about the tomato rig')
    write_json(out / 'manifest.json', manifest)
    return dict(manifest=manifest, rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--rays', type=int, default=20_000_000)
    ap.add_argument('--core', type=Path, default=CORE)
    ap.add_argument('--java', default='java')
    a = ap.parse_args()
    r = build(a.out, a.rays, a.core, a.java)
    print(json.dumps(dict(status=r['manifest']['status'], checks=r['manifest']['checks'],
                          worst_sigma=r['manifest']['worst_groimp_deviation_in_sigma'],
                          rows=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                 if k in ('receiver', 'incidence_deg', 'analytic_umol_s', 'python_umol_s', 'groimp_run1_umol_s',
                                          'groimp_run2_umol_s', 'groimp_vs_analytic', 'expected_mc_sigma')} for row in r['rows']]),
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
