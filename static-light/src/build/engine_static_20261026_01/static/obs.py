"""Observed / model plants in the cabinet (CONTRACT.md §1–4): placement, receivers, sensors, cabinet traces, aggregation."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from . import light as L
from .geometry import Surfels, EngineSnapshot
from .tracer import Scene, trace_cabinet, coincident

REPO = Path(__file__).resolve().parents[3]
R31B_PLY = REPO / 'build/p1_r31b_reuse_20261024_01/source/R3-1b_surfels_full.ply'
R31B_IDS = REPO / 'build/p1_vp_cleanup_20261026_01/surfel_leaf_ids.csv'
RECON = REPO / 'build/p1_recon_all_20261025_01/outputs'
OBSERVED = ['R3-1b', 't2_02', 't2_04', 't2_05', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06']
USABLE = {1, 7}
LEAVES = set(range(1, 9))
RIM_Z = 0.070
MEDIUM_Z = 0.060          # grow frame: rim 70 mm - 10 mm (pot_spec); board z taken as grow z (declared)
POT_SIDE = 0.059
SCALE_LABEL = 'conditional_on_unvalidated_scale'   # pot not in the reconstruction products: rim check impossible
SCENE_XY = np.array([0.280, 0.210])


def ply_path(pid: str) -> Path:
    return R31B_PLY if pid == 'R3-1b' else RECON / pid / f'R3-1b-style_{pid}_surfels_full.ply'


def r31b_labels(n: int) -> np.ndarray:
    lab = np.full(n, -2, np.int64)
    with open(R31B_IDS, encoding='utf-8') as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith('#')][1:]
    for r in rows:
        lab[int(r[0])] = int(r[1])
    return lab


_CACHE: dict = {}


def load_plant(pid: str, footprint: float = 1.0) -> Surfels:
    key = (pid, footprint)
    if key not in _CACHE:
        labels = None
        if pid == 'R3-1b':
            from .geometry import read_ply
            labels = r31b_labels(len(read_ply(R31B_PLY)))
        _CACHE[key] = Surfels(ply_path(pid), scale=footprint, labels=labels)
    return _CACHE[key]


def base_xy(s: Surfels) -> np.ndarray:
    """Stem-base proxy for the pot centre: area-weighted xy of surfels within 5 mm of the plant's lowest 0.5 % z."""
    z0 = np.percentile(s.centre[:, 2], 0.5); m = s.centre[:, 2] <= z0 + 0.005
    w = s.area_b[m]
    return (s.centre[m, :2] * w[:, None]).sum(0) / w.sum()


class Placed:
    """One plant placed at grow xy `pos` (metres)."""

    def __init__(self, pid: str, pos: np.ndarray, footprint: float = 1.0):
        self.pid = pid; self.s = load_plant(pid, footprint); self.pos = np.asarray(pos, float)
        b = base_xy(self.s); self.offset = np.array([pos[0] - b[0], pos[1] - b[1], 0.0])
        self.centre = self.s.centre + self.offset
        if self.s.labels is not None:
            self.leaf = np.isin(self.s.labels, list(LEAVES)); self.leaf_id = np.where(self.leaf, self.s.labels, -1)
        else:
            self.leaf = np.ones(len(self.s), bool); self.leaf_id = np.full(len(self.s), 0)   # canopy_surface (no split)

        self.normal = self.s.normal; self.area = self.s.area_b; self.area_a = self.s.area_a
        self.own_local = np.arange(len(self.s)); self.delta = None; self.coincident_half = False

    def prims(self):
        return self.s.prims(self.offset)


class PlacedModel:
    """One engine plant (consumed triangles of one state) placed with its stem base at grow xy `pos` and engine z = 0 on the medium surface.
    Receivers: `samples` points per leaflet triangle; coincident receivers (mirrored pair overlap) get credit 1/2 as in gate S1 v2."""

    def __init__(self, snap: EngineSnapshot, plant: str, base_engine: np.ndarray, pos: np.ndarray, samples: int = 8, seed: int = 5):
        self.pid = 'model_' + plant; self.pos = np.asarray(pos, float)
        keep = np.array([snap.plant.get(o) == plant for o in snap.owner]); self.keep = keep
        self.offset = np.array([pos[0] - base_engine[0], pos[1] - base_engine[1], MEDIUM_Z - base_engine[2]])
        P, k = snap.prims(self.offset, keep); self._P, self._k = P, k
        R, N, w, own, organ = snap.leaf_receivers(samples, seed, self.offset)
        mine = keep[own]; local = np.cumsum(keep) - 1
        self.centre = R[mine]; self.normal = N[mine]; self.area = w[mine]; self.area_a = w[mine]; self.own_local = local[own[mine]]
        self.leaf = np.ones(mine.sum(), bool); self.leaf_id = np.zeros(mine.sum(), np.int64); self.organ = organ[mine]
        self.delta = 1e-7; self.coincident_half = True
        self.s = self

    def __len__(self):
        return len(self.centre)

    def prims(self):
        return self._P, self._k


def sensors_for(pos: np.ndarray, top_z: float, spacing: float = 0.002, layer: float = 0.005):
    """Horizontal point sensors on a 2 mm grid over the pot square at each 5 mm layer mid-height above the medium surface."""
    g = np.arange(-POT_SIDE / 2 + spacing / 2, POT_SIDE / 2, spacing)
    gx, gy = np.meshgrid(g, g)
    nlay = int(math.ceil((top_z - MEDIUM_Z) / layer))
    pts, lay = [], []
    for k in range(nlay):
        z = MEDIUM_Z + (k + 0.5) * layer
        p = np.stack([gx.ravel() + pos[0], gy.ravel() + pos[1], np.full(gx.size, z)], 1); pts.append(p); lay.append(np.full(len(p), k))
    return np.concatenate(pts), np.concatenate(lay)


class FieldFast:
    """Vectorised copy of tools.empirical_light_field.evaluate (bilinear per plane, linear z, no extrapolation); checked against it."""

    def __init__(self):
        self.mf = L.MeasuredField(); S = self.mf.field['samples']; self.planes = {}
        for z in (70.0, 150.0):
            pl = [s for s in S if s['z_mm'] == z]; xs = np.array(sorted({s['x_mm'] for s in pl})); ys = np.array(sorted({s['y_mm'] for s in pl}))
            v = np.zeros((len(xs), len(ys)))
            for s in pl: v[np.searchsorted(xs, s['x_mm']), np.searchsorted(ys, s['y_mm'])] = s['ppfd_umol_m2_s']
            self.planes[z] = (xs, ys, v)

    def _plane(self, z, x, y):
        xs, ys, v = self.planes[z]
        i = np.clip(np.searchsorted(xs, x, side='right') - 1, 0, len(xs) - 2); j = np.clip(np.searchsorted(ys, y, side='right') - 1, 0, len(ys) - 2)
        tx = (x - xs[i]) / (xs[i + 1] - xs[i]); ty = (y - ys[j]) / (ys[j + 1] - ys[j])
        return (v[i, j] * (1 - tx) * (1 - ty) + v[i + 1, j] * tx * (1 - ty) + v[i, j + 1] * (1 - tx) * ty + v[i + 1, j + 1] * tx * ty)

    def U(self, p_m: np.ndarray):
        p = np.asarray(p_m) * 1000.0; ok = L.MeasuredField.in_support(p); out = np.full(len(p), np.nan)
        x, y, z = p[ok, 0], p[ok, 1], p[ok, 2]; tz = (z - 70.0) / 80.0
        out[ok] = ((1 - tz) * self._plane(70.0, x, y) + tz * self._plane(150.0, x, y)) / L.F_MAX
        return out, ok

    def check(self, n: int = 300, seed: int = 3) -> float:
        from tools.empirical_light_field import evaluate
        rng = np.random.default_rng(seed)
        p = np.stack([rng.uniform(*L.SUPPORT['x'], n), rng.uniform(*L.SUPPORT['y'], n), rng.uniform(*L.SUPPORT['z'], n)], 1)
        fast, _ = self.U(p / 1000.0)
        ref = np.array([evaluate(self.mf.field, list(q))['ppfd_umol_m2_s'] / L.F_MAX for q in p])
        return float(np.max(np.abs(fast - ref)))


def hdir(p: np.ndarray, bars: np.ndarray, m: float) -> np.ndarray:
    """Unoccluded model direct horizontal (absolute model units), same expression as the tracer's wsum_h."""
    v = bars[None, :, :] - p[:, None, :]; d = np.linalg.norm(v, axis=2); uz = v[:, :, 2] / d
    return np.where(uz > 0, uz ** m * uz / d ** 2, 0.0).sum(1)


def unshaded_U(field: FieldFast, p: np.ndarray, bars_ref: np.ndarray, bars_s: np.ndarray | None, m: float, f_d: float = L.DIFFUSE):
    """U on the measured route (in support) and with the model extension (outside: clamped measured x model ratio); scenario ratio applied."""
    U, ok = field.U(p)
    pm = p * 1000.0
    c = np.stack([np.clip(pm[:, 0], *L.SUPPORT['x']), np.clip(pm[:, 1], *L.SUPPORT['y']), np.clip(pm[:, 2], *L.SUPPORT['z'])], 1) / 1000.0
    Uc, _ = field.U(c)
    href = hdir(p, bars_ref, m); href_c = hdir(c, bars_ref, m)
    ext = Uc * ((1 - f_d) * href / href_c + f_d)
    Uext = np.where(ok, U, ext)
    if bars_s is not None:
        r = (1 - f_d) * hdir(p, bars_s, m) / href + f_d
        U = U * r; Uext = Uext * r
    return U, Uext, ok


def trace_plants(placed: list[Placed], scene_set: list[Placed], seed: int, dz_mm: float = 0.0, dy_mm: float = 0.0, delta: float = 0.005,
                 sensors: bool = True, field: FieldFast | None = None):
    """Cabinet kernel for every receiver of `placed` plants inside the occluder set `scene_set`; returns per-plant arrays."""
    Ps, ks = [], []
    owner_offset, start = {}, 0
    for q in scene_set:
        P, k = q.prims(); Ps.append(P); ks.append(k); owner_offset[q.pid] = start; start += len(P)
    if not Ps:
        Ps, ks = [np.zeros((0, 14))], [np.zeros(0, np.int8)]
    scene = Scene(np.concatenate(Ps), np.concatenate(ks), h=0.003) if Ps else Scene(np.zeros((0, 14)), np.zeros(0, np.int8))
    m = L.cos_m(); bars = L.bar_points(seed, dz_mm=dz_mm, dy_mm=dy_mm); bars_ref = L.bar_points(seed); dm = L.dome(seed)
    out = {}
    for q in placed:
        own = (q.own_local + owner_offset[q.pid]) if q.pid in owner_offset else np.full(len(q.centre), -1)
        ks_, ku_, hd, parts = trace_cabinet(scene, q.centre, q.normal, True, own, delta if q.delta is None else q.delta, bars, m, dm, L.DIFFUSE)
        if q.coincident_half and len(scene.P):
            half = coincident(scene, q.centre, q.normal, own); parts = parts.copy(); parts[half] *= 0.5
        else:
            half = None
        inpot = (np.abs(q.centre[:, 0] - q.pos[0]) <= POT_SIDE / 2) & (np.abs(q.centre[:, 1] - q.pos[1]) <= POT_SIDE / 2)
        rec = dict(coincident=half, parts=parts, area=q.area, area_a=q.area_a, leaf=q.leaf, leaf_id=q.leaf_id, z=q.centre[:, 2], inpot=inpot)
        if field is not None:
            rec['U'], rec['Uext'], rec['ok'] = unshaded_U(field, q.centre, bars_ref, None if (dz_mm == 0 and dy_mm == 0) else bars, m)
        if field is not None:
            g = np.arange(-POT_SIDE / 2 + 0.001, POT_SIDE / 2, 0.002); gx, gy = np.meshgrid(g, g)
            G = np.stack([gx.ravel() + q.pos[0], gy.ravel() + q.pos[1], np.full(gx.size, 0.070)], 1)
            rec['U_ground'], _, _ = unshaded_U(field, G, bars_ref, None if (dz_mm == 0 and dy_mm == 0) else bars, m)
        if sensors:
            S, lay = sensors_for(q.pos, q.centre[:, 2].max())
            Ns = np.tile([0, 0, 1.0], (len(S), 1))
            s_s, s_u, _, sp = trace_cabinet(scene, S, Ns, False, np.full(len(S), -1), 1e-7, bars, m, dm, L.DIFFUSE)
            rec['sensor_parts'] = sp; rec['sensor_layer'] = lay; rec['sensor_xyz'] = S
        out[q.pid] = rec
    return out


def kernel(parts: np.ndarray, f_d: float = L.DIFFUSE, shaded: bool = True) -> np.ndarray:
    return (1 - f_d) * parts[:, 0 if shaded else 1] + f_d * parts[:, 2 if shaded else 3]


def summarize(rec: dict, pid: str, f_d: float = L.DIFFUSE, pitch: float | None = None) -> dict:
    """Per-plant (and R3-1b per usable leaf) numbers from one trace. Relative units (r.u.); LUE conditional on the meter scale."""
    K = kernel(rec['parts'], f_d); Ku = kernel(rec['parts'], f_d, shaded=False)
    a = rec['area']; leaf = rec['leaf']
    out = dict(plant=pid, receivers=int(len(a)), leaf_area_b_m2=float(a[leaf].sum()), leaf_area_a_m2=float(rec['area_a'][leaf].sum()),
               ratio_b_over_a=float(a[leaf].sum() / rec['area_a'][leaf].sum()), surface_area_b_m2=float(a.sum()),
               lai_pot_top=float(a[leaf].sum() / POT_SIDE ** 2), lai_tray_cell=float(a[leaf].sum() / ((pitch or 0.060) ** 2)),
               shaded_fraction_leaf=float(1 - (K[leaf] * a[leaf]).sum() / (Ku[leaf] * a[leaf]).sum()),
               mean_kernel_leaf=float((K[leaf] * a[leaf]).sum() / a[leaf].sum()))
    if rec.get('coincident') is not None:
        c = rec['coincident']
        out['coincident_area_fraction'] = float(a[c].sum() / a.sum()); out['leaf_area_unique_m2'] = float(a.sum() - 0.5 * a[c].sum())
    if 'U' in rec:
        ok = rec['ok']; U, Ue = rec['U'], rec['Uext']
        inc_in = np.where(ok, U * K, 0.0); inc_ext = Ue * K
        out.update(area_fraction_outside_support=float(a[leaf & ~ok].sum() / a[leaf].sum()),
                   absorbed_in_support_ru_m2=float((inc_in * a)[leaf].sum()), absorbed_with_extension_ru_m2=float((inc_ext * a)[leaf].sum()),
                   mean_incident_with_extension_ru=float((inc_ext * a)[leaf].sum() / a[leaf].sum()))
        q = inc_ext * L.F_MAX
        g = L.gross_gC(q[leaf], a[leaf]); mol = L.absorbed_mol(q[leaf], a[leaf])
        out['lue_gC_mol_conditional_on_meter_scale'] = float(g.sum() / mol.sum()) if mol.sum() > 0 else None
        ground = np.array(rec.get('U_ground', [np.nan]))
        out['intercepted_per_pot_top_area'] = float(out['absorbed_with_extension_ru_m2'] / (np.nanmean(ground) * POT_SIDE ** 2)) if np.isfinite(np.nanmean(ground)) else None
    if pid == 'R3-1b':
        per_leaf = {}
        for lid in sorted(USABLE):
            m = rec['leaf_id'] == lid
            d = dict(area_b_m2=float(a[m].sum()), area_a_m2=float(rec['area_a'][m].sum()), mean_kernel=float((K[m] * a[m]).sum() / a[m].sum()),
                     shaded_fraction=float(1 - (K[m] * a[m]).sum() / (Ku[m] * a[m]).sum()), z_mean_mm=float(1000 * (rec['z'][m] * a[m]).sum() / a[m].sum()))
            if 'U' in rec:
                inc = rec['Uext'][m] * K[m]
                d.update(absorbed_with_extension_ru_m2=float((inc * a[m]).sum()), mean_incident_ru=float((inc * a[m]).sum() / a[m].sum()),
                         area_fraction_outside_support=float(a[m & ~rec['ok']].sum() / a[m].sum()),
                         lue_gC_mol_conditional_on_meter_scale=float(L.gross_gC(inc * L.F_MAX, a[m]).sum() / L.absorbed_mol(inc * L.F_MAX, a[m]).sum()))
            per_leaf[f'leaf_{lid}'] = d
        m = leaf & ~np.isin(rec['leaf_id'], list(USABLE))
        per_leaf['other_leaves'] = dict(area_b_m2=float(a[m].sum()), mean_kernel=float((K[m] * a[m]).sum() / a[m].sum()))
        out['per_leaf'] = per_leaf
    if 'sensor_parts' in rec:
        Ts = kernel(rec['sensor_parts'], f_d); lay = rec['sensor_layer']
        prof = []
        z_rel = (rec['z'] - MEDIUM_Z)
        for k in range(int(lay.max()) + 1):
            zlo, zhi = 0.005 * k, 0.005 * (k + 1)
            sel = leaf & (z_rel >= zlo) & (z_rel < zhi); sp = sel & rec['inpot']
            prof.append(dict(layer=k, z_mm=[1000 * zlo, 1000 * zhi], transmittance=float(Ts[lay == k].mean()),
                             leaf_area_m2=float(a[sel].sum()), leaf_area_above_pot_m2=float(a[sp].sum()), mean_kernel=float((K[sel] * a[sel]).sum() / a[sel].sum()) if a[sel].sum() > 0 else None))
        # k diagnostic: sensors cover the pot square, so only leaf area above the pot square enters the cumulative LAI
        lai_above = np.cumsum([p['leaf_area_above_pot_m2'] for p in prof][::-1])[::-1] / POT_SIDE ** 2
        for p, la in zip(prof, lai_above): p['lai_above_pot_top'] = float(la - p['leaf_area_above_pot_m2'] / POT_SIDE ** 2 / 2)
        out['leaf_area_fraction_above_pot_square'] = float(a[leaf & rec['inpot']].sum() / a[leaf].sum())
        T = np.array([p['transmittance'] for p in prof]); X = np.array([p['lai_above_pot_top'] for p in prof]); okk = (T > 0) & (X > 0)
        out['k_extinction_diagnostic'] = float(-(X[okk] * np.log(T[okk])).sum() / (X[okk] ** 2).sum()) if okk.sum() >= 2 else None
        for p in prof: p['below_pot_rim_walls_not_modelled'] = bool(p['z_mm'][1] <= 1000 * (RIM_Z - MEDIUM_Z))
        rim = [p for p in prof if not p['below_pot_rim_walls_not_modelled']]
        out['transmittance_at_rim_layer'] = rim[0]['transmittance'] if rim else None   # first 5 mm layer above the rim (pot walls absent)
        out['transmittance_lowest_layer_inside_pot'] = prof[0]['transmittance']
        out['profile'] = prof
    return out
