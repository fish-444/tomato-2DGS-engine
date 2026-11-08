"""Light routes for STATIC mode (CONTRACT.md §2–3): measured-field relative U, cabinet bar + dome kernel samples, scenario ratio, NRH gross."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIELD = REPO / 'build/p2_empirical_light_field_20261013_01/as_recorded/field/field.json'
FIELD_CONFIG = REPO / 'build/p2_empirical_light_field_20261013_01/as_recorded_configuration.json'
F_MAX = 367.416                               # canopy-plane maximum of the measured field (umol m-2 s-1, meter scale) = 1 r.u.
SUPPORT = dict(x=(122.5, 437.5), y=(105.0, 315.0), z=(70.0, 150.0))   # mm, grow frame, shelf floor
BARS = dict(centre_x_mm=300.0, centre_y_mm=105.0, spacing_mm=50.0, length_mm=500.0, z_mm=300.0, count=3)   # user_assumed pose
HALF_ANGLE_DEG = 60.0                         # user_assumed half-intensity half-angle
DIFFUSE = 0.6                                 # declared; 0.5 / 0.7 spread
AMAX, ALPHA, THETA, HOURS = 13.57, 0.048, 0.92, 14.0


def cos_m(half_angle_deg: float = HALF_ANGLE_DEG) -> float:
    from tomato3d.light.scene import cosine_exponent
    return cosine_exponent(math.radians(half_angle_deg))


class MeasuredField:
    def __init__(self):
        from tools.empirical_light_field import load_field
        self.missing_provenance = []
        try:
            self.field = load_field(FIELD, FIELD_CONFIG)
        except FileNotFoundError:
            self.field = self._load_with_missing_notes()

    def _load_with_missing_notes(self) -> dict:
        """load_field stops at the first absent provenance file. Two auxiliary notes of 10-13 (observation clarification, lamp input
        update) are no longer on disk; every file that exists must still match its hash, the code hashes must match, and the samples must
        re-derive exactly from the raw grid. The absent names are kept in `missing_provenance` and reported."""
        import json
        from tools.empirical_light_field import read_json, digest, sample_records
        from tools.leaf_light.common import file_hash
        field = json.loads(FIELD.read_text(encoding='utf-8'))
        for name, expected in {**field['source_hashes'], **field['code_hashes']}.items():
            if not Path(name).exists():
                self.missing_provenance.append(name); continue
            if file_hash(Path(name)) != expected:
                raise RuntimeError('source_or_code_hash_mismatch:' + name)
        if digest(read_json(FIELD_CONFIG)) != field['configuration_sha256']:
            raise RuntimeError('configuration_changed_remeasurement_required')
        if any(field['inputs'][k] in self.missing_provenance for k in ('grid', 'configuration')):
            raise RuntimeError('field input missing')
        if field['samples'] != sample_records(Path(field['inputs']['grid']), field['configuration']):
            raise RuntimeError('derived_samples_do_not_match_raw_grid')
        return field

    @staticmethod
    def in_support(p_mm: np.ndarray) -> np.ndarray:
        return ((p_mm[:, 0] >= SUPPORT['x'][0]) & (p_mm[:, 0] <= SUPPORT['x'][1]) & (p_mm[:, 1] >= SUPPORT['y'][0]) & (p_mm[:, 1] <= SUPPORT['y'][1])
                & (p_mm[:, 2] >= SUPPORT['z'][0]) & (p_mm[:, 2] <= SUPPORT['z'][1]))

    def U(self, p_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Relative unshaded horizontal light (r.u.) and the support mask; outside support = NaN (bounded, no extrapolation)."""
        from tools.empirical_light_field import evaluate
        p = np.asarray(p_m) * 1000.0; ok = self.in_support(p); out = np.full(len(p), np.nan)
        # the field is bilinear in xy per plane and linear in z: evaluate on a fine lattice and interpolate exactly the same way
        for i in np.nonzero(ok)[0]:
            out[i] = evaluate(self.field, list(p[i]))['ppfd_umol_m2_s'] / F_MAX
        return out, ok

    def U_clamped(self, p_m: np.ndarray) -> np.ndarray:
        p = np.asarray(p_m) * 1000.0
        c = np.stack([np.clip(p[:, 0], *SUPPORT['x']), np.clip(p[:, 1], *SUPPORT['y']), np.clip(p[:, 2], *SUPPORT['z'])], 1)
        return self.U(c / 1000.0)[0], c / 1000.0


def bar_points(seed: int, samples_per_bar: int = 8, dz_mm: float = 0.0, dy_mm: float = 0.0) -> np.ndarray:
    """Jittered sample points on the three user_assumed bars (metres, grow frame). dz/dy shift the lamp for scenarios."""
    rng = np.random.default_rng(seed)
    pts = []
    for b in range(BARS['count']):
        y = BARS['centre_y_mm'] + (b - (BARS['count'] - 1) / 2) * BARS['spacing_mm'] + dy_mm
        t = (np.arange(samples_per_bar) + rng.random(samples_per_bar)) / samples_per_bar
        x = BARS['centre_x_mm'] - BARS['length_mm'] / 2 + t * BARS['length_mm']
        for xx in x:
            pts.append((xx, y, BARS['z_mm'] + dz_mm))
    return np.array(pts) / 1000.0


def dome(seed: int, n_mu: int = 6, n_phi: int = 12) -> np.ndarray:
    """Isotropic upper-hemisphere directions, stratified uniform in cos(zenith) and azimuth, jittered by seed."""
    rng = np.random.default_rng(seed + 7919)
    out = []
    for i in range(n_mu):
        for j in range(n_phi):
            mu = (i + rng.random()) / n_mu; phi = 2 * math.pi * (j + rng.random()) / n_phi; s = math.sqrt(max(0.0, 1 - mu * mu))
            out.append((s * math.cos(phi), s * math.sin(phi), max(mu, 1e-6)))
    return np.array(out)


def nrh(q: np.ndarray) -> np.ndarray:
    """Engine leaf law (umol CO2 m-2 s-1) at absorbed PPFD q (umol m-2 s-1); identical to vtc_development_io's independent check."""
    a, b, t = ALPHA, AMAX, THETA
    return (a * q + b - np.sqrt((a * q + b) ** 2 - 4 * t * a * q * b)) / (2 * t)


def gross_gC(q: np.ndarray, area: np.ndarray, hours: float = HOURS) -> np.ndarray:
    return nrh(q) * area * hours * 3600 * 12.011e-6


def absorbed_mol(q: np.ndarray, area: np.ndarray, hours: float = HOURS) -> np.ndarray:
    return q * area * hours * 3600 / 1e6
