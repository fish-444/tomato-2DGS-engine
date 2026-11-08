"""Kaggle CPU kernel: A1-A3 on reconstructed plants + section-3 synthetic recovery (CONTRACT.md, Amendment 1). One run."""
import json, os, sys, time, types, shutil
from pathlib import Path
import numpy as np
import pandas as pd

T0 = time.time()
IN = None
for dp, dn, fn in os.walk('/kaggle/input'):
    if 'run_root.txt' in fn:
        IN = Path(dp); break
assert IN is not None, 'dataset root not found'
WORK = Path('/kaggle/working/root'); shutil.copytree(IN, WORK, dirs_exist_ok=True); os.chdir(WORK); sys.path.insert(0, str(WORK))
OUT = Path('/kaggle/working/out'); OUT.mkdir(exist_ok=True)

# engine_points imports recon-only modules it does not use in sample()/meshes(); stub them (CONTRACT section 3)
rec = types.ModuleType('tomato3d.recon'); rec.__path__ = []
ca = types.ModuleType('tomato3d.recon.charuco_a3'); rs = types.ModuleType('tomato3d.recon.r31_score'); rs.rle_decode = None
rec.charuco_a3, rec.r31_score = ca, rs
sys.modules.update({'tomato3d.recon': rec, 'tomato3d.recon.charuco_a3': ca, 'tomato3d.recon.r31_score': rs})
import tomato3d; tomato3d.recon = rec
import importlib.util
spec = importlib.util.spec_from_file_location('engine_points', WORK / 'build/abs_engine_points_20261031_01/engine_points.py'); ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)
from tomato3d.synth import microtom_v2 as mv
from tomato3d.traits import recon_traits as rt
import cv2

LABELS = ['recon_trait', 'conditional_on_unvalidated_scale', 'after_checkpoint_v1', 'validation_status_unconfirmed', 'age_unknown']
PLANTS = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05']
log = open(OUT / 'run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer): return int(o)
    return o

# ------------------------------------------------------------------ real plants: A1-A3
real = {}; sigma_n = None
for p in PLANTS:
    d = pd.read_csv(WORK / 'data' / p / 'static_surfels.csv', comment='#')
    z = np.load(WORK / 'data' / p / 'per_surfel_light.npz', allow_pickle=True)
    xyz = d[['x_mm', 'y_mm', 'z_mm']].to_numpy(float); n = rt.plus_z(d[['nx', 'ny', 'nz']].to_numpy(float))
    w = d['area_b_mm2'].to_numpy(float); fb = int(np.isnan(w).sum()); w = np.where(np.isnan(w), d['area_a_mm2'].to_numpy(float), w)
    flags = []
    if len(z['z_mm']) == len(d) and np.max(np.abs(z['z_mm'] - xyz[:, 2])) <= 0.01:
        h = np.asarray(z['h_above_base_mm'], float)
    else:
        h = xyz[:, 2] - xyz[:, 2].min(); flags.append('h_from_z_min')
    ax = np.asarray(z['axis_like']).astype(bool) if len(z['axis_like']) == len(d) else np.zeros(len(d), bool)
    if len(z['axis_like']) != len(d): flags.append('axis_like_misaligned')
    P = (d['organ'] != 'stem').to_numpy(); S = P & ~ax
    th = rt.inclination(n)
    r = dict(plant=p, labels=LABELS, flags=flags, n_surfels=len(d), n_P=int(P.sum()), n_S=int(S.sum()), area_a_fallback=fb,
             axis_like_share_of_P=float(ax[P].mean()), stem_class=bool((d['organ'] == 'stem').any()))
    r['A1_P'] = rt.lad(th[P], w[P]); r['A1_S'] = rt.lad(th[S], w[S]); r['A1_P_thirds'] = rt.lad_by_thirds(th[P], w[P], h[P])
    r['A2'] = rt.size_traits(xyz[P], w[P], h[P], h)
    r['A3_P'] = rt.azimuth_traits(xyz[P], n[P], w[P]); r['A3_S'] = rt.azimuth_traits(xyz[S], n[S], w[S])
    nn = __import__('scipy.spatial', fromlist=['cKDTree']).cKDTree(xyz[P]).query(xyz[P], k=2)[0][:, 1]; r['nn_spacing_median_mm'] = float(np.median(nn))
    if p == 'R3-1b':
        sigma_n, cnt = rt.neighbour_normal_noise(xyz[P], n[P]); r['neighbour_normal_noise_deg'] = sigma_n; r['neighbour_normal_noise_n'] = cnt
    real[p] = clean(r); say(p, 'A done', round(time.time() - T0, 1))
json.dump(real, open(OUT / 'real_A_traits.json', 'w'), indent=1)
say('sigma_n (R3-1b) =', sigma_n)

# ------------------------------------------------------------------ synthetic recovery
GRID = [('default', {})] + [(f'{k}={v}', {k: v}) for k, vs in [('phyllo', (130, 145)), ('bend', (0.1, 0.5)), ('cup', (0.1, 0.45)), ('droop', (0.15, 0.6))] for v in vs] \
     + [('elev=60/30', dict(elev_low=60, elev_high=30)), ('elev=80/55', dict(elev_low=80, elev_high=55))]
SEEDS = [101, 102, 103, 104]
BLADE = {'blade_P', 'blade_T', 'blade_I', 'cotyledon'}

def noisy(xyz, n, sig, rng):
    n = n / np.linalg.norm(n, axis=1, keepdims=True); xyz = xyz + n * rng.uniform(-0.75, 0.75, (len(n), 1))
    a = rng.normal(size=(len(n), 3)); a -= (a * n).sum(1, keepdims=True) * n; a /= np.linalg.norm(a, axis=1, keepdims=True)
    ang = np.radians(np.abs(rng.normal(0, sig, len(n))))[:, None]; n2 = n * np.cos(ang) + np.cross(a, n) * np.sin(ang)
    return xyz, rt.plus_z(n2)

def noisy_poly(poly, rng):
    poly = np.asarray(poly, float); S = rt.arclen(poly); s = np.unique(np.r_[np.arange(0, S[-1], 20.0), S[-1]])
    return rt.at_s(poly, s) + rng.normal(0, 1.0, (len(s), 3))

def tri_data(ms, cats):
    C, N, A = [], [], []
    for m in ms:
        if m['cat'] not in cats: continue
        a, b, c = (m['V'][m['F'][:, i]] for i in range(3)); cr = np.cross(b - a, c - a); ar = 0.5 * np.linalg.norm(cr, axis=1); k = ar > 0
        C.append(((a + b + c) / 3)[k]); N.append(cr[k] / (2 * ar[k, None])); A.append(ar[k])
    if not C: return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(C), rt.plus_z(np.concatenate(N)), np.concatenate(A)

def footprint(ms, cell=0.25):
    V = np.concatenate([m['V'] for m in ms]); lo = V[:, :2].min(0) - 1; hi = V[:, :2].max(0) + 1
    W, H = (np.ceil((hi - lo) / cell).astype(int) + 1); img = np.zeros((H, W), np.uint8)
    for m in ms:
        T = np.round((m['V'][:, :2] - lo) / cell * 16).astype(np.int32)
        for f in m['F']: cv2.fillConvexPoly(img, T[f], 1, lineType=cv2.LINE_8, shift=4)
    return float(img.sum() * cell * cell)

rows_plant, rows_leaf = [], []
for sig_name in ('primary', 'stress15'):
    sig = sigma_n if sig_name == 'primary' else 15.0
    for cname, over in GRID:
        for seed in SEEDS:
            tree = mv.generate(mv.params_for(**over, seed=seed)); ms = ep.meshes(tree)
            Pt = ep.sample(tree, np.zeros(3), density=4.0, seed=seed); rng = np.random.default_rng(seed * 7 + (0 if sig_name == 'primary' else 1))
            xyz, n = noisy(Pt['xyz'] * 1000.0, Pt['normal'], sig, rng); kind = Pt['kind']; leaf = Pt['leaf']; w = np.full(len(xyz), 1 / 4.0)
            stem_true = np.asarray(tree['axes'][0]['tube']['polyline'], float); base_z = stem_true[0, 2]
            nonstem = kind != 'stem'; blade = np.isin(kind, list(BLADE))
            Ct, Nt, At = tri_data(ms, BLADE); Vall = np.concatenate([m['V'] for m in ms])
            ex1 = rt.lad(rt.inclination(n[nonstem]), w[nonstem]); exb = rt.lad(rt.inclination(n[blade]), w[blade]); tr1 = rt.lad(rt.inclination(Nt), At)
            h = xyz[:, 2] - base_z; ht = Vall[:, 2] - base_z
            ex_th = rt.lad_by_thirds(rt.inclination(n[nonstem]), w[nonstem], h[nonstem]); tr_th = rt.lad_by_thirds(rt.inclination(Nt), At, Ct[:, 2] - base_z)
            sz = rt.size_traits(xyz[nonstem], w[nonstem], h[nonstem], h)
            a3e = rt.azimuth_traits(xyz[nonstem], n[nonstem], w[nonstem]); a3b = rt.azimuth_traits(xyz[blade], n[blade], w[blade]); a3t = rt.azimuth_traits(Ct, Nt, At)
            row = dict(noise=sig_name, sigma_n=sig, config=cname, seed=seed,
                       A1_mean_ext=ex1['mean_deg'], A1_mean_ext_blades=exb['mean_deg'], A1_mean_true=tr1['mean_deg'],
                       A1_sd_ext=ex1['sd_deg'], A1_sd_true=tr1['sd_deg'], A1_chi_ext=ex1['ellipsoidal_chi'], A1_chi_true=tr1['ellipsoidal_chi'],
                       height_ext=sz['height_p995_mm'], height_true=float(ht.max()),
                       proj1_ext=sz['proj_area_1mm_mm2'], proj2_ext=sz['proj_area_2mm_mm2'], proj_true=footprint(ms),
                       width_x_ext=sz['width_x_mm'], width_x_true=float(np.ptp(Vall[:, 0])), width_y_ext=sz['width_y_mm'], width_y_true=float(np.ptp(Vall[:, 1])),
                       A3_rbar_ext=a3e['rbar_surfel'], A3_rbar_ext_blades=a3b['rbar_surfel'], A3_rbar_true=a3t['rbar_surfel'],
                       A3_dec_ext=a3e['decision'], A3_dec_true=a3t['decision'])
            for i in range(3):
                row[f'A1_third{i+1}_ext'] = ex_th[i].get('mean_deg'); row[f'A1_third{i+1}_true'] = tr_th[i].get('mean_deg')
            # B / C per main-axis leaf
            stem_noisy = noisy_poly(stem_true, rng)
            main = sorted([lf for lf in tree['leaves'] if lf['axis'] == 'M'], key=lambda lf: int(lf['rank']))
            psi_t, psi_e, s_t, s_e = [], [], [], []
            for lf in main:
                pt = np.asarray(lf['rachis']['polyline'], float)
                if rt.arclen(pt)[-1] < 25: continue
                sel_r = (kind == 'rachis') & (leaf == lf['id']); pe = rt.densify(noisy_poly(pt, rng), xyz[sel_r])
                at_, ae = rt.arch_traits(pt), rt.arch_traits(pe); bt, be = rt.leaf_base_traits(pt, stem_true), rt.leaf_base_traits(pe, stem_noisy)
                psi_t.append(bt['base_azimuth_deg']); psi_e.append(be['base_azimuth_deg']); s_t.append(bt['s_on_stem_mm']); s_e.append(be['s_on_stem_mm'])
                lr = dict(noise=sig_name, config=cname, seed=seed, leaf=lf['id'], rank=int(lf['rank']))
                for k in ('L_mm', 'theta0_deg', 'theta_tip_deg', 'dtheta_deg', 'curv_deg_per_mm', 's_apex_rel', 'tip_drop_mm'):
                    lr[k + '_ext'] = ae[k]; lr[k + '_true'] = at_[k]
                lr['insertion_deg_ext'] = be['insertion_deg']; lr['insertion_deg_true'] = bt['insertion_deg']
                for kind_ in ('arc', 'quad'):
                    for k in ('direct_max_dev_mm', 'direct_tip_err_mm', 'fit_rms_mm'):
                        lr[f'{kind_}_{k}_ext'] = ae[kind_][k]; lr[f'{kind_}_{k}_true'] = at_[kind_][k]
                lr['better_ext'] = ae['better']; lr['better_true'] = at_['better']
                bl = blade & (leaf == lf['id']); Ctl, Ntl, Atl = tri_data([m for m in ms if m['leaf'] == lf['id']], BLADE)
                if bl.sum() > 50 and len(Ctl) > 10:
                    ce, ct = rt.phi_psi(xyz[bl], n[bl], w[bl]), rt.phi_psi(Ctl, Ntl, Atl)
                    lr.update(phi_ext=ce['phi_deg'], phi_true=ct['phi_deg'], psi_ext=ce['psi_deg'], psi_true=ct['psi_deg'], c1_flags_ext='|'.join(ce['flags']))
                    ll = (leaf == lf['id']) & nonstem
                    lr.update(C3_incl_ext=float(np.average(rt.inclination(n[ll]), weights=w[ll])), C3_incl_true=float(np.average(rt.inclination(Ntl), weights=Atl)))
                rows_leaf.append(lr)
            dv_t, dv_e = rt.divergence(psi_t), rt.divergence(psi_e)
            row.update(div_ext=dv_e.get('circ_mean_deg'), div_true=dv_t.get('circ_mean_deg'), div_sd_ext=dv_e.get('circ_sd_deg'), div_sd_true=dv_t.get('circ_sd_deg'),
                       hand_ext=dv_e.get('handedness'), hand_true=dv_t.get('handedness'), n_main_leaves=len(psi_t))
            for a_, b_ in zip(rt.internodes(s_e), rt.internodes(s_t)):
                rows_leaf.append(dict(noise=sig_name, config=cname, seed=seed, leaf='internode', internode_mm_ext=a_, internode_mm_true=b_))
            rows_plant.append(row)
        say(sig_name, cname, 'done', round(time.time() - T0, 1))
pd.DataFrame(rows_plant).to_csv(OUT / 'synthetic_recovery_plant.csv', index=False)
pd.DataFrame(rows_leaf).to_csv(OUT / 'synthetic_recovery_leaf.csv', index=False)

# ------------------------------------------------------------------ gate (CONTRACT section 3)
dp, dl = pd.DataFrame(rows_plant), pd.DataFrame(rows_leaf)
ANG = dict(A1_mean=dp, A1_sd=dp, A1_third1=dp, A1_third2=dp, A1_third3=dp, theta0_deg=dl, theta_tip_deg=dl, dtheta_deg=dl, insertion_deg=dl, phi=dl, psi=dl, C3_incl=dl)
LEN = dict(height=dp, proj1=dp, width_x=dp, width_y=dp, L_mm=dl, tip_drop_mm=dl, internode_mm=dl, curv_deg_per_mm=dl)
gate = {}
for noise in ('primary', 'stress15'):
    g = {}
    for k, df in ANG.items():
        x = df[df.noise == noise][[k + '_ext', k + '_true']].dropna(); b = x[k + '_ext'] - x[k + '_true']
        g[k] = dict(kind='angle_deg', n=len(x), bias=float(b.mean()), sd=float(b.std()), passed=bool(abs(b.mean()) <= 5.0))
    for k, df in LEN.items():
        x = df[df.noise == noise][[k + '_ext', k + '_true']].dropna(); rb = float((x[k + '_ext'] - x[k + '_true']).sum() / x[k + '_true'].abs().sum())
        g[k] = dict(kind='relative', n=len(x), rel_bias=rb, passed=bool(abs(rb) <= 0.10))
    x = dp[dp.noise == noise]; d = ((x.div_ext - x.div_true + 180) % 360) - 180
    g['divergence'] = dict(kind='angle_deg_circular', n=int(d.notna().sum()), bias=float(d.mean()), sd=float(d.std()), passed=bool(abs(d.mean()) <= 5.0),
                           handedness_agree=float((x.hand_ext == x.hand_true).mean()))
    b = x.A3_rbar_ext - x.A3_rbar_true; g['A3_rbar'] = dict(kind='rbar', n=len(x), bias=float(b.mean()), sd=float(b.std()), passed=bool(abs(b.mean()) <= 0.05),
                                                          decision_agree=float((x.A3_dec_ext == x.A3_dec_true).mean()))
    y = dl[(dl.noise == noise) & dl.leaf.ne('internode')]
    g['arch_better'] = dict(ext_arc_share=float((y.better_ext == 'arc').mean()), true_arc_share=float((y.better_true == 'arc').mean()),
                            agree=float((y.better_ext == y.better_true).mean()))
    gate[noise] = g
json.dump(clean(dict(labels=LABELS, sigma_n_primary=sigma_n, gate=gate)), open(OUT / 'gate.json', 'w'), indent=1)
say('DONE', round(time.time() - T0, 1))
