"""Compound leaves by 2D rachis assignment per photo, then a per-surfel 3D vote (no 3D growth, no graph).

Contract: build/p1_compound_2dvote_20261025_01/CONTRACT.md. The user's rachis traces (db `rachis_traces`, exported
read-only) are oriented in memory by the declared rule, projected into every training view with the Revision-1 splat
visibility, each H2(M1) leaflet mask is attached in 2D to the nearest visible projected rachis (D_assign / M_margin in mm),
and every blade surfel takes the majority of the compound ids of the assigned masks covering it in the views where it is
visible (V_min / S_min). The four parameters are fitted on the development leaves (odd numbers) only; held-out leaves
(even numbers) are scored once.

usage: python -m tomato3d.organs.vote2d [--out DIR] [--stage a|b|all] [--page]
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import contract as C
from tomato3d.organs import light_organs as LO
from tomato3d.organs import seed2d as S2
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_compound_2dvote_20261025_01"
R5_DIR = CC.OUT2 / "r31b_r5"
PHOTOS = LO.DEFAULTS["poses"].parent / "undist_3024"
STATUS = "R3-1b_unaccepted"; GROUPING = "user_seeded_2d_vote"
LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping=GROUPING)
CENTRE = np.array([CC.POT_CENTRE[0], CC.POT_CENTRE[1]]); POT_TOP = CC.POT_TOP
H2 = S2.H2_GRID
# declared (contract sections 2-6)
VISP = dict(res=S2.VIS["res"], tol_m=C.R3_FUSION["vis_tol_m"], radius_sigma=S2.VIS["radius_sigma"], max_r_px=S2.VIS["max_r_px"], step_mm=0.5, usable_share=0.5)
GRID = dict(D_assign_mm=(10, 15, 20, 25), M_margin_mm=(0, 5, 10), V_min=(2, 3), S_min=(0.5, 0.6))
GATE = dict(membership_min=0.70, merge_max=0.10)
NEIGH_MM = 1.5 * CC.R2["leaflet_mm"]          # section 5: att_mult 1.5 x 18.6 mm = 27.9 mm (section 11 neighbourhood)
TIE_MM = 0.5
DEV_PARITY = 1                                 # odd leaf numbers = development
PAL = {1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25), 4: (0, 130, 200), 5: (245, 130, 48), 6: (145, 30, 180), 7: (70, 240, 240), 8: (240, 50, 230),
       9: (210, 245, 60), 10: (250, 190, 212), 11: (0, 128, 128), 12: (220, 190, 255)}
PAGE_PX = 900


def is_dev(leaf):
    return leaf % 2 == DEV_PARITY


# ---------------------------------------------------------------- section 1: orientation and stem fit
def orient_traces(traces, centre=CENTRE, tie_mm=TIE_MM):
    """Base = end nearer the central vertical (horizontal distance). Returns oriented copies, flipped keys, tie keys."""
    out, flipped, ties = [], [], []
    for t in traces:
        p = np.asarray(t["pts"], float); d0 = float(np.hypot(*(p[0, :2] - centre))); d1 = float(np.hypot(*(p[-1, :2] - centre)))
        rec = dict(t, pts=p, end_dist_mm=(round(d0 * 1000, 1), round(d1 * 1000, 1)), flipped=False)
        if abs(d0 - d1) * 1000 < tie_mm:
            ties.append(t["key"])
        elif d1 < d0:
            rec.update(pts=p[::-1].copy(), surf=list(reversed(t.get("surf", []))), unc=list(reversed(t.get("unc", []))), flipped=True); flipped.append(t["key"])
        out.append(rec)
    return out, flipped, ties


def stem_fit(traces, ransac_axis):
    """Round-2 rule with oriented traces: LS point nearest the base lines of the development traces; reference axis = smallest
    mean distance to the held-out oriented base ends."""
    devT = [t for t in traces if is_dev(t["leaf"])]; heldT = [t for t in traces if not is_dev(t["leaf"])]
    lines = [(t["pts"][0], t["pts"][0] - t["pts"][1]) for t in devT if len(t["pts"]) >= 2]
    x = CC.line_point_ls(lines) if len(lines) >= 2 else None
    axes = {"ransac_q1_refit": (np.asarray(ransac_axis[0], float), np.asarray(ransac_axis[1], float)), "pot_centre_vertical": (np.array([CENTRE[0], CENTRE[1], POT_TOP]), np.array([0, 0, 1.0]))}
    if x is not None:
        axes["trace_fit_vertical"] = (np.array([x[0], x[1], POT_TOP]), np.array([0, 0, 1.0]))
    hb = np.array([t["pts"][0] for t in heldT]) if heldT else np.zeros((0, 3))
    score = {k: (float(CC.dist_to_line(hb, p, d).mean() * 1000) if len(hb) else None) for k, (p, d) in axes.items()}
    chosen = min((k for k in axes if score[k] is not None), key=lambda k: score[k]) if len(hb) else "pot_centre_vertical"
    rep = dict(candidates={k: dict(point=[round(float(v), 5) for v in axes[k][0]], direction=[round(float(v), 4) for v in axes[k][1]], mean_dist_to_heldout_bases_mm=(round(score[k], 2) if score[k] is not None else None)) for k in axes},
               chosen=chosen, rule="smallest mean distance to the held-out ORIENTED base ends", development_lines=len(lines), held_out_bases=int(len(hb)))
    return rep, axes[chosen]


# ---------------------------------------------------------------- section 2: projection, visibility
def project_view(xyz, sc, po, view, res=VISP["res"]):
    K = ca.scaled_K(np.array(po["K_undist"]), po["undistorted_res"], res); R = np.array(po["views"][view]["R_cw"]); t = np.array(po["views"][view]["t_cw"])
    xc = xyz @ R.T + t; z = xc[:, 2]; zs = np.where(z > 0, z, 1e-9)
    u = np.round(K[0, 0] * xc[:, 0] / zs + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / zs + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res); idx = np.nonzero(ok)[0]
    r_px = np.minimum(VISP["radius_sigma"] * sc[idx].max(1) * K[0, 0] / z[idx], VISP["max_r_px"])
    zb = S2.splat_zbuffer(u[idx], w[idx], z[idx], r_px, res)
    vis = np.zeros(len(xyz), bool); vis[idx[z[idx] <= zb[w[idx], u[idx]] + VISP["tol_m"]]] = True
    return dict(K=K, R=R, t=t, u=u, w=w, z=z, zb=zb, vis=vis, res=res)


def project_points(P, pv):
    xc = P @ pv["R"].T + pv["t"]; z = xc[:, 2]; zs = np.where(z > 0, z, 1e-9)
    return pv["K"][0, 0] * xc[:, 0] / zs + pv["K"][0, 2], pv["K"][1, 1] * xc[:, 1] / zs + pv["K"][1, 2], z


def trace_visibility(dens, pv):
    u, w, z = project_points(dens, pv); res = pv["res"]; ui = np.round(u).astype(int); wi = np.round(w).astype(int)
    inside = (z > 0) & (ui >= 0) & (wi >= 0) & (ui < res) & (wi < res); vis = np.zeros(len(z), bool)
    vis[inside] = z[inside] <= pv["zb"][wi[inside], ui[inside]] + VISP["tol_m"]
    return u, w, z, vis


def mask_distances(lab, nlab, uu, ww, zz, f_px_h2):
    """Per H2 label: min pixel distance to the rachis raster (mm at the trace depth) and whether the mask contains a rachis pixel."""
    import cv2
    ras = np.ones((H2, H2), np.uint8); ui = np.clip(np.round(uu).astype(int), 0, H2 - 1); wi = np.clip(np.round(ww).astype(int), 0, H2 - 1); ras[wi, ui] = 0
    dt = cv2.distanceTransform(ras, cv2.DIST_L2, 5)
    dmin = np.full(nlab + 1, np.inf); np.minimum.at(dmin, lab.ravel(), dt.ravel().astype(np.float64))
    px_per_mm = f_px_h2 * 0.001 / float(zz.mean())
    return dmin / px_per_mm, np.bincount(lab[wi, ui], minlength=nlab + 1) > 0


def stage_a(xyz, sc, po, views, traces, labels_dir, out: Path):
    """Per view: H2 label under every visible surfel, trace visibility shares, per-mask distances to each usable leaf's rachis."""
    res = VISP["res"]; n = len(xyz); H2id = np.full((n, len(views)), -1, np.int16); per_view = {}
    dens = {t["key"]: CC.densify(t["pts"], VISP["step_mm"] / 1000) for t in traces}
    for k, v in enumerate(views):
        pv = project_view(xyz, sc, po, v, res)
        lab = np.asarray(Image.open(labels_dir / (Path(v).stem + ".png"))).astype(np.int32); nlab = int(lab.max())
        vi = np.nonzero(pv["vis"])[0]; H2id[vi, k] = lab[pv["w"][vi] * H2 // res, pv["u"][vi] * H2 // res]
        tr_vis, rach = {}, {}
        for t in traces:
            u, w, z, vis = trace_visibility(dens[t["key"]], pv); tr_vis[t["key"]] = dict(share=round(float(vis.mean()), 4), n=int(len(z)), leaf=t["leaf"])
            if vis.mean() >= VISP["usable_share"]:
                rach.setdefault(t["leaf"], []).append((u[vis] * H2 / res, w[vis] * H2 / res, z[vis]))
        dist_mm, crossed, rpts = {}, {}, {}
        for L, parts in rach.items():
            uu = np.concatenate([p[0] for p in parts]); ww = np.concatenate([p[1] for p in parts]); zz = np.concatenate([p[2] for p in parts])
            d, c = mask_distances(lab, nlab, uu, ww, zz, pv["K"][0, 0] * H2 / res); dist_mm[L] = np.where(np.isfinite(d), d, 1e9).round(2).tolist(); crossed[L] = c.tolist()
            step = max(1, len(uu) // 60); rpts[L] = np.c_[uu[::step], ww[::step]].round(1).tolist()
        per_view[v] = dict(nlab=nlab, n_visible_surfels=int(pv["vis"].sum()), trace_vis=tr_vis, usable=sorted(rach), dist_mm={str(L): x for L, x in dist_mm.items()},
                           crossed={str(L): x for L, x in crossed.items()}, rachis_px={str(L): x for L, x in rpts.items()})
        print(f"[stage a] {k + 1}/{len(views)} {v[9:18]} masks {nlab} usable leaves {sorted(rach)} visible surfels {int(pv['vis'].sum())}", flush=True)
    np.savez_compressed(out / "stage_a_h2id.npz", H2id=H2id, views=np.array(views))
    json.dump(per_view, open(out / "stage_a_views.json", "w", encoding="utf-8"))
    return H2id, per_view


# ---------------------------------------------------------------- section 3-4: assignment and vote
def assign_view(info, D, M):
    """int16 map H2 label -> leaf id (0 = unassigned_in_view, -1 = ambiguous_in_view)."""
    nlab = info["nlab"]; m = np.zeros(nlab + 1, np.int16); Ls = [int(L) for L in info["usable"]]
    if not Ls or nlab == 0:
        return m
    Dm = np.array([info["dist_mm"][str(L)] for L in Ls], float); Cr = np.array([info["crossed"][str(L)] for L in Ls], bool)
    for k in range(1, nlab + 1):
        if Cr[:, k].sum() >= 2:
            m[k] = -1; continue
        o = np.argsort(Dm[:, k], kind="stable"); d1 = Dm[o[0], k]
        if d1 <= D and (len(o) == 1 or Dm[o[1], k] - d1 >= M):
            m[k] = Ls[o[0]]
    return m


def vote(H2id, maps, blade, V_min, S_min, n_ids=16):
    """Majority of assigned-mask leaf ids over the views where a surfel is visible and covered. Returns labels (0 = unassigned), reasons, counts."""
    n, V = H2id.shape; cnt = np.zeros((n, n_ids + 1), np.int32)
    for k in range(V):
        h = H2id[:, k]; ok = np.nonzero(h > 0)[0]; lv = maps[k][h[ok]]; sel = lv > 0
        np.add.at(cnt, (ok[sel], lv[sel].astype(int)), 1)
    tot = cnt.sum(1); win = cnt.argmax(1); top = cnt.max(1); srt = np.sort(cnt, 1); second = srt[:, -2]
    share = np.where(tot > 0, top / np.maximum(tot, 1), 0.0)
    lab = np.where((tot >= V_min) & (share >= S_min) & (top > second), win, 0).astype(np.int16); lab[~blade] = 0
    reason = np.full(n, "axis", dtype=object); b = blade
    reason[b & (tot == 0)] = "no_votes"; reason[b & (tot > 0) & (tot < V_min)] = "too_few_votes"
    reason[b & (tot >= V_min) & (share < S_min)] = "low_share"; reason[b & (tot >= V_min) & (share >= S_min) & (top == second)] = "tie"; reason[lab > 0] = "assigned"
    return lab, reason, cnt


# ---------------------------------------------------------------- section 5-6: neighbourhoods, measures, fit
def neighbourhoods(xyz, blade_idx, blob_full, traces, lim_m=NEIGH_MM / 1000):
    blob = blob_full[blade_idx]; valid = blob >= 0; nb = int(blob.max()) + 1 if valid.any() else 0
    cents = np.array([xyz[blade_idx[blob == b]].mean(0) if (blob == b).any() else [np.nan] * 3 for b in range(nb)])
    N = {}
    for L in sorted({t["leaf"] for t in traces}):
        polys = [t["pts"] for t in traces if t["leaf"] == L]; d = np.min([CC.poly_dist(cents, p)[0] for p in polys], axis=0) if nb else np.zeros(0)
        sel = np.nonzero(d <= lim_m)[0]; N[L] = blade_idx[np.isin(blob, sel)]
    return N, cents


def measures(lab, N, group):
    rows = {}
    for L in group:
        idx = N[L]; l = lab[idx]; n = max(len(idx), 1); asg = int((l > 0).sum())
        rows[L] = dict(n_surfels=int(len(idx)), membership=float((l == L).sum() / n), unassigned=float((l == 0).sum() / n), membership_among_assigned=(float((l == L).sum() / asg) if asg else None),
                       merge={str(L2): float((l == L2).sum() / n) for L2 in group if L2 != L}, other_ids=float(((l > 0) & (l != L) & ~np.isin(l, list(group))).sum() / n))
    return rows


def summary(rows):
    mem = [r["membership"] for r in rows.values()]; mg = [v for r in rows.values() for v in r["merge"].values()]
    return dict(mean_membership=float(np.mean(mem)) if mem else None, max_merge=float(max(mg)) if mg else 0.0, mean_merge=float(np.mean(mg)) if mg else 0.0)


def grid_points():
    return [dict(D_assign_mm=D, M_margin_mm=M, V_min=V, S_min=S) for D in GRID["D_assign_mm"] for M in GRID["M_margin_mm"] for V in GRID["V_min"] for S in GRID["S_min"]]


def choose(results):
    """results: [(params, summary)] on development leaves. Section 6: (a) max mean membership with every merge <= 0.10; (b) else max (membership - mean merge);
    ties -> larger V_min, larger S_min, smaller D, larger M."""
    tie = lambda p: (p["V_min"], p["S_min"], -p["D_assign_mm"], p["M_margin_mm"])
    ok = [(p, s) for p, s in results if s["max_merge"] <= GATE["merge_max"] and s["mean_membership"] is not None]
    if ok:
        best = max(ok, key=lambda ps: (ps[1]["mean_membership"], tie(ps[0]))); return best[0], "a_merge_ok_max_membership"
    best = max(results, key=lambda ps: ((ps[1]["mean_membership"] or 0) - ps[1]["mean_merge"], tie(ps[0]))); return best[0], "b_no_merge_free_point"


def run_params(H2id, per_view, views, blade, p):
    maps = [assign_view(per_view[v], p["D_assign_mm"], p["M_margin_mm"]) for v in views]
    lab, reason, cnt = vote(H2id, maps, blade, p["V_min"], p["S_min"])
    return lab, reason, cnt, maps


# ---------------------------------------------------------------- section 7: per-leaf outputs
def leaf_outputs(xyz, nrm, tu, tv, sc, lab, traces, axis, per_view, maps, views, H):
    rows = []
    for L in sorted({t["leaf"] for t in traces}):
        m = lab == L; Ls = [t for t in traces if t["leaf"] == L]
        base_h = float(np.mean([t["pts"][0][2] for t in Ls]) - POT_TOP)
        counts = [int((maps[k] == L).sum()) for k, v in enumerate(views) if L in per_view[v]["usable"]]
        rec = dict(leaf=L, split=("development" if is_dev(L) else "held_out"), traces=len(Ls), n_surfels=int(m.sum()), base_height_mm=round(base_h * 1000, 1),
                   views_usable=len(counts), count_2d_estimate=(float(np.median(counts)) if counts else None), **LABELS)
        if m.sum() >= 3:
            P = xyz[m]; cen, w, e1, e2, nn = LO.pca_frame(P); area, length, width, n_px = LO.union_area(P, tu[m], tv[m], sc[m], cen, e1, e2)
            wts = np.pi * LO.FLOORS["ellipse_k"] ** 2 * sc[m, 0] * sc[m, 1]; nr = nrm[m] * np.where(nrm[m][:, 2:3] < 0, -1, 1); mn = (nr * wts[:, None]).sum(0); mn /= max(np.linalg.norm(mn), 1e-12)
            incl = float(np.degrees(np.arccos(np.clip(mn[2], -1, 1)))); p0, d0 = axis
            rec.update(area_m2=round(float(area), 8), area_rel_H2=round(float(area) / H ** 2, 6), length_m=round(length, 5), width_m=round(width, 5), mean_normal=[round(float(v), 4) for v in mn],
                       inclination_area_weighted_deg=round(incl, 2), azimuth_deg=round(float(LO.deg_azimuth(cen[0] - p0[0], cen[1] - p0[1])), 1), centroid_m=[round(float(v), 5) for v in cen],
                       inclination_pca_plane_deg=round(float(np.degrees(np.arccos(np.clip(abs(nn[2]), -1, 1)))), 2))
        rows.append(rec)
    return rows


# ---------------------------------------------------------------- overlays and page
def _hex(c):
    return "#%02x%02x%02x" % c


def draw_overlay(view, info, mp, photos: Path, labels_dir: Path, px=PAGE_PX):
    img = Image.open(photos / view).convert("RGB"); img.thumbnail((px, px)); W, Hh = img.size; sx, sy = W / H2, Hh / H2
    lab = np.asarray(Image.open(labels_dir / (Path(view).stem + ".png"))).astype(np.int32)
    ov = np.zeros((H2, H2, 4), np.uint8)
    for k in range(1, info["nlab"] + 1):
        L = int(mp[k]); mk = lab == k
        if L > 0:
            ov[mk] = (*PAL.get(L, (200, 200, 200)), 120)
        elif L < 0:
            ov[mk] = (255, 0, 255, 70)
        else:
            ov[mk] = (160, 160, 160, 55)
    ovi = Image.fromarray(ov).resize((W, Hh), Image.BILINEAR); img = Image.alpha_composite(img.convert("RGBA"), ovi).convert("RGB"); dr = ImageDraw.Draw(img)
    for Ls, pts in info["rachis_px"].items():
        c = PAL.get(int(Ls), (255, 255, 255)); P = [(x * sx, y * sy) for x, y in pts]
        if len(P) >= 2: dr.line(P, fill=c, width=3)
        if P: dr.ellipse([P[0][0] - 6, P[0][1] - 6, P[0][0] + 6, P[0][1] + 6], fill=c, outline="white", width=2); dr.text((P[0][0] + 8, P[0][1] - 14), f"잎 {Ls}", fill="white")
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=72); return buf.getvalue(), (W, Hh)


PAGE_HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2D 엽축 배정 뷰</title>
<style>:root{--bg:#111;--fg:#eee;--line:#333}body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif}header{padding:8px 16px;background:#7a1f1f;font-size:12px}
main{display:grid;grid-template-columns:260px 1fr;gap:12px;padding:12px 16px}@media(max-width:800px){main{grid-template-columns:1fr}}
#list{max-height:80vh;overflow:auto;font-size:12px}#list div{padding:3px 4px;cursor:pointer;display:flex;justify-content:space-between}#list div.on{background:#333}
img{max-width:100%;height:auto;display:block}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid var(--line);padding:2px 6px}i.sw{display:inline-block;width:10px;height:10px;margin-right:4px}</style></head><body>
<header id="hd"></header><main><aside><div style="font-size:12px;opacity:.8;margin-bottom:6px">보기(훈련 뷰) — 클릭</div><div id="list"></div></aside>
<section><div id="leg" style="margin-bottom:6px"></div><img id="im" alt="overlay"><div id="info" style="margin-top:8px"></div></section></main>
<script id="D" type="application/json">__DATA__</script>
<script>const D=JSON.parse(document.getElementById('D').textContent);document.getElementById('hd').textContent=D.header;
document.getElementById('leg').innerHTML=Object.entries(D.palette).map(([L,c])=>`<i class="sw" style="background:${c}"></i>잎 ${L}${D.dev.includes(+L)?' (개발)':' (보류)'} &nbsp;`).join('')+'<i class="sw" style="background:#a0a0a0"></i>미배정 &nbsp;<i class="sw" style="background:#f0f"></i>모호(엽축 2개 교차) · 선 = 보이는 엽축 투영, 점 = 밑동';
const list=document.getElementById('list');let cur=0;function show(i){cur=i;const v=D.views[i];document.getElementById('im').src='data:image/jpeg;base64,'+v.jpg;
document.getElementById('info').innerHTML=`<b>${v.id}</b> · 마스크 ${v.nlab} · 사용 가능 엽축 ${v.usable.join(', ')||'없음'}<br><table><tr><th>잎</th><th>배정 마스크</th><th>엽축 가시 비율</th></tr>${Object.entries(v.counts).map(([L,c])=>`<tr><td><i class="sw" style="background:${D.palette[L]}"></i>${L}</td><td>${c}</td><td>${(v.vis[L]||[]).map(s=>s.toFixed(2)).join(' / ')}</td></tr>`).join('')}</table>미배정 ${v.unassigned} · 모호 ${v.ambiguous}`;
[...list.children].forEach((d,j)=>d.className=j===i?'on':'');}
list.innerHTML=D.views.map((v,i)=>`<div data-i="${i}"><span>${v.id}</span><span style="opacity:.7">${v.assigned}/${v.nlab}</span></div>`).join('');list.onclick=e=>{const d=e.target.closest('[data-i]');if(d)show(+d.dataset.i);};show(0);</script></body></html>"""


def build_page(out: Path, views, per_view, maps, photos: Path, labels_dir: Path, params, traces):
    data_views = []
    for k, v in enumerate(views):
        info = per_view[v]; mp = maps[k]; jpg, _ = draw_overlay(v, info, mp, photos, labels_dir)
        counts = {str(L): int((mp == L).sum()) for L in info["usable"]}
        vis = {}
        for key, tv in info["trace_vis"].items():
            vis.setdefault(str(tv["leaf"]), []).append(tv["share"])
        data_views.append(dict(id=v[9:18], nlab=info["nlab"], usable=info["usable"], counts=counts, vis=vis, assigned=int((mp[1:] > 0).sum()), unassigned=int((mp[1:] == 0).sum()), ambiguous=int((mp[1:] < 0).sum()),
                               jpg=base64.b64encode(jpg).decode()))
    leaves = sorted({t["leaf"] for t in traces})
    data = dict(header=f"{STATUS} · 2D 엽축 배정 (사진마다 H2 마스크 → 가장 가까운 보이는 사용자 엽축; D_assign {params['D_assign_mm']} mm, M_margin {params['M_margin_mm']} mm) · 형태만, {LO.SCALE_TAG} · 계측 아님",
                palette={str(L): _hex(PAL.get(L, (200, 200, 200))) for L in leaves}, dev=[L for L in leaves if is_dev(L)], views=data_views)
    (out / "viewer").mkdir(exist_ok=True); p = out / "viewer" / "overlay_page.html"; p.write_text(PAGE_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8"); return p


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--stage", choices=("a", "b", "all"), default="all"); ap.add_argument("--page", action="store_true")
    a = ap.parse_args(argv); out = a.out; out.mkdir(parents=True, exist_ok=True)
    P = CC.load_r31b(); sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); rows = P["rows"]; xyz = P["xyz"]; sc = sf["scale"][rows].astype(np.float64); tu, tv = sf["tu"][rows].astype(np.float64), sf["tv"][rows].astype(np.float64)
    blade = ~P["is_axis"]; po = json.load(open(LO.DEFAULTS["poses"])); views = json.load(open(LO.DEFAULTS["views"]))["train"]; H = float(xyz[:, 2].max() - POT_TOP)
    raw = CC.load_traces(out / "traces_export" / "rachis_traces"); traces, flipped, ties = orient_traces(raw)
    r5 = json.load(open(R5_DIR / "scores_r5.json", encoding="utf-8")); rc = r5["stem_axis"]["candidates"]["ransac_q1_refit"]
    stem, axis = stem_fit(traces, (rc["point"], rc["direction"]))
    json.dump(dict(statement="엽축의 밑동은 식물 중심(줄기) 쪽 끝이다.", rule="base = end nearer the pot-centre vertical (horizontal distance); ties < 0.5 mm keep stored order",
                   traces=[dict(key=t["key"], leaf=t["leaf"], split=("development" if is_dev(t["leaf"]) else "held_out"), points=len(t["pts"]), end_dist_mm=t["end_dist_mm"], flipped=t["flipped"]) for t in traces],
                   flipped=flipped, ties=ties, stem_fit=stem, previous_r5_stem_fit=r5["stem_axis"]), open(out / "traces_oriented.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if a.stage in ("a", "all"):
        H2id, per_view = stage_a(xyz, sc, po, views, traces, LO.DEFAULTS["h2"], out)
    else:
        H2id = np.load(out / "stage_a_h2id.npz")["H2id"]; per_view = json.load(open(out / "stage_a_views.json", encoding="utf-8"))
    if a.stage == "a":
        return
    # neighbourhoods (section 5) from the section-11 blobs
    z5 = np.load(R5_DIR / "surfel_clusters.npz"); assert np.array_equal(z5["rows"], rows); N, cents = neighbourhoods(xyz, np.nonzero(blade)[0], z5["blob"], traces)
    dev_leaves = sorted({t["leaf"] for t in traces if is_dev(t["leaf"])}); held_leaves = sorted({t["leaf"] for t in traces if not is_dev(t["leaf"])})
    # fit on development leaves (section 6)
    results = []
    for p in grid_points():
        lab, reason, cnt, maps = run_params(H2id, per_view, views, blade, p); md = measures(lab, N, dev_leaves); s = summary(md)
        results.append((p, s)); print(f"[fit] {p} dev mean membership {s['mean_membership']:.3f} max merge {s['max_merge']:.3f} assigned {(lab > 0).sum()}", flush=True)
    best, rule = choose(results)
    json.dump(dict(grid=GRID, rule=rule, chosen=best, points=[dict(params=p, **s) for p, s in results], development_leaves=dev_leaves, note="development leaves only; held-out leaves untouched here"),
              open(out / "fit_grid.json", "w", encoding="utf-8"), indent=1)
    # scored once with the chosen point
    lab, reason, cnt, maps = run_params(H2id, per_view, views, blade, best)
    md = measures(lab, N, dev_leaves); mh = measures(lab, N, held_leaves)
    gate = {str(L): dict(membership=round(mh[L]["membership"], 4), membership_ok=mh[L]["membership"] >= GATE["membership_min"], max_merge=round(max(mh[L]["merge"].values()) if mh[L]["merge"] else 0.0, 4),
                         merge_ok=all(v <= GATE["merge_max"] for v in mh[L]["merge"].values())) for L in held_leaves}
    passed = all(g["membership_ok"] and g["merge_ok"] for g in gate.values()) and bool(held_leaves)
    # per-step loss accounting (section 7): visibility, assignment, vote
    tr_views = {t["key"]: sum(1 for v in views if per_view[v]["trace_vis"][t["key"]]["share"] >= VISP["usable_share"]) for t in traces}
    tot_masks = sum(per_view[v]["nlab"] for v in views); n_asg = sum(int((m[1:] > 0).sum()) for m in maps); n_amb = sum(int((m[1:] < 0).sum()) for m in maps)
    views_with_usable = sum(1 for v in views if per_view[v]["usable"])
    held_cov = {}
    for L in held_leaves:
        idx = N[L]; vis_any = (H2id[idx] >= 0).sum(1); cov_any = (cnt[idx].sum(1) > 0)
        held_cov[str(L)] = dict(n=int(len(idx)), visible_in_ge1_view=float((vis_any > 0).mean()), covered_by_assigned_mask_ge1=float(cov_any.mean()), votes_ge_Vmin=float((cnt[idx].sum(1) >= best["V_min"]).mean()),
                                assigned_any_id=float((lab[idx] > 0).mean()), this_id=float((lab[idx] == L).mean()))
    rs = Counter(reason[blade].tolist())
    scores = dict(**LABELS, params=best, fit_rule=rule, gate=GATE, gate_rows=gate, gate_pass=passed, development=md, held_out=mh, held_out_pairs_merged=[(L, L2) for L in held_leaves for L2 in held_leaves if L < L2 and (mh[L]["merge"][str(L2)] > GATE["merge_max"] or mh[L2]["merge"][str(L)] > GATE["merge_max"])],
                  comparison=dict(R5_majority_among_assigned=[0.51, 0.36, 0.45], R4_majority_among_assigned=[0.39, 0.46, 0.54], note="section 11 numbers are shares among assigned surfels; this run also reports membership_among_assigned"),
                  step_losses=dict(trace_views_usable=tr_views, views_with_any_usable_rachis=views_with_usable, views_total=len(views), masks_total=tot_masks, masks_assigned=n_asg, masks_ambiguous=n_amb,
                                   blade_vote_reasons=dict(rs), held_out_neighbourhood_coverage=held_cov),
                  blade_surfels=int(blade.sum()), assigned_blade_surfels=int((lab > 0).sum()), per_leaf_surfels={int(L): int((lab == L).sum()) for L in sorted(set(lab[lab > 0].tolist()))},
                  stem_fit=stem, flipped_traces=flipped, ties=ties, neighbourhood_mm=NEIGH_MM, bench="not run: the topology bench has no rendered views / per-view H2 masks (R3-1b only)")
    json.dump(scores, open(out / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
    # per-leaf outputs (written whether or not the gate passes; labelled)
    rows_out = leaf_outputs(xyz, P["normal"], tu, tv, sc, lab, traces, axis, per_view, maps, views, H)
    tag = "gate_passed" if passed else "gate_failed_viewing_only"
    with open(out / "leaves.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {GROUPING}; {tag}; area = kind (1) union of surfel ellipses; count_2d_estimate = median assigned masks over usable views; untraced leaves unassigned\n")
        keys = sorted({k for r in rows_out for k in r}, key=lambda k: (k not in ("leaf", "split"), k)); w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(rows_out)
    full = np.full(len(xyz), -1); full[lab > 0] = lab[lab > 0]
    np.savez_compressed(out / "surfel_clusters.npz", rows=rows, leaf=full, axis_kind=z5["axis_kind"], reason=np.array(reason.tolist()), votes_total=cnt.sum(1))
    json.dump({v: maps[k].tolist() for k, v in enumerate(views)}, open(out / "assignments_chosen.json", "w", encoding="utf-8"))
    print(json.dumps(dict(params=best, rule=rule, gate=gate, gate_pass=passed, assigned=int((lab > 0).sum()), per_leaf=scores["per_leaf_surfels"]), indent=1), flush=True)
    if a.page:
        print("page", build_page(out, views, per_view, maps, PHOTOS, LO.DEFAULTS["h2"], best, traces), flush=True)


if __name__ == "__main__":
    main()
