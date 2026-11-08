"""Local leaf-label diffusion into grey (unassigned) blade surfels only, R3-1b. Contract: build/p1_local_diffusion_20261027_01/CONTRACT.md (+ Amendment D1).

Base (attempt 1) = the viewer's '2D 엽축 배정' set (vote2d, trace numbering); attempt 2 (`--base vp`, tuned_on_R3-1b) = the V-P reference row (map-v3 numbering).
Coloured surfels are never relabelled. Labels spread from labelled surfels and the user's rachis traces along the stacked-sheet-retry sheet graph (never across
sheets), one pass, with a reach limit per leaf and a competition margin. Human labels are used only to score; scoring code imported.

usage: python -m tomato3d.organs.local_diffusion [--base vote2d|vp]
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import light_organs as LO
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import trace_reconcile as TR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2
from tomato3d.organs import vp_recluster as RC

ROOT = LO.ROOT; OUT = ROOT / "build/p1_local_diffusion_20261027_01"
STATUS = "R3-1b_unaccepted"
BASES = dict(vote2d=dict(file=V2.OUT / "surfel_clusters.npz", label="vote2d (viewer '2D 엽축 배정'), trace numbering", attempt="attempt_1"),
             vp=dict(file=TR.VP_DIR / "surfel_clusters.npz", label="V-P reference row (user_labelled_propagated), map-v3 numbering", attempt="tuned_on_R3-1b (Amendment D1)"))
T_SHEET_MM = 1.529
D = dict(candidate_mm=20.0, edge_mm=3.0, angle_deg=30.0, offset_mm=max(T_SHEET_MM, 1.0), trace_seed_mm=4.0, crease_deg=20.0, crease_penalty=2.0, margin_mm=4.0,
         reach_pct=95, reach_mult=1.2, reach_default_mm=12.0, w_min=0.1, trace_map_min=0.5)
SENS = dict(margin_mm=(2.0, 4.0, 6.0), reach_mult=(1.0, 1.2, 1.5))
SRC = ("grey", "base", "diffused", "contested", "out_of_reach")


def trace_map_vp():
    """Trace -> V-P leaf from the reconcile's matrix_T (before_VP rows): the label holding >= 50 % of the trace's non-zero labels, else unmapped (0)."""
    out = {}
    for r in csv.DictReader(open(TR.OUT / "matrix_T.csv", encoding="utf-8")):
        if r["state"] != "before_VP": continue
        cnt = {int(k.split("_")[1]): int(v) for k, v in r.items() if k.startswith("label_") and int(k.split("_")[1]) > 0}; tot = sum(cnt.values()); L, n = max(cnt.items(), key=lambda kv: kv[1])
        out[r["trace"]] = dict(leaf=(L if tot and n / tot >= D["trace_map_min"] else 0), share=(round(n / tot, 3) if tot else 0.0), trace_leaf=int(r["leaf"]))
    return out


def load(base_kind="vote2d"):
    pl = UL.load_plant(); b = np.nonzero(pl["blade"])[0]; xyz = pl["xyz"][b]
    zb = np.load(BASES[base_kind]["file"], allow_pickle=True); assert np.array_equal(zb["rows"], pl["rows"]); base = np.where(zb["leaf"] > 0, zb["leaf"], 0).astype(np.int16); axis = zb["axis_kind"]
    zs = np.load(SR.OUT / "sheets" / "sheets.npz"); assert np.array_equal(zs["rows"], pl["rows"][b]); S = dict(normal=zs["normal"].astype(float), n_sheets=zs["n_sheets"], index=zs["index"])
    traces = CC.load_traces(TR.TRACES); tmap = {}
    for t in traces: t["trace_leaf"] = t["leaf"]
    if base_kind == "vp":
        tmap = trace_map_vp()
        for t in traces: t["leaf"] = tmap[t["key"]]["leaf"]
    W = np.load(SR.FIRST / "contributions.npz")["W"]; VISb = (W >= D["w_min"])[b]
    return dict(pl=pl, b=b, xyz=xyz, base=base, axis=axis, S=S, traces=traces, VISb=VISb, kind=base_kind, trace_map=tmap)


def sheet_graph(xyz, S):
    E = SR.edges_3mm(xyz); ok = SR.edge_ok(xyz, S, E, D["offset_mm"]); E = E[ok]; i, j = E[:, 0], E[:, 1]
    L = np.linalg.norm(xyz[i] - xyz[j], axis=1) * 1000; ni, nj = S["normal"][i], S["normal"][j]
    ang = np.degrees(np.arccos(np.clip(np.abs((ni * nj).sum(1)), 0, 1))); cost = L * (1 + np.where(ang > D["crease_deg"], D["crease_penalty"], 0.0)); plain = L
    n = len(xyz); G = coo_matrix((np.r_[cost, cost], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr(); Gp = coo_matrix((np.r_[plain, plain], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    return G, Gp, int(len(E))


def seeds_and_reach(B, Gp):
    """Seeds per leaf (labelled surfels + sheet-graph neighbourhood of the mapped trace points) and the reach limit per leaf."""
    xyz, base_b = B["xyz"], B["base"][B["b"]]; tree = cKDTree(xyz); seeds = {}; reach = {}; info = {}
    for L in sorted(set(base_b[base_b > 0].tolist()) | {t["leaf"] for t in B["traces"] if t["leaf"] > 0}):
        s = set(np.nonzero(base_b == L)[0].tolist()); tr = [t for t in B["traces"] if t["leaf"] == L]; dense = np.vstack([CC.densify(t["pts"]) for t in tr]) if tr else None; nseed_tr = 0
        if dense is not None:
            d, j = tree.query(dense); entries = np.unique(j[d <= D["trace_seed_mm"] / 1000])
            if len(entries):
                dist = dijkstra(Gp, directed=False, indices=entries, min_only=True, limit=D["trace_seed_mm"]); near = np.nonzero(np.isfinite(dist))[0]; nseed_tr = len(set(near.tolist()) - s); s |= set(near.tolist())
        nbase = int((base_b == L).sum())
        if dense is not None and nbase:
            dd, _ = cKDTree(dense).query(xyz[base_b == L]); reach[L] = float(np.percentile(dd * 1000, D["reach_pct"]) * D["reach_mult"]); rule = "p95_trace_distance_x1.2"
        else: reach[L] = D["reach_default_mm"]; rule = "default_12mm" + ("_trace_without_base_surfels" if dense is not None else "_no_trace")
        pending = (not V2.is_dev(L)) if B["kind"] == "vote2d" else False
        seeds[L] = np.array(sorted(s), int); info[L] = dict(base_surfels=nbase, traces=[t["key"] for t in tr], trace_seed_surfels=nseed_tr, reach_mm=round(reach[L], 2), reach_rule=rule, pending=pending)
    return seeds, reach, info


def costs(G, seeds):
    leaves = sorted(seeds); C = np.full((len(leaves), G.shape[0]), np.inf)
    for k, L in enumerate(leaves):
        if len(seeds[L]): C[k] = dijkstra(G, directed=False, indices=seeds[L], min_only=True)
    return leaves, C


def assign(leaves, C, cand, reach, margin_mm, reach_mult=D["reach_mult"]):
    """Per candidate: best leaf if within reach and ahead of the runner-up by the margin; else contested / out_of_reach."""
    n = C.shape[1]; lab = np.zeros(n, np.int16); src = np.zeros(n, np.int8)              # 0 grey
    order = np.argsort(C, axis=0); best = order[0]; cb = C[best, np.arange(n)]; c2 = C[order[1], np.arange(n)] if len(leaves) > 1 else np.full(n, np.inf)
    rl = np.array([reach[L] * reach_mult / D["reach_mult"] if reach[L] != D["reach_default_mm"] else reach[L] for L in leaves])[best]
    ok_reach = np.isfinite(cb) & (cb <= rl); ok_margin = np.zeros(n, bool); ok_margin[ok_reach] = (c2[ok_reach] - cb[ok_reach]) >= margin_mm     # inf runner-up = infinite margin
    take = cand & ok_reach & ok_margin; lab[take] = np.array(leaves)[best[take]]; src[take] = 2
    src[cand & ok_reach & ~ok_margin] = 3; src[cand & ~ok_reach] = 4
    return lab, src


def fill_purity(B, lab_new, src, prj, umaps):
    hit = tot = 0; per = {}
    for vid, p in prj.items():
        m = B["VISb"][:, p["k"]] & p["inside"] & (src == 2); ul = umaps[vid][p["wi"], p["ui"]]; m &= ul > 0
        eq = lab_new[m] == ul[m]; hit += int(eq.sum()); tot += int(m.sum())
        for L in np.unique(lab_new[m]): mm = lab_new[m] == L; per.setdefault(int(L), [0, 0]); per[int(L)][0] += int(eq[mm].sum()); per[int(L)][1] += int(mm.sum())
    return (round(hit / tot, 4) if tot else None), tot, {str(L): dict(purity=round(h / t, 4), labelled_projections=t) for L, (h, t) in per.items()}


def images(B, base_b, lab_b, src_b, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xyz = B["xyz"] * 1000; out.mkdir(exist_ok=True)
    def col(lb, sr):
        c = np.full((len(lb), 3), 0.5)
        for L in np.unique(lb[lb > 0]): c[lb == L] = np.array(V2.PAL.get(int(L), (200, 200, 200))) / 255
        d = sr == 2; c[d] = c[d] * 0.55 + 0.45; c[sr == 3] = 1.0; return c
    t5 = next(t for t in B["traces"] if t["trace_leaf"] == 5)["pts"] * 1000; t4 = next(t for t in B["traces"] if t["trace_leaf"] == 4)["pts"] * 1000
    L6 = next(t for t in B["traces"] if t["trace_leaf"] == 6)["leaf"] or 6; c6 = xyz[base_b == L6].mean(0) if (base_b == L6).any() else t4.mean(0)
    for name, c, r in [("leaf5", t5.mean(0), 45), ("leaf4_leaf6", (t4.mean(0) + c6) / 2, 50)]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6)); m = (np.abs(xyz[:, 0] - c[0]) < r) & (np.abs(xyz[:, 1] - c[1]) < r); o = np.argsort(xyz[m, 2])
        for a, (lb, sr, ttl) in zip(ax, ((base_b, np.where(base_b > 0, 1, 0), "before (base)"), (lab_b, src_b, "after (diffused lighter, contested white)"))):
            a.scatter(xyz[m, 0][o], xyz[m, 1][o], c=col(lb[m], sr[m])[o], s=3, linewidths=0); a.set_aspect("equal"); a.set_title(ttl); a.set_facecolor("#202020")
            for t in B["traces"]: a.plot(t["pts"][:, 0] * 1000, t["pts"][:, 1] * 1000, color="#ffd400", lw=1.5); a.text(*(t["pts"][-1][:2] * 1000), f"{t['trace_leaf']}" + (f"→{t['leaf']}" if t["leaf"] != t["trace_leaf"] else ""), color="#ffd400", fontsize=9)
            a.set_xlim(c[0] - r, c[0] + r); a.set_ylim(c[1] - r, c[1] + r)
        fig.suptitle(f"{STATUS} · {B['kind']} base · top view, {name} ({2 * r} mm window) · gate_failed_viewing_only unless stated"); fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=110); plt.close(fig)


VIEWER = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>국소 확산 뷰어</title>
<style>:root{--bg:#111;--fg:#eee}body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}#hd{position:fixed;top:0;left:0;right:0;background:#7a1f1f;font-size:12px;padding:4px 10px;z-index:3}
#side{position:fixed;top:26px;right:0;width:250px;bottom:0;background:#1b1b1b;padding:8px;overflow:auto;z-index:2;font-size:12px}#cv{position:fixed;top:26px;left:0;right:250px;bottom:0}label{display:block;margin:3px 0}
@media(max-width:700px){#side{width:100%;top:auto;height:35%}#cv{right:0;bottom:35%}}</style></head><body><div id="hd">__NOTE__</div><canvas id="cv"></canvas>
<div id="side"><b>읽기 전용</b><br>색 = 기준 라벨, 연한 색 = 확산으로 새로 붙은 점, 흰색 = 경합(여백 부족), 어두운 회색 = 닿지 않음/회색 유지, 노란 선 = 엽축<br>
<label><input type="checkbox" id="showDiff" checked> 확산 점 보이기</label><label><input type="checkbox" id="showCont" checked> 경합 점 흰색</label><div id="leg"></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script><script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>const D=__DATA__;const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));const P=new Float32Array(b64(D.pos).buffer),C0=b64(D.col),CD=b64(D.cold),SRC=b64(D.src);const N=D.n;
const cv=document.getElementById('cv');const R=new THREE.WebGLRenderer({canvas:cv,antialias:true});const S=new THREE.Scene();const cam=new THREE.PerspectiveCamera(45,1,1,2000);cam.position.set(0,-220,140);cam.up.set(0,0,1);
const ctl=new THREE.OrbitControls(cam,cv);function size(){const w=cv.clientWidth,h=cv.clientHeight;R.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}window.addEventListener('resize',size);size();
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(P,3));const col=new Float32Array(N*3);g.setAttribute('color',new THREE.BufferAttribute(col,3));S.add(new THREE.Points(g,new THREE.PointsMaterial({size:1.6,vertexColors:true})));
for(const t of D.traces){const lg=new THREE.BufferGeometry().setFromPoints(t.pts.map(p=>new THREE.Vector3(p[0]*1000,p[1]*1000,p[2]*1000)));S.add(new THREE.Line(lg,new THREE.LineBasicMaterial({color:0xffd400})));}
function paint(){const sd=document.getElementById('showDiff').checked,sc=document.getElementById('showCont').checked;for(let i=0;i<N;i++){let r=C0[i*3],gg=C0[i*3+1],bb=C0[i*3+2];const s=SRC[i];if(s===2&&sd){r=CD[i*3];gg=CD[i*3+1];bb=CD[i*3+2];}if(s===3&&sc){r=gg=bb=255;}col[i*3]=r/255;col[i*3+1]=gg/255;col[i*3+2]=bb/255;}g.attributes.color.needsUpdate=true;}
document.getElementById('showDiff').onchange=paint;document.getElementById('showCont').onchange=paint;document.getElementById('leg').innerHTML=D.legend;paint();(function loop(){ctl.update();R.render(S,cam);requestAnimationFrame(loop);})();window.__ld={n:N,src:SRC};</script></body></html>"""


def write_viewer(B, lab, src, info, path, note):
    pl = B["pl"]; xyz = pl["xyz"]; c = xyz.mean(0); n = len(xyz); col = np.full((n, 3), 45, np.uint8); col[B["b"]] = 110; cold = col.copy()
    lab_f = np.zeros(n, np.int16); lab_f[B["b"]] = lab; src_f = np.zeros(n, np.int8); src_f[B["b"]] = src; lab, src = lab_f, src_f
    for L in np.unique(lab[lab > 0]):
        c3 = np.array(V2.PAL.get(int(L), (200, 200, 200))); m = lab == L; col[m & (src == 1)] = c3; cold[m] = (c3 * 0.55 + 255 * 0.45).astype(np.uint8)
    traces = [dict(leaf=t["leaf"], pts=(t["pts"] - c).round(5).tolist()) for t in B["traces"]]
    leg = "".join(f'<div><span style="color:#%02x%02x%02x">●</span> 잎 {L} {"(보류)" if i["pending"] else ""} · 기준 {i["base_surfels"]} · 확산 +{i["diffused"]} · 닿음 한도 {i["reach_mm"]} mm</div>' % tuple(V2.PAL.get(L, (200, 200, 200))) for L, i in sorted(info.items()))
    data = dict(n=n, pos=base64.b64encode(((xyz - c) * 1000).astype(np.float32).tobytes()).decode(), col=base64.b64encode(col.tobytes()).decode(), cold=base64.b64encode(cold.tobytes()).decode(),
                src=base64.b64encode(src.astype(np.uint8).tobytes()).decode(), traces=traces, legend=leg)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(VIEWER.replace("__DATA__", json.dumps(data)).replace("__NOTE__", note), encoding="utf-8"); return path


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--base", choices=tuple(BASES), default="vote2d"); ap.add_argument("--out", type=Path, default=None); a = ap.parse_args(argv)
    out = a.out or (OUT if a.base == "vote2d" else OUT / "second_attempt"); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    B = load(a.base); pl, b, xyz = B["pl"], B["b"], B["xyz"]; base_b = B["base"][b]; nb = len(b)
    G, Gp, nedges = sheet_graph(xyz, B["S"]); print(f"sheet graph: {nedges} edges on {nb} blade surfels ({time.time() - t0:.0f}s)", flush=True)
    axis_b = B["axis"][b]; dense_all = np.vstack([CC.densify(t["pts"]) for t in B["traces"] if t["leaf"] > 0]); lab_pts = np.vstack([xyz[base_b > 0], dense_all])
    dnear, _ = cKDTree(lab_pts).query(xyz); cand = (base_b == 0) & (axis_b == 0) & (dnear <= D["candidate_mm"] / 1000)
    seeds, reach, info = seeds_and_reach(B, Gp); leaves, C = costs(G, seeds); print(f"leaves {leaves}, candidates {int(cand.sum())} of {int((base_b == 0).sum())} grey ({time.time() - t0:.0f}s)", flush=True)
    lab_new, src = assign(leaves, C, cand, reach, D["margin_mm"]); src[base_b > 0] = 1; lab = np.where(base_b > 0, base_b, lab_new).astype(np.int16)
    # human maps for the training evaluation (in the base's numbering) and the held-out truth
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; raw, _ = UL.label_maps(UL.OUT, chosen + [held])
    nmap_v3 = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]; perms = {v: {int(k): x for k, x in p.items()} for v, p in json.load(open(TR.OUT / "scores.json", encoding="utf-8"))["permutations"].items()}
    nmap_train = TR.renumber_map(perms, chosen) if a.base == "vote2d" else nmap_v3
    umaps = UL.remap({v: raw[v] for v in chosen}, nmap_train); umaps = {v: np.where(umaps[v] > 0, umaps[v], 0).astype(np.int16) for v in chosen}; prj = SR.projections(pl, b, chosen)
    fp, fp_n, fp_leaf = fill_purity(B, lab, src, prj, umaps); grey0 = int((base_b == 0).sum()); per_leaf = {}
    for L in leaves:
        nd = int(((src == 2) & (lab == L)).sum()); per_leaf[L] = dict(**info[L], diffused=nd, share_of_grey=round(nd / grey0, 4), from_pending_leaf=(nd if info[L]["pending"] else 0))
    truth_orig = UL.remap({held: raw[held]}, nmap_v3)[held]; truth_orig = np.where(truth_orig > 0, truth_orig, 0).astype(np.int16)
    truth_rec = UL.remap({held: raw[held]}, {f"{held}|{L}": t for L, t in perms[held].items()})[held]; truth_rec = np.where(truth_rec > 0, truth_rec, 0).astype(np.int16)
    H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = AP.view_file(held); col = H2id[:, pl["views"].index(hv)]
    def full(lb): f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; return f
    def score(lb):
        if a.base == "vote2d": return TR.score_both(pl, full(lb), hv, col, truth_rec, truth_orig)[0]           # judged on 'reconciled'
        ren = UL.render_ids(pl, full(lb), hv, col); rows, g = UL.score_heldout(truth_orig, ren); cov = float(((ren > 0) & (truth_orig > 0)).sum() / max((truth_orig > 0).sum(), 1))
        fl = {str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rows.items()}
        return dict(original=dict(rows={str(k): v for k, v in rows.items()}, summary=g, coverage=round(cov, 4), flags=fl))
    judged = "reconciled" if a.base == "vote2d" else "original"
    sc_base = score(base_b); sc_diff = score(lab); gb, gd = sc_base[judged]["summary"], sc_diff[judged]["summary"]
    passed = gd["median_iou"] > gb["median_iou"] and gd["max_confusion"] <= gb["max_confusion"] and (fp or 0) >= 0.85; beats = gd["median_iou"] > 0.517 and gd["max_confusion"] <= 0.393
    sens = {}
    for mm in SENS["margin_mm"]:
        for rm in SENS["reach_mult"]:
            l2, s2 = assign(leaves, C, cand, reach, mm, rm); g2 = score(np.where(base_b > 0, base_b, l2).astype(np.int16))[judged]["summary"]
            sens[f"margin{mm:g}_reach{rm:g}"] = dict(diffused=int((s2 == 2).sum()), contested=int((s2 == 3).sum()), median_iou=round(g2["median_iou"], 4), max_confusion=round(g2["max_confusion"], 4))
    res = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping="local_diffusion_into_grey", base=BASES[a.base]["label"], attempt=BASES[a.base]["attempt"], trace_map=B["trace_map"],
               params=D, t_sheet_mm=T_SHEET_MM, sheet_graph_edges=nedges, blade_surfels=nb, grey_before=grey0, candidates=int(cand.sum()), leaves=leaves,
               source_counts=dict(Counter(SRC[s] for s in src.tolist())), grey_after=int(((src == 0) | (src == 3) | (src == 4)).sum()), share_grey_newly_labelled=round(int((src == 2).sum()) / grey0, 4),
               share_contested=round(int((src == 3).sum()) / grey0, 4), share_out_of_reach=round(int((src == 4).sum()) / grey0, 4), share_not_candidate=round(int((src == 0).sum()) / grey0, 4),
               from_pending_leaf=int(sum(v["from_pending_leaf"] for v in per_leaf.values())), fill_purity=dict(purity=fp, labelled_projections=fp_n, per_leaf=fp_leaf), per_leaf={str(L): v for L, v in per_leaf.items()},
               held_out=dict(view=held, judged_on=judged, base=sc_base, base_plus_diffusion=sc_diff), reference_best=dict(median_iou=0.517, max_confusion=0.393, numbering="V-P (map v3)"),
               pass_rule="median IoU > base AND max confusion <= base AND fill purity >= 0.85", passed=bool(passed), beats_reference_on_both=bool(beats), sensitivity=sens,
               tag=("gate_passed" if gd["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    src_full = np.zeros(len(pl["xyz"]), np.int8); src_full[b] = src; pend = np.zeros(len(pl["xyz"]), bool); pend[b] = (src == 2) & np.isin(lab, [L for L in leaves if info[L]["pending"]])
    np.savez_compressed(out / "labels_diffused.npz", rows=pl["rows"], base=B["base"], label=full(lab), source=src_full, source_names=np.array(SRC), from_pending_leaf=pend)
    with open(out / "per_leaf.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); keys = ["base_surfels", "traces", "trace_seed_surfels", "reach_mm", "reach_rule", "pending", "diffused", "share_of_grey", "from_pending_leaf"]
        w.writerow(["leaf"] + keys + ["iou_base", "conf_base", "iou_after", "conf_after", "fill_purity"])
        for L in leaves:
            rb = sc_base[judged]["rows"].get(str(L), {}); rd = sc_diff[judged]["rows"].get(str(L), {})
            w.writerow([L] + [per_leaf[L][k] for k in keys] + [round(rb.get("iou", float("nan")), 4), round(rb.get("confusion", float("nan")), 4), round(rd.get("iou", float("nan")), 4), round(rd.get("confusion", float("nan")), 4), fp_leaf.get(str(L), {}).get("purity")])
    images(B, base_b, lab, src, out / "images")
    note = f"{STATUS} · conditional_on_unvalidated_scale · {res['tag']} — 국소 확산, 기준 = {BASES[a.base]['label']} ({res['attempt']}; 보류 사진 IoU {gb['median_iou']:.3f}→{gd['median_iou']:.3f}, 혼동 {gb['max_confusion']:.3f}→{gd['max_confusion']:.3f}, 채움 순도 {fp}) · 읽기 전용"
    write_viewer(B, lab, src, per_leaf, OUT / "viewer" / ("diffusion_viewer.html" if a.base == "vote2d" else "diffusion_viewer_vp.html"), note)
    print(json.dumps({k: v for k, v in res.items() if k not in ("held_out", "sensitivity", "per_leaf", "params")}, indent=1, default=float, ensure_ascii=False))
    print("held_out base", gb, "| diffusion", gd); print("rows after", {k: (round(v["iou"], 3), round(v["confusion"], 3)) for k, v in sc_diff[judged]["rows"].items()}); print("sens", json.dumps(sens))


if __name__ == "__main__":
    main()
