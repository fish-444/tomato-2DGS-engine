"""Colour-mark individual leaflets only, from the existing masks (viewing task, 2026-10-25). No inference, no mask, vote,
threshold or grouping change; every output reads an existing run and draws it. Labels on every output:
R3-1b_unaccepted, conditional_on_unvalidated_scale.

Part A  2D overlays per ring view from build/p1_leaflet_2dseed_20261024_01/leaflet_instances (H2(M1) reused), colour =
        hash of the cross-view id of Step 2's linking (leaflet_links.json); view-local ids hatched.
Part B  viewer from the R3-1b height template with a leaflet colour mode, sources clustering_only (Track A) and 2d_masks
        (vote); the per-surfel unavailable / background state and agreement fractions are recovered by re-running the
        2D-seed voting deterministically (checked equal to the stored labels) because the run did not persist them.
Part C  the 13 tray / dwt photos of H2(M1) run 05 with the human answers as outline styles.

usage: python -m tomato3d.organs.leaflet_colour [--parts A B C] [--out DIR]
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.leafseg import pixels as px
from tomato3d.organs import light_organs as LO
from tomato3d.organs import seed2d as S
from tomato3d.organs.light_viewer import TEMPLATE, _b64

ROOT = LO.ROOT
OUT = ROOT / "build/p1_leaflet_colour_20261025_01"
SEED_RUN = S.OUT
TRACK_A = LO.DEFAULTS["out"]
PHOTOS = LO.DEFAULTS["poses"].parent / "undist_3024"
STATUS = "R3-1b_unaccepted"
SCALE_TAG = LO.SCALE_TAG
PAGE = 1512                 # overlay resolution
FILL_ALPHA = 0.45
LOW_SCORE = 0.5             # declared here: "low score" = the proposal's SAM score below 0.5 (the run's own M1 cut 0.15 is not in the copied files)
# Paul Tol's muted + light qualitative hues (colour-blind safe), 12 hues x 3 lightness steps; no pure red/green pair
HUES = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#6699CC", "#EE8866", "#BBCC33"]
GREY_UNAVAILABLE = (128, 128, 128)      # no vote or conflicting votes
GREY_BACKGROUND = (70, 70, 70)          # voted background in every view where visible (and axis surfels)
GREY_DIM = (205, 205, 205)              # viewer: everything else when a legend entry is isolated


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def palette():
    cols = []
    for step in (1.0, 0.78, 1.18):
        for h in HUES:
            r, g, b = _hex(h); cols.append(tuple(int(min(255, max(0, c * step))) for c in (r, g, b)))
    return cols


PAL = palette()


def colour_of(key: str):
    """Stable colour of an id: md5 hash -> palette index (36 entries)."""
    return PAL[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(PAL)]


# ---------------------------------------------------------------- drawing helpers
def _font(size=18):
    for name in ("arial.ttf", "DejaVuSans.ttf", "malgun.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def resize_mask(m, size):
    return cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def contours(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2) for c in cs if len(c) >= 3]


def draw_outline(img, m, colour, thickness=2, style="solid", gap=6):
    for c in contours(m):
        if style == "solid":
            cv2.polylines(img, [c.reshape(-1, 1, 2)], True, colour, thickness, cv2.LINE_AA)
        else:                                                     # dashed / dotted: alternate runs along the contour
            on, off = (gap, gap) if style == "dashed" else (2, 4)
            k = 0
            while k < len(c):
                seg = c[k:k + on]
                if len(seg) >= 2:
                    cv2.polylines(img, [seg.reshape(-1, 1, 2)], False, colour, thickness, cv2.LINE_AA)
                k += on + off


def hatch(img, m, colour, spacing=7):
    h, w = m.shape; lines = np.zeros((h, w), np.uint8)
    for d in range(-h, w, spacing):
        cv2.line(lines, (d, 0), (d + h, h), 1, 1)
    sel = m & (lines > 0); img[sel] = colour


def blend_fill(img, m, colour, alpha=FILL_ALPHA):
    img[m] = (img[m] * (1 - alpha) + np.array(colour, float) * alpha).astype(np.uint8)


def badge(pil_draw, xy, text, fill, font):
    x, y = xy; pil_draw.rectangle([x - 1, y - 1, x + 13, y + 15], fill=fill); pil_draw.text((x + 1, y - 1), text, fill="black", font=font)


def overlay_image(rgb, items, link_of, font_size=18):
    """items: [(id, mask, score, flags)] with flags keys merge_suspect / human / view_local. Returns image, stats."""
    img = rgb.copy(); n = len(items); order = sorted(range(n), key=lambda i: (items[i][2] if items[i][2] is not None else 0))
    cover = np.zeros(img.shape[:2], np.int16)
    for i in order:
        iid, m, score, fl = items[i]; col = fl["colour"]
        blend_fill(img, m, col)
        if fl.get("view_local"):
            hatch(img, m, col)
        cover[m] += 1
    overlap = cover >= 2
    for i in order:
        iid, m, score, fl = items[i]; col = fl["colour"]; thin = score is not None and score < LOW_SCORE
        if fl.get("human"):
            draw_outline(img, m, (255, 255, 255), 4); draw_outline(img, m, col, 2)
        elif fl.get("merge_suspect"):
            draw_outline(img, m, col, 1 if thin else 2); draw_outline(img, m, (0, 0, 0), 2, style="dashed")
        else:
            draw_outline(img, m, col, 1 if thin else 2)
    if overlap.any():
        draw_outline(img, overlap, (255, 255, 255), 1)
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil); font = _font(font_size); small = _font(13)
    for iid, m, score, fl in items:
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        cx, cy = int(xs.mean()), int(ys.mean()); label = fl.get("label", iid)
        d.text((cx + 1, cy + 1), label, fill="black", font=font); d.text((cx, cy), label, fill="white", font=font)
        bx = cx + 4 + 9 * len(label)
        if fl.get("merge_suspect"):
            badge(d, (bx, cy), "M", (255, 220, 0), small); bx += 18
        if fl.get("human"):
            badge(d, (bx, cy), "H", (255, 255, 255), small)
    tinted = (cover > 0).sum()
    return np.asarray(pil), dict(instances=n, overlap_px=int(overlap.sum()), tinted_px=int(tinted), overlap_share=float(overlap.sum() / max(tinted, 1)))


def banner(img, text):
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil); d.rectangle([0, 0, pil.width, 26], fill=(122, 31, 31)); d.text((6, 4), text, fill="white", font=_font(16)); return np.asarray(pil)


# ---------------------------------------------------------------- Part A
def part_a(out: Path, corrections=None):
    links = json.load(open(SEED_RUN / "leaflet_links.json", encoding="utf-8"))["leaflet_to_2d_ids"]
    link_of = {}
    for lf, lst in links.items():
        for s in lst:
            link_of[s] = lf
    surf = json.load(open(SEED_RUN / "leaflet_surfels.json", encoding="utf-8")); n_surf = {k: len(v["surfel_rows"]) for k, v in surf.items() if not k.startswith("_")}
    views = json.load(open(LO.DEFAULTS["views"]))["train"]; (out / "overlays").mkdir(parents=True, exist_ok=True)
    stats, legend_rows, thumbs = {}, [], []
    for v in views:
        vid = v[9:18]; j = json.load(open(SEED_RUN / "leaflet_instances" / (Path(v).stem + ".json"), encoding="utf-8"))
        rgb = np.asarray(Image.open(PHOTOS / v).convert("RGB").resize((PAGE, PAGE), Image.BILINEAR))
        items = []
        for inst in j["instances"]:
            key = f"{vid}:{inst['id']}"; lf = link_of.get(key); human = str(inst.get("source", "")).startswith("human") or (corrections is not None and f"{vid}|{inst['id']}" in corrections)
            colour = colour_of(lf) if lf else colour_of(key)
            fl = dict(colour=colour, view_local=lf is None, merge_suspect=bool(inst.get("merge_suspect")), human=human, label=(lf if lf else inst["id"]))
            items.append((inst["id"], resize_mask(px.rle_decode(inst["rle"]), PAGE), inst.get("score"), fl))
            legend_rows.append(dict(view=vid, id_2d=inst["id"], cross_view_id=(lf or "view_local"), colour="#%02x%02x%02x" % colour, surfel_count=(n_surf.get(lf) if lf else ""),
                                    merge_suspect=bool(inst.get("merge_suspect")), p_merged=inst.get("p_merged"), source=("human" if human else inst.get("source", "H2(M1)")), score=inst.get("score")))
        img, st = overlay_image(rgb, items, link_of); st["view_local"] = sum(1 for it in items if it[3]["view_local"]); st["merge_suspect"] = sum(1 for it in items if it[3]["merge_suspect"]); stats[vid] = st
        img = banner(img, f"{vid} · {STATUS} · leaflets only (H2(M1) 2D masks; colour = cross-view id, hatched = view-local; M = merge_suspect) · {SCALE_TAG}")
        Image.fromarray(img).save(out / "overlays" / f"{vid}.jpg", quality=85); thumbs.append((vid, cv2.resize(img, (378, 378), interpolation=cv2.INTER_AREA)))
    cols = 8; rows = int(np.ceil(len(thumbs) / cols)); sheet = np.full((rows * 378, cols * 378, 3), 255, np.uint8)
    for k, (vid, t) in enumerate(thumbs):
        r, c = divmod(k, cols); sheet[r * 378:(r + 1) * 378, c * 378:(c + 1) * 378] = t
    Image.fromarray(sheet).save(out / "contact_sheet_ring_views.jpg", quality=80)
    with open(out / "legend_2d_ids.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {STATUS}; {SCALE_TAG}; viewing output of build/p1_leaflet_2dseed_20261024_01 (H2(M1) 2D masks, Step-2 links); no new claim\n")
        w = csv.DictWriter(f, fieldnames=list(legend_rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(legend_rows)
    # legend sheet per cross-view id
    lfs = {r["leaflet_id"]: r for r in S.read_csv(SEED_RUN / "leaflets.csv")}
    pil = Image.new("RGB", (900, 40 + 28 * (len(lfs) + 2)), "white"); d = ImageDraw.Draw(pil); f = _font(16)
    d.text((10, 8), f"{STATUS} · 2d_masks vote · cross-view leaflet ids (same colour in every view) · {SCALE_TAG}", fill="black", font=f)
    for k, (lf, r) in enumerate(sorted(lfs.items())):
        y = 40 + 28 * k; col = colour_of(lf); d.rectangle([10, y + 4, 34, y + 24], fill=col)
        d.text((44, y + 4), f"{lf}   surfels {r['n_surfels']}   2D masks linked {len(links.get(lf, []))}   merge_suspect {'yes' if 'merge_suspect' in r['flags'] else 'no'}   source H2(M1)", fill="black", font=f)
    y = 40 + 28 * len(lfs); d.rectangle([10, y + 4, 34, y + 24], fill=(255, 255, 255), outline="black"); d.text((44, y + 4), f"hatched = view-local 2D id (not linked to any 3D leaflet): {sum(s['view_local'] for s in stats.values())} of {sum(s['instances'] for s in stats.values())} proposals", fill="black", font=f)
    pil.save(out / "legend_sheet.png")
    json.dump(dict(status=STATUS, scale=SCALE_TAG, source="build/p1_leaflet_2dseed_20261024_01/leaflet_instances (H2(M1) reused, run 2026-10-22; links from the 2d_masks vote)", low_score_cut=LOW_SCORE, per_view=stats,
                   totals=dict(views=len(stats), proposals=sum(s["instances"] for s in stats.values()), view_local=sum(s["view_local"] for s in stats.values()),
                               merge_suspect=sum(s["merge_suspect"] for s in stats.values()), overlap_share_median=float(np.median([s["overlap_share"] for s in stats.values()])))),
              open(out / "part_a_summary.json", "w", encoding="utf-8"), indent=1)
    return stats


# ---------------------------------------------------------------- Part B
def recover_vote_state():
    """Re-run the 2D-seed voting deterministically (same code, same inputs) to recover per-surfel reasons and agreement."""
    a = argparse.Namespace(surfels=LO.DEFAULTS["surfels"], track_a=TRACK_A, poses=LO.DEFAULTS["poses"], views=LO.DEFAULTS["views"], out=SEED_RUN, step1b=True, human_masks=None,
                           visibility="splat1512", pot_top=LO.POT_TOP_Z, root=list(S.C.POT_CENTRE_M), group_order="current", erode_px=0, watershed=False, masks=LO.DEFAULTS["masks"])
    sf = sio.read_surfels_full(a.surfels); ta = np.load(TRACK_A / "surfel_labels.npz"); rows = ta["rows"]; is_axis = ta["is_axis"].astype(bool)
    xyz = sf["xyz"][rows].astype(np.float64); sc = sf["scale"][rows]; po = json.load(open(a.poses)); views = json.load(open(a.views))["train"]
    crit = S.Criterion(S.library_rows()); maps, _ = S.score_2d_proposals(views, crit, Path("/nonexistent_dir_no_annotation"), None)   # no file annotation (out path does not exist)
    V = S.votes_per_view(xyz, po, views, S.VIS["res"], S.VIS["tol_m"], sc=sc, mode="splat", label_maps=maps)
    cand = np.nonzero(~is_axis)[0]; Vc = V[cand]; comp, _ = S.colabel_components(xyz[cand], Vc); link, _ = S.link_ids(Vc, comp, views); final, reason, ren = S.confirm(Vc, comp, link)
    cl = np.full(len(xyz), -1); cl[cand] = final; reasons = np.full(len(xyz), "axis", dtype=object); reasons[cand] = reason; reasons[cand[final >= 0]] = "leaflet"
    labelled = (V > 0).sum(1); agree = np.zeros(len(xyz))
    for q, s in enumerate(cand):
        ids = Vc[q]; agree[s] = sum(1 for k in np.nonzero(ids > 0)[0] if link.get((int(k), int(ids[k]))) is not None and ren.get(link[(int(k), int(ids[k]))]) == final[q])
    stored = np.load(SEED_RUN / "surfel_labels.npz"); same = bool((stored["cl"] == cl).all())
    return dict(cl=cl, reasons=reasons, V=V, agree=agree, labelled=labelled, views=views, rows=rows, is_axis=is_axis, identical_to_stored=same)


def source_arrays(n_all, rows, cl, state, leaflet_ids, highlight=None):
    """Per-PLY-row uint16 leaflet index (0 none) and rgb for one source."""
    idx = np.zeros(n_all, np.uint16); col = np.tile(np.array(GREY_BACKGROUND, np.uint8), (n_all, 1))
    col[rows[state == "unavailable"]] = GREY_UNAVAILABLE
    for c, lid in enumerate(leaflet_ids):
        sel = rows[cl == c]; idx[sel] = c + 1; col[sel] = colour_of(lid)
    return idx, col


def part_b(out: Path):
    sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); n = len(sf["xyz"]); html = TEMPLATE.read_text(encoding="utf-8")
    m = re.search(r'<script id="D" type="application/json">(.*?)</script>', html, re.S); M = json.loads(m.group(1))
    rec = recover_vote_state(); rows = rec["rows"]; views = rec["views"]
    # source 1: 2d_masks (vote)
    lab2 = np.load(SEED_RUN / "surfel_labels.npz"); ids2 = [x for x in lab2["leaflet_id"].tolist()]
    state2 = np.where(rec["reasons"] == "leaflet", "leaflet", np.where(np.isin(rec["reasons"], ["no_component", "conflict", "few_views"]), "unavailable", "background"))
    idx2, col2 = source_arrays(n, rows, lab2["cl"], state2, ids2)
    links = json.load(open(SEED_RUN / "leaflet_links.json", encoding="utf-8"))["leaflet_to_2d_ids"]; L2 = S.read_csv(SEED_RUN / "leaflets.csv")
    meta2 = []
    for c, lid in enumerate(ids2):
        sel = lab2["cl"] == c; r = next(x for x in L2 if x["leaflet_id"] == lid)
        agree_frac = float(np.median(rec["agree"][sel] / np.maximum(rec["labelled"][sel], 1)))
        meta2.append(dict(id=lid, n=int(sel.sum()), merge_suspect="merge_suspect" in r["flags"], sources=["H2(M1)"], views=sorted({s.split(":")[0] for s in links.get(lid, [])}), agreement=round(agree_frac, 3), colour="#%02x%02x%02x" % colour_of(lid)))
    # source 0: clustering_only (Track A): clusters coloured, unclustered blade = unavailable grey, axis = background grey
    labA = np.load(TRACK_A / "surfel_labels.npz"); idsA = labA["leaflet_id"].tolist(); LA = S.read_csv(TRACK_A / "leaflets.csv")
    stateA = np.where(labA["cl"] >= 0, "leaflet", np.where(labA["is_axis"], "background", "unavailable"))
    idxA, colA = source_arrays(n, labA["rows"], labA["cl"], stateA, idsA)
    critA = json.load(open(SEED_RUN / "criterion.json", encoding="utf-8"))["flags_3d"].get("track_a", {})          # Step-1b flags of Track A's instances, as computed by the 2D-seed run
    metaA = [dict(id=lid, n=int((labA["cl"] == c).sum()), merge_suspect=bool(critA.get(lid, {}).get("merge_suspect")), sources=["3D clustering (no 2D mask)"], views=[], agreement=None, colour="#%02x%02x%02x" % colour_of(lid)) for c, lid in enumerate(idsA)]
    cls = np.zeros(n, np.uint8); cls[rows] = 1; lo = sf["xyz"].min(0); rng = np.maximum(sf["xyz"].max(0) - lo, 1e-9)
    M["discs"] = dict(n=n, n_plant=int(cls.sum()), H=float(rng[2]), lo=lo.tolist(), rng=rng.tolist(), pos=_b64(np.round((sf["xyz"] - lo) / rng * 65535).astype(np.uint16)),
                      tu=_b64(np.round(np.clip(sf["tu"], -1, 1) * 32767).astype(np.int16)), tv=_b64(np.round(np.clip(sf["tv"], -1, 1) * 32767).astype(np.int16)),
                      s=_b64(sf["scale"].astype(np.float16)), op=_b64(np.round(np.clip(sf["opacity"], 0, 1) * 255).astype(np.uint8)), col=_b64(sf["rgb"].astype(np.uint8)), cls=_b64(cls))
    M["leaflet"] = dict(sources=[dict(key="clustering_only", label="clustering_only (Track A, pre_topology_edit)", idx=_b64(idxA), col=_b64(colA), meta=metaA),
                                 dict(key="2d_masks", label="2d_masks (vote)", idx=_b64(idx2), col=_b64(col2), meta=meta2)],
                        greys=dict(unavailable="#808080 no vote / conflicting votes", background="#464646 voted background (and axis)"), dim="#cdcdcd",
                        note="same colour = same leaflet only within one source; no compound leaf, rank or axis")
    M["ctr"] = (lo + rng / 2).tolist(); M["note"] = f"{STATUS}: leaflets only; colour mode '소엽' adds per-surfel leaflet colours (sources clustering_only / 2d_masks vote); mid-grey = unavailable, dark grey = background; {SCALE_TAG}"
    html = html[:m.start(1)] + json.dumps(M, separators=(",", ":")) + html[m.end(1):]
    title = f"{STATUS} · 소엽만 색칠 (leaflets only)"
    html = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", html); html = re.sub(r'<h1 id="ttl">[^<]*</h1>', f'<h1 id="ttl">{title}</h1>', html)
    html = re.sub(r'(z-index:50;background:#7a1f1f;[^>]*>)[^<]*(</div>)', lambda mm: mm.group(1) + f"{STATUS} · 소엽만 (복엽·순위·축 없음) · 형태만, 척도 미검증 ({SCALE_TAG}) · 계측 아님" + mm.group(2), html, count=1)
    ui = ('<label>소엽 색 (출처)</label><div class="row" id="lfsrc"><button data-ls="-1" class="on">끔 (사진 색)</button><button data-ls="0">clustering_only</button><button data-ls="1">2d_masks (vote)</button></div>'
          '<div id="lfleg" style="display:none"><div style="font-size:11px;opacity:.8"><span style="display:inline-block;width:10px;height:10px;background:#808080"></span> unavailable (no / conflicting votes) &nbsp;<span style="display:inline-block;width:10px;height:10px;background:#464646"></span> background · 같은 색 = 같은 소엽 (한 출처 안에서만)</div>'
          '<label>최소 서펠 수 <span id="lfminN">0</span></label><input type="range" id="lfmin" min="0" max="3000" step="10" value="0">'
          '<div id="lflist" style="max-height:170px;overflow:auto;font-size:11px"></div><div id="lfpanel" style="font-size:12px;border-top:1px solid var(--line);margin-top:6px;padding-top:6px">서펠을 클릭하면 소엽 정보가 여기 나옵니다.</div></div>')
    html = html.replace('<label>불투명도 배율', ui + '<label>불투명도 배율', 1)
    js = r"""<script>(function(){const LF=M.leaflet;const RGB=C;let cur=-1,hl=null,minN=0;const TEX={};
function build(si){const s=LF.sources[si];const idx=new Uint16Array(b64(s.idx).buffer);const col=b64(s.col);const C2=new Uint8Array(TW*TH*4);
 for(let i=0;i<N;i++){let r=col[i*3],g=col[i*3+1],b=col[i*3+2];const k=idx[i];if(k>0){const m=s.meta[k-1];if((hl!==null&&hl!==k-1)||m.n<minN){r=g=b=205;}}C2[i*4]=r;C2[i*4+1]=g;C2[i*4+2]=b;C2[i*4+3]=255;}return {idx,C2};}
const _um=gl.uniformMatrix4fv.bind(gl);gl.uniformMatrix4fv=function(loc,t,m){window.__lastMVP=m;return _um(loc,t,m);};
function apply(){if(cur<0){tex(3,RGB,false);document.getElementById('lfleg').style.display='none';}else{const t=build(cur);TEX[cur]=t;tex(3,t.C2,false);document.getElementById('lfleg').style.display='block';renderList();}
 document.querySelectorAll('#lfsrc button').forEach(b=>b.className=(+b.dataset.ls===cur)?'on':'');if(window.__viewer){window.__viewer.set({cmode:0});}}
function renderList(){const s=LF.sources[cur];document.getElementById('lflist').innerHTML=`<div style="margin:4px 0">${s.label} · 소엽 ${s.meta.length}</div>`+s.meta.map((m,i)=>`<div data-lf="${i}" style="cursor:pointer;display:flex;gap:6px;align-items:center;padding:1px 0;${hl===i?'font-weight:600':''}"><i style="width:10px;height:10px;background:${m.colour};flex:none"></i>${m.id} <span style="opacity:.7">${m.n}${m.merge_suspect?' · M':''}</span></div>`).join('');}
document.getElementById('lfsrc').addEventListener('click',e=>{const b=e.target.closest('[data-ls]');if(!b)return;cur=+b.dataset.ls;hl=null;apply();});
document.getElementById('lflist').addEventListener('click',e=>{const d=e.target.closest('[data-lf]');if(!d)return;const i=+d.dataset.lf;hl=(hl===i)?null:i;apply();showPanel(i);});
document.getElementById('lfmin').addEventListener('input',e=>{minN=+e.target.value;document.getElementById('lfminN').textContent=minN;if(cur>=0)apply();});
document.querySelectorAll('#cm button').forEach(b=>b.addEventListener('click',()=>{if(+b.dataset.cm===0&&cur>=0){tex(3,TEX[cur]?TEX[cur].C2:build(cur).C2,false);}}));
function showPanel(i){const s=LF.sources[cur];const m=s.meta[i];document.getElementById('lfpanel').innerHTML=`<b>${m.id}</b> · ${s.label}<br>surfels ${m.n} · merge_suspect ${m.merge_suspect?'yes':'no'}<br>2D mask sources: ${m.sources.join(', ')}<br>views voting: ${m.views.length?m.views.join(' '):'—'}<br>agreement (median over surfels): ${m.agreement===null?'—':m.agreement}`;}
cv.addEventListener('click',e=>{if(cur<0||!window.__lastMVP)return;const r=cv.getBoundingClientRect();const mx=(e.clientX-r.left)*(cv.width/r.width),my=(e.clientY-r.top)*(cv.height/r.height);const M4=window.__lastMVP;const t=TEX[cur]||build(cur);
 let best=-1,bd=144,bz=1e9;for(let i=0;i<N;i++){if(qk[i]!==1||t.idx[i]===0)continue;const x=PX[i*3],y=PX[i*3+1],z=PX[i*3+2];const cw=M4[3]*x+M4[7]*y+M4[11]*z+M4[15];if(cw<=0)continue;const cx=(M4[0]*x+M4[4]*y+M4[8]*z+M4[12])/cw,cy=(M4[1]*x+M4[5]*y+M4[9]*z+M4[13])/cw;const sx=(cx+1)/2*cv.width,sy=(1-cy)/2*cv.height;const d=(sx-mx)*(sx-mx)+(sy-my)*(sy-my);if(d<bd&&(d<bd-20||cw<bz)){bd=d;best=i;bz=cw;}}
 if(best>=0){const k=t.idx[best]-1;showPanel(k);}});
window.__leaflet={set:(si)=>{cur=si;apply();},state:()=>({cur,hl,minN,sources:LF.sources.map(s=>s.meta.length)})};})();</script>"""
    html = html.replace("</body>", js + "</body>", 1) if "</body>" in html else html + js
    (out / "viewer").mkdir(parents=True, exist_ok=True); p = out / "viewer" / "leaflet_colour_viewer.html"; p.write_text(html, encoding="utf-8")
    state_counts = {k: dict(Counter(v.tolist())) for k, v in dict(clustering_only=stateA, **{"2d_masks": state2}).items()}
    json.dump(dict(status=STATUS, scale=SCALE_TAG, viewer=str(p), vote_state_recovered_identical_to_stored=rec["identical_to_stored"], surfel_states=state_counts,
                   sources=dict(clustering_only=str(TRACK_A), **{"2d_masks": str(SEED_RUN)}), human_corrected="not available (mask_corrections store empty at build time)"),
              open(out / "part_b_summary.json", "w", encoding="utf-8"), indent=1)
    return p


# ---------------------------------------------------------------- Part C
def part_c(out: Path):
    from tomato3d.leafseg import run as r2
    from tomato3d.leaflib import answers as an
    ans = json.load(open(ROOT / "build/p1_leaflib_realism_20261022_01/library/answers_resolved.json", encoding="utf-8"))["agreed"]
    style = {"1": ("whole", "solid"), "2": ("mostly_one", "dashed"), "3": ("merged", "dashed_black"), "4": ("fragment", "dotted")}
    (out / "tray_overlays").mkdir(parents=True, exist_ok=True); thumbs = []; stats = {}
    for p in sorted((ROOT / an.RUNS["run05"] / "proposals").glob("*_leafseg.json")):
        j = json.load(open(p, encoding="utf-8")); photo = p.stem.replace("_leafseg", ""); held = photo in an.HELD_OUT
        rgb = r2.load_rgb(ROOT / j["image"]["file"]); rgb = cv2.resize(np.asarray(rgb), (PAGE, PAGE), interpolation=cv2.INTER_AREA)
        img = rgb.copy(); cover = np.zeros((PAGE, PAGE), np.int16); items = []
        for inst in sorted(j["instances"], key=lambda x: x.get("score") or 0):
            m = resize_mask(px.rle_decode(inst["visible_segmentation"]), PAGE); col = colour_of(f"{photo}|{inst['id']}"); a = ans.get(f"run05|{photo}|{inst['id']}"); k = a["answers"][0] if a else None
            blend_fill(img, m, col); cover[m] += 1; items.append((inst, m, col, k))
        for inst, m, col, k in items:
            if k == "3": draw_outline(img, m, col, 2); draw_outline(img, m, (0, 0, 0), 2, style="dashed")
            elif k == "2": draw_outline(img, m, col, 1, style="dashed")
            elif k == "4": draw_outline(img, m, col, 2, style="dotted")
            else: draw_outline(img, m, col, 2 if k == "1" else 1)
        ov = cover >= 2
        if ov.any(): draw_outline(img, ov, (255, 255, 255), 1)
        pil = Image.fromarray(img); d = ImageDraw.Draw(pil); f = _font(14)
        for inst, m, col, k in items:
            ys, xs = np.nonzero(m)
            if len(xs): d.text((int(xs.mean()), int(ys.mean())), inst["id"], fill="white", font=f, stroke_width=1, stroke_fill="black")
        img = banner(np.asarray(pil), f"{photo}{' · HELD-OUT (shown, not altered)' if held else ''} · H2(M1) run 05 · outline: solid whole / dashed mostly-one / black dashed merged / dotted fragment / thin = no answer · leaflets only")
        Image.fromarray(img).save(out / "tray_overlays" / f"{photo}.jpg", quality=85); thumbs.append(cv2.resize(img, (504, 504), interpolation=cv2.INTER_AREA))
        stats[photo] = dict(instances=len(items), answered=sum(1 for *_, k in items if k), by_answer=dict(Counter(style.get(k, ("none",))[0] for *_, k in items)), held_out=held, overlap_share=float(ov.sum() / max((cover > 0).sum(), 1)))
    cols = 5; rows_ = int(np.ceil(len(thumbs) / cols)); sheet = np.full((rows_ * 504, cols * 504, 3), 255, np.uint8)
    for k, t in enumerate(thumbs):
        r, c = divmod(k, cols); sheet[r * 504:(r + 1) * 504, c * 504:(c + 1) * 504] = t
    Image.fromarray(sheet).save(out / "contact_sheet_tray_dwt.jpg", quality=80)
    json.dump(dict(source="build/p1_leafseg_20261020_05/H2/proposals (H2(M1) run 05) + answers_resolved.json (human answers, read only)", per_photo=stats), open(out / "part_c_summary.json", "w", encoding="utf-8"), indent=1)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--parts", nargs="+", default=["A", "B", "C"]); ap.add_argument("--corrections", type=Path, default=None)
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    corr = json.load(open(a.corrections, encoding="utf-8")) if a.corrections else None
    if "A" in a.parts: st = part_a(out, corr); print("A", len(st), "views")
    if "B" in a.parts: print("B", part_b(out))
    if "C" in a.parts: st = part_c(out); print("C", len(st), "photos")


if __name__ == "__main__":
    main()
