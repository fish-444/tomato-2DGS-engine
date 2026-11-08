"""Local labelling page for user-labelled compound leaves (contract section 3): photo + UNION masks, leaf buttons, click = label,
empty click = SAM point prompt, split = SAM re-segmentation, undo = removal record. Records are append-only human records in
build/p1_compound_userlabel_20261025_01/labels/leaf_labels.json; the session never edits them.

usage: python -m tomato3d.organs.userlabel_page [--port 8824] [--no-sam]   (127.0.0.1 only)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2

OUT = UL.OUT
PAGE_PX = 1200
H2 = UL.H2
LEAF_MAX = 12
STATE = dict(sam=None, lock=threading.Lock(), cur_view=None, rgb={}, det={}, labels=None, vids=[], held=None)


# ---------------------------------------------------------------- data
def load_labels():
    f = OUT / "labels" / "leaf_labels.json"; f.parent.mkdir(exist_ok=True)
    return json.load(open(f, encoding="utf-8")) if f.exists() else {}


def save_record(rec):
    """Append-only: a new id per record; removal is a record with removed=true that names the earlier one."""
    with STATE["lock"]:
        recs = load_labels(); rid = f"{rec['view']}_{int(time.time() * 1000):x}_{len(recs)}"; rec = dict(rec, id=rid, at=time.strftime("%Y-%m-%dT%H:%M:%S")); recs[rid] = rec
        tmp = OUT / "labels" / "leaf_labels.json.tmp"; json.dump(recs, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=0); tmp.replace(OUT / "labels" / "leaf_labels.json")
    return rec


def polys_of(mask, scale):
    import cv2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(c) < 30: continue
        a = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2) * scale; out.append(a.round(1).tolist())
    return out


def view_payload(vid):
    det = json.load(open(OUT / "detect" / f"{vid}.json", encoding="utf-8")); view = det["view"]
    img = Image.open(AP.PHOTOS / view).convert("RGB"); W0 = img.size[0]; img.thumbnail((PAGE_PX, PAGE_PX)); W, Hh = img.size; sc = W / H2
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=80)
    masks = [dict(id=m["id"], prop=m["prop"], score=m["score"], polys=polys_of(UL.unrle(m["rle"]), sc)) for m in det["masks"]]
    # rachis hints: oriented traces projected (no visibility test; hint only)
    tr = json.load(open(V2.OUT / "traces_oriented.json", encoding="utf-8")); raw = {t["key"]: t for t in V2.CC.load_traces(V2.OUT / "traces_export" / "rachis_traces")}
    pl = STATE["pl"]; pv = V2.project_view(pl["xyz"][:1], pl["sc"][:1], pl["po"], view, V2.VISP["res"]); hints = []
    for t in tr["traces"]:
        pts = np.asarray(raw[t["key"]]["pts"], float); pts = pts[::-1] if t["flipped"] else pts
        u, w, z = V2.project_points(pts, pv); hints.append(dict(leaf=t["leaf"], pts=(np.c_[u, w] * H2 / V2.VISP["res"] * sc).round(1).tolist()))
    recs = [r for r in load_labels().values() if r["view"] == vid]
    for r in recs:
        if "rle" in r and "polys" not in r: r["polys"] = polys_of(UL.unrle(r["rle"]), sc)
    return dict(vid=vid, view=view, jpg=base64.b64encode(buf.getvalue()).decode(), W=W, H=Hh, scale_page_per_full=W / W0, masks=masks, hints=hints, records=recs, held_out=(vid == STATE["held"]))


# ---------------------------------------------------------------- SAM
def sam_set_view(vid):
    if STATE["sam"] is None: return None
    if STATE["cur_view"] != vid:
        view = AP.view_file(vid); rgb = STATE["rgb"].get(vid)
        if rgb is None: rgb = np.asarray(Image.open(AP.PHOTOS / view).convert("RGB")); STATE["rgb"][vid] = rgb
        STATE["sam"].predictor.set_image(rgb); STATE["cur_view"] = vid
    return STATE["rgb"][vid]


def sam_mask(pos_full, neg_full, box_full=None):
    from tomato3d.leafseg import engine as eng
    sam = STATE["sam"]; pts = np.vstack([pos_full, neg_full]) if len(neg_full) else np.asarray(pos_full, np.float32)
    lab = np.r_[np.ones(len(pos_full)), np.zeros(len(neg_full))].astype(np.int32)
    masks, ious, _ = sam.predictor.predict(point_coords=pts.astype(np.float32), point_labels=lab, box=(None if box_full is None else np.asarray(box_full, np.float32)), multimask_output=True)
    best = None
    for m, q in zip(masks, ious):
        g = eng._to_grid(m > 0, (H2, H2)); s = float(q) * eng._solidity(g)
        if best is None or s > best[1]: best = (g, s, float(q))
    return best


def api_point(body):
    vid = body["vid"]; k = body["full_per_page"]; pos = [[body["x"] * k, body["y"] * k]]; neg = [[p[0] * k, p[1] * k] for p in body.get("negatives", [])]
    with STATE["lock"]:
        rgb = sam_set_view(vid)
        if rgb is None: return dict(error="SAM unavailable on this server")
        g, s, q = sam_mask(pos, neg)
    sc = body["page_per_grid"]; return dict(rle=UL.rle(g), polys=polys_of(g, sc), score=round(q, 3), area_px=int(g.sum()))


def api_split(body):
    vid = body["vid"]; k = body["full_per_page"]; det = json.load(open(OUT / "detect" / f"{vid}.json", encoding="utf-8")); by_id = {m["id"]: m for m in det["masks"]}
    parent = UL.unrle(by_id[body["mask_id"]]["rle"]) if body["mask_id"] in by_id else UL.unrle(body["rle"])
    pts = np.array([[p[0] * k, p[1] * k] for p in body["points"]], np.float32); ys, xs = np.nonzero(parent); gf = 3024 / H2 if parent.any() else 1.0
    box = [xs.min() * gf, ys.min() * gf, (xs.max() + 1) * gf, (ys.max() + 1) * gf] if parent.any() else None
    out = []
    with STATE["lock"]:
        if sam_set_view(vid) is None: return dict(error="SAM unavailable on this server")
        for i in range(len(pts)):
            g, s, q = sam_mask(pts[i:i + 1], np.delete(pts, i, 0), box); g = g & parent
            if g.sum() >= 30: out.append(dict(rle=UL.rle(g), polys=polys_of(g, body["page_per_grid"]), score=round(q, 3), area_px=int(g.sum())))
    return dict(parts=out)


# ---------------------------------------------------------------- page
HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>복엽 라벨</title>
<style>:root{--bg:#141414;--fg:#eee;--line:#333}body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif}
#top{position:sticky;top:0;z-index:9;background:#1c1c1c;border-bottom:1px solid var(--line);padding:6px 10px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:5px;padding:4px 9px;cursor:pointer;font:inherit}button.on{outline:2px solid #fff}
#wrap{overflow:auto;position:relative}#stage{position:relative;transform-origin:0 0}canvas{display:block}#msg{font-size:12px;opacity:.85}.leaf{min-width:34px}
#hd{background:#7a1f1f;font-size:12px;padding:4px 10px}</style></head><body>
<div id="hd">R3-1b_unaccepted · 사람 라벨 (잎 번호는 사람만 정함) · 아래 잎부터 1, 2, … (엽축 추적·자 측정과 같은 번호) · 보류 사진은 채점에만 쓰임</div>
<div id="top"><span id="nav"></span><button id="prev">◀ 이전</button><button id="next">다음 ▶</button><span style="margin-left:8px">잎:</span><span id="leaves"></span><button id="unk" class="leaf">모름</button>
<span style="margin-left:8px">모드:</span><button id="mLabel" class="on">칠하기</button><button id="mSplit">나누기</button><button id="mPoint">점 추가(SAM)</button><button id="undo">되돌리기</button>
<label><input type="checkbox" id="hint" checked> 엽축 힌트</label><label>확대 <input id="zoom" type="range" min="1" max="3" step="0.25" value="1"></label><span id="cnt"></span><span id="msg"></span></div>
<div id="wrap"><div id="stage"><canvas id="cv"></canvas></div></div>
<script>
const PAL={1:'#e6194b',2:'#3cb44b',3:'#ffe119',4:'#0082c8',5:'#f58230',6:'#911eb4',7:'#46f0f0',8:'#f032e6',9:'#d2f53c',10:'#fabebe',11:'#008080',12:'#dcbeff'};
let VIDS=[],idx=0,V=null,leaf=1,mode='label',splitPts=[],splitMask=null,busy=false;const cv=document.getElementById('cv'),cx=cv.getContext('2d');const img=new Image();
const lv=document.getElementById('leaves');for(let i=1;i<=12;i++){const b=document.createElement('button');b.className='leaf';b.textContent=i;b.style.borderColor=PAL[i];b.dataset.l=i;b.onclick=()=>{leaf=i;document.getElementById('unk').className='leaf';[...lv.children].forEach(x=>x.className='leaf'+(+x.dataset.l===i?' on':''));msg('잎 '+i);};lv.appendChild(b);}
document.getElementById('unk').onclick=()=>{leaf='모름';[...lv.children].forEach(x=>x.className='leaf');document.getElementById('unk').className='leaf on';};lv.children[0].className='leaf on';
function msg(t){document.getElementById('msg').textContent=t;}
function setMode(m){mode=m;splitPts=[];splitMask=null;['mLabel','mSplit','mPoint'].forEach(k=>document.getElementById(k).className='');document.getElementById({label:'mLabel',split:'mSplit',point:'mPoint'}[m]).className='on';msg({label:'마스크를 클릭하면 현재 잎으로 칠함 (다시 클릭 = 지움)',split:'나눌 마스크를 클릭한 뒤 소엽마다 한 점씩 클릭, 끝나면 "나누기 실행"',point:'마스크가 없는 소엽 위를 클릭하면 SAM 이 마스크를 만들어 현재 잎으로 저장'}[m]);if(m==='split'){const b=document.getElementById('runSplit')||Object.assign(document.createElement('button'),{id:'runSplit',textContent:'나누기 실행'});b.onclick=runSplit;document.getElementById('top').appendChild(b);}else{const b=document.getElementById('runSplit');if(b)b.remove();}draw();}
document.getElementById('mLabel').onclick=()=>setMode('label');document.getElementById('mSplit').onclick=()=>setMode('split');document.getElementById('mPoint').onclick=()=>setMode('point');
document.getElementById('zoom').oninput=e=>{document.getElementById('stage').style.transform='scale('+e.target.value+')';document.getElementById('stage').style.width=cv.width*e.target.value+'px';document.getElementById('stage').style.height=cv.height*e.target.value+'px';};
async function load(i){idx=i;V=await (await fetch('/api/view/'+VIDS[i])).json();img.onload=()=>{cv.width=V.W;cv.height=V.H;draw();};img.src='data:image/jpeg;base64,'+V.jpg;
 document.getElementById('nav').textContent=`사진 ${i+1}/${VIDS.length} · ${V.vid}${V.held_out?' (보류: 채점용)':''}`;}
function active(){const st={};for(const r of V.records.sort((a,b)=>a.at<b.at?-1:1)){const key=r.mask_id||r.id;if(r.removed){delete st[r.removed];continue;}st[key]=r;}return st;}
function draw(){if(!V)return;cx.drawImage(img,0,0);const st=active();const polyOf=(m)=>{const p=new Path2D();for(const poly of m.polys){poly.forEach((q,j)=>j?p.lineTo(q[0],q[1]):p.moveTo(q[0],q[1]));p.closePath();}return p;};
 for(const m of V.masks){const p=polyOf(m);cx.lineWidth=1;cx.strokeStyle='rgba(230,230,230,.7)';cx.stroke(p);}
 for(const r of Object.values(st)){const m=r.mask_id?V.masks.find(x=>x.id===r.mask_id):r;if(!m||!m.polys)continue;const p=polyOf(m);cx.fillStyle=(r.leaf==='모름'?'rgba(200,200,200,.45)':PAL[r.leaf]+'88');cx.fill(p);cx.lineWidth=2;cx.strokeStyle=r.leaf==='모름'?'#ccc':PAL[r.leaf];cx.stroke(p);const c=cent(m.polys[0]);cx.font='bold 14px sans-serif';cx.fillStyle='#fff';cx.strokeStyle='#000';cx.lineWidth=3;cx.strokeText(r.leaf,c[0],c[1]);cx.fillText(r.leaf,c[0],c[1]);}
 if(document.getElementById('hint').checked)for(const h of V.hints){cx.beginPath();h.pts.forEach((q,j)=>j?cx.lineTo(q[0],q[1]):cx.moveTo(q[0],q[1]));cx.setLineDash([6,4]);cx.lineWidth=2;cx.strokeStyle='rgba(255,212,0,.55)';cx.stroke();cx.setLineDash([]);const q=h.pts[0];if(q){cx.fillStyle='rgba(255,212,0,.8)';cx.beginPath();cx.arc(q[0],q[1],5,0,6.3);cx.fill();cx.font='12px sans-serif';cx.fillText('엽축 '+h.leaf,q[0]+6,q[1]-6);}}
 if(splitMask){const p=polyOf(splitMask);cx.lineWidth=3;cx.strokeStyle='#ff8c00';cx.stroke(p);}for(const q of splitPts){cx.fillStyle='#ff8c00';cx.beginPath();cx.arc(q[0],q[1],5,0,6.3);cx.fill();}
 const n=Object.values(st).filter(r=>r.leaf!=='모름').length;document.getElementById('cnt').textContent=`이 사진에서 칠한 소엽 ${n} / 검출 ${V.masks.length}`;}
function cent(poly){let x=0,y=0;for(const q of poly){x+=q[0];y+=q[1];}return [x/poly.length,y/poly.length];}
function hit(x,y){const st=active();let best=null;const test=(m,key)=>{const p=new Path2D();for(const poly of m.polys){poly.forEach((q,j)=>j?p.lineTo(q[0],q[1]):p.moveTo(q[0],q[1]));p.closePath();}if(cx.isPointInPath(p,x,y)){const a=area(m.polys[0]);if(!best||a<best.a)best={m,key,a};}};
 for(const m of V.masks)test(m,m.id);for(const r of Object.values(st))if(!r.mask_id&&r.polys)test(r,r.id);return best;}
function area(poly){let a=0;for(let i=0;i<poly.length;i++){const p=poly[i],q=poly[(i+1)%poly.length];a+=p[0]*q[1]-q[0]*p[1];}return Math.abs(a)/2;}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}
cv.addEventListener('click',async e=>{if(busy||!V)return;const r=cv.getBoundingClientRect();const z=+document.getElementById('zoom').value;const x=(e.clientX-r.left)/z*(cv.width/(r.width/z)),y=(e.clientY-r.top)/z*(cv.height/(r.height/z));const h=hit(x,y);const st=active();
 if(mode==='label'){if(!h){msg('마스크 없음 — "점 추가(SAM)" 모드를 쓰세요');return;}const ex=st[h.key];if(ex&&ex.leaf===leaf){const rec=await post('/api/label',{view:V.vid,removed:h.key,source:'user_remove'});V.records.push(rec);msg('지움');}else{const body=h.m.polys&&!h.m.id?{view:V.vid,leaf,source:'user_relabel',rle:h.m.rle,polys:h.m.polys,mask_id:null}:{view:V.vid,leaf,source:'user_click',mask_id:h.key};if(ex){const rm=await post('/api/label',{view:V.vid,removed:h.key,source:'user_remove'});V.records.push(rm);}const rec=await post('/api/label',body);V.records.push(rec);msg('잎 '+leaf+' ← '+h.key);}draw();return;}
 if(mode==='point'){busy=true;msg('SAM 계산 중…');const res=await post('/api/sam_point',{vid:V.vid,x,y,full_per_page:1/V.scale_page_per_full,page_per_grid:V.W/1536});busy=false;if(res.error){msg(res.error);return;}const rec=await post('/api/label',{view:V.vid,leaf,source:'user_point_sam',rle:res.rle,polys:res.polys,point_page:[x,y]});V.records.push(rec);msg('SAM 마스크 추가 → 잎 '+leaf+' ('+res.area_px+' px)');draw();return;}
 if(mode==='split'){if(!splitMask){if(!h){msg('나눌 마스크를 먼저 클릭');return;}splitMask=h.m;splitMask._key=h.key;msg('소엽마다 한 점씩 클릭한 뒤 "나누기 실행"');}else{splitPts.push([x,y]);msg('점 '+splitPts.length);}draw();}});
async function runSplit(){if(!splitMask||splitPts.length<2){msg('점이 2개 이상 필요');return;}busy=true;msg('SAM 나누기 계산 중…');const res=await post('/api/split',{vid:V.vid,mask_id:splitMask.id||null,rle:splitMask.rle||null,points:splitPts,full_per_page:1/V.scale_page_per_full,page_per_grid:V.W/1536});busy=false;if(res.error){msg(res.error);return;}
 const st=active();if(st[splitMask._key]){const rm=await post('/api/label',{view:V.vid,removed:splitMask._key,source:'user_remove'});V.records.push(rm);}
 for(const p of res.parts){const rec=await post('/api/label',{view:V.vid,leaf,source:'user_split_sam',rle:p.rle,polys:p.polys,parent:splitMask._key});V.records.push(rec);}msg('나눔: '+res.parts.length+' 조각 → 잎 '+leaf);splitMask=null;splitPts=[];draw();}
document.getElementById('undo').onclick=async()=>{const st=active();const last=Object.values(st).sort((a,b)=>a.at<b.at?1:-1)[0];if(!last){msg('되돌릴 것 없음');return;}const rec=await post('/api/label',{view:V.vid,removed:last.mask_id||last.id,source:'user_undo'});V.records.push(rec);msg('되돌림');draw();};
document.getElementById('prev').onclick=()=>load((idx+VIDS.length-1)%VIDS.length);document.getElementById('next').onclick=()=>load((idx+1)%VIDS.length);document.getElementById('hint').onchange=draw;
(async()=>{const s=await (await fetch('/api/state')).json();VIDS=s.vids;await load(0);setMode('label');window.__ul={state:()=>({idx,leaf,mode,records:V&&V.records.length,masks:V&&V.masks.length}),hit,active,load,post,setLeaf:(l)=>{leaf=l;},setMode,masks:()=>V.masks,V:()=>V};})();
</script></body></html>"""


# ---------------------------------------------------------------- cross-photo number map (contract section 8)
REF_VIEW = "060313903"


def map_data():
    import numpy as np
    pl = STATE["pl"]; H2id = np.load(V2.OUT / "stage_a_h2id.npz")["H2id"]; vids = STATE["vids"]
    maps, counts = UL.label_maps(OUT, vids); loc = UL.local_surfel_labels(pl, maps, vids, H2id); prop, alts, groups = UL.propose_mapping(loc, REF_VIEW)
    desc = {v: {L: UL.group_descriptors(pl["xyz"], loc[v] == L) for L in groups[v]} for v in vids}
    for v in vids:                                   # contract section 8 topology check; flagged auto rows go back to check
        for L in groups[v]:
            p = prop[v][L]; ref_d = desc[REF_VIEW].get(p["to"]) if (v != REF_VIEW and p["to"] is not None) else None
            p["desc"] = desc[v][L]; p["flags"] = UL.topo_flags(desc[v][L], ref_d)
            if p.get("status") == "auto" and any(f in p["flags"] for f in ("match_mismatch", "over_painted", "too_few_surfels")): p["status"] = "check"
    STATE["desc"] = desc
    geo = {}                                          # rows where the saved number disagrees with the nearest reference leaf in 3D (prefill only)
    gf = OUT / "number_map_vs_geometry.json"
    if gf.exists():
        for o in json.load(open(gf, encoding="utf-8")):
            if o["user"] > 0 and not o["agree"]:
                geo[(o["photo"], o["local"])] = o
    for v in vids:
        for L in groups[v]:
            o = geo.get((v, L))
            if o: prop[v][L].update(status="geo_diff", geo=dict(best=o["geo_best"], dist=o["geo_dist"], second=o["geo_2nd"], saved=o["user"], saved_dist=o["user_dist"]))
    pal = {i: tuple(int(PALH[i][k:k + 2], 16) for k in (1, 3, 5)) for i in PALH}; photos = []
    for vid in [REF_VIEW] + [v for v in vids if v != REF_VIEW]:
        img = Image.open(AP.PHOTOS / AP.view_file(vid)).convert("RGB"); img.thumbnail((760, 760)); W, Hh = img.size
        lab = np.asarray(Image.fromarray(maps[vid].astype(np.int32)).resize((W, Hh), Image.NEAREST)); arr = np.asarray(img).astype(np.float32)
        for L in groups[vid] + [-1]:
            m = lab == L
            if m.any(): arr[m] = arr[m] * 0.45 + np.array(pal.get(L, (200, 200, 200))) * 0.55
        im = Image.fromarray(arr.astype(np.uint8)); from PIL import ImageDraw; dr = ImageDraw.Draw(im)
        if vid != REF_VIEW:                       # help overlay: reference leaf groups projected with the board poses (visible surfels only)
            import cv2
            view = AP.view_file(vid); k = pl["views"].index(view); vis = H2id[:, k] >= 0
            pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"]); sx = W / V2.VISP["res"]
            for L in groups[REF_VIEW]:
                sel = vis & (loc[REF_VIEW] == L)
                if sel.sum() < 10: continue
                ras = np.zeros((Hh, W), np.uint8); uu = np.clip((pv["u"][sel] * sx).astype(int), 0, W - 1); ww = np.clip((pv["w"][sel] * sx).astype(int), 0, Hh - 1); ras[ww, uu] = 255
                ras = cv2.morphologyEx(cv2.dilate(ras, np.ones((7, 7), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
                cnts, _ = cv2.findContours(ras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); col = pal.get(L, (255, 255, 255))
                for c in cnts:
                    if cv2.contourArea(c) < 40: continue
                    pts = [tuple(int(v) for v in q[0]) for q in c]; dr.line(pts + [pts[0]], fill=(255, 255, 255), width=4); dr.line(pts + [pts[0]], fill=col, width=2)
                big = max(cnts, key=cv2.contourArea) if cnts else None
                if big is not None:
                    x, y, _, _ = cv2.boundingRect(big); dr.rectangle([x, y - 15, x + 38, y], fill=col); dr.text((x + 2, y - 14), f"기준{L}", fill=(0, 0, 0))
        cents = {}
        for L in groups[vid]:
            ys, xs = np.nonzero(lab == L)
            if len(xs): cents[L] = (float(np.median(xs)), float(np.median(ys))); x, y = cents[L]; dr.rectangle([x - 11, y - 11, x + 11, y + 11], fill=(0, 0, 0)); dr.text((x - 5, y - 7), str(L), fill=(255, 255, 255))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
        photos.append(dict(vid=vid, ref=(vid == REF_VIEW), held_out=(vid == STATE["held"]), jpg=base64.b64encode(buf.getvalue()).decode(),
                           groups=[dict(local=int(L), leaflets=int(counts[vid].get(L, 0)), surfels=int((loc[vid] == L).sum()), proposal=prop[vid][L]["to"], via=prop[vid][L]["via"], shared=prop[vid][L]["shared"], status=prop[vid][L].get("status"), ratio=prop[vid][L].get("ratio"), desc=prop[vid][L].get("desc"), flags=prop[vid][L].get("flags", []), geo=prop[vid][L].get("geo"),
                                        alts=[dict(shared=s, photo=b, local=int(B)) for s, b, B in alts[f"{vid}|{L}"]]) for L in groups[vid]]))
    saved = json.load(open(OUT / "labels" / "number_map.json", encoding="utf-8")) if (OUT / "labels" / "number_map.json").exists() else None
    return dict(photos=photos, ref=REF_VIEW, saved=saved, palette=PALH)


PALH = {1: '#e6194b', 2: '#3cb44b', 3: '#ffe119', 4: '#0082c8', 5: '#f58230', 6: '#911eb4', 7: '#46f0f0', 8: '#f032e6', 9: '#d2f53c', 10: '#fabebe', 11: '#008080', 12: '#dcbeff'}


def save_map(body):
    """Append-only: every save is a new version; the latest version is the decision."""
    f = OUT / "labels" / "number_map.json"; doc = json.load(open(f, encoding="utf-8")) if f.exists() else dict(versions=[])
    doc["versions"].append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), map=body["map"], auto_rows=body.get("auto", []), changed_from_proposal=body.get("changed", []), note="human decision; per photo local number -> plant leaf number (0 = fragment/other part, -1 = unknown)"))
    doc["latest"] = doc["versions"][-1]; tmp = f.with_suffix(".tmp"); json.dump(doc, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1); tmp.replace(f); return dict(ok=True, versions=len(doc["versions"]))


MAP_HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>잎 번호 맞추기</title>
<style>body{margin:0;background:#141414;color:#eee;font:13px system-ui,sans-serif}#hd{background:#7a1f1f;font-size:12px;padding:6px 12px}main{padding:10px 14px}
.ph{display:grid;grid-template-columns:minmax(0,760px) 1fr;gap:12px;border-bottom:1px solid #333;padding:10px 0}@media(max-width:900px){.ph{grid-template-columns:1fr}}img{max-width:100%;display:block}
table{border-collapse:collapse}td,th{border:1px solid #333;padding:3px 6px;vertical-align:top}select{font:inherit;background:#222;color:#eee}i.sw{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle}
button{font:inherit;padding:6px 14px;background:#2d6a3e;color:#fff;border:0;border-radius:5px;cursor:pointer}.alt{font-size:11px;opacity:.75}#bar{position:sticky;top:0;background:#1c1c1c;padding:8px 14px;border-bottom:1px solid #333;z-index:5}</style></head><body>
<div id="hd">R3-1b_unaccepted · 사진 사이 잎 번호 맞추기 — 칠한 묶음은 그대로, 사진마다 "원래 번호 → 식물 잎 번호"만 정합니다. 제안 = 3D 서펠 공유량(제안일 뿐). 0 = 조각/다른 부위, ? = 모름.</div>
<div id="bar"><button id="save">저장</button> <span id="st"></span> · 기준 사진(첫 번째)의 번호를 식물 잎 번호로 씁니다. 아래 잎부터 1, 2, … 가 아니면 기준 사진 표에서 고치세요.</div><main id="m"></main>
<script>let D=null;const OPTS=[...Array(12).keys()].map(i=>i+1);
function sel(id,v){return `<select data-k="${id}">`+['<option value="-1">? 모름</option>','<option value="0">0 조각/다른 부위</option>'].concat(OPTS.map(o=>`<option value="${o}" ${+v===o?'selected':''}>${o}</option>`)).join('').replace(`value="${v}"`,`value="${v}" selected`)+'</select>';}
(async()=>{D=await (await fetch('/api/map_data')).json();const saved=D.saved&&D.saved.latest?D.saved.latest.map:{};
const nGeo=D.photos.flatMap(p=>p.groups).filter(g=>g.status==='geo_diff').length;const nCheck=D.photos.flatMap(p=>p.groups).filter(g=>g.status==='check').length;const nAuto=D.photos.flatMap(p=>p.groups).filter(g=>g.status==='auto').length;
const badge=g=>g.status==='geo_diff'?`<b style="color:#ff5050">3D 위치 불일치</b><br><span class="alt">저장값 ${g.geo.saved} (거리 ${g.geo.saved_dist}) → 3D 최근접 ${g.geo.best} (${g.geo.dist}), 다음 ${g.geo.second}</span>`:g.status==='check'?'<b style="color:#ffb000">확인 필요</b>':g.status==='auto'?'<span style="color:#7fd48c">자동</span>':'기준';
document.getElementById('m').innerHTML=`<p style="background:#4a1010;padding:6px"><b>빨간 줄 ${nGeo}개</b>: 저장한 번호가 3D 위치(높이·방위·거리, 보드 투영)에서 가장 가까운 기준 잎과 다릅니다. 칸에는 3D 최근접 번호를 미리 채웠습니다 — 사진과 "기준N" 테두리를 보고 맞으면 그대로, 아니면 고친 뒤 저장. 거리 1 이하 = 문턱(높이 10 mm·방위 30°·거리 15 mm) 안. 다른 줄은 지난번 저장값 그대로입니다.</p><p>자동 확정 ${nAuto}개 (공유 서펠 1등 ≥ 2등 × 3, 사진 안에서 겹치는 번호 없음) · <b style="color:#ffb000">확인 필요 ${nCheck}개</b>. 사진 위 테두리선 = 기준 사진의 잎 묶음을 보드 자세로 이 사진에 투영한 것(색·"기준N" = 기준 번호). 칠한 색 + 검은 사각형 숫자 = 이 사진의 원래 번호.</p>`+D.photos.map(p=>`<div class="ph"><img src="data:image/jpeg;base64,${p.jpg}"><div><b>${p.vid}</b> ${p.ref?'(기준)':''}${p.held_out?' (보류: 채점용)':''}<table><tr><th>원래 번호</th><th>소엽</th><th>→ 식물 잎 번호</th><th>상태</th><th>위치 h·θ·r<br>(높이 범위·크기)</th><th>근거 (공유 서펠)</th></tr>`+p.groups.map(g=>{const k=p.vid+'|'+g.local;const v=g.geo?g.geo.best:((k in saved)?saved[k]:(g.proposal==null?-1:g.proposal));const d=g.desc;const FL={match_mismatch:'기준 잎과 위치 다름',over_painted:'과대 칠함 의심(두 잎 걸침)',too_few_surfels:'3D 점 너무 적음'};return `<tr style="${g.status==='geo_diff'?'background:#4a1010':g.status==='check'?'background:#3a2c00':g.status==='auto'?'opacity:.7':''}"><td><i class="sw" style="background:${D.palette[g.local]}"></i>${g.local}</td><td>${g.leaflets}</td><td>${sel(k,v)}</td><td>${badge(g)}${(g.flags||[]).map(f=>'<br><span style="color:#ff6b6b">⚠ '+(FL[f]||f)+'</span>').join('')}</td><td class="alt">${d?`${d.h} mm · ${d.theta}° · ${d.r} mm<br>IQR ${d.h_iqr} · 크기 ${d.extent}${d.bimodal?' · 높이 두 덩어리':''}`:'—'}</td><td class="alt">${g.via}${g.shared?' '+g.shared:''}${g.ratio?' (×'+g.ratio+')':''}<br>${g.alts.map(a=>a.photo+'#'+a.local+' '+a.shared).join(' · ')}</td></tr>`;}).join('')+'</table></div></div>').join('');
document.getElementById('st').textContent=D.saved?('저장된 판 '+D.saved.versions.length+'개 (최신 불러옴)'):'아직 저장 안 됨 — 제안이 채워져 있음';})();
document.getElementById('save').onclick=async()=>{const map={},auto=[],changed=[];const P={};D.photos.forEach(p=>p.groups.forEach(g=>{P[p.vid+'|'+g.local]=g;}));document.querySelectorAll('select[data-k]').forEach(s=>{const k=s.dataset.k;map[k]=+s.value;const g=P[k];if(g&&g.status==='auto')auto.push(k);if(g&&g.geo&&+s.value===g.geo.best)changed.push(k+':geo_prefill_kept');if(g&&(g.proposal==null?-1:g.proposal)!==+s.value)changed.push(k);});const r=await (await fetch('/api/map_save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({map,auto,changed})})).json();document.getElementById('st').textContent=r.ok?('저장됨 (판 '+r.versions+')'):JSON.stringify(r);};
window.__map={data:()=>D};</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            b = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if self.path.startswith("/map"):
            b = MAP_HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if self.path == "/api/map_data":
            return self._json(map_data())
        if self.path == "/api/state":
            return self._json(dict(vids=STATE["vids"], held_out=STATE["held"], sam=STATE["sam"] is not None))
        if self.path.startswith("/api/view/"):
            return self._json(view_payload(self.path.rsplit("/", 1)[1]))
        self._json(dict(error="not found"), 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/label":
                return self._json(save_record(body))
            if self.path == "/api/map_save":
                return self._json(save_map(body))
            if self.path == "/api/sam_point":
                return self._json(api_point(body))
            if self.path == "/api/split":
                return self._json(api_split(body))
        except Exception as e:
            return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
        self._json(dict(error="not found"), 404)

    def log_message(self, *a):
        pass


def warm(vids):
    for vid in vids:
        with STATE["lock"]:
            try: sam_set_view(vid)
            except Exception as e: print("warm failed", vid, e, flush=True)
    print("sam warm done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8824); ap.add_argument("--no-sam", action="store_true"); a = ap.parse_args(argv)
    vc = json.load(open(OUT / "views_chosen.json", encoding="utf-8")); STATE["vids"] = vc["chosen"] + [vc["held_out"]]; STATE["held"] = vc["held_out"]; STATE["pl"] = UL.load_plant()
    if not a.no_sam:
        from tomato3d.leafseg import run as r2
        from tomato3d.topview.measure2d import load_sam
        STATE["sam"] = load_sam(r2.SAMCFG); threading.Thread(target=warm, args=(STATE["vids"],), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); print(f"serving http://127.0.0.1:{a.port}/  views {STATE['vids']}  sam={'on' if STATE['sam'] else 'off'}", flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()
