"""3D grouping page for the nine other individuals (contract build/p1_all_3dfirst_20261027_01 section 7): pick leaflet groups and give them the number of their compound leaf.

One page, an individual selector (individuals appear as soon as their `merge/` output exists), groups coloured by id, by confidence or by flag, a flag side list, append-only clicks in
build/p1_all_3dfirst_20261027_01/user_3d_group_click.jsonl (provenance `user_3d_group_click`, one record per click, never rewritten). Read-only on every input. 127.0.0.1:8835 only,
light work (one individual's surfel positions at a time). Copy of the 8833 page design; 8833 itself is not touched.

Base-click mode (user 2026-10-31: the user clicks the base of every compound leaf on the reconstructed plants): append-only `user_base_click.jsonl`, `add` / `undo` records.

Base-only view: open the page with ?mode=base (only the individual selector, the base-click box and messages are shown; base mode is always on).

usage: python -m tomato3d.organs.all3d_page [--port 8835] [--store PATH] [--store-base PATH] [--lan] [--token T]

--lan: also listen on the local network (tablet) - only clients with a private / loopback address are served and every request needs the random token printed at start (URL ?k=...).
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import csv
import ipaddress
import json
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/p1_all_3dfirst_20261027_01"
STORE = OUT / "user_3d_group_click.jsonl"
STORE_BASE = OUT / "user_base_click.jsonl"                        # base clicks (one per compound leaf), append-only, provenance `user_base_click`
ORDER = ["n_04", "t2_02", "n_01", "t2_05", "n_06", "n_05", "t2_04", "n_02", "n_03", "e_05", "e_01", "e_04", "e_07", "e_06", "e_03", "e_02"]
FLAG_COL = [(1, "unassigned", (130, 130, 130)), (2, "contested", (255, 255, 255)), (16, "sam_conflict", (255, 140, 0)), (4, "multi_sheet_patch", (220, 40, 40)),
            (32, "oversize", (230, 50, 200)), (8, "low_vote", (0, 200, 220)), (64, "low_visibility_region", (60, 70, 170))]
PAL = {k: "#%02x%02x%02x" % tuple(int(255 * x) for x in colorsys.hsv_to_rgb((k * 0.137) % 1.0, 0.75, 0.95)) for k in range(1, 13)}
STATE = {}


def individuals():
    rows = []
    for i in ORDER:
        f = OUT / i / "merge" / "summary.json"
        s = json.load(open(f, encoding="utf-8")) if f.exists() else None
        rows.append(dict(id=i, ready=bool(s), groups=(s or {}).get("groups"), minutes=(s or {}).get("estimated_user_minutes"), unassigned=(s or {}).get("unassigned_share"),
                         voting_views=(s or {}).get("voting_views")))
    return rows


def base_state(iid):
    """Replay of the append-only base clicks of one individual: `add` appends, `undo` removes the last alive one."""
    alive = []
    for l in (STORE_BASE.read_text(encoding="utf-8").splitlines() if STORE_BASE.exists() else []):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("individual") != iid:
            continue
        if r["action"] == "add":
            alive.append(r)
        elif r["action"] == "undo" and alive:
            alive.pop()
        elif r["action"] == "reset_leaves":                                  # user 2026-11-01 19:40: leaf skeleton restarted, main axis kept
            alive = []
    return alive


def load(iid):
    d = OUT / iid / "merge"; z = np.load(d / "groups.npz"); xyz = z["xyz"].astype(np.float64); c = xyz.mean(0); g = z["group"].astype(np.int64); conf = z["confidence"]; fl = z["flags"]; ax = z["axis_like"]
    ids = sorted(set(int(x) for x in g if x >= 0)); gcol = {u: [int(255 * x) for x in colorsys.hsv_to_rgb((k * 0.61803398875) % 1.0, 0.65 + 0.3 * ((k % 3) / 2), 0.95)] for k, u in enumerate(ids)}
    cg = np.full((len(xyz), 3), 70, np.uint8); cc = cg.copy(); cf = cg.copy()
    for u in ids:
        cg[g == u] = gcol[u]
    for code, rgb in ((3, (40, 200, 70)), (2, (240, 220, 40)), (1, (230, 60, 60))):
        cc[(g >= 0) & (conf == code)] = rgb
    cf[(g >= 0) & (fl == 0)] = (60, 200, 90)
    for bit, _, rgb in reversed(FLAG_COL):
        cf[(fl & bit) != 0] = rgb
    for a in (cg, cc, cf):
        a[ax] = (45, 45, 45)
    enc = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    flags = []
    if (d / "flags.csv").exists():
        flags = [dict(flag=r["flag"], group=(int(r["group"]) if r["group"] != "" else None), n=int(r["n_surfels"]), at=[float(r["x_m"]) - c[0], float(r["y_m"]) - c[1], float(r["z_m"]) - c[2]]) for r in csv.DictReader(open(d / "flags.csv", encoding="utf-8"))]
    summ = json.load(open(d / "summary.json", encoding="utf-8")); disc = None
    ply = ROOT / "build/p1_recon_all_20261025_01/outputs" / iid / f"R3-1b-style_{iid}_surfels_full.ply"
    if ply.exists():                                                                 # oriented discs like the shared viewer: tangent axes tu, tv and the two scales (mm)
        from tomato3d.fuse3d import surfels_io as sio
        sf = sio.read_surfels_full(ply); r = z["rows"]
        if len(sf["xyz"]) > int(r.max()) and np.allclose(sf["xyz"][r], xyz, atol=1e-6):
            q = lambda a: np.round(np.clip(a, -1, 1) * 32767).astype(np.int16)
            disc = dict(tu=enc(q(sf["tu"][r])), tv=enc(q(sf["tv"][r])), s=enc((sf["scale"][r] * 1000.0).astype(np.float32)))
    from tomato3d.organs import skeleton_assign as SK
    sk_s, sk_z = SK.latest(iid); col_skel = None; skel_info = None
    if sk_z is not None and np.array_equal(sk_z["rows"], z["rows"]):
        lab_ = sk_z["label"].astype(int); st_ = sk_z["status"].astype(int); cs = np.full((len(xyz), 3), 60, np.uint8)
        for k in range(1, int(lab_.max()) + 1 if lab_.size else 1):
            cs[lab_ == k] = [int(255 * x) for x in colorsys.hsv_to_rgb((k * 0.61803398875) % 1.0, 0.75, 0.95)]
        cs[st_ == 2] = (255, 255, 255); cs[ax] = np.where(lab_[ax, None] > 0, cs[ax], 45); col_skel = enc(cs)
        skel_info = {k: sk_s[k] for k in ("branches", "clicks", "assigned_share", "contested_share", "none_share", "at")}
    return dict(id=iid, n=int(len(xyz)), units=len(ids), disc=disc, col_skel=col_skel, skel_info=skel_info, axis=enc(ax.astype(np.uint8)), centre_mm=[float(v) for v in c * 1000.0], pos=enc(((xyz - c) * 1000).astype(np.float32)), ids=enc(g.astype(np.int32)), col_groups=enc(cg), col_conf=enc(cc), col_flags=enc(cf),
                flags=flags, flag_legend=[dict(name=n, rgb="#%02x%02x%02x" % rgb) for _, n, rgb in FLAG_COL], palette=PAL,
                summary={k: summ[k] for k in ("groups", "group_confidence", "unassigned_share", "flagged_surfel_shares", "flagged_regions", "voting_views", "layers_used", "estimated_user_minutes", "pose_dropped_views") if k in summ},
                note="R3-1b-style_unaccepted · conditional_on_unvalidated_scale · gate_failed_viewing_only — 3D-first groups; leaf numbers come only from your clicks")


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>9개체 복엽 묶기</title>
<style>:root{--bg:#111;--fg:#eee}body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}#hd{position:fixed;top:0;left:0;right:0;background:#7a1f1f;font-size:12px;padding:4px 10px;z-index:3}
#side{position:fixed;top:26px;right:0;width:310px;bottom:0;background:#1b1b1b;padding:8px;overflow:auto;z-index:2}button,select{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:5px;padding:4px 8px;margin:2px;cursor:pointer;font:inherit}
.lf{min-width:34px}button.on{background:#555;border-color:#ddd}#cv{position:fixed;top:26px;left:0;width:calc(100vw - 310px);height:calc(100vh - 26px);touch-action:none}@media(pointer:coarse){button,select{padding:9px 12px;margin:3px}.lf{min-width:44px}#fl div,#sets div{padding:6px 4px;font-size:13px}}#box{position:fixed;border:1px dashed #fff;background:rgba(255,255,255,.12);display:none;z-index:2;pointer-events:none}
#msg{font-size:12px;opacity:.85;margin-top:6px}#fl div,#sets div{cursor:pointer;padding:1px 3px;font-size:11px}#fl div:hover,#sets div:hover{background:#333}
@media(max-width:700px){#side{width:100%;top:auto;height:45%}#cv{width:100vw;height:calc(55vh - 26px)}}</style></head><body>
<div id="hd"></div><button id="dsc" style="position:fixed;left:8px;top:32px;z-index:4">점으로 보기</button><canvas id="cv"></canvas><div id="box"></div><div id="side"><div><select id="who"></select> <span id="info" style="font-size:11px"></span></div>
<div style="margin-top:4px"><b>여러 그룹을 한 번에 고르기</b>: <button id="bx">상자 선택 (B)</button> 켜고 화면을 드래그 · Shift = 더하기 · 클릭 = 하나씩</div>
<div style="font-size:11px;opacity:.8">선택한 그룹이 2개 이상이면 잎 번호 버튼이 <b>한 묶음(복엽)으로 등록</b>하고 그 번호를 붙입니다. 묶음 안의 그룹을 클릭하면 묶음 전체가 선택됩니다 (Ctrl+클릭 = 그 그룹만).</div><div id="btns"></div>
<div id="modes" style="margin:4px 0"><button data-m="groups" class="on">그룹</button><button data-m="conf">신뢰도</button><button data-m="flags">플래그</button><button data-m="height">높이</button></div>
<div id="basebox" style="margin:4px 0;padding:4px;border:1px solid #666"><button id="bm">골격 찍기 모드</button><button id="bnew">다음 잎 (선 끝)</button><button id="bundo">마지막 점 취소</button> <span id="bcount"></span><div style="margin-top:4px"><button id="bcalc">골격으로 배정 계산</button><button id="bview">배정 색 보기</button> <span id="binfo" style="font-size:11px"></span></div><div id="skbox" style="margin-top:4px;font-size:12px">기존 선: <select id="skl"></select> <button id="sext">이 잎에 점 이어 찍기</button><button id="sdel">마지막 점 삭제</button><button id="slost">잃음 표시</button> 합칠 잎 <select id="skl2"></select><button id="smerge">합치기</button> 점 <select id="skpi"></select><button id="smove">점 옮기기</button> <button id="ssplit">이 점에서 나누기</button> <button id="sflip">방향 뒤집기</button><button id="sconf">방향 맞음</button> <span id="skinfo"></span><div style="font-size:11px;opacity:.85">주황 공 = 나누기 제안(점 사이 25 mm 초과, 자동 적용 안 함). 선 이름 옆 "방향 제안: 뒤집기"는 기부가 끝 쪽일 것 같다는 자동 제안일 뿐이니, 확인하고 "방향 뒤집기" 또는 "방향 맞음"을 누르세요. "완료"를 누르면 이 골격으로 잎 라벨 계산(T-prox, Kaggle 약 15–25분)이 시작됩니다.</div></div><div id="axbox" style="margin-top:4px;font-size:12px;border-top:1px solid #555;padding-top:4px"><b>주축</b> <button id="axm">주축 그리기</button><button id="axl">곁가지 그리기</button><button id="axu">마지막 점 취소</button> 점 <select id="axpi"></select><button id="axmv">점 옮기기</button> <span id="axinfo"></span><div style="font-size:11px;opacity:.85">"주축 그리기"를 켜고 보이는 줄기를 <b>아래(화분 쪽)에서 위로</b> 5–8번 찍으세요(맨 위 생장점이 보이면 거기까지). 화분 중심이 알려진 개체는 첫 점이 자동이고, 아니면 첫 클릭 = 흙 위 줄기 기부입니다. 곁가지: 첫 클릭은 주축 가까이(붙은 곳), 그다음 곁가지를 따라. 초록 실선 = 찍은 곳(관측), 빨간 점선 = 모델로 채운 곳, 빨간 × = 모델이 추정한 마디(관측 아님).</div></div><div id="skbanner" style="display:none;background:#5a3a00;color:#ffe9a8;padding:4px;margin-top:4px;font-size:12px"></div>
<div style="margin-top:4px"><button id="sdone">완료 → 확인 점 10개</button> <span id="hobox"></span></div>
<div style="font-size:11px;opacity:.85">한 선 = 복엽 하나. 기부(줄기 쪽)부터 끝 쪽으로 찍고, 다음 복엽으로 넘어갈 때 "다음 잎 (선 끝)"을 누르세요. 점 사이가 20 mm를 넘으면 경고만 하고 선은 끊지 않습니다. 기존 선을 고르면 그 잎에 점을 이어 찍거나, 마지막 점을 지우거나, 잃음으로 표시하거나, 다른 선과 합칠 수 있습니다(모두 기록에 추가만 됨, 되돌리기 가능). 다 찍으면 "완료": 무작위 점 10개를 하나씩 분홍 공으로 보여 주니 어느 잎인지 고르세요(모름 가능). 이 확인 점은 씨앗으로 쓰지 않습니다. 큰 빨간 공 = 기부.</div></div>
<div id="legend" style="font-size:11px"></div>
<div><button id="join">번호 없이 묶기</button><button id="unj">묶음 풀기</button><button id="acc">그대로 수락</button><button id="spl">여기서 나눠야 함</button><button id="clr">선택 해제</button></div>
<div id="sel"></div><div id="cnt"></div><div id="msg"></div><div style="font-size:11px;margin-top:6px">묶음 목록 (클릭하면 선택)</div><div id="sets"></div><div style="font-size:11px;margin-top:6px">플래그 목록 (클릭하면 이동)</div><div id="fl"></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>(async function(){const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));const J=async u=>await (await fetch(u)).json();
const cv=document.getElementById('cv'),boxEl=document.getElementById('box');const R=new THREE.WebGLRenderer({canvas:cv,antialias:true});const S=new THREE.Scene();const cam=new THREE.PerspectiveCamera(45,1,1,3000);cam.position.set(0,-260,160);cam.up.set(0,0,1);
const ctl=new THREE.OrbitControls(cam,cv);function size(){const w=cv.clientWidth,h=cv.clientHeight;R.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}window.addEventListener('resize',size);size();
let D=null,pts=null,g=null,col=null,ID=null,P=null,C={},A={},sel=new Set(),MODE='groups',N=0,BOX=false,CEN={},BASEMODE=false,BASES=[],AX=null,AXI=[],DISC=false,LINE=1;const GAP_MM=1e9,GAP_WARN=20,DZ_MM=10;const baseGroup=new THREE.Group();S.add(baseGroup);
const hex=h=>[parseInt(h.slice(1,3),16)/255,parseInt(h.slice(3,5),16)/255,parseInt(h.slice(5,7),16)/255];
const setCol=k=>{const h=((k%997)*0.61803398875)%1;const q=Math.floor(h*6),f=h*6-q,v=0.95,pp=v*0.45,qq=v*(1-0.55*f),tt=v*(1-0.55*(1-f));return [[v,tt,pp],[qq,v,pp],[pp,v,tt],[pp,qq,v],[tt,pp,v],[v,pp,qq]][q%6];};
const sid=u=>(A[u]&&A[u].action!=='unjoin'&&A[u].set_id)||null;
function sets(){const m={};for(const u in A){const s=sid(+u);if(s)(m[s]=m[s]||[]).push(+u);}return m;}
function paint(){if(!D)return;const K=C[MODE];for(let i=0;i<N;i++){let r=K[i*3]/255,gg=K[i*3+1]/255,bb=K[i*3+2]/255;const u=ID[i];
if(MODE==='groups'&&u>=0&&A[u]){if(A[u].leaf&&A[u].leaf!=='모름'){[r,gg,bb]=hex(D.palette[A[u].leaf]);}else if(sid(u)){[r,gg,bb]=setCol(sid(u));}}
if(u>=0&&sel.has(u)){r=1;gg=1;bb=1;}col[i*3]=r;col[i*3+1]=gg;col[i*3+2]=bb;}
markCol();document.getElementById('sel').textContent='선택: '+sel.size+'개 그룹 ('+[...sel].slice(0,12).join(', ')+(sel.size>12?' …':'')+')';
const SS=sets();const nl=Object.values(A).filter(a=>a.leaf&&a.leaf!=='모름').length;document.getElementById('cnt').textContent='잎 번호 붙인 그룹 '+nl+' / '+D.units+' · 묶음 '+Object.keys(SS).length+'개 · 기록된 다른 조치 '+Object.values(A).filter(a=>['accept','split_request'].includes(a.action)).length;
const sl=document.getElementById('sets');sl.innerHTML='';Object.keys(SS).forEach(k=>{const ms=SS[k];const lf=A[ms[0]].leaf;const d=document.createElement('div');d.innerHTML='<span style="color:'+(lf&&lf!=='모름'?D.palette[lf]:'#ccc')+'">■</span> 묶음 '+k.slice(-5)+' · '+ms.length+'개 그룹 · '+(lf?'잎 '+lf:'번호 없음');d.onclick=()=>{sel.clear();ms.forEach(m=>sel.add(m));paint();};sl.appendChild(d);});
const lg=document.getElementById('legend');lg.innerHTML=MODE==='flags'?D.flag_legend.map(f=>'<span style="color:'+f.rgb+'">■</span> '+f.name).join(' · ')+' · <span style="color:#3cc85a">■</span> 플래그 없음 · 진회색 = 엽축/줄기류':(MODE==='conf'?'<span style="color:#28c846">■</span> 높음 · <span style="color:#f0dc28">■</span> 중간 · <span style="color:#e63c3c">■</span> 낮음(플래그 있음)':'');}
async function show(id){D=await J('/api/data?id='+id);N=D.n;P=new Float32Array(b64(D.pos).buffer);buildGeom();ID=new Int32Array(b64(D.ids).buffer);C={groups:b64(D.col_groups),conf:b64(D.col_conf),flags:b64(D.col_flags)};if(D.col_skel)C.skel=b64(D.col_skel);else if(MODE==='skel')MODE='groups';document.getElementById('binfo').textContent=D.skel_info?('가지 '+D.skel_info.branches+' · 배정 '+(100*D.skel_info.assigned_share).toFixed(1)+' % · 애매(흰색) '+(100*D.skel_info.contested_share).toFixed(1)+' %'):'아직 계산 안 함';{let zmin=1e9,zmax=-1e9;for(let i=0;i<N;i++){const z=P[i*3+2];if(z<zmin)zmin=z;if(z>zmax)zmax=z;}const H=new Uint8Array(N*3);for(let i=0;i<N;i++){const t=(P[i*3+2]-zmin)/Math.max(zmax-zmin,1e-6);H[i*3]=Math.round(255*t);H[i*3+1]=Math.round(255*(1-Math.abs(2*t-1)));H[i*3+2]=Math.round(255*(1-t));}C.height=H;}AX=b64(D.axis);AXI=[];for(let i=0;i<N;i++)if(AX[i])AXI.push(i);BASES=(await J('/api/base?id='+id)).map(b=>({n:b.n,p:b.p,surfel:b.surfel,line:b.line||1,legacy:!!b.legacy}));LINE=Math.max(1,...BASES.map(b=>b.line));renderBases();sel.clear();A={};for(const r of await J('/api/assign?id='+id))A[r.unit_id]=r;
CEN={};const cnt={};for(let i=0;i<N;i++){const u=ID[i];if(u<0)continue;const c=CEN[u]||(CEN[u]=[0,0,0]);c[0]+=P[i*3];c[1]+=P[i*3+1];c[2]+=P[i*3+2];cnt[u]=(cnt[u]||0)+1;}for(const u in CEN){CEN[u]=CEN[u].map(x=>x/cnt[u]);}
document.getElementById('hd').textContent=D.note;const s=D.summary;document.getElementById('info').textContent='그룹 '+s.groups+' · 미배정 '+(100*s.unassigned_share).toFixed(1)+' % · 예상 '+s.estimated_user_minutes+' 분 · 투표 사진 '+s.voting_views;
const fl=document.getElementById('fl');fl.innerHTML='';D.flags.forEach(f=>{const d=document.createElement('div');d.textContent=f.flag+(f.group!==null?' · 그룹 '+f.group:'')+' · '+f.n;d.onclick=()=>{ctl.target.set(f.at[0]*1000,f.at[1]*1000,f.at[2]*1000);if(f.group!==null){sel.clear();sel.add(f.group);}paint();};fl.appendChild(d);});paint();}
const ind=await J('/api/individuals');const who=document.getElementById('who');ind.forEach(x=>{const o=document.createElement('option');o.value=x.id;o.textContent=x.id+(x.ready?' ('+x.groups+'그룹)':' — 대기 중');o.disabled=!x.ready;who.appendChild(o);});
const first=ind.find(x=>x.ready);if(first){who.value=first.id;await show(first.id);}else document.getElementById('hd').textContent='아직 완료된 개체가 없습니다';who.onchange=()=>show(who.value);
async function J2(u,b){return await (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json();}
async function write(action,leaf,set_id){const ids=[...sel];for(const s of ids){const cur=A[s]||{};const r=await J2('/api/assign',{id:D.id,unit_id:s,action:action,leaf:(leaf!==undefined?leaf:(cur.leaf||null)),set_id:(set_id!==undefined?set_id:(cur.action==='unjoin'?null:(cur.set_id||null))),with:ids});A[s]=r;}return ids.length;}
async function assign(action,leaf){if(!sel.size){document.getElementById('msg').textContent='먼저 그룹을 클릭하거나 상자 선택';return;}
let set_id;if(action==='leaf'&&sel.size>=2){const ss=new Set([...sel].map(u=>sid(u)));set_id=(ss.size===1&&[...ss][0])?[...ss][0]:Date.now();}
const n=await write(action,leaf,set_id);document.getElementById('msg').textContent=n+'개 그룹 → '+(action==='leaf'?(n>=2?'한 묶음으로 등록, 잎 '+leaf:'잎 '+leaf):action);sel.clear();paint();}
async function join(){if(sel.size<2){document.getElementById('msg').textContent='2개 이상 선택하세요';return;}const n=await write('join',undefined,Date.now());document.getElementById('msg').textContent=n+'개 그룹을 한 묶음으로 등록 (번호는 아직 없음)';sel.clear();paint();}
async function unjoin(){if(!sel.size)return;const n=await write('unjoin',undefined,null);document.getElementById('msg').textContent=n+'개 그룹을 묶음에서 풀었습니다';sel.clear();paint();}
const bt=document.getElementById('btns');for(let k=1;k<=12;k++){const x=document.createElement('button');x.className='lf';x.textContent=k;x.style.borderColor=D?D.palette[k]:'#888';x.onclick=()=>assign('leaf',k);bt.appendChild(x);}const u0=document.createElement('button');u0.textContent='모름';u0.onclick=()=>assign('leaf','모름');bt.appendChild(u0);
document.getElementById('join').onclick=join;document.getElementById('unj').onclick=unjoin;document.getElementById('acc').onclick=()=>assign('accept');document.getElementById('spl').onclick=()=>assign('split_request');document.getElementById('clr').onclick=()=>{sel.clear();paint();};
function setBox(on){BOX=on;ctl.enabled=!on;document.getElementById('bx').classList.toggle('on',on);cv.style.cursor=on?'crosshair':'';}
document.getElementById('bx').onclick=()=>setBox(!BOX);window.addEventListener('keydown',e=>{if((e.key==='b'||e.key==='B')&&!/SELECT|INPUT/.test((e.target||{}).tagName))setBox(!BOX);});
document.querySelectorAll('#modes button').forEach(x=>x.onclick=()=>{MODE=x.dataset.m;document.querySelectorAll('#modes button').forEach(y=>y.classList.toggle('on',y===x));paint();});
function renderBases(){while(baseGroup.children.length)baseGroup.remove(baseGroup.children[0]);if(!D)return;const lines={};
BASES.forEach(b=>{const key=(b.legacy?'old':'')+b.line;(lines[key]=lines[key]||[]).push(Object.assign(new THREE.Vector3(b.p[0]-D.centre_mm[0],b.p[1]-D.centre_mm[1],b.p[2]-D.centre_mm[2]),{legacy:b.legacy}));});
let nseg=0;Object.keys(lines).forEach(k=>{const v=lines[k];const segs=[];v.forEach((q,i)=>{if(i===0||q.distanceTo(v[i-1])>GAP_MM||(q.legacy&&Math.abs(q.z-v[i-1].z)>DZ_MM))segs.push([]);segs[segs.length-1].push(q);});   // old clicks: a vertical step > 10 mm starts a new compound-leaf axis
segs.forEach(sg=>{nseg++;sg.forEach((q,i)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(i===0?2.8:1.4,12,12),new THREE.MeshBasicMaterial({color:i===0?0xff2020:0xffd400}));m.position.copy(q);baseGroup.add(m);});
if(sg.length>1)baseGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(sg),new THREE.LineBasicMaterial({color:0xffd400})));});});
document.getElementById('bcount').textContent='가지 '+nseg+'개 · 점 '+BASES.length+'개 · 지금 선 '+LINE;}
async function addBase(h){const idx=hid(h),snap=0,far=false;                      // user 2026-11-01: the clicked surfel itself, anywhere (no snap to the axis class)
const pt=[P[idx*3]+D.centre_mm[0],P[idx*3+1]+D.centre_mm[1],P[idx*3+2]+D.centre_mm[2]];if(!STARTED){STARTED=true;J2('/api/skel_edit',{id:D.id,action:'start'});}
if(await axisClick(pt,idx))return;
if(MOVE){const r=await skEdit({action:'move_point',leaf_uid:MOVE.uid,index:MOVE.index,point_mm:pt,surfel:idx});if(r)document.getElementById('msg').textContent=MOVE.uid+' 점 '+(MOVE.index+1)+' 옮김';MOVE=null;document.getElementById('smove').classList.remove('on');return;}
if(EXT){const r=await skEdit({action:'extend',leaf_uid:EXT,point_mm:pt,surfel:idx});if(r)document.getElementById('msg').textContent=EXT+'에 점을 이었습니다';return;}
const prev=BASES.filter(b=>b.line===LINE).slice(-1)[0];const gap=prev?Math.hypot(pt[0]-prev.p[0],pt[1]-prev.p[1],pt[2]-prev.p[2]):0;
const r=await J2('/api/base',{id:D.id,action:'add',surfel:idx,point_mm:pt,snap_mm:+snap.toFixed(2),snap_far:far,line:LINE});BASES.push({n:r.n,p:pt,surfel:idx,line:LINE});renderBases();loadSkel();
document.getElementById('msg').textContent=(gap>GAP_WARN?'점 사이가 '+gap.toFixed(0)+' mm 넘습니다 — 같은 잎이 맞나요? (선은 끊지 않음) · ':'')+'선 '+LINE+'에 점 저장';}
async function undoBase(){if(!BASES.length)return;const b=BASES.pop();await J2('/api/base',{id:D.id,action:'undo',surfel:b.surfel});renderBases();document.getElementById('msg').textContent='마지막 점을 취소했습니다';}
document.getElementById('bm').onclick=()=>{BASEMODE=!BASEMODE;document.getElementById('bm').classList.toggle('on',BASEMODE);if(BASEMODE)setBox(false);paint();};document.getElementById('bundo').onclick=undoBase;
document.getElementById('bcalc').onclick=async()=>{document.getElementById('msg').textContent='배정 계산 중…';const r=await J2('/api/skeleton',{id:D.id});if(r.error){document.getElementById('msg').textContent='찍은 골격이 없습니다';return;}const cur=D.id;await show(cur);MODE='skel';paint();document.getElementById('msg').textContent='배정 완료: 가지 '+r.branches+'개, 흰색 = 두 가지 사이에서 애매';};
document.getElementById('bview').onclick=()=>{if(!C.skel){document.getElementById('msg').textContent='먼저 "골격으로 배정 계산"';return;}MODE=(MODE==='skel'?'groups':'skel');document.getElementById('bview').textContent=MODE==='skel'?'원래 색 보기':'배정 색 보기';paint();};document.getElementById('bnew').onclick=async()=>{const used=BASES.some(b=>b.line===LINE);if(used){await J2('/api/skel_edit',{id:D.id,action:'end_line',line:LINE});LINE=Math.max(...BASES.map(b=>b.line))+1;}EXT=null;document.getElementById('sext').classList.remove('on');renderBases();loadSkel();document.getElementById('msg').textContent='선 '+LINE+' 시작: 다음 잎의 기부부터 찍으세요';};
// ---- main / lateral axis (skeleton_v1.1, Amendment A6): user stem clicks, bottom -> top; the fit, nodes and model-completed parts come back from the server
let AXMODE=null,AXMOVE=null;const axGroup=new THREE.Group();S.add(axGroup);
function v3(q){return new THREE.Vector3(q[0]-D.centre_mm[0],q[1]-D.centre_mm[1],q[2]-D.centre_mm[2]);}
function curAxis(){const ax=(SKEL&&SKEL.axes)||[];return ax.find(x=>x.uid===(AXMODE&&AXMODE.uid))||ax.find(x=>x.type==='main');}
function renderAxes(){while(axGroup.children.length)axGroup.remove(axGroup.children[0]);if(!D)return;const ax=(SKEL&&SKEL.axes)||[];const main=ax.find(a=>a.type==='main');
ax.forEach(a=>{a.points.forEach(p=>{const m=new THREE.Mesh(new THREE.SphereGeometry(p.source==='user_click'?1.8:1.2,10,10),new THREE.MeshBasicMaterial({color:p.source==='user_click'?0x3399ff:0x99ccff}));m.position.copy(v3(p.xyz_mm));axGroup.add(m);});
 if(a.curve_mm&&a.curve_mm.length>1){const sC=[0];for(let i=1;i<a.curve_mm.length;i++){const q=a.curve_mm[i],o=a.curve_mm[i-1];sC.push(sC[i-1]+Math.hypot(q[0]-o[0],q[1]-o[1],q[2]-o[2]));}
  a.segments.forEach(sg=>{const v=[];a.curve_mm.forEach((q,i)=>{if(sC[i]>=sg.s0_mm-1.5&&sC[i]<=sg.s1_mm+1.5)v.push(v3(q));});if(v.length<2)return;const obs=sg.provenance==='observed';
   const L=new THREE.Line(new THREE.BufferGeometry().setFromPoints(v),obs?new THREE.LineBasicMaterial({color:0x22c55e}):new THREE.LineDashedMaterial({color:0xef4444,dashSize:2,gapSize:1.5}));if(!obs)L.computeLineDistances();axGroup.add(L);});}
 a.nodes.forEach(n=>{if(n.kind==='fspm_node_hypothesis'){const g=new THREE.Group();[[1,1],[1,-1]].forEach(([x,y])=>{g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-2*x,-2*y,0),new THREE.Vector3(2*x,2*y,0)]),new THREE.LineBasicMaterial({color:0xef4444})));});g.position.copy(v3(n.xyz_mm));axGroup.add(g);}
  else{const m=new THREE.Mesh(new THREE.SphereGeometry(1.5,10,10),new THREE.MeshBasicMaterial({color:0xffffff}));m.position.copy(v3(n.xyz_mm));axGroup.add(m);}});});
const cur=curAxis();document.getElementById('axpi').innerHTML=cur?cur.points.map((p,i)=>p.source==='user_click'?'<option value="'+i+'">'+(i+1)+'</option>':'').join(''):'';
const sm=SKEL&&SKEL.axis_summary;const ms=main&&sm?sm[main.uid]:null;
document.getElementById('axinfo').textContent=main?('주축 점 '+main.points.filter(p=>p.source==='user_click').length+'개'+(ms?' · 길이 '+ms.length_mm+' mm 중 관측 '+(100*(ms.observed_share||0)).toFixed(0)+' % · 마디 관측 '+ms.nodes_observed+' / 추정 '+ms.nodes_hypothesised:'')+(sm&&sm.unattached&&sm.unattached.length?' · 축에 안 붙은 잎 '+sm.unattached.length+'개':'')+(main.flags.length?' · '+main.flags.join(', '):'')):'주축 없음';}
async function axisClick(pt,idx){
 if(AXMOVE){const r=await skEdit({action:'axis_move_point',axis_uid:AXMOVE.uid,index:AXMOVE.index,point_mm:pt,surfel:idx});AXMOVE=null;document.getElementById('axmv').classList.remove('on');if(r)document.getElementById('msg').textContent='주축 점을 옮겼습니다';return true;}
 if(!AXMODE)return false;
 if(!AXMODE.uid){const main=((SKEL&&SKEL.axes)||[]).find(a=>a.type==='main');
  if(AXMODE.type==='main'&&main){AXMODE.uid=main.uid;}
  else{const r=await skEdit({action:'axis_start',type:AXMODE.type,parent_axis_uid:(AXMODE.type==='lateral'&&main)?main.uid:null});if(!r)return true;const ax=((SKEL&&SKEL.axes)||[]).filter(a=>a.type===AXMODE.type);AXMODE.uid=ax.length?ax[ax.length-1].uid:null;}}
 const r=await skEdit({action:'axis_add',axis_uid:AXMODE.uid,point_mm:pt,surfel:idx,source:'user_click'});if(r)document.getElementById('msg').textContent=(AXMODE.type==='main'?'주축':'곁가지')+' 점 저장 — 계속 위로 찍으세요';return true;}
function setAx(type){const on=!(AXMODE&&AXMODE.type===type);AXMODE=on?{type:type,uid:null}:null;EXT=null;MOVE=null;document.getElementById('axm').classList.toggle('on',on&&type==='main');document.getElementById('axl').classList.toggle('on',on&&type==='lateral');
 document.getElementById('msg').textContent=AXMODE?(type==='main'?'주축: 보이는 줄기를 아래에서 위로 찍으세요':'곁가지: 먼저 주축 가까이(붙은 곳), 다음 곁가지를 따라'):'주축 그리기 끝';}
document.getElementById('axm').onclick=()=>setAx('main');document.getElementById('axl').onclick=()=>setAx('lateral');
document.getElementById('axu').onclick=async()=>{const a=curAxis();if(!a)return;const ui=a.points.map((p,i)=>p.source==='user_click'?i:-1).filter(i=>i>=0);if(!ui.length)return;if(await skEdit({action:'axis_delete_point',axis_uid:a.uid,index:ui[ui.length-1]}))document.getElementById('msg').textContent='주축 마지막 점 취소';};
document.getElementById('axmv').onclick=()=>{const a=curAxis();const i=parseInt(document.getElementById('axpi').value);if(!a||isNaN(i)){document.getElementById('msg').textContent='옮길 주축 점을 고르세요';return;}AXMOVE=(AXMOVE&&AXMOVE.index===i)?null:{uid:a.uid,index:i};document.getElementById('axmv').classList.toggle('on',!!AXMOVE);document.getElementById('msg').textContent=AXMOVE?'주축 점 '+(i+1)+': 새 자리를 클릭하세요':'점 옮기기 해제';};
// ---- coloured skeleton lines (2026-11-02, colour only since the user asked for no leaf numbers): the pink-point answer swatches use the same colour, so a pick is matched to a line by colour
const skLineGroup=new THREE.Group();S.add(skLineGroup);
function skColour(uid){const k=parseInt(uid.split('-L')[1]);const h=((k*0.61803398875)%1),s=0.75,v=0.95,i=Math.floor(h*6),f=h*6-i,p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);const rgb=[[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6];return new THREE.Color(rgb[0],rgb[1],rgb[2]);}    // same hsv rule as the assignment view, so a swatch equals the leaf colour on screen
function numSprite(txt,col){const c=document.createElement('canvas');c.width=64;c.height=64;const g=c.getContext('2d');g.fillStyle='rgba(0,0,0,0.65)';g.beginPath();g.arc(32,32,30,0,7);g.fill();g.fillStyle='#'+col.getHexString();g.beginPath();g.arc(32,32,17,0,7);g.fill();      // colour only, no leaf number
 const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));sp.scale.set(7,7,1);return sp;}
function renderSkelLines(){while(skLineGroup.children.length)skLineGroup.remove(skLineGroup.children[0]);if(!D||!SKEL||!SKEL.leaves)return;
 SKEL.leaves.forEach(l=>{if(l.status!=='active'||!l.poly_mm.length)return;const col=skColour(l.uid);const v=l.poly_mm.map(q=>v3(q));
  if(v.length>1)skLineGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(v),new THREE.LineBasicMaterial({color:col,linewidth:2})));
  const b=new THREE.Mesh(new THREE.SphereGeometry(2.0,10,10),new THREE.MeshBasicMaterial({color:col}));b.position.copy(v[0]);skLineGroup.add(b);
  const sp=numSprite(l.uid.split('-L')[1],col);sp.position.copy(v[0]).add(new THREE.Vector3(0,0,5));skLineGroup.add(sp);});}
// ---- dz10 review aids (display only): suggested cuts, direction proposal; split / flip / confirm are appended records
const cutGroup=new THREE.Group();S.add(cutGroup);
function renderCuts(){while(cutGroup.children.length)cutGroup.remove(cutGroup.children[0]);if(!D||!SKEL||!SKEL.leaves)return;SKEL.leaves.forEach(l=>{if(l.status!=='active'||!l.cuts)return;l.cuts.forEach(i=>{const a=l.poly_mm[i-1],b=l.poly_mm[i];if(!a||!b)return;const m=new THREE.Mesh(new THREE.SphereGeometry(2.2,12,12),new THREE.MeshBasicMaterial({color:0xff8c00}));m.position.copy(v3([(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2]));cutGroup.add(m);});});}
document.getElementById('ssplit').onclick=async()=>{const u=document.getElementById('skl').value,i=parseInt(document.getElementById('skpi').value);if(!u||isNaN(i)||i<1){document.getElementById('msg').textContent='나눌 선과 점(2번째 이후)을 고르세요: 고른 점부터 새 잎이 됩니다';return;}if(await skEdit({action:'split',leaf_uid:u,index:i}))document.getElementById('msg').textContent=u+' 를 점 '+(i+1)+'에서 나눴습니다 (뒤쪽 = 새 잎)';};
document.getElementById('sflip').onclick=async()=>{const u=document.getElementById('skl').value;if(u&&await skEdit({action:'flip_direction',leaf_uid:u}))document.getElementById('msg').textContent=u+' 방향을 뒤집었습니다 (기부 ↔ 끝)';};
document.getElementById('sconf').onclick=async()=>{const u=document.getElementById('skl').value;if(u&&await skEdit({action:'confirm_direction',leaf_uid:u}))document.getElementById('msg').textContent=u+' 방향 확인';};
// ---- skeleton editor (skeleton_v1 layer, 2026-11-01): existing lines with leaf_uid, extend / delete last point / mark lost / merge, then held-out picks. Every action = one appended log record.
let SKEL=null,EXT=null,MOVE=null,HO=[],HOI=-1,STARTED=false;const hoGroup=new THREE.Group();S.add(hoGroup);
async function loadSkel(){if(!D)return;try{SKEL=await (await fetch('/api/skel?id='+D.id)).json();}catch(e){return;}const s=document.getElementById('skl'),s2=document.getElementById('skl2');if(!SKEL||!SKEL.leaves){s.innerHTML='';s2.innerHTML='';return;}
const opts=SKEL.leaves.map(l=>'<option value="'+l.uid+'">'+l.uid+' ('+l.status+', 점 '+l.n+(l.flags.length?', '+l.flags.join('/'):'')+(l.dir==='observed'?', 방향 확인됨':(l.dir_auto==='flip'?', 방향 제안: 뒤집기':''))+(l.cuts&&l.cuts.length?', 나누기 제안 '+l.cuts.length:'')+')</option>').join('');const keep=s.value;s.innerHTML=opts;s2.innerHTML=opts;if(keep)s.value=keep;
document.getElementById('skinfo').textContent='잎 '+SKEL.leaves.filter(l=>l.status==='active').length+'개 · 확인 점 '+SKEL.heldout+'개'+(SKEL.minutes&&SKEL.minutes.minutes!=null?' · 찍은 시간 '+SKEL.minutes.minutes+'분':'');fillPts();renderAxes();renderCuts();renderSkelLines();
const nleg=SKEL.leaves.filter(l=>l.status==='active'&&(l.flags.indexOf('legacy_split_20mm')>=0||l.flags.indexOf('legacy_split_dz10mm')>=0)).length,bn=document.getElementById('skbanner');bn.style.display=nleg?'block':'none';if(nleg)bn.textContent='어제 찍은 점을 세로 변화 10 mm 규칙으로 나눈 선이 '+nleg+'개입니다 — 같은 잎이면 합치기, 아니면 그대로 두고 \'완료\'로 확인하세요.';}
function fillPts(){const s=document.getElementById('skl'),p=document.getElementById('skpi');const l=SKEL&&SKEL.leaves?SKEL.leaves.find(x=>x.uid===s.value):null;p.innerHTML=l?l.poly_mm.map((q,i)=>'<option value="'+i+'">'+(i+1)+(i===0?' (기부)':i===l.poly_mm.length-1?' (끝)':'')+'</option>').join(''):'';if(l)p.value=String(l.poly_mm.length-1);}
async function skEdit(b){const r=await J2('/api/skel_edit',Object.assign({id:D.id},b));if(r.error){document.getElementById('msg').textContent=r.error;return null;}await loadSkel();return r;}
document.getElementById('sext').onclick=()=>{const u=document.getElementById('skl').value;if(!u)return;EXT=(EXT===u?null:u);document.getElementById('sext').classList.toggle('on',!!EXT);document.getElementById('msg').textContent=EXT?('다음 클릭은 '+EXT+'에 이어 붙습니다'):'이어 찍기 해제';};
document.getElementById('sdel').onclick=async()=>{const u=document.getElementById('skl').value;if(u&&await skEdit({action:'delete_point',leaf_uid:u,index:-1}))document.getElementById('msg').textContent=u+' 마지막 점 삭제';};
document.getElementById('slost').onclick=async()=>{const u=document.getElementById('skl').value;if(u&&await skEdit({action:'mark_lost',leaf_uid:u}))document.getElementById('msg').textContent=u+' 잃음 표시';};
document.getElementById('skl').onchange=fillPts;
document.getElementById('smove').onclick=()=>{const u=document.getElementById('skl').value,i=parseInt(document.getElementById('skpi').value);if(!u||isNaN(i)){document.getElementById('msg').textContent='옮길 선과 점을 고르세요';return;}MOVE=(MOVE&&MOVE.uid===u&&MOVE.index===i)?null:{uid:u,index:i};EXT=null;document.getElementById('sext').classList.remove('on');document.getElementById('smove').classList.toggle('on',!!MOVE);document.getElementById('msg').textContent=MOVE?(u+' 점 '+(i+1)+': 새 자리를 클릭하세요'):'점 옮기기 해제';};
document.getElementById('smerge').onclick=async()=>{const k=document.getElementById('skl').value,a=document.getElementById('skl2').value;if(!k||!a||k===a){document.getElementById('msg').textContent='서로 다른 두 선을 고르세요';return;}if(await skEdit({action:'merge',keep_uid:k,absorb_uid:a}))document.getElementById('msg').textContent=a+' → '+k+' 합침';};
function showHO(){while(hoGroup.children.length)hoGroup.remove(hoGroup.children[0]);const box=document.getElementById('hobox');if(HOI<0||HOI>=HO.length){box.innerHTML=(HO.length&&HOI>=HO.length)?'확인 점 끝. 고맙습니다.':'';return;}const h=HO[HOI];
const m=new THREE.Mesh(new THREE.SphereGeometry(3.8,14,14),new THREE.MeshBasicMaterial({color:0xff69b4,depthTest:false}));m.renderOrder=999;m.position.set(h.point_mm[0]-D.centre_mm[0],h.point_mm[1]-D.centre_mm[1],h.point_mm[2]-D.centre_mm[2]);hoGroup.add(m);const ring=new THREE.Mesh(new THREE.SphereGeometry(5.6,14,14),new THREE.MeshBasicMaterial({color:0xffffff,wireframe:true,depthTest:false}));ring.renderOrder=998;ring.position.copy(m.position);hoGroup.add(ring);
const act=(SKEL&&SKEL.leaves?SKEL.leaves.filter(l=>l.status==='active'):[]);box.innerHTML='<b>분홍점 '+(HOI+1)+'/'+HO.length+'</b> 분홍 점은 어느 잎? 그 잎 선과 <b>같은 색</b>을 누르세요 (번호 없음) <div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0">'+act.map(l=>'<button data-u="'+l.uid+'" aria-label="잎 색" style="width:44px;height:34px;border-radius:6px;border:2px solid #fff;background:#'+skColour(l.uid).getHexString()+'"></button>').join('')+'<button data-u="" style="height:34px">모름</button></div>'+(C.skel?'':'<div style="color:#ffb74d;font-size:12px">먼저 "골격으로 배정 계산"을 누르면 잎이 같은 색으로 칠해집니다.</div>');if(C.skel&&MODE!=='skel'){MODE='skel';document.getElementById('bview').textContent='원래 색 보기';paint();}
box.querySelectorAll('button').forEach(b=>{b.onclick=async()=>{const r=await skEdit({action:'heldout_pick',surfel:h.surfel,leaf_uid:b.dataset.u||null,point_mm:h.point_mm});HOI++;showHO();if(r&&r.grade){document.getElementById('hobox').textContent='채점: '+r.grade;document.getElementById('msg').textContent='채점: '+r.grade;}};});}
document.getElementById('sdone').onclick=async()=>{const rd=await skEdit({action:'done'});if(rd&&rd.grade)document.getElementById('skinfo').textContent=rd.grade;HO=(await (await fetch('/api/heldout?id='+D.id)).json()).filter(h=>!h.done);HOI=HO.length?0:-1;if(!HO.length)document.getElementById('msg').textContent='남은 확인 점이 없습니다';showHO();};
document.getElementById('who').addEventListener('change',()=>{SKEL=null;EXT=null;AXMODE=null;AXMOVE=null;STARTED=false;HO=[];HOI=-1;showHO();setTimeout(loadSkel,1500);});
function hid(h){return h.instanceId!==undefined?h.instanceId:h.index;}
function markCol(){if(!pts)return;if(pts.isInstancedMesh)pts.instanceColor.needsUpdate=true;else pts.geometry.attributes.color.needsUpdate=true;}
function buildGeom(){if(pts){S.remove(pts);pts.geometry.dispose();pts=null;}
if(DISC&&D.disc){const tu=new Int16Array(b64(D.disc.tu).buffer),tv=new Int16Array(b64(D.disc.tv).buffer),sc=new Float32Array(b64(D.disc.s).buffer);const m=new THREE.InstancedMesh(new THREE.CircleGeometry(1,8),new THREE.MeshBasicMaterial({side:THREE.DoubleSide}),N);const M4=new THREE.Matrix4();
for(let i=0;i<N;i++){const ax=tu[i*3]/32767,ay=tu[i*3+1]/32767,az=tu[i*3+2]/32767,bx=tv[i*3]/32767,by=tv[i*3+1]/32767,bz=tv[i*3+2]/32767;const nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;const r0=1.5*sc[i*2],r1=1.5*sc[i*2+1];
M4.set(ax*r0,bx*r1,nx,P[i*3],ay*r0,by*r1,ny,P[i*3+1],az*r0,bz*r1,nz,P[i*3+2],0,0,0,1);m.setMatrixAt(i,M4);}
m.instanceColor=new THREE.InstancedBufferAttribute(new Float32Array(N*3),3);col=m.instanceColor.array;pts=m;}
else{const g2=new THREE.BufferGeometry();g2.setAttribute('position',new THREE.BufferAttribute(P,3));col=new Float32Array(N*3);g2.setAttribute('color',new THREE.BufferAttribute(col,3));pts=new THREE.Points(g2,new THREE.PointsMaterial({size:1.6,vertexColors:true}));}
S.add(pts);document.getElementById('dsc').textContent=(DISC&&D.disc)?'점으로 보기':'원판으로 보기';}
document.getElementById('dsc').onclick=()=>{DISC=!DISC;buildGeom();paint();};
const rc=new THREE.Raycaster();rc.params.Points.threshold=1.5;let down=null;
cv.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];if(BOX){boxEl.style.display='block';boxEl.style.left=e.clientX+'px';boxEl.style.top=e.clientY+'px';boxEl.style.width='0px';boxEl.style.height='0px';}});
cv.addEventListener('pointermove',e=>{if(BOX&&down){const x0=Math.min(down[0],e.clientX),y0=Math.min(down[1],e.clientY);boxEl.style.left=x0+'px';boxEl.style.top=y0+'px';boxEl.style.width=Math.abs(e.clientX-down[0])+'px';boxEl.style.height=Math.abs(e.clientY-down[1])+'px';}});
function addUnit(u){const s=sid(u);if(s){for(const k in A)if(sid(+k)===s)sel.add(+k);}else sel.add(u);}
cv.addEventListener('pointerup',e=>{if(!down||!pts)return;const r=cv.getBoundingClientRect();const dx=Math.hypot(e.clientX-down[0],e.clientY-down[1]);
if(BOX&&dx>4){boxEl.style.display='none';const x0=Math.min(down[0],e.clientX),x1=Math.max(down[0],e.clientX),y0=Math.min(down[1],e.clientY),y1=Math.max(down[1],e.clientY);if(!e.shiftKey)sel.clear();cam.updateMatrixWorld();const v=new THREE.Vector3();let n=0;
for(const u in CEN){v.set(CEN[u][0],CEN[u][1],CEN[u][2]).project(cam);if(v.z>1)continue;const px=(v.x+1)/2*r.width+r.left,py=(1-v.y)/2*r.height+r.top;if(px>=x0&&px<=x1&&py>=y0&&py<=y1){addUnit(+u);n++;}}
document.getElementById('msg').textContent='상자 안 그룹 '+n+'개 선택 (뒤쪽 면 그룹도 포함 — 필요하면 돌려서 다시)';down=null;paint();return;}
boxEl.style.display='none';if(dx>4){down=null;return;}down=null;const m=new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
rc.setFromCamera(m,cam);const h=rc.intersectObject(pts)[0];if(!h)return;if(BASEMODE){addBase(h);return;}const id=ID[hid(h)];if(id<0){document.getElementById('msg').textContent='미배정 점 또는 엽축 (그룹 아님)';return;}
if(!e.shiftKey&&!e.ctrlKey)sel.clear();if(sel.has(id)&&(e.shiftKey||e.ctrlKey))sel.delete(id);else if(e.ctrlKey)sel.add(id);else addUnit(id);
document.getElementById('msg').textContent='그룹 '+id+(A[id]&&A[id].leaf?' (현재 잎 '+A[id].leaf+')':'')+(sid(id)?' · 묶음 전체 선택':'');paint();});
if(new URLSearchParams(location.search).get('mode')==='base'){const keep=document.getElementById('who').parentElement;document.querySelectorAll('#side > *').forEach(el=>{if(el!==keep&&el.id!=='basebox'&&el.id!=='msg')el.style.display='none';});document.getElementById('bm').style.display='none';['skbox','axbox','skbanner'].forEach(x=>{const e=document.getElementById(x);if(e)e.style.setProperty('display','none','important');});BASEMODE=true;document.title='잎 골격 찍기';const _sk=setInterval(()=>{if(D){clearInterval(_sk);loadSkel();}},700);document.getElementById('basebox').insertAdjacentHTML('afterbegin','<div style="font-size:15px;font-weight:600;margin-bottom:4px">잎 골격 찍기</div><div style="font-size:13px;margin-bottom:6px">잎마다: <b>기부(줄기 쪽) → 끝 쪽</b>으로 2–5번 찍고 <b>"다음 잎"</b>. 틀리면 "마지막 점 취소". 주축(초록)은 그대로 둡니다.</div>');paint();}
(function loop(){ctl.update();R.render(S,cam);requestAnimationFrame(loop);})();window.__p={ho:(a)=>{HO=a;HOI=a.length?0:-1;showHO();},sel:()=>[...sel],A:()=>A,pick:(id)=>{sel.add(id);paint();},assign,join,unjoin,show,box:setBox};})();</script></body></html>"""


def skel_rebuild(i, save=False):
    """skeleton_v1 of one individual re-derived from the two append-only logs (skeleton_store.rebuild); saved only after an edit."""
    import numpy as np
    from tomato3d.organs import skeleton_propagate as SP
    from tomato3d.organs import skeleton_store as SS
    z = np.load(OUT / i / "merge" / "groups.npz")
    return SS.rebuild(SP.ROOT, STORE_BASE, i, SP.capture_date(i), SP.reconstruction_id(i), z["xyz"].astype(float), save=save), z


def skel_payload(i, save=False):
    import numpy as np
    (sk, flags, skipped), _ = skel_rebuild(i, save)
    return dict(date=sk["capture_date"], heldout=len(sk["heldout"]), minutes=sk.get("minutes"), skipped=len(skipped), flags=[list(f) for f in flags],
                leaves=[dict(uid=l["leaf_uid"], status=l["status"], n=l["n_clicks"], flags=l["flags"], base_mm=(np.asarray(l["base_xyz"]) * 1000).round(1).tolist(),
                             poly_mm=(np.asarray(l["polyline_xyz"]) * 1000).round(1).tolist(), dir=l.get("attrs", {}).get("direction"), dir_auto=l.get("attrs", {}).get("direction_auto"),
                             cuts=l.get("attrs", {}).get("suggested_cuts", [])) for l in sk["leaves"]], skelver=sk.get("skelver"),
                axes=[dict(uid=a["axis_uid"], type=a["type"], status=a["status"], flags=a.get("flags", []), points=[dict(xyz_mm=(np.asarray(p["xyz"]) * 1000).round(1).tolist(), source=p["source"]) for p in a["points"]],
                           curve_mm=(a["fit"]["curve_mm"][::2] + a["fit"]["curve_mm"][-1:]) if a.get("fit") else None, segments=a.get("segments", []), nodes=a.get("nodes", [])) for a in sk.get("axes", []) if a["status"] == "active"],
                axis_summary=sk.get("axis_summary"))


SKEL_ACTIONS = ("extend", "move_point", "delete_point", "mark_lost", "merge", "heldout_pick", "end_line", "start", "done", "axis_start", "axis_add", "axis_move_point", "axis_delete_point", "split", "flip_direction", "confirm_direction")


DZ10_TRIGGER_FROM = "2026-11-01T19:40:00"


def spawn_labels(i, grade_only=False):
    """Detached label / grade job (contract build/p1_dz10_labels_20261101_01); its own log is job_<id>.log there."""
    import subprocess
    import sys
    args = [sys.executable, "-X", "utf8", "-m", "tomato3d.organs.skeleton_dz10_labels", "--id", i] + (["--grade"] if grade_only else [])
    out = ROOT / "build/p1_dz10_labels_20261101_01"; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"spawn_{i}.log", "a", encoding="utf-8") as f:
        subprocess.Popen(args, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))     # hidden console: the Kaggle CLI child died with 0xC000013A under DETACHED_PROCESS


class H(BaseHTTPRequestHandler):
    def _allowed(self):
        """Loopback always; with --lan only private-network clients that carry the token (query ?k= or the cookie it sets)."""
        ip = ipaddress.ip_address(self.client_address[0])
        if ip.is_loopback:
            return True
        tok = STATE.get("token")
        if not tok or not ip.is_private:
            return False
        q = parse_qs(urlparse(self.path).query).get("k", [""])[0]; ck = dict(x.strip().split("=", 1) for x in self.headers.get("Cookie", "").split(";") if "=" in x).get("k", "")
        return secrets.compare_digest(q or ck, tok)

    def _j(self, o, code=200):
        b = json.dumps(o, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if not self._allowed():
            return self._j(dict(error="forbidden"), 403)
        if u.path in ("/", "/index.html"):
            b = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            if STATE.get("token"):
                self.send_header("Set-Cookie", f"k={STATE['token']}; Path=/; SameSite=Strict")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if u.path == "/api/individuals":
            return self._j(individuals())
        if u.path == "/api/data":
            i = q.get("id", [""])[0]
            if i not in ORDER or not (OUT / i / "merge" / "groups.npz").exists():
                return self._j(dict(error="not ready"), 404)
            if STATE.get("id") != i:
                STATE["id"], STATE["data"] = i, load(i)
            return self._j(STATE["data"])
        if u.path in ("/api/skel", "/api/heldout"):
            i = q.get("id", [""])[0]
            if i not in ORDER or not (OUT / i / "merge" / "groups.npz").exists():
                return self._j(dict(error="not ready"), 404)
            if u.path == "/api/skel":
                return self._j(skel_payload(i))
            from tomato3d.organs import skeleton_store as SS
            (sk, _, _), z = skel_rebuild(i); return self._j(SS.heldout_candidates(sk, z["xyz"].astype(float)))
        if u.path == "/api/base":
            return self._j([dict(n=k + 1, p=r["point_mm"], surfel=r["surfel"], line=int(r.get("line") or 1), legacy=("line" not in r), snap_mm=r.get("snap_mm"), snap_far=r.get("snap_far")) for k, r in enumerate(base_state(q.get("id", [""])[0]))])
        if u.path == "/api/assign":
            i = q.get("id", [""])[0]; recs = [json.loads(l) for l in STORE.read_text(encoding="utf-8").splitlines() if l.strip()] if STORE.exists() else []
            last = {}
            for r in recs:
                if r.get("individual") == i:
                    last[r["unit_id"]] = r
            return self._j(list(last.values()))
        self._j(dict(error="not found"), 404)

    def do_POST(self):
        if not self._allowed():
            return self._j(dict(error="forbidden"), 403)
        if self.path == "/api/skel_edit":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"); i = body.get("id")
            if i not in ORDER or body.get("action") not in SKEL_ACTIONS:
                return self._j(dict(error="bad request"), 400)
            from tomato3d.organs import skeleton_propagate as SP
            from tomato3d.organs import skeleton_store as SS
            rec = SS.append_edit(SP.ROOT, i, SP.capture_date(i), {k: v for k, v in body.items() if k != "id"}); skel = skel_payload(i, save=True); grade = None
            if body["action"] == "done" and rec.get("at", "") >= DZ10_TRIGGER_FROM:                  # dz10 Part A: "완료" -> distance + T-prox labels for this skeleton version (detached job)
                spawn_labels(i); grade = "잎 라벨 계산을 시작했습니다 (T-prox는 Kaggle, 약 15–25분). 그다음 확인 점 10개를 찍으면 자동 채점합니다."
            if body["action"] == "heldout_pick" and skel["heldout"] >= SS.HELDOUT["n"]:
                spawn_labels(i, grade_only=True)
            if body["action"] == "heldout_pick" and skel["heldout"] >= SS.HELDOUT["n"]:              # 10 picks done -> distance route + held-out grade (A4), told to the user in Korean
                try:
                    summ = SP.step_distance(i)[0]; grade = summ.get("grade_ko")
                except Exception as e:
                    grade = f"채점 실패: {type(e).__name__}"
            return self._j(dict(ok=True, rec=rec, skel=skel, grade=grade))
        if self.path == "/api/skeleton":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"); i = body.get("id")
            if i not in ORDER:
                return self._j(dict(error="bad request"), 400)
            from tomato3d.organs import skeleton_assign as SK
            summ = SK.assign(i, STORE_BASE); STATE["id"] = None                      # drop the cached page data so the new colours load
            return self._j(summ or dict(error="no skeleton clicks"))
        if self.path not in ("/api/assign", "/api/base"):
            return self._j(dict(error="not found"), 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/base":
            if body.get("id") not in ORDER or body.get("action") not in ("add", "undo"):
                return self._j(dict(error="bad request"), 400)
            alive = base_state(body["id"]); rec = dict(individual=body["id"], action=body["action"], n=(len(alive) + 1 if body["action"] == "add" else len(alive)), surfel=int(body.get("surfel", -1)), point_mm=body.get("point_mm"), line=int(body.get("line") or 1),
                                                      snap_mm=body.get("snap_mm"), snap_far=body.get("snap_far"), provenance="user_base_click", at=time.strftime("%Y-%m-%dT%H:%M:%S"), t=time.time())
            with open(STORE_BASE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return self._j(rec)
        if body.get("id") not in ORDER:
            return self._j(dict(error="unknown individual"), 400)
        rec = dict(individual=body["id"], unit="group", unit_id=int(body["unit_id"]), action=body.get("action", "leaf"), leaf=body.get("leaf"), set_id=body.get("set_id"), together_with=body.get("with"),
                   provenance="user_3d_group_click", at=time.strftime("%Y-%m-%dT%H:%M:%S"), t=time.time())
        with open(STORE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._j(rec)

    def log_message(self, *a):
        pass


def main(argv=None):
    global STORE, STORE_BASE
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8835); ap.add_argument("--store", type=Path, default=None); ap.add_argument("--store-base", type=Path, default=None); ap.add_argument("--lan", action="store_true"); ap.add_argument("--token", default=None); a = ap.parse_args(argv)
    if a.store_base:
        STORE_BASE = a.store_base                         # tests only
    if a.store:
        STORE = a.store                                  # tests only: never the user's record file
    if a.lan:
        STATE["token"] = a.token or secrets.token_urlsafe(9)                # --token keeps a tablet link valid across restarts
    srv = ThreadingHTTPServer(("0.0.0.0" if a.lan else "127.0.0.1", a.port), H); print(f"serving http://127.0.0.1:{a.port}/" + (f" and on the local network, token {STATE['token']}" if a.lan else ""), flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()
