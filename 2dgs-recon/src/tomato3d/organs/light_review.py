"""Human review page of the light-organs run (acceptance A4) that doubles as the declared grouping fallback (section 8).

Four declared photos; every leaflet's visible surfels are projected with the run's board poses and drawn as an outline
coloured by compound leaf. The reviewer marks each leaflet correct / merged / split, marks missed leaflets on the photo,
judges each compound leaf correct / merged / split, and - in grouping mode - clicks leaflets into human compound leaves.
Answers go to the artifact's db (collections `leaflet_answers`, `leaf_answers`, `missed`, `human_groups`, `session`);
the page never changes any parameter. Written for the Artifact tool (no doctype / html / body wrappers).

usage: python -m tomato3d.organs.light_review --run <run dir> [--surfels ply] [--poses poses.json] [--photos dir]
                                               [--views 060430586 060343438 060603671 060528683]
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs.light_organs import DEFAULTS, REVIEW_LIMITS
from tomato3d.organs.light_viewer import palette, read_csv
from tomato3d.recon import charuco_a3 as ca

PHOTOS = DEFAULTS["poses"].parent / "undist_3024"
VIEWS = ["060430586", "060343438", "060603671", "060528683"]
PAGE_PX = 1200          # photo long side in the page
VIS_RES = 756           # z-buffer resolution for the visible-part test
VIS_TOL = 0.002


def project(xyz, po, view, res):
    R = np.array(po["views"][view]["R_cw"]); t = np.array(po["views"][view]["t_cw"]); K = ca.scaled_K(np.array(po["K_undist"]), po["undistorted_res"], res)
    xc = xyz @ R.T + t; z = xc[:, 2]; u = K[0, 0] * xc[:, 0] / z + K[0, 2]; v = K[1, 1] * xc[:, 1] / z + K[1, 2]
    return u, v, z


def visible(u, v, z, res, tol):
    ui = np.round(u).astype(int); vi = np.round(v).astype(int); ok = (z > 0) & (ui >= 0) & (vi >= 0) & (ui < res) & (vi < res)
    idx = np.nonzero(ok)[0]; zb = np.full((res, res), np.inf); order = idx[np.argsort(-z[idx], kind="stable")]; zb[vi[order], ui[order]] = z[order]
    vis = np.zeros(len(u), bool); vis[idx] = z[idx] <= zb[vi[idx], ui[idx]] + tol
    return vis


def outlines(run: Path, surfels: Path, poses: Path, views, photos: Path):
    sf = sio.read_surfels_full(surfels); po = json.load(open(poses))
    lab = np.load(run / "surfel_labels.npz"); rows = lab["rows"]; cl = lab["cl"]; leaflet_of_cl = lab["leaflet_id"]; leaf_of_cl = lab["leaf_id"]
    xyz = sf["xyz"][rows]
    leaflets = read_csv(run / "leaflets.csv"); leaves = read_csv(run / "compound_leaves.csv")
    pal = dict(zip([L["leaf_id"] for L in leaves], palette(len(leaves))))
    out_views = []
    for v in views:
        name = next(k for k in po["views"] if v in k); u, v2, z = project(xyz, po, name, VIS_RES); vis = visible(u, v2, z, VIS_RES, VIS_TOL)
        scale = PAGE_PX / VIS_RES; polys = []
        for c in range(cl.max() + 1):
            m = (cl == c) & vis
            if m.sum() < 6:
                continue
            P2 = np.c_[u[m], v2[m]] * scale
            try:
                h = ConvexHull(P2); pts = P2[h.vertices]
            except Exception:
                continue
            polys.append(dict(leaflet=str(leaflet_of_cl[c]), leaf=str(leaf_of_cl[c]), n_vis=int(m.sum()), pts=np.round(pts, 1).tolist()))
        img = Image.open(photos / f"20261021_{v}_iOS.jpg").convert("RGB"); img.thumbnail((PAGE_PX, PAGE_PX)); buf = io.BytesIO(); img.save(buf, "JPEG", quality=78)
        out_views.append(dict(id=v, w=img.size[0], h=img.size[1], jpg=base64.b64encode(buf.getvalue()).decode(), polys=polys))
    return dict(views=out_views, leaflets=[dict(id=x["leaflet_id"], leaf=x["leaf_id"], uncertain=x["uncertain"], flags=x["flags"]) for x in leaflets],
                leaves=[dict(id=L["leaf_id"], n=int(L["n_leaflets"]), members=L["leaflet_ids"].split(";"), colour="rgb(%d,%d,%d)" % pal[L["leaf_id"]]) for L in leaves],
                limits=REVIEW_LIMITS)


HTML = r"""<title>R3-1b 소엽·복엽 검토</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* layout: photo stage left, verdict rail right; stacks at phone width */
:root{--bg:#f6f4ef;--fg:#1f2321;--mut:#6b716d;--panel:#ffffff;--line:#d9d5cc;--acc:#2f6f4f;--warn:#b7791f;--bad:#b23a3a;--ok:#2f6f4f;--chip:#ecebe4;--display:"IBM Plex Sans KR",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181a;--fg:#e6e8e6;--mut:#9aa39e;--panel:#1d2124;--line:#343a3e;--acc:#5fae86;--warn:#d9a441;--bad:#e06c6c;--ok:#5fae86;--chip:#262b2f;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15181a;--fg:#e6e8e6;--mut:#9aa39e;--panel:#1d2124;--line:#343a3e;--acc:#5fae86;--warn:#d9a441;--bad:#e06c6c;--ok:#5fae86;--chip:#262b2f;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.5 var(--display);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:12px}
h1{font-size:18px;margin:0 0 4px;text-wrap:balance}.sub{color:var(--mut);font-size:12px;margin:0 0 10px}
.banner{background:#7a1f1f;color:#fff;border-radius:8px;padding:6px 10px;font-size:12px;margin-bottom:10px}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:14px}@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.stage{min-width:0}.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
button{background:var(--chip);color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:5px 9px;font:inherit;font-size:12px;cursor:pointer}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.photo{position:relative;width:100%;max-width:100%;background:#000;border-radius:8px;overflow:hidden}
.photo img{display:block;width:100%;height:auto}.photo svg{position:absolute;inset:0;width:100%;height:100%}
polygon{fill-opacity:.22;stroke-width:2;cursor:pointer}polygon.sel{stroke-width:4;fill-opacity:.45}polygon.hide{display:none}
circle.miss{fill:var(--bad);stroke:#fff;stroke-width:2;cursor:pointer}
.rail{display:flex;flex-direction:column;gap:10px;min-width:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card h2{font-size:13px;margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.k{font-family:var(--mono);font-variant-numeric:tabular-nums}
.list{max-height:260px;overflow:auto;font-size:12px}.list div{display:flex;gap:6px;align-items:center;padding:2px 0;border-bottom:1px dashed var(--line)}
.sw{width:10px;height:10px;border-radius:2px;flex:none}.v{margin-left:auto;font-family:var(--mono)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.stat{display:grid;grid-template-columns:1fr 1fr;gap:6px}.stat b{display:block;font-size:20px;font-family:var(--mono)}
label{font-size:12px;color:var(--mut)}#toast{position:fixed;left:50%;bottom:16px;transform:translateX(-50%);background:var(--fg);color:var(--bg);padding:6px 12px;border-radius:8px;font-size:12px;display:none}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono&display=swap">
<h1>R3-1b 소엽·복엽 검토</h1>
<p class="sub">소엽 윤곽선 = 재구성 표면을 네 장의 사진에 투영한 볼록 외곽(가려진 부분은 제외). 색 = 자동 복엽 묶음. 목표 15분. 답은 저장만 되며 어떤 파라미터도 바꾸지 않습니다.</p>
<div class="banner">R3-1b_unaccepted · 형태만 · 척도 미검증 · 축/마디/화방 = unavailable · 계측 아님</div>
<div class="wrap"><div class="stage">
<div class="tabs" id="tabs"></div>
<div class="row" style="margin-bottom:8px"><button id="mReview" class="on">판정 모드</button><button id="mGroup">그룹 모드 (대안)</button><button id="mMiss">놓친 소엽 표시</button><button id="hideDone">판정 끝난 것 숨기기</button></div>
<div class="photo" id="photo"><img id="img" alt="review photo"><svg id="ov"></svg></div>
<p class="sub" id="hint">판정 모드: 소엽을 클릭하면 correct → merged → split → (지움) 순으로 바뀝니다. 놓친 소엽 모드: 윤곽선이 없는 소엽 위를 클릭해 표시합니다 (표시 점을 다시 클릭하면 지움).</p>
</div><div class="rail">
<div class="card"><h2>진행</h2><div class="stat"><div><label>소엽 판정</label><b class="k" id="sLf">0</b></div><div><label>소엽 불일치율</label><b class="k" id="sLfR">–</b></div><div><label>복엽 판정</label><b class="k" id="sCl">0</b></div><div><label>복엽 불일치율</label><b class="k" id="sClR">–</b></div></div>
<p class="sub" id="verdict" style="margin:6px 0 0"></p><p class="sub" style="margin:4px 0 0">한도(계약): 소엽 ≤ 0.25, 복엽 ≤ 0.35. 불일치 = (merged + split + missed) / 판정 수.</p>
<div class="row"><label>경과</label><span class="k" id="clock">00:00</span><button id="finish">검토 완료 저장</button></div><p class="sub" id="store"></p></div>
<div class="card"><h2>선택한 소엽</h2><div id="selInfo" class="sub">사진에서 소엽을 클릭하세요.</div><div class="row" id="selBtns"></div></div>
<div class="card"><h2>복엽 판정 (<span id="nCl"></span>)</h2><div class="list" id="clList"></div></div>
<div class="card" id="grpCard" hidden><h2>사람 그룹 (대안)</h2><p class="sub">그룹을 고른 뒤 사진에서 소엽을 클릭하면 그 그룹에 들어갑니다. 자동 묶음은 바뀌지 않고, 사람 그룹은 별도로 저장됩니다.</p><div class="row" id="grpBtns"></div><div class="list" id="grpList"></div></div>
<div class="card"><h2>소엽 목록</h2><div class="list" id="lfList"></div></div>
</div></div><div id="toast"></div>
<script id="D" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('D').textContent);
const LEAF_COL={};D.leaves.forEach(l=>LEAF_COL[l.id]=l.colour);LEAF_COL['unassigned']='rgb(235,235,235)';
const S={view:0,mode:'review',sel:null,hideDone:false,group:null,lf:{},cl:{},miss:{},grp:{},t0:Date.now()};
let db=null,user=null,uid='local';
const $=id=>document.getElementById(id);
function toast(m){const t=$('toast');t.textContent=m;t.style.display='block';clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',1800);}
// ---- storage: artifact db when available, else localStorage (per-viewer only)
function lsSave(){try{localStorage.setItem('lo_review',JSON.stringify({lf:S.lf,cl:S.cl,miss:S.miss,grp:S.grp}));}catch(e){}}
function lsLoad(){try{const d=JSON.parse(localStorage.getItem('lo_review')||'null');if(d){Object.assign(S,{lf:d.lf||{},cl:d.cl||{},miss:d.miss||{},grp:d.grp||{}});}}catch(e){}}
async function put(col,id,data){if(db){try{if(data===null)await db.doc(col+'/'+id).delete();else await db.doc(col+'/'+id).set(Object.assign({at:new Date().toISOString(),by:uid},data));$('store').textContent='저장: artifact db';}catch(e){$('store').textContent='저장 실패: '+(e.code||e.message);}}else lsSave();}
async function loadDb(){const cols={leaflet_answers:'lf',leaf_answers:'cl',missed:'miss',human_groups:'grp'};for(const [c,k] of Object.entries(cols)){const q=await db.collection(c).get();S[k]={};q.docs.forEach(d=>{S[k][d.id]=d.data();});}}
// ---- rendering
function render(){const V=D.views[S.view];$('img').src='data:image/jpeg;base64,'+V.jpg;const ov=$('ov');ov.setAttribute('viewBox','0 0 '+V.w+' '+V.h);
let s='';V.polys.forEach((p,i)=>{const a=S.lf[p.leaflet]&&S.lf[p.leaflet].verdict;const hide=S.hideDone&&a;const col=S.mode==='group'&&S.grp[p.leaflet]?grpCol(S.grp[p.leaflet].group):LEAF_COL[p.leaf]||LEAF_COL.unassigned;
s+=`<polygon data-i="${i}" class="${S.sel===p.leaflet?'sel':''} ${hide?'hide':''}" points="${p.pts.map(q=>q.join(',')).join(' ')}" fill="${col}" stroke="${a==='merged'?'#e03030':a==='split'?'#e0a020':a==='correct'?'#30c060':col}"><title>${p.leaflet} · ${p.leaf}${a?' · '+a:''}</title></polygon>`;});
Object.entries(S.miss).forEach(([id,m])=>{if(m.view===V.id)s+=`<circle class="miss" data-m="${id}" cx="${m.x}" cy="${m.y}" r="9"><title>missed leaflet</title></circle>`;});
ov.innerHTML=s;$('tabs').innerHTML=D.views.map((v,i)=>`<button data-v="${i}" class="${i===S.view?'on':''}">사진 ${v.id} · 윤곽 ${v.polys.length}</button>`).join('');
$('mReview').className=S.mode==='review'?'on':'';$('mGroup').className=S.mode==='group'?'on':'';$('mMiss').className=S.mode==='miss'?'on':'';$('hideDone').className=S.hideDone?'on':'';$('grpCard').hidden=S.mode!=='group';
renderRail();}
function grpCol(g){const i=parseInt(g.replace(/\D/g,''))||0;return `hsl(${(i*61)%360} 70% 50%)`;}
function renderRail(){const judged=Object.values(S.lf).filter(x=>x.verdict);const bad=judged.filter(x=>x.verdict!=='correct').length+Object.keys(S.miss).length;const den=judged.length+Object.keys(S.miss).length;
$('sLf').textContent=judged.length+' / '+D.leaflets.length;$('sLfR').textContent=den?(bad/den).toFixed(2):'–';$('sLfR').className='k '+(den?(bad/den<=D.limits.leaflet_disagreement_max?'ok':'bad'):'');
const cj=Object.values(S.cl).filter(x=>x.verdict);const cb=cj.filter(x=>x.verdict!=='correct').length;$('sCl').textContent=cj.length+' / '+D.leaves.length;$('sClR').textContent=cj.length?(cb/cj.length).toFixed(2):'–';$('sClR').className='k '+(cj.length?(cb/cj.length<=D.limits.leaf_disagreement_max?'ok':'bad'):'');
let v='';if(den&&bad/den>D.limits.leaflet_disagreement_max)v+='소엽 한도 초과 → 계약 §8: 튠 금지, 대안(사람 마스크 투영)은 이 개체에 없음. ';if(cj.length&&cb/cj.length>D.limits.leaf_disagreement_max)v+='복엽 한도 초과 → 그룹 모드로 사람 묶음을 기록하세요.';$('verdict').textContent=v;
$('nCl').textContent=D.leaves.length;$('clList').innerHTML=D.leaves.map(l=>{const a=S.cl[l.id]&&S.cl[l.id].verdict;return `<div><i class="sw" style="background:${l.colour}"></i>${l.id} <span class="sub" style="margin:0">${l.n}枚</span><span class="v">${['correct','merged','split'].map(k=>`<button data-cl="${l.id}" data-k="${k}" class="${a===k?'on':''}" style="padding:2px 5px">${k[0].toUpperCase()}</button>`).join('')}</span></div>`;}).join('');
$('lfList').innerHTML=D.leaflets.map(x=>{const a=S.lf[x.id]&&S.lf[x.id].verdict;return `<div data-lf="${x.id}" style="cursor:pointer"><i class="sw" style="background:${LEAF_COL[x.leaf]||LEAF_COL.unassigned}"></i>${x.id} <span class="sub" style="margin:0">${x.leaf}${x.uncertain==='true'?' · uncertain':''}</span><span class="v ${a==='correct'?'ok':a?'bad':''}">${a||''}</span></div>`;}).join('');
const groups=[...new Set(Object.values(S.grp).map(g=>g.group))].sort();$('grpBtns').innerHTML=groups.map(g=>`<button data-g="${g}" class="${S.group===g?'on':''}" style="border-color:${grpCol(g)}">${g}</button>`).join('')+`<button data-g="__new">+ 새 그룹</button>`;
$('grpList').innerHTML=groups.map(g=>`<div><i class="sw" style="background:${grpCol(g)}"></i>${g}<span class="v">${Object.entries(S.grp).filter(([k,v])=>v.group===g).map(([k])=>k).join(' ')}</span></div>`).join('');
const sel=S.sel;if(sel){const x=D.leaflets.find(l=>l.id===sel);const a=S.lf[sel]&&S.lf[sel].verdict;$('selInfo').innerHTML=`<b>${sel}</b> · 자동 복엽 ${x?x.leaf:'?'} · 플래그 ${x&&x.flags||'–'} · 현재 ${a||'미판정'}`;$('selBtns').innerHTML=['correct','merged','split'].map(k=>`<button data-sv="${k}" class="${a===k?'on':''}">${k}</button>`).join('')+`<button data-sv="">지움</button>`;}else{$('selInfo').textContent='사진에서 소엽을 클릭하세요.';$('selBtns').innerHTML='';}}
// ---- events
document.addEventListener('click',async e=>{const t=e.target.closest('[data-v],[data-cl],[data-lf],[data-sv],[data-g]');if(!t)return;
if(t.dataset.v!==undefined){S.view=+t.dataset.v;render();return;}
if(t.dataset.cl){const id=t.dataset.cl,k=t.dataset.k;const cur=S.cl[id]&&S.cl[id].verdict;S.cl[id]={verdict:cur===k?null:k};if(!S.cl[id].verdict){delete S.cl[id];await put('leaf_answers',id,null);}else await put('leaf_answers',id,S.cl[id]);render();return;}
if(t.dataset.lf){S.sel=t.dataset.lf;render();return;}
if(t.dataset.sv!==undefined){const id=S.sel;if(!id)return;if(t.dataset.sv){S.lf[id]={verdict:t.dataset.sv};await put('leaflet_answers',id,S.lf[id]);}else{delete S.lf[id];await put('leaflet_answers',id,null);}render();return;}
if(t.dataset.g){if(t.dataset.g==='__new'){const n=[...new Set(Object.values(S.grp).map(g=>g.group))].length+1;S.group='HG'+String(n).padStart(2,'0');}else S.group=t.dataset.g;render();}});
$('ov').addEventListener('click',async e=>{const p=e.target.closest('polygon');const V=D.views[S.view];
if(e.target.classList.contains('miss')){const id=e.target.dataset.m;delete S.miss[id];await put('missed',id,null);render();return;}
if(p){const poly=V.polys[+p.dataset.i];const id=poly.leaflet;S.sel=id;
 if(S.mode==='review'){const order=[null,'correct','merged','split'];const cur=S.lf[id]&&S.lf[id].verdict;const nx=order[(order.indexOf(cur||null)+1)%4];if(nx){S.lf[id]={verdict:nx,view:V.id};await put('leaflet_answers',id,S.lf[id]);}else{delete S.lf[id];await put('leaflet_answers',id,null);}}
 else if(S.mode==='group'){if(!S.group){toast('먼저 그룹을 고르세요');}else{S.grp[id]={group:S.group,auto_leaf:poly.leaf};await put('human_groups',id,S.grp[id]);}}
 else if(S.mode==='miss'){/* a click on a polygon in miss mode does nothing: missed means no outline */ toast('윤곽선이 없는 곳을 클릭하세요');}
 render();return;}
if(S.mode==='miss'){const r=$('ov').getBoundingClientRect();const x=Math.round((e.clientX-r.left)/r.width*V.w),y=Math.round((e.clientY-r.top)/r.height*V.h);const id='M'+Date.now().toString(36);S.miss[id]={view:V.id,x,y};await put('missed',id,S.miss[id]);render();}});
$('mReview').onclick=()=>{S.mode='review';render();};$('mGroup').onclick=()=>{S.mode='group';render();};$('mMiss').onclick=()=>{S.mode='miss';render();};$('hideDone').onclick=()=>{S.hideDone=!S.hideDone;render();};
$('finish').onclick=async()=>{const judged=Object.values(S.lf).filter(x=>x.verdict);const bad=judged.filter(x=>x.verdict!=='correct').length+Object.keys(S.miss).length;const den=judged.length+Object.keys(S.miss).length;const cj=Object.values(S.cl).filter(x=>x.verdict);const cb=cj.filter(x=>x.verdict!=='correct').length;
 await put('session','summary',{elapsed_s:Math.round((Date.now()-S.t0)/1000),leaflets_judged:judged.length,leaflet_disagreement:den?bad/den:null,missed:Object.keys(S.miss).length,leaves_judged:cj.length,leaf_disagreement:cj.length?cb/cj.length:null,human_groups:Object.keys(S.grp).length,finished:true});toast('저장했습니다');};
setInterval(()=>{const s=Math.round((Date.now()-S.t0)/1000);$('clock').textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');},1000);
lsLoad();render();
(async()=>{try{if(window.claude&&window.claude.use){db=await claude.use('db');user=await claude.use('user');if(user){uid=await user.id();}if(db){await loadDb();$('store').textContent='저장: artifact db (불러옴)';render();}else $('store').textContent='artifact db 없음 → 이 브라우저에만 저장';}}catch(e){$('store').textContent='db 오류: '+(e.message||e);}})();
</script>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, default=DEFAULTS["out"]); ap.add_argument("--surfels", type=Path, default=DEFAULTS["surfels"])
    ap.add_argument("--poses", type=Path, default=DEFAULTS["poses"]); ap.add_argument("--photos", type=Path, default=PHOTOS); ap.add_argument("--views", nargs="+", default=VIEWS)
    a = ap.parse_args(argv)
    data = outlines(a.run, a.surfels, a.poses, a.views, a.photos)
    out = a.run / "review"; out.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (out / "review_page.html").write_text(html, encoding="utf-8")
    json.dump(dict(views=[dict(id=v["id"], polygons=len(v["polys"])) for v in data["views"]], leaflets=len(data["leaflets"]), leaves=len(data["leaves"])), open(out / "review_inputs.json", "w"), indent=1)
    print(out / "review_page.html", len(html), [(v["id"], len(v["polys"])) for v in data["views"]])


if __name__ == "__main__":
    main()
