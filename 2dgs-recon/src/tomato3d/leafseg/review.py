"""Gate A review page (single self-contained HTML; answers go to the artifact's db, never into proposals/).

usage: python -m tomato3d.leafseg.review --run build/p1_leafseg_20261019_01 --out build/p1_leafseg_20261019_01/review/leafseg_review.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VIEW_PX = 1200          # overlay width shown (JPEG)
HIT_PX = 512            # click map width (8-bit PNG of instance index + 1)

TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* Layout: photo strip on top, then photo + answer panel side by side (stacked on phones). */
:root{--bg:#f3f4f1;--panel:#ffffff;--ink:#1d2420;--mute:#5f6b64;--line:#d6dbd5;--leaf:#2f6b4f;--leaf-soft:#e2eee7;--warn:#b35c00;--warn-soft:#fbead6;--bad:#a8323b;--bad-soft:#f6dfe1;
--f-display:"IBM Plex Sans KR","Noto Sans KR",system-ui,sans-serif;--f-body:"Noto Sans KR",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121614;--panel:#1b201d;--ink:#e6ebe7;--mute:#9aa69f;--line:#2e3631;--leaf:#7cc4a0;--leaf-soft:#1f3329;--warn:#f0a050;--warn-soft:#3a2a17;--bad:#f08088;--bad-soft:#3a1d21;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121614;--panel:#1b201d;--ink:#e6ebe7;--mute:#9aa69f;--line:#2e3631;--leaf:#7cc4a0;--leaf-soft:#1f3329;--warn:#f0a050;--warn-soft:#3a2a17;--bad:#f08088;--bad-soft:#3a1d21;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--f-body)}
.wrap{max-width:1280px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px;display:grid;gap:18px}
h1{font:600 22px/1.25 var(--f-display);margin:0;text-wrap:balance}
.lede{color:var(--mute);margin:4px 0 0;max-width:72ch}
.strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
.chip{flex:0 0 auto;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:6px 10px;font:13px var(--f-mono);cursor:pointer;display:grid;gap:2px;text-align:left}
.chip small{color:var(--mute);font:11px var(--f-body)}
.chip[aria-pressed="true"]{border-color:var(--leaf);background:var(--leaf-soft)}
.chip .done{color:var(--leaf)}
.main{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:18px;align-items:start}
@media (max-width:860px){.main{grid-template-columns:1fr}}
.photo{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.photo canvas{display:block;width:100%;height:auto;cursor:crosshair;touch-action:manipulation}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--mute);padding:8px 12px;border-top:1px solid var(--line)}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:4px}
.panel{display:grid;gap:14px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:grid;gap:10px;min-width:0}
.card h2{font:600 14px var(--f-display);margin:0;letter-spacing:.01em}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;font-variant-numeric:tabular-nums}
.stat b{display:block;font:600 18px var(--f-mono)}.stat span{font-size:11px;color:var(--mute)}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;border:1px solid var(--line);color:var(--mute)}
.badge.held{border-color:var(--warn);color:var(--warn)}
.sel{display:grid;gap:8px}
.sel .who{font:600 15px var(--f-mono)}
.meta{font-size:12px;color:var(--mute);display:flex;flex-wrap:wrap;gap:4px 10px}
.btns{display:flex;flex-wrap:wrap;gap:6px}
button.ans{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:7px;padding:6px 12px;font:13px var(--f-body);cursor:pointer}
button.ans:focus-visible,.chip:focus-visible,.lf:focus-visible{outline:2px solid var(--leaf);outline-offset:1px}
button.ans.on.o{background:var(--leaf-soft);border-color:var(--leaf);color:var(--leaf)}
button.ans.on.x{background:var(--bad-soft);border-color:var(--bad);color:var(--bad)}
button.ans.on.q{background:var(--warn-soft);border-color:var(--warn);color:var(--warn)}
.grid{display:flex;flex-wrap:wrap;gap:5px}
.lf{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:2px 6px;font:12px var(--f-mono);cursor:pointer;font-variant-numeric:tabular-nums}
.lf.occ{border-style:dashed;border-color:var(--warn)}
.lf.O{background:var(--leaf-soft);border-color:var(--leaf)}.lf.X{background:var(--bad-soft);border-color:var(--bad)}
.lf.cur{outline:2px solid var(--ink)}
.pairs{display:grid;gap:6px;max-height:340px;overflow:auto}
.pair{display:flex;flex-wrap:wrap;align-items:center;gap:6px;font:12px var(--f-mono)}
.pair .t{min-width:118px}
textarea{width:100%;min-height:64px;font:13px var(--f-body);border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--ink);padding:8px}
.status{font-size:12px;color:var(--mute)}
.fail{font-size:12px;color:var(--warn)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="wrap">
 <header><h1>__H1__</h1>
 <p class="lede">모델 제안(__MODELS__)입니다. 정확도 주장은 없습니다. 사진에서 소엽을 누르거나 번호를 고른 뒤 O(맞음)/X(틀림)를 표시해 주세요. 위아래 화살표(위→아래 잎)도 맞음/반대/모름으로 답할 수 있습니다. 답은 이 페이지 저장소에만 저장되고 제안 파일은 바뀌지 않습니다.</p></header>
 <nav class="strip" id="strip" aria-label="사진 목록"></nav>
 <section class="main">
  <div class="photo"><canvas id="cv" width="1200" height="1200" aria-label="분할 오버레이"></canvas>
   <div class="legend"><span><i class="sw" style="background:#d84a5c"></i>보이는 소엽(색 = 개체)</span><span><i class="sw" style="background:repeating-linear-gradient(45deg,#ff8c00 0 3px,transparent 3px 6px);border:1px solid #ff8c00"></i>가려진 소엽</span><span><i class="sw" style="background:#808080"></i>배정 안 된 식물 영역</span><span><i class="sw" style="background:#ff00ff"></i>엽축 골격</span><span><i class="sw" style="background:#00e5e5"></i>위→아래 화살표</span></div></div>
  <div class="panel">
   <div class="card"><h2 id="title"></h2><div class="stats" id="stats"></div><div class="fail" id="fails"></div></div>
   <div class="card sel" id="selcard"><h2>선택한 소엽</h2><div class="who" id="who">사진에서 소엽을 누르세요</div><div class="meta" id="meta"></div>
     <div class="btns"><button class="ans o" data-v="O" id="bO">O 맞음</button><button class="ans x" data-v="X" id="bX">X 틀림</button><button class="ans q" data-v="merged" id="bM">여러 장이 합쳐짐</button><button class="ans q" data-v="part" id="bP">일부만 잡힘</button><button class="ans" data-v="" id="bC">지우기</button></div></div>
   <div class="card"><h2>소엽 목록</h2><div class="grid" id="grid"></div></div>
   <div class="card"><h2>위·아래 순서 (위 → 아래)</h2><div class="pairs" id="pairs"></div></div>
   <div class="card"><h2>메모 (놓친 잎, 기타)</h2><textarea id="note" placeholder="예: 가운데 아래쪽 큰 잎 3장이 잡히지 않음"></textarea><div class="status" id="status">저장소 연결 중…</div></div>
  </div>
 </section>
</div>
<script id="D" type="application/json">/*__DATA__*/</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById('D').textContent);
let cur=0,sel=null,db=null,ans={};
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
const imgs=D.images.map(r=>{const i=new Image();i.src='data:image/jpeg;base64,'+r.overlay;return i;});
const hits=D.images.map(r=>{const i=new Image();i.src='data:image/png;base64,'+r.hit;return i;});
const hitData={};
function hitOf(k){if(hitData[k])return hitData[k];const h=hits[k];if(!h.complete||!h.naturalWidth)return null;const c=document.createElement('canvas');c.width=h.naturalWidth;c.height=h.naturalHeight;const g=c.getContext('2d');g.drawImage(h,0,0);hitData[k]={w:c.width,h:c.height,d:g.getImageData(0,0,c.width,c.height).data};return hitData[k];}
function A(){const n=D.images[cur].name;ans[n]=ans[n]||{leaflets:{},pairs:{},note:""};return ans[n];}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function strip(){const s=document.getElementById('strip');s.innerHTML='';D.images.forEach((r,k)=>{const b=document.createElement('button');b.className='chip';b.setAttribute('aria-pressed',k===cur);const a=ans[r.name];const n=a?Object.keys(a.leaflets).length:0;
 b.innerHTML=esc(r.name)+'<small>'+(r.split==='held_out'?'평가용':'개발용')+' · 소엽 '+r.instances.length+(n?' · <span class="done">답 '+n+'</span>':'')+'</small>';b.onclick=()=>{cur=k;sel=null;render();};s.appendChild(b);});}
function draw(){const r=D.images[cur],im=imgs[cur];if(!im.complete){im.onload=draw;return;}cv.width=im.naturalWidth;cv.height=im.naturalHeight;cx.drawImage(im,0,0);
 if(sel){const i=r.instances.find(x=>x.id===sel);if(i&&i.c){const s=cv.width/r.grid_w;cx.strokeStyle='#ffffff';cx.lineWidth=3;cx.beginPath();cx.arc(i.c[0]*s,i.c[1]*s,18,0,Math.PI*2);cx.stroke();cx.strokeStyle='#000';cx.lineWidth=1;cx.stroke();}}}
function render(){const r=D.images[cur];strip();draw();
 document.getElementById('title').innerHTML=esc(r.name)+' <span class="badge'+(r.split==='held_out'?' held':'')+'">'+(r.split==='held_out'?'평가용(held-out)':'개발용')+'</span>';
 const st=r.stats;document.getElementById('stats').innerHTML=[['소엽',st.leaflets],['가려짐',st.occluded],['순서 쌍',st.edges],['미배정 면적',Math.round(st.unassigned*100)+'%']].map(([k,v])=>'<div class="stat"><b>'+v+'</b><span>'+k+'</span></div>').join('');
 document.getElementById('fails').textContent=r.fail;
 const g=document.getElementById('grid');g.innerHTML='';const a=A();
 r.instances.forEach(i=>{const b=document.createElement('button');const v=a.leaflets[i.id]||'';b.className='lf'+(i.occluded?' occ':'')+(v==='O'?' O':v?' X':'')+(i.id===sel?' cur':'');b.textContent=i.id.slice(1);b.title=i.id+(i.occluded?' (가려짐)':'');b.onclick=()=>{sel=i.id;render();};g.appendChild(b);});
 const p=document.getElementById('pairs');p.innerHTML='';if(!r.edges.length)p.innerHTML='<div class="status">이 사진에는 순서 쌍이 없습니다.</div>';
 r.edges.forEach(e=>{const key=e.over+'>'+e.under,v=a.pairs[key]||'';const d=document.createElement('div');d.className='pair';d.innerHTML='<span class="t">'+e.over.slice(1)+' → '+e.under.slice(1)+'</span>';
  [['ok','맞음','o'],['reversed','반대','x'],['unsure','모름','q']].forEach(([val,lab,cls])=>{const b=document.createElement('button');b.className='ans '+cls+(v===val?' on':'');b.textContent=lab;b.onclick=()=>{a.pairs[key]=(v===val?'':val);if(!a.pairs[key])delete a.pairs[key];save();render();};d.appendChild(b);});p.appendChild(d);});
 const i=r.instances.find(x=>x.id===sel);document.getElementById('who').textContent=i?i.id+(i.occluded?' · 가려짐':''):'사진에서 소엽을 누르세요';
 document.getElementById('meta').innerHTML=i?[i.src==='D1_lowconf_recovery'?'복원(D1)':i.src==='D2_split'?'분리(D2, '+i.splitFrom+')':'','면 '+i.face,'복엽 '+i.compound,(i.variety==='green'?'초록':'은빛')+' 계열',i.occluders.length?'가린 잎 '+i.occluders.map(x=>x.slice(1)).join(','):'',i.flags.join(' ')].filter(Boolean).map(esc).map(s=>'<span>'+s+'</span>').join(''):'';
 ['bO','bX','bM','bP'].forEach(id=>{const b=document.getElementById(id);b.disabled=!i;b.classList.toggle('on',!!i&&a.leaflets[i.id]===b.dataset.v);});document.getElementById('bC').disabled=!i;
 const t=document.getElementById('note');if(document.activeElement!==t)t.value=a.note||'';}
document.querySelectorAll('#selcard button.ans').forEach(b=>b.onclick=()=>{if(!sel)return;const a=A();if(b.dataset.v)a.leaflets[sel]=b.dataset.v;else delete a.leaflets[sel];save();render();});
cv.addEventListener('click',ev=>{const r=D.images[cur],h=hitOf(cur);if(!h)return;const rc=cv.getBoundingClientRect();const x=Math.floor((ev.clientX-rc.left)/rc.width*h.w),y=Math.floor((ev.clientY-rc.top)/rc.height*h.h);const v=h.d[(y*h.w+x)*4];sel=v>0&&r.instances[v-1]?r.instances[v-1].id:null;render();});
let nt=null;document.getElementById('note').addEventListener('input',e=>{A().note=e.target.value;clearTimeout(nt);nt=setTimeout(save,800);});
const status=t=>document.getElementById('status').textContent=t;
let saving=Promise.resolve();
function save(){const n=D.images[cur].name,body=JSON.parse(JSON.stringify(A()));if(!db){status('저장소에 연결되지 않아 이 브라우저에서만 보입니다.');try{localStorage.setItem('leafseg_'+n,JSON.stringify(body));}catch(e){}return;}
 saving=saving.then(()=>db.doc('reviews/'+n).set(Object.assign(body,{image:n,source:'human_review_page',updated:new Date().toISOString()}))).then(()=>status('저장됨 · '+new Date().toLocaleTimeString()),e=>status('저장 실패: '+(e&&e.message||e)+' (다시 눌러 주세요)'));}
render();
(async()=>{try{db=window.claude&&await window.claude.use('db');}catch(e){db=null;}
 if(!db){status('저장소를 쓸 수 없는 보기입니다. 답은 이 브라우저에만 남습니다.');D.images.forEach(r=>{try{const s=localStorage.getItem('leafseg_'+r.name);if(s)ans[r.name]=JSON.parse(s);}catch(e){}});render();return;}
 status('저장소 연결됨 · 답은 자동 저장됩니다');
 db.collection('reviews').onSnapshot(snap=>{snap.docs.forEach(d=>{const v=d.data();if(v&&v.image)ans[v.image]={leaflets:v.leaflets||{},pairs:v.pairs||{},note:v.note||''};});render();},e=>status('불러오기 실패: '+(e&&e.message||e)));})();
</script>
"""


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="소엽 분할 검토 Gate A"); ap.add_argument("--models", default="Option A: 튜닝된 OBB v5 + SAM2 v5"); a = ap.parse_args(argv)
    run = Path(a.run); gate = json.load(open(run / "gateA.json")); order = [f"tray_top_0{i}" for i in range(1, 10)] + ["dwt_101", "dwt_102", "dwt_103", "dwt_104"]
    imgs = []
    for name in [n for n in order if n in gate]:
        doc = json.load(open(run / "proposals" / f"{name}_leafseg.json")); g = gate[name]
        ov = Image.open(run / "overlays" / f"{name}.jpg").convert("RGB"); w0 = ov.width
        ov = ov.resize((VIEW_PX, round(ov.height * VIEW_PX / w0)), Image.LANCZOS); buf = io.BytesIO(); ov.save(buf, "JPEG", quality=80)
        lab = np.load(run / "layers" / f"{name}.npz")["label"]; hit = cv2.resize(np.clip(lab, 0, 255).astype(np.uint8), (HIT_PX, round(lab.shape[0] * HIT_PX / lab.shape[1])), interpolation=cv2.INTER_NEAREST)
        hb = io.BytesIO(); Image.fromarray(hit).convert("RGB").save(hb, "PNG")
        fc = g["failure_candidates"]
        fail = "실패 후보(자동, 확인 필요): " + ", ".join(x for x in [f"합쳐짐 {len(fc['merged'])}" if fc["merged"] else "", f"접힘 분리 {len(fc['split_folded'])}" if fc["split_folded"] else "",
                                                      f"엽축을 잎으로 잡아 제외 {fc['rachis_kept_as_leaf_dropped']}" if fc["rachis_kept_as_leaf_dropped"] else "",
                                                      f"사진 가장자리에서 잘림 {len(fc['truncated_by_image'])}" if fc["truncated_by_image"] else ""] if x) if any(fc.values()) else ""
        imgs.append(dict(name=name, split=doc["image"]["split"], grid_w=doc["grid_shape"][1], overlay=b64(buf.getvalue()), hit=b64(hb.getvalue()),
                         stats=dict(leaflets=g["leaflets"], occluded=g["occluded"], edges=g["order_edges"], unassigned=g["unassigned_area_fraction"]), fail=fail,
                         instances=[dict(id=i["id"], occluded=i["occluded"], face=i["face"], src=i.get("source"), splitFrom=i.get("split_from"), compound=i.get("compound_box_id", i["compound_leaf_id"]), variety=i["variety_group"],
                                         occluders=i["occluders"], flags=i["flags"], c=i["centroid"]) for i in doc["instances"]],
                         edges=[dict(over=e["over"], under=e["under"], p=round(e["p"], 2)) for e in doc["order_edges"]]))
    html = TEMPLATE.replace("__TITLE__", a.title).replace("__H1__", a.title.replace("검토 ", "검토 · ", 1)).replace("__MODELS__", a.models)
    html = html.replace("/*__DATA__*/", json.dumps(dict(images=imgs), ensure_ascii=False, separators=(",", ":")))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(html, encoding="utf-8")
    print(out, out.stat().st_size, len(imgs), "images")


if __name__ == "__main__":
    main()
