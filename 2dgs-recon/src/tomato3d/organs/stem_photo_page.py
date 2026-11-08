"""Local review / measure page for the stem reference photos (port 8839). Every click is appended to stem_clicks.jsonl in full-resolution (3024 px)
image coordinates; nothing is overwritten; reload resumes. Geometric candidates (stem_cands/<photo>.json, Revision 2) are drawn suspects-first; ✓ writes the candidate as a click record
(source "auto", cand id), ✗ writes a cand_verdict. On QC photos the centre candidate stays hidden until the user has clicked >= 3 centre points.

usage: python -m tomato3d.organs.stem_photo_page [--port 8839]
"""

from __future__ import annotations

import argparse
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from tomato3d.organs import light_organs as LO

D = LO.ROOT / "build/p1_stem_gapfill_20261029_01"
CLICKS = D / "stem_clicks.jsonl"
# value order: clear lower stem + rim first, then whole-plant views with rim, the close-up, upper views, hand-covered last
ORDER = ["0735", "0736", "0737", "0753", "0740", "0739", "0738", "0725", "0726", "0727", "0728", "0729", "0730", "0731", "0732", "0734", "0724",
         "0741", "0742", "0743", "0744", "0745", "0746", "0747", "0748", "0749", "0750", "0751", "0752"]
QC = ["0736", "0727", "0731", "0740", "0753"]                 # declared: the five quality-check photos (user's centreline vs proposal); no other use
VIEW_PX = 1512


def photos():
    return [f"IMG_{k}" for k in ORDER if (D / "photos_jpg" / f"IMG_{k}.jpg").exists()]


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>줄기 사진 측정</title>
<style>body{margin:0;background:#141414;color:#eee;font:13px system-ui,sans-serif}#hd{background:#3a2c00;padding:6px 12px;font-size:12px}
main{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:10px;padding:8px 12px}@media(max-width:900px){main{grid-template-columns:1fr}}
#wrap{overflow:auto;max-height:88vh;border:1px solid #333}canvas{display:block;cursor:crosshair}button{background:#2a2a2a;color:#eee;border:1px solid #555;border-radius:5px;padding:5px 9px;margin:2px;cursor:pointer;font:inherit}
button.on{outline:2px solid #ffd400}.sec{border-top:1px solid #333;margin-top:8px;padding-top:6px}#list div{padding:2px 4px;cursor:pointer}#list div.cur{background:#333}small{opacity:.75}</style></head><body>
<div id="hd">줄기 참고 사진 측정 · 어느 개체인지 정하지 않음 (모든 개체 공통 참고 자료) · 클릭은 원본 3024 px 좌표로 저장 · 손이 측정 부분을 가리면 "손 가림"</div>
<main><div><div id="wrap"><canvas id="cv"></canvas></div></div><aside>
<div><b id="name"></b> <span id="qc"></span><br><span id="prog"></span></div>
<div class="sec">도구 (한 번에 하나)<br>
<button data-m="centre">중심선 점 (3–6)</button><button data-m="width">줄기 폭 (왼→오 한 쌍)</button><button data-m="node">마디 (밑동→엽병 방향 한 쌍)</button>
<button data-m="petw">엽병 밑 폭 (한 쌍)</button><button data-m="branch">곁가지 (밑동→방향 한 쌍)</button><button data-m="rim">화분 테두리 폭 (왼→오)</button><button data-m="grid">격자 한 칸 (왼→오)</button>
<div id="hint"><small>도구를 고르고 사진을 클릭하세요.</small></div></div>
<div class="sec">사진 표시<br><button id="hand">손 가림</button><button id="blur">흐림</button><button id="done">이 사진 끝</button><br>
</div>
<div class="sec"><b>자동 후보</b> <small>(의심 먼저 · ✓ 맞음 / ✗ 틀림 · 빨강 = 의심)</small><br><button id="okall">의심 아닌 후보 모두 맞음</button><div id="cl"></div></div>
<div class="sec"><button id="undo">되돌리기</button><button id="prev">◀</button><button id="next">▶</button> 확대 <input id="z" type="range" min="0.4" max="1.6" step="0.1" value="0.6"></div>
<div class="sec" id="list"></div></aside></main>
<script>
let P=[],i=0,mode='centre',pend=null,R=[],prop=null,Z=0.6,C=[],isqc=false,sel=null;const cv=document.getElementById('cv'),cx=cv.getContext('2d'),img=new Image();
const COL={centre:'#ff4040',width:'#40c0ff',node:'#ffd400',petw:'#ff9a40',branch:'#c070ff',rim:'#ffffff',grid:'#80ff80'};
const PAIR={width:1,node:1,petw:1,branch:1,rim:1,grid:1};
async function post(o){const r=await fetch('/api/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});return r.json();}
async function load(k){i=k;pend=null;const s=await (await fetch('/api/photo/'+P[i])).json();R=s.records;prop=s.proposal;C=(s.cands&&s.cands.cands)||[];isqc=s.qc;sel=null;img.onload=()=>{draw();};img.src='/img/'+P[i]+'?'+Date.now();
 document.getElementById('name').textContent=P[i];document.getElementById('qc').textContent=s.qc?'· 품질 점검용 (중심선 3–6점 꼭)':'';list();}
const KO={rim:'화분 테',centre:'중심선',width:'줄기 폭',node:'마디'};
function decided(cid){return act().some(r=>r.cand===cid);}
function manualCentre(){return act().filter(r=>r.kind==='centre'&&r.source!=='auto').length;}
function vis(){return C.filter(c=>!decided(c.cid)&&!(isqc&&c.kind==='centre'&&manualCentre()<3)).sort((a,b)=>(b.suspect-a.suspect)||(a.conf-b.conf));}
async function accept(c){if(c.kind==='centre'){for(const p of c.pts){const r=await post({photo:P[i],kind:'centre',pts:[p],source:'auto',cand:c.cid});R.push(r);}}else{const r=await post({photo:P[i],kind:c.kind,pts:c.pts,source:'auto',cand:c.cid});R.push(r);}}
async function reject(c){const r=await post({photo:P[i],kind:'cand_verdict',value:'bad',cand:c.cid,pts:[]});R.push(r);}
function clist(){const V=vis();document.getElementById('cl').innerHTML=(V.length?'':'<small>남은 후보 없음'+(isqc?' (품질 점검 사진: 중심선 3점을 직접 찍으면 중심선 후보가 나타남)':'')+'</small>')+V.map(c=>`<div style="color:${c.suspect?'#ff7070':'#7fffd4'}" data-c="${c.cid}"><button data-a="1">✓</button><button data-a="0">✗</button> ${c.cid} ${KO[c.kind]} ${c.conf.toFixed(2)} <small>${c.why.join('; ')}</small></div>`).join('');
 document.querySelectorAll('#cl [data-c]').forEach(d=>{const c=C.find(x=>x.cid===d.dataset.c);d.onmouseenter=()=>{sel=c.cid;draw(true);};d.querySelectorAll('button').forEach(b=>b.onclick=async()=>{if(b.dataset.a==='1')await accept(c);else await reject(c);draw();});});}
function act(){const a={},rm=new Set(R.filter(r=>r.kind==='remove').map(r=>r.target));for(const r of R){if(r.kind!=='remove'&&!rm.has(r.id))a[r.id]=r;}return Object.values(a);}
function draw(nolist){const s=Z*VW/3024;cv.width=Math.round(3024*s);cv.height=Math.round(3024*s);cx.drawImage(img,0,0,cv.width,cv.height);
 cx.setLineDash([8,6]);cx.font='14px system-ui';for(const c of vis()){cx.strokeStyle=c.suspect?'#ff5050':'#00ffcc';cx.fillStyle=cx.strokeStyle;cx.lineWidth=c.cid===sel?5:2;cx.beginPath();c.pts.forEach((p,j)=>j?cx.lineTo(p[0]*s,p[1]*s):cx.moveTo(p[0]*s,p[1]*s));cx.stroke();c.pts.forEach(p=>{cx.beginPath();cx.arc(p[0]*s,p[1]*s,5,0,6.3);cx.fill();});cx.fillText(c.cid,c.pts[0][0]*s+7,c.pts[0][1]*s-7);}cx.setLineDash([]);
 const A=act();const cen=A.filter(r=>r.kind==='centre').sort((a,b)=>a.t-b.t);if(cen.length>1){cx.strokeStyle=COL.centre;cx.lineWidth=2;cx.beginPath();cen.forEach((r,j)=>j?cx.lineTo(r.pts[0][0]*s,r.pts[0][1]*s):cx.moveTo(r.pts[0][0]*s,r.pts[0][1]*s));cx.stroke();}
 for(const r of A){cx.strokeStyle=COL[r.kind]||'#fff';cx.fillStyle=cx.strokeStyle;cx.lineWidth=2;r.pts.forEach(p=>{cx.beginPath();cx.arc(p[0]*s,p[1]*s,4,0,6.3);cx.fill();});if(r.pts.length===2){cx.beginPath();cx.moveTo(r.pts[0][0]*s,r.pts[0][1]*s);cx.lineTo(r.pts[1][0]*s,r.pts[1][1]*s);cx.stroke();}}
 if(pend){cx.fillStyle='#fff';cx.beginPath();cx.arc(pend[0]*s,pend[1]*s,5,0,6.3);cx.fill();}
 const fl=A.filter(r=>r.kind==='flag').map(r=>r.value);document.getElementById('prog').textContent=`중심선 ${cen.length} · 폭 ${A.filter(r=>r.kind==='width').length} · 마디 ${A.filter(r=>r.kind==='node').length} · 테두리 ${A.filter(r=>r.kind==='rim').length} ${fl.join(' ')}`;if(!nolist)clist();}
const VW=1512;
cv.addEventListener('click',async e=>{const b=cv.getBoundingClientRect();const s=Z*VW/3024;const p=[Math.round((e.clientX-b.left)/s*10)/10,Math.round((e.clientY-b.top)/s*10)/10];
 if(PAIR[mode]){if(!pend){pend=p;draw();return;}const r=await post({photo:P[i],kind:mode,pts:[pend,p]});R.push(r);pend=null;}else{const r=await post({photo:P[i],kind:mode,pts:[p]});R.push(r);}draw();});
document.querySelectorAll('[data-m]').forEach(b=>b.onclick=()=>{mode=b.dataset.m;pend=null;document.querySelectorAll('[data-m]').forEach(x=>x.className=x===b?'on':'');document.getElementById('hint').innerHTML='<small>'+({centre:'줄기 가운데를 아래→위로 3–6번 클릭',width:'같은 높이에서 줄기 왼쪽 가장자리, 오른쪽 가장자리',node:'마디(엽병이 붙는 곳) 클릭 후 엽병을 따라 1–2 cm 위 한 점',petw:'마디 바로 위 엽병 양쪽 가장자리',branch:'곁가지 밑동 클릭 후 곁가지 방향 한 점',rim:'화분 테두리 바깥 왼쪽 끝, 오른쪽 끝 (같은 면)',grid:'격자 한 칸의 왼쪽 선, 오른쪽 선 (줄기 근처 높이)'})[mode]+'</small>';draw();});
document.querySelector('[data-m=centre]').className='on';
for(const [id,v] of [['hand','hand_occluded'],['blur','blurred'],['done','done']])document.getElementById(id).onclick=async()=>{const r=await post({photo:P[i],kind:'flag',value:v,pts:[]});R.push(r);draw();list();};
document.getElementById('okall').onclick=async()=>{for(const c of vis().filter(c=>!c.suspect))await accept(c);draw();};
document.getElementById('undo').onclick=async()=>{const a=act().filter(r=>r.kind!=='flag').sort((x,y)=>y.t-x.t)[0];if(!a)return;const r=await post({photo:P[i],kind:'remove',target:a.id,pts:[]});R.push(r);draw();};
document.getElementById('prev').onclick=()=>load((i+P.length-1)%P.length);document.getElementById('next').onclick=()=>load((i+1)%P.length);
document.getElementById('z').oninput=e=>{Z=+e.target.value;draw();};
async function list(){const s=await (await fetch('/api/status')).json();document.getElementById('list').innerHTML=P.map((p,j)=>`<div data-j="${j}" class="${j===i?'cur':''}">${p} ${s[p]||''}</div>`).join('');document.querySelectorAll('#list [data-j]').forEach(d=>d.onclick=()=>load(+d.dataset.j));}
(async()=>{P=(await (await fetch('/api/state')).json()).photos;await load(0);window.__sp={state:()=>({i,mode,records:R.length,photo:P[i]}),post,load};})();
</script></body></html>"""


def records():
    if not CLICKS.exists(): return []
    return [json.loads(l) for l in CLICKS.read_text(encoding="utf-8").splitlines() if l.strip()]


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/": return self._send(HTML.encode(), "text/html; charset=utf-8")
        if p == "/api/state": return self._send(json.dumps(dict(photos=photos(), qc=QC)).encode(), "application/json")
        if p == "/api/status":
            st = {}
            for r in records():
                if r["kind"] == "flag": st[r["photo"]] = (st.get(r["photo"], "") + " " + {"done": "✓", "hand_occluded": "손", "blurred": "흐림"}.get(r["value"], "")).strip()
            return self._send(json.dumps(st, ensure_ascii=False).encode(), "application/json")
        if p.startswith("/api/photo/"):
            name = p.rsplit("/", 1)[1]; pf = D / "stem_masks" / f"{name}.json"; cf = D / "stem_cands" / f"{name}.json"
            return self._send(json.dumps(dict(records=[r for r in records() if r["photo"] == name], qc=name[4:] in QC, proposal=(json.load(open(pf)) if pf.exists() else None), cands=(json.load(open(cf)) if cf.exists() else None))).encode(), "application/json")
        if p.startswith("/img/"):
            name = p.rsplit("/", 1)[1]; im = Image.open(D / "photos_jpg" / f"{name}.jpg"); im = im.resize((VIEW_PX, VIEW_PX)); b = io.BytesIO(); im.save(b, "JPEG", quality=88)
            return self._send(b.getvalue(), "image/jpeg")
        self._send(b"{}", "application/json", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        rec = dict(body, id=f"c{int(time.time() * 1000):x}", t=time.time(), at=time.strftime("%Y-%m-%dT%H:%M:%S"), coords="full-resolution px (3024)")
        with open(CLICKS, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._send(json.dumps(rec).encode(), "application/json")

    def log_message(self, *a):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8839); a = ap.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); print(f"serving http://127.0.0.1:{a.port}/ photos {len(photos())}", flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()
