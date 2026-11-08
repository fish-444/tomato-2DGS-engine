"""P1 session A box viewer + click page (127.0.0.1:8840): surfels coloured by leaf (contested grey, outside white, stem dark), semi-transparent oriented boxes with petiole base and rachis line,
opacity slider, box info on click, "이 잎 맞음 / 틀림" + correct rachis point. Clicks are appended to <individual>/box_clicks.jsonl (never rewritten); they feed a re-run only after
the user writes "<individual> 박스 수정 완료". Read-only on every input.

usage: python -m tomato3d.organs.leafbox_page [--individual R3-1b] [--port 8840] [--write-only]
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from tomato3d.organs import light_organs as LO
from tomato3d.organs import vote2d as V2

ROOT = LO.ROOT / "build/p1_leafbox_20261029_01"; STATE = {}


def page_data(ind):
    d = ROOT / ind; bj = json.load(open(d / "boxes.json", encoding="utf-8")); z = np.load(d / "labels_leafbox.npz"); lab = z["label"].astype(int)
    X = np.load(LO.ROOT / "build/p1_pc_absorb_20261027_01/kaggle_data/r8pc_bundle.npz")["xyz"] * 1000.0; c = X.mean(0); n = len(X)
    col = np.full((n, 3), 150, np.uint8); col[lab == -1] = (128, 128, 128); col[lab == -2] = (255, 255, 255); col[lab == -3] = (55, 55, 55); col[lab == -4] = (175, 175, 175)
    for L in np.unique(lab[lab > 0]): col[lab == L] = V2.PAL.get(int(L), (220, 220, 220))
    boxes = []
    for lf in bj["leaves"]:
        b = lf["box"]; boxes.append(dict(leaf=lf["leaf_id"], centre=(np.array(b["centre"]) - c).tolist(), R=b["R"], half=b["half_extent_mm"], base=(np.array(lf["petiole_base_xyz"]) - c).tolist(),
                                         poly=(np.array(lf["rachis_polyline"]) - c).tolist(), n=lf["n_surfels"], area=lf["area_b_mm2_range"], overlap=lf["overlap_with"], flags=lf["flags"],
                                         colour="#%02x%02x%02x" % V2.PAL.get(int(lf["leaf_id"]), (220, 220, 220))))
    cnt = {k: int((lab == v).sum()) for k, v in (("contested", -1), ("outside", -2), ("stem", -3), ("rachis_unpainted", -4))}
    return dict(individual=ind, n=n, pos=base64.b64encode((X - c).astype(np.float32).tobytes()).decode(), col=base64.b64encode(col.tobytes()).decode(), boxes=boxes, counts=cnt, centre=c.tolist(),
                note=f"{ind} · " + " · ".join(bj["labels"]) + " · " + bj["scale_status"] + " — 상자 = 대략의 그릇 (겹침 허용), 소속은 규칙으로 정함")


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>복엽 상자</title>
<style>:root{--bg:#111;--fg:#eee}body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}#hd{position:fixed;top:0;left:0;right:0;background:#7a1f1f;font-size:12px;padding:4px 10px;z-index:3}
#side{position:fixed;top:26px;right:0;width:290px;bottom:0;background:#1b1b1b;padding:8px;overflow:auto;z-index:2}#cv{position:fixed;top:26px;left:0;width:calc(100vw - 290px);height:calc(100vh - 26px);display:block}button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:5px;padding:4px 8px;margin:2px;cursor:pointer;font:inherit}
button.on{background:#555;border-color:#ddd}label{display:block;margin:4px 0}#info{border-top:1px solid #444;margin-top:6px;padding-top:6px;font-size:12px}@media(max-width:700px){#side{width:100%;top:auto;height:42%}#cv{width:100vw;height:calc(58vh - 26px)}}</style></head><body>
<div id="hd"></div><canvas id="cv"></canvas><div id="side"><div id="leg" style="font-size:12px"></div>
<label><input type="checkbox" id="showBox" checked> 상자 보이기</label><label>상자 불투명도 <input type="range" id="op" min="0" max="0.8" step="0.05" value="0.25"></label><label><input type="checkbox" id="showPts" checked> 서펠 보이기</label>
<div id="info">상자를 클릭하면 정보가 나옵니다.</div><div id="act" style="display:none"><button id="ok">이 잎 맞음</button><button id="bad">틀림</button><div id="badmsg" style="font-size:11px;opacity:.85"></div></div><div id="msg" style="font-size:11px;margin-top:6px"></div>
<p style="font-size:11px;opacity:.75">틀림 → 그 잎의 올바른 엽축 위치(서펠)를 한 번 클릭. 기록은 box_clicks.jsonl 에 쌓이고, "<개체> 박스 수정 완료" 라고 쓰셔야 다시 계산합니다.</p></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script><script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>(async function(){const D=__DATA__;document.getElementById('hd').textContent=D.note;const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));const P=new Float32Array(b64(D.pos).buffer),C=b64(D.col);const N=D.n;
const cv=document.getElementById('cv');const R=new THREE.WebGLRenderer({canvas:cv,antialias:true});R.setPixelRatio(Math.min(window.devicePixelRatio||1,2));const S=new THREE.Scene();const cam=new THREE.PerspectiveCamera(45,1,1,3000);cam.position.set(0,-260,160);cam.up.set(0,0,1);
const ctl=new THREE.OrbitControls(cam,cv);function size(){const w=cv.clientWidth,h=cv.clientHeight;R.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}window.addEventListener('resize',size);size();
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(P,3));const col=new Float32Array(N*3);for(let i=0;i<N*3;i++)col[i]=C[i]/255;g.setAttribute('color',new THREE.BufferAttribute(col,3));
const pts=new THREE.Points(g,new THREE.PointsMaterial({size:1.6,vertexColors:true}));S.add(pts);let rr=0;for(let i=0;i<N;i+=7){const d=Math.hypot(P[i*3],P[i*3+1],P[i*3+2]);if(d>rr)rr=d;}const dist=rr*2.1/Math.tan(22.5*Math.PI/180)/2;cam.position.set(0,-dist*0.85,dist*0.5);ctl.update();const meshes=[];
for(const b of D.boxes){const geo=new THREE.BoxGeometry(2*b.half[0],2*b.half[1],2*b.half[2]);const mat=new THREE.MeshBasicMaterial({color:b.colour,transparent:true,opacity:0.25,depthWrite:false});const m=new THREE.Mesh(geo,mat);
const M4=new THREE.Matrix4();const r=b.R;M4.set(r[0][0],r[1][0],r[2][0],b.centre[0],r[0][1],r[1][1],r[2][1],b.centre[1],r[0][2],r[1][2],r[2][2],b.centre[2],0,0,0,1);m.applyMatrix4(M4);m.userData=b;S.add(m);meshes.push(m);
const ed=new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color:b.colour}));ed.applyMatrix4(M4);S.add(ed);m.userData.edges=ed;
const ln=new THREE.Line(new THREE.BufferGeometry().setFromPoints(b.poly.map(p=>new THREE.Vector3(p[0],p[1],p[2]))),new THREE.LineBasicMaterial({color:0xffd400}));S.add(ln);
const sp=new THREE.Mesh(new THREE.SphereGeometry(1.8,12,12),new THREE.MeshBasicMaterial({color:0xff3030}));sp.position.set(b.base[0],b.base[1],b.base[2]);S.add(sp);}
document.getElementById('leg').innerHTML='잎 색 = 소속, <span style="color:#888">■</span> 경합 '+D.counts.contested+' · □ 밖 '+D.counts.outside+' · <span style="color:#373737">■</span> 줄기 '+D.counts.stem+' · 빨간 점 = 잎자루 밑, 노란 선 = 엽축<br>'+D.boxes.map(b=>`<span style="color:${b.colour}">■</span> 잎 ${b.leaf}`).join(' ');
document.getElementById('op').oninput=e=>meshes.forEach(m=>m.material.opacity=+e.target.value);document.getElementById('showBox').onchange=e=>meshes.forEach(m=>{m.visible=e.target.checked;m.userData.edges.visible=e.target.checked;});
document.getElementById('showPts').onchange=e=>{pts.visible=e.target.checked;};
let cur=null,wantPoint=false;const rc=new THREE.Raycaster();rc.params.Points.threshold=1.5;let down=null;cv.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
async function post(o){const r=await fetch('/api/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});return r.ok?await r.json():{error:'저장 실패 (서버 없음?)'};}
cv.addEventListener('pointerup',async e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;const r=cv.getBoundingClientRect();rc.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),cam);
if(wantPoint&&cur){const h=rc.intersectObject(pts)[0];if(!h)return;const p=[P[h.index*3]+D.centre[0],P[h.index*3+1]+D.centre[1],P[h.index*3+2]+D.centre[2]];const o=await post({individual:D.individual,leaf:cur.leaf,verdict:'wrong',rachis_point_mm:p,surfel:h.index});
document.getElementById('msg').textContent=o.error||('잎 '+cur.leaf+' 틀림 + 엽축 점 저장');wantPoint=false;return;}
const hs=rc.intersectObjects(meshes.filter(m=>m.visible));if(!hs.length)return;cur=hs[0].object.userData;
document.getElementById('info').innerHTML=`<b>잎 ${cur.leaf}</b> · 서펠 ${cur.n} · 면적(b) ${cur.area[0]}–${cur.area[2]} mm²<br>겹침: ${cur.overlap.map(o=>'잎 '+o.leaf_id+' (부피 '+(o.volume_share*100).toFixed(0)+' %, 서펠 '+o.n_surfels+')').join(', ')||'없음'}<br>표시: ${cur.flags.join(', ')||'없음'}`;
document.getElementById('act').style.display='block';document.getElementById('badmsg').textContent='';});
document.getElementById('ok').onclick=async()=>{if(!cur)return;const o=await post({individual:D.individual,leaf:cur.leaf,verdict:'correct'});document.getElementById('msg').textContent=o.error||('잎 '+cur.leaf+' 맞음 저장');};
document.getElementById('bad').onclick=()=>{if(!cur)return;wantPoint=true;document.getElementById('badmsg').textContent='이제 올바른 엽축 위치의 서펠을 클릭하세요.';};
(function loop(){ctl.update();R.render(S,cam);requestAnimationFrame(loop);})();})();</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = STATE["html"].encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/api/click": self.send_response(404); self.end_headers(); return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        rec = dict(individual=STATE["ind"], leaf=int(body["leaf"]), verdict=str(body["verdict"]), rachis_point_mm=body.get("rachis_point_mm"), surfel=body.get("surfel"), provenance="user_box_click", at=time.strftime("%Y-%m-%dT%H:%M:%S"), t=time.time())
        with open(STATE["store"], "a", encoding="utf-8") as fh: fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        b = json.dumps(rec, ensure_ascii=False).encode(); self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--individual", default="R3-1b"); ap.add_argument("--port", type=int, default=8840); ap.add_argument("--write-only", action="store_true"); a = ap.parse_args(argv)
    html = HTML.replace("__DATA__", json.dumps(page_data(a.individual))); (ROOT / a.individual / "box_viewer.html").write_text(html, encoding="utf-8")
    if a.write_only: return
    STATE.update(html=html, ind=a.individual, store=ROOT / a.individual / "box_clicks.jsonl"); srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"serving http://127.0.0.1:{a.port}/ individual={a.individual}", flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()
