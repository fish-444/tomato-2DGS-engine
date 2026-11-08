"""User 3D grouping page (contract Amendment A1, step 4d): R3-1b blade surfels coloured by 3D patch (or 4c group); the user selects patches and presses a
leaf-number button. Assignments are appended (never rewritten) to build/p1_stacked_sheets_retry_20261026_01/oversegment/user_3d_group_click.jsonl with
provenance `user_3d_group_click`. Read-only on every input. Serves 127.0.0.1:8833 only.

usage: python -m tomato3d.organs.group_page [--unit patch|group] [--port 8833]
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import light_viewer as LV
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2

OUT = SR.OUT / "oversegment"; STORE = OUT / "user_3d_group_click.jsonl"
STATE = {}


def page_data(unit):
    pl = UL.load_plant(); b = np.nonzero(pl["blade"])[0]; z = np.load(OUT / "groups_4c.npz"); unit_id = z["patch"] if unit == "patch" else z["group"]
    xyz = pl["xyz"]; c = xyz.mean(0); ids = np.full(len(xyz), -1); ids[b] = unit_id
    units = sorted(set(int(x) for x in unit_id if x >= 0)); pal = LV.palette(max(len(units), 1)); col = np.full((len(xyz), 3), 70, np.uint8)
    for k, u in enumerate(units): col[ids == u] = pal[k]
    col[~pl["blade"]] = (45, 45, 45)
    traces = [dict(leaf=t["leaf"], pts=(np.asarray(t["pts"]) - c).round(5).tolist()) for t in CC.load_traces(V2.OUT / "traces_export" / "rachis_traces")]
    return dict(unit=unit, n=int(len(xyz)), units=len(units), pos=base64.b64encode(((xyz - c) * 1000).astype(np.float32).tobytes()).decode(),
                col=base64.b64encode(col.tobytes()).decode(), ids=base64.b64encode(ids.astype(np.int32).tobytes()).decode(), traces=traces,
                palette={str(L): "#%02x%02x%02x" % V2.PAL.get(L, (200, 200, 200)) for L in range(1, 13)},
                note="R3-1b_unaccepted · conditional_on_unvalidated_scale · gate_failed_viewing_only — 3D patches (4a); numbers come only from your clicks", layer=proposal_layer(pl))


def proposal_layer(pl):
    """Read-only colours: R7 labels, R7 + proposals (revision 1) and, when present, R7 + proposals with the 3D-surroundings evidence (revision 2)."""
    d0 = SR.ROOT / "build/p1_local_diffusion_20261027_01/proposal_blue_leaf_2026-10-27"
    if not (d0 / "proposal.npz").exists(): return None
    out = one_layer(pl, d0); d3 = d0 / "with_3d_surroundings"
    if (d3 / "proposal.npz").exists():
        o3 = one_layer(pl, d3); out.update(prop3=o3["prop"], kind3=o3["kind"], panel3=o3["panel"])
    r8 = SR.ROOT / "build/p1_pc_absorb_20261027_01/R8_pc/labels.npz"
    if r8.exists(): out.update(r8_layer(pl, r8))
    rx = SR.ROOT / "build/p1_pc_absorb_20261027_01/R8_pc_rachis/labels_rachis.npz"
    if rx.exists(): out.update(rx_layer(pl, rx))
    return out


def rx_layer(pl, f):
    """R8_pc_rachis: rachis-like surfels absorbed into the leaves (darker leaf colour); differences view (trace seeds / walker / still unpainted)."""
    z = np.load(f); lab = z["label"].astype(int); r8 = z["r8_pc"].astype(int); ax = z["axis_kind"]; tg = z["target"]; src = z["source"]; n = len(lab)
    c = np.full((n, 3), 128, np.uint8); c[ax == 1] = (60, 60, 60); c[(ax == 2) & (lab <= 0)] = (165, 165, 165); c[(lab <= 0) & (ax == 0)] = (255, 255, 255)
    for L in np.unique(lab[lab > 0]):
        c3 = np.array(V2.PAL.get(int(L), (220, 220, 220))); c[(lab == L) & ~tg] = c3; c[(lab == L) & tg] = (c3 * 0.6).astype(np.uint8)
    d = np.full((n, 3), 70, np.uint8); d[ax == 1] = (40, 40, 40); ts = tg & (src == 2); wk = tg & (src > 2) & (src < 10) & (lab > 0); un = tg & (lab <= 0)
    d[ts] = (30, 150, 60); d[wk] = (60, 220, 90); d[un] = (255, 160, 0)
    enc = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    panel = (f"엽축 흡수 (R8_pc 위, 사진·SAM 미사용) · 잎 색 어둡게 = 그 잎의 엽축으로 흡수 {int((tg & (lab > 0)).sum())} · 차이 색: <span style='color:#1e963c'>■</span> 사람 엽축 3 mm 안 {int(ts.sum())} · "
             f"<span style='color:#3cdc5a'>■</span> 걷기 흡수 {int(wk.sum())} · <span style='color:#ffa000'>■</span> 못 칠함 {int(un.sum())} · 연회색 = 못 칠한 엽축")
    return dict(rx=enc(c), rxdiff=enc(d), rxpanel=panel)


def r8_layer(pl, f):
    """R8_pc (point-cloud absorption) colours and the R7-vs-R8 difference colours (newly painted / label changed / still unpainted)."""
    z = np.load(f); lab = z["label"].astype(int); r7 = z["r7"].astype(int); ax = np.load(SR.ROOT / "build/p1_compound_userlabel_20261025_01/map_v3_charuco/user_labelled_propagated/surfel_clusters.npz")["axis_kind"]; n = len(pl["xyz"])
    c = np.full((n, 3), 128, np.uint8); c[ax == 1] = (60, 60, 60); c[ax == 2] = (165, 165, 165); c[(lab <= 0) & (ax == 0)] = (255, 255, 255)
    for L in np.unique(lab[lab > 0]): c[lab == L] = V2.PAL.get(int(L), (220, 220, 220))
    d = np.full((n, 3), 70, np.uint8); d[ax == 1] = (40, 40, 40); d[(ax == 2) & (lab <= 0)] = (110, 110, 110)
    new = (lab > 0) & (r7 <= 0); chg = (lab > 0) & (r7 > 0) & (lab != r7); still = (lab <= 0) & (r7 <= 0) & ((ax == 0) | z["node"]); lost = (lab <= 0) & (r7 > 0)
    d[new] = (60, 220, 90); d[chg] = (230, 50, 200); d[still] = (255, 255, 255); d[lost] = (255, 160, 0)
    enc = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    panel = (f"점군 흡수 R8_pc (세트 {str(z['set_name'])}, 사진·SAM 미사용) · 차이 색: <span style='color:#3cdc5a'>■</span> 새로 칠함 {int(new.sum())} · <span style='color:#e632c8'>■</span> R7 과 다른 잎 {int(chg.sum())} · "
             f"<span style='color:#ffa000'>■</span> R7 에선 칠했는데 R8 미배정 {int(lost.sum())} · □ 둘 다 미배정 {int(still.sum())} · 어두운 회색 = 같음")
    return dict(r8=enc(c), r8diff=enc(d), r8panel=panel)


def one_layer(pl, d):
    z = np.load(d / "proposal.npz"); res = json.load(open(d / "result.json", encoding="utf-8")); ax = z["axis_kind"]; n = len(pl["xyz"])
    def colours(lab):
        c = np.full((n, 3), 128, np.uint8); c[ax == 1] = (60, 60, 60); c[ax == 2] = (165, 165, 165); c[(lab <= 0) & (ax == 0)] = (255, 255, 255)
        for L in np.unique(lab[lab > 0]): c[lab == L] = V2.PAL.get(int(L), (220, 220, 220))
        return c
    before = colours(z["r7_label"]); after = colours(z["proposed_label"]); k = z["proposal_kind"]
    for L in np.unique(z["proposed_label"][k > 0]):
        m = (k > 0) & (z["proposed_label"] == L); after[m] = (np.array(V2.PAL.get(int(L), (220, 220, 220))) * 0.55 + 255 * 0.45).astype(np.uint8)
    rows = []
    for L, pc in res["per_candidate"].items():
        rows.append(f"<b>잎 {L}</b> 둘레 12 mm 미배정 {pc['region']['total']} (흰 {pc['region']['white']} · 축1 {pc['region']['axis1']} · 축2 {pc['region']['axis2']})<br>제안 (a) 축2→엽신 {pc['proposals']['a_axis2_to_blade']} · (b) 흰색 엄격 규칙 통과 {pc['proposals']['b_white_strict']}")
        rows += [f"· {g['group']}: {g['n']}개, {g['area_mm2']} mm², 사용자 가중 통과 {g['user']['n_pass']} / 사진 우선 {g['photo_first']['n_pass']}" for g in pc["groups"]]
    enc = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    return dict(r7=enc(before), prop=enc(after), kind=enc(k.astype(np.uint8)), panel="<br>".join(rows))


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>3D 잎 묶기</title>
<style>:root{--bg:#111;--fg:#eee}body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}#hd{position:fixed;top:0;left:0;right:0;background:#7a1f1f;font-size:12px;padding:4px 10px;z-index:3}
#side{position:fixed;top:26px;right:0;width:270px;bottom:0;background:#1b1b1b;padding:8px;overflow:auto;z-index:2}button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:5px;padding:4px 8px;margin:2px;cursor:pointer;font:inherit}
.lf{min-width:34px}button.on{background:#555;border-color:#ddd}#cv{position:fixed;top:26px;left:0;right:270px;bottom:0}#msg{font-size:12px;opacity:.85;margin-top:6px}@media(max-width:700px){#side{width:100%;top:auto;height:40%}#cv{right:0;bottom:40%}}</style></head><body>
<div id="hd"></div><canvas id="cv"></canvas><div id="side"><div>패치 클릭(Shift = 여러 개) → 잎 번호 버튼 (아래 잎부터 1, 2, …; 엽축 번호와 같은 번호 체계를 쓰세요)</div><div id="btns"></div>
<div id="lyrow" style="margin:4px 0"></div><div id="propinfo" style="font-size:11px;opacity:.85;max-height:220px;overflow:auto"></div><button id="clr">선택 해제</button><label><input type="checkbox" id="showall" checked> 번호 붙은 패치 색 = 잎 색</label><div id="sel"></div><div id="cnt"></div><div id="msg"></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>(async function(){const D=await (await fetch('/api/data')).json();document.getElementById('hd').textContent=D.note+' · 단위 '+(D.unit==='patch'?'패치':'그룹')+' '+D.units+'개';
const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));const P=new Float32Array(b64(D.pos).buffer),C0=b64(D.col),ID=new Int32Array(b64(D.ids).buffer);const N=D.n;
const cv=document.getElementById('cv');const R=new THREE.WebGLRenderer({canvas:cv,antialias:true});const S=new THREE.Scene();const cam=new THREE.PerspectiveCamera(45,1,1,2000);cam.position.set(0,-220,140);cam.up.set(0,0,1);
const ctl=new THREE.OrbitControls(cam,cv);function size(){const w=cv.clientWidth,h=cv.clientHeight;R.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}window.addEventListener('resize',size);size();
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(P,3));const col=new Float32Array(N*3);g.setAttribute('color',new THREE.BufferAttribute(col,3));
const pts=new THREE.Points(g,new THREE.PointsMaterial({size:1.6,vertexColors:true}));S.add(pts);
for(const t of D.traces){const lg=new THREE.BufferGeometry().setFromPoints(t.pts.map(p=>new THREE.Vector3(p[0]*1000,p[1]*1000,p[2]*1000)));S.add(new THREE.Line(lg,new THREE.LineBasicMaterial({color:0xffd400})));}
let A={},sel=new Set();const st=await (await fetch('/api/assign')).json();for(const r of st)A[r.unit_id]=r.leaf;
function hex(h){return [parseInt(h.slice(1,3),16)/255,parseInt(h.slice(3,5),16)/255,parseInt(h.slice(5,7),16)/255];}
let LY='patch';const CR7=D.layer?b64(D.layer.r7):null,CPR=D.layer?b64(D.layer.prop):null,PK=D.layer?b64(D.layer.kind):null,CP3=D.layer&&D.layer.prop3?b64(D.layer.prop3):null,PK3=D.layer&&D.layer.kind3?b64(D.layer.kind3):null,CR8=D.layer&&D.layer.r8?b64(D.layer.r8):null,CR8D=D.layer&&D.layer.r8diff?b64(D.layer.r8diff):null,CRX=D.layer&&D.layer.rx?b64(D.layer.rx):null,CRXD=D.layer&&D.layer.rxdiff?b64(D.layer.rxdiff):null;
if(D.layer){const ly=document.getElementById('lyrow');ly.innerHTML='<button data-ly="patch" class="on">패치 (클릭용)</button><button data-ly="r7">R7 라벨</button><button data-ly="prop">제안 보기</button>'+(CP3?'<button data-ly="prop3">제안 보기 (3D 주위)</button>':'')+(CR8?'<div style="font-size:11px;margin-top:4px">점군 흡수 (기하+색)</div><button data-ly="r7">R7</button><button data-ly="r8">R8_pc</button><button data-ly="r8d">차이만</button>':'')+(CRX?'<div style="font-size:11px;margin-top:4px">엽축 흡수 (R8_pc 위)</div><button data-ly="rx">엽축 흡수 후</button><button data-ly="rxd">엽축 차이만</button>':'')+'<label style="display:block"><input type="checkbox" id="hlprop"> 제안 노란색 강조</label>';
ly.querySelectorAll('button').forEach(x=>x.onclick=()=>{LY=x.dataset.ly;ly.querySelectorAll('button').forEach(y=>y.classList.toggle('on',y===x));y0();paint();});document.getElementById('hlprop').onchange=()=>paint();}
function y0(){document.getElementById('propinfo').innerHTML=(LY==='patch'||!D.layer)?'':'읽기 전용 · 흰색 = 미배정 엽신, 진회색 = 축1(줄기), 연회색 = 축2(엽축), '+(LY!=='r7'?'연한 색 = 제안':'R7 그대로')+'<br>'+(LY==='rx'||LY==='rxd'?D.layer.rxpanel:(LY==='r8'||LY==='r8d'?D.layer.r8panel:(LY==='prop3'?D.layer.panel3:D.layer.panel)));}
function paint(){const all=document.getElementById('showall').checked;const K0=LY==='patch'?C0:(LY==='r7'?CR7:(LY==='prop3'?CP3:(LY==='r8'?CR8:(LY==='r8d'?CR8D:(LY==='rx'?CRX:(LY==='rxd'?CRXD:CPR))))));const KK=LY==='prop3'?PK3:PK;const HL=(LY==='prop'||LY==='prop3')&&document.getElementById('hlprop')&&document.getElementById('hlprop').checked;for(let i=0;i<N;i++){let r=K0[i*3]/255,gg=K0[i*3+1]/255,bb=K0[i*3+2]/255;const u=ID[i];if(HL&&KK[i]>0){r=1;gg=0.9;bb=0;}if(LY==='patch'&&u>=0&&all&&A[u]!==undefined&&A[u]!=='모름'){[r,gg,bb]=hex(D.palette[A[u]]);}if(u>=0&&sel.has(u)){r=1;gg=1;bb=1;}col[i*3]=r;col[i*3+1]=gg;col[i*3+2]=bb;}g.attributes.color.needsUpdate=true;
document.getElementById('sel').textContent='선택: '+[...sel].join(', ');const done=Object.keys(A).length;document.getElementById('cnt').textContent='번호 붙인 단위 '+done+' / '+D.units;}
const bt=document.getElementById('btns');for(let k=1;k<=12;k++){const x=document.createElement('button');x.className='lf';x.textContent=k;x.style.borderColor=D.palette[k];x.onclick=()=>assign(k);bt.appendChild(x);}
const u=document.createElement('button');u.textContent='모름';u.onclick=()=>assign('모름');bt.appendChild(u);
async function assign(k){if(!sel.size){document.getElementById('msg').textContent='먼저 패치를 클릭';return;}for(const s of sel){const r=await (await fetch('/api/assign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({unit:D.unit,unit_id:s,leaf:k})})).json();A[s]=k;}
document.getElementById('msg').textContent=sel.size+'개 → 잎 '+k;sel.clear();paint();}
document.getElementById('clr').onclick=()=>{sel.clear();paint();};document.getElementById('showall').onchange=paint;
const rc=new THREE.Raycaster();rc.params.Points.threshold=1.5;let down=null;cv.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
cv.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;const r=cv.getBoundingClientRect();const m=new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
rc.setFromCamera(m,cam);const h=rc.intersectObject(pts)[0];if(!h)return;const id=ID[h.index];if(id<0){document.getElementById('msg').textContent='미배정 점 (패치 아님)';return;}
if(!e.shiftKey)sel.clear();sel.has(id)?sel.delete(id):sel.add(id);document.getElementById('msg').textContent='패치 '+id+(A[id]!==undefined?' (현재 잎 '+A[id]+')':'');paint();});
paint();(function loop(){ctl.update();R.render(S,cam);requestAnimationFrame(loop);})();window.__gp={sel:()=>[...sel],A:()=>A,pick:(id)=>{sel.add(id);paint();},assign,units:D.units};})();</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _j(self, o, code=200):
        b = json.dumps(o, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if self.path == "/api/data": return self._j(STATE["data"])
        if self.path == "/api/assign":
            recs = [json.loads(l) for l in STORE.read_text(encoding="utf-8").splitlines() if l.strip()] if STORE.exists() else []
            last = {}
            for r in recs:
                if r.get("unit") == STATE["data"]["unit"]: last[r["unit_id"]] = r
            return self._j(list(last.values()))
        self._j(dict(error="not found"), 404)

    def do_POST(self):
        if self.path != "/api/assign": return self._j(dict(error="not found"), 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        rec = dict(unit=body["unit"], unit_id=int(body["unit_id"]), leaf=body["leaf"], provenance="user_3d_group_click", at=time.strftime("%Y-%m-%dT%H:%M:%S"), t=time.time())
        with open(STORE, "a", encoding="utf-8") as fh: fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._j(rec)

    def log_message(self, *a):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--unit", choices=("patch", "group"), default="patch"); ap.add_argument("--port", type=int, default=8833); ap.add_argument("--store", type=Path, default=None); a = ap.parse_args(argv)
    global STORE
    if a.store: STORE = a.store                      # tests only: never the user record file
    STATE["data"] = page_data(a.unit); srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); print(f"serving http://127.0.0.1:{a.port}/ unit={a.unit} units={STATE['data']['units']}", flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()
