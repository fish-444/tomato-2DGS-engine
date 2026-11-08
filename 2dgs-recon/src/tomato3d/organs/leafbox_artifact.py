"""Read-only artifact page for the session-A compound-leaf boxes: 3D view (surfels by leaf, oriented boxes, petiole base, rachis line), per-leaf table and the method.
Writes build/p1_leafbox_20261029_01/<individual>/box_artifact.html (no server, no clicks).

usage: python -m tomato3d.organs.leafbox_artifact [--individual R3-1b]
"""

from __future__ import annotations

import argparse
import html
import json

from tomato3d.organs import leafbox_page as LP

PAGE = r"""<title>복엽 상자 R3-1b</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: specimen sheet — the 3D view as the plate, then a measurement table and the method as numbered steps (the steps are a real sequence). */
:root{--bg:#f4f6f3;--panel:#ffffff;--fg:#1d2621;--muted:#5b6a61;--line:#d5ddd7;--accent:#2f7d5b;--warn:#b5651d;--plate:#121814;
--f-body:"IBM Plex Sans KR",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121714;--panel:#1a211d;--fg:#e2e9e4;--muted:#9aaba1;--line:#2c3631;--accent:#5fbf92;--warn:#e0a060;--plate:#0b0f0d;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121714;--panel:#1a211d;--fg:#e2e9e4;--muted:#9aaba1;--line:#2c3631;--accent:#5fbf92;--warn:#e0a060;--plate:#0b0f0d;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:15px;line-height:1.6}
.wrap{max-width:1100px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px;display:grid;gap:22px}
h1{font-size:1.6rem;font-weight:600;margin:0;text-wrap:balance}h2{font-size:1.15rem;font-weight:600;margin:0 0 8px;text-wrap:balance}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}.tag{font:500 .75rem var(--f-mono);letter-spacing:.02em;border:1px solid var(--line);border-radius:4px;padding:1px 6px;color:var(--muted)}
.tag.warn{color:var(--warn);border-color:var(--warn)}
.plate{position:relative;background:var(--plate);border-radius:6px;overflow:hidden;aspect-ratio:16/10;max-width:100%}
#cv{width:100%;height:100%;display:block}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:.85rem;color:var(--muted)}
.ctl label{display:flex;gap:6px;align-items:center}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:.82rem}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
#info{font-size:.88rem;min-height:2.6em;color:var(--fg)}
.tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-weight:500;color:var(--muted);font-size:.78rem}td:first-child,th:first-child{text-align:left}tr:last-child td{border-bottom:none}
.bad{color:var(--warn);font-weight:500}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:22px}.grid2>*{min-width:0}
ol.steps{margin:0;padding-left:1.3em;display:grid;gap:8px}ol.steps li{max-width:68ch}
.note{font-size:.85rem;color:var(--muted);max-width:75ch}
code{font-family:var(--f-mono);font-size:.82em}
input[type=range]{accent-color:var(--accent)}input:focus-visible{outline:2px solid var(--accent)}
</style>
<div class="wrap">
<header><h1>복엽 상자 · R3-1b</h1><div class="tags">__TAGS__</div>
<p class="note">복엽마다 방향 있는 상자 하나를 씌운 결과입니다. 왜소종이라 잎이 겹쳐서 상자도 겹칠 수 있습니다. 상자는 대략의 그릇일 뿐이고, 서펠의 소속은 아래 규칙으로 정합니다. 사진·SAM·색은 근거로 쓰지 않았습니다.</p></header>
<section><div class="plate"><canvas id="cv" aria-label="복엽 상자 3D 보기"></canvas></div>
<div class="ctl" style="margin-top:10px"><label><input type="checkbox" id="showBox" checked> 상자</label><label><input type="checkbox" id="showPts" checked> 서펠</label><label>상자 불투명도 <input type="range" id="op" min="0" max="0.8" step="0.05" value="0.25"></label><span>끌어서 회전 · 휠로 확대 · 상자를 누르면 정보</span></div>
<div class="legend" id="leg" style="margin-top:8px"></div><div id="info" style="margin-top:6px">상자를 누르면 잎 번호, 서펠 수, 면적, 겹친 잎이 여기 나옵니다.</div></section>
<section><h2>잎별 상자</h2><div class="tbl"><table><thead><tr><th>잎</th><th>서펠</th><th>면적(b) mm² 자기 / +경합</th><th>반폭 u·v·w mm</th><th>잎자루 밑 σ mm</th><th>엽축 선 안</th><th>u / 엽축 길이</th><th>경합 면적 · 빛 몫</th><th>가장 큰 겹침</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="note" style="margin-top:8px">__SUMMARY__</p></section>
<section class="grid2"><div><h2>방법</h2><ol class="steps">
<li><b>씨앗.</b> R8_pc 의 엽신 라벨과 엽축 흡수 결과(사람 엽축 자취 기반)를 그 잎의 서펠로 씁니다. 씨앗 라벨은 바꾸지 않습니다.</li>
<li><b>상자 축.</b> u 는 그 잎 엽축 서펠의 주축(잎자루 밑 → 끝; 줄기에 가까운 끝이 밑), w 는 엽신 법선의 면적 가중 평균을 u 에 직교화한 것, v = u × w 입니다.</li>
<li><b>상자 크기.</b> 자기 서펠의 2–98 % 범위에 시트 두께 1.5 mm 와 위치 오차 1 mm 를 더합니다. 다른 여유는 없습니다.</li>
<li><b>소속.</b> R8_pc 에서 칠하지 못한 엽신마다 상자까지 거리를 잽니다. 상자 하나 안 / 여러 상자 겹침 / 밖(10 mm 안 후보)으로 나눕니다. 상자 안이라는 것만으로는 배정하지 않습니다.</li>
<li><b>규칙 순서.</b> (a) 같은 층: 그 잎 서펠과 시트 그래프로 이어져야 후보 → (b) 엽축까지 시트 그래프 경로가 2등보다 1.5 배 이상 짧으면 그 잎 → (c) 4·8·12 mm 반경 면적 밀도가 세 반경 모두 같은 잎이고 차이 ≥ 0.25 이면 그 잎 → (d) 아니면 경합(빈칸, 그래도 빛 계산에서는 가림).</li>
<li><b>반복.</b> 배정된 서펠로 상자를 다시 맞추고 소속을 다시 정합니다. 최대 4 회, 바뀐 비율이 0.5 % 미만이면 멈춥니다(이번 3 회).</li>
<li><b>검사.</b> H2 구조(엽축 선이 상자 안 ≥ 95 %, 상자 u 길이 / 엽축 선 길이 ≥ 0.9), H3 겹침 부피(몬테카를로 20,000 점), H4 경합이 가져갈 수 있는 면적·빛의 상한, H1 정확도.</li>
</ol></div>
<div><h2>결과 읽는 법</h2><ol class="steps" style="list-style:none;padding-left:0">
<li><b>태그 leafbox_weak.</b> 잎 1–7 이 H2 에서 떨어졌습니다. 엽축 선(2 mm 칸마다 중앙값)이 넓은 엽축 서펠 구름(적합 잔차 4–11 mm) 속에서 지그재그로 꺾여 길이가 부풀었기 때문입니다. 잎자루 밑–끝 직선거리로 보면 잎 1–5·7 은 0.99–1.39 입니다(참고값, 규칙 아님).</li>
<li><b>겹침이 큽니다.</b> 잎 2–6 은 부피 0.90, 3–6 0.74, 4–5 0.73 이 겹칩니다. 잎 6 상자 안 서펠의 72 % 는 다른 잎 것입니다. 경합 서펠은 겹친 곳에 몰립니다(747 대 140).</li>
<li><b>경합의 영향은 작습니다.</b> 경합이 가져갈 수 있는 몫은 면적 최대 15 %(잎 5), 흡수 빛 최대 2 %(잎 4)입니다.</li>
<li><b>정확도.</b> 사람 보류 클릭이 없어 계산할 수 없습니다. 대신 R8_pc 경계 서펠 20 % 를 숨겨 되찾는 대리 검사는 0.778 [0.768, 0.788] 로 R8_pc 의 0.833 보다 낮습니다(정답과 절차가 달라 직접 비교는 아님).</li>
<li><b>줄기 붙이기(세션 B) 입력.</b> 잎마다 잎자루 밑 좌표, 엽축 방향·선, 상자를 넘깁니다. 잎자루 밑 불확실성 σ 가 6.6–13.2 mm 라 줄기 붙이기도 그만큼 불확실하게 시작합니다.</li>
</ol></div></section>
<p class="note">척도 미검증(보드 좌표 mm, R3-1b 화분 테 맞춤 기록 없음). 지정이지 채택 아님. 계약: <code>build/p1_leafbox_20261029_01/CONTRACT.md</code> · 결과: <code>R3-1b/boxes.json</code>, <code>labels_leafbox.npz</code> · Kaggle CPU 커널에서 35 초.</p>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>(function(){const D=__DATA__;const b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));const P=new Float32Array(b64(D.pos).buffer),C=b64(D.col),N=D.n;
const cv=document.getElementById('cv');const R=new THREE.WebGLRenderer({canvas:cv,antialias:true});R.setPixelRatio(Math.min(window.devicePixelRatio||1,2));const S=new THREE.Scene();const cam=new THREE.PerspectiveCamera(45,1,1,3000);cam.up.set(0,0,1);
const ctl=new THREE.OrbitControls(cam,cv);function size(){const w=cv.clientWidth,h=cv.clientHeight;if(!w||!h)return;R.setSize(w,h,false);cam.aspect=w/h;cam.updateProjectionMatrix();}new ResizeObserver(size).observe(cv);
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(P,3));const col=new Float32Array(N*3);for(let i=0;i<N*3;i++)col[i]=C[i]/255;g.setAttribute('color',new THREE.BufferAttribute(col,3));
const pts=new THREE.Points(g,new THREE.PointsMaterial({size:1.6,vertexColors:true}));S.add(pts);let rr=0;for(let i=0;i<N;i+=7){const d=Math.hypot(P[i*3],P[i*3+1],P[i*3+2]);if(d>rr)rr=d;}const dist=rr*2.1/Math.tan(22.5*Math.PI/180)/2;cam.position.set(0,-dist*0.85,dist*0.5);
const meshes=[];for(const b of D.boxes){const geo=new THREE.BoxGeometry(2*b.half[0],2*b.half[1],2*b.half[2]);const m=new THREE.Mesh(geo,new THREE.MeshBasicMaterial({color:b.colour,transparent:true,opacity:0.25,depthWrite:false}));
const M4=new THREE.Matrix4();const r=b.R;M4.set(r[0][0],r[1][0],r[2][0],b.centre[0],r[0][1],r[1][1],r[2][1],b.centre[1],r[0][2],r[1][2],r[2][2],b.centre[2],0,0,0,1);m.applyMatrix4(M4);m.userData=b;S.add(m);meshes.push(m);
const ed=new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color:b.colour}));ed.applyMatrix4(M4);S.add(ed);m.userData.edges=ed;
S.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(b.poly.map(p=>new THREE.Vector3(p[0],p[1],p[2]))),new THREE.LineBasicMaterial({color:0xffd400})));
const sp=new THREE.Mesh(new THREE.SphereGeometry(1.8,12,12),new THREE.MeshBasicMaterial({color:0xff3030}));sp.position.set(b.base[0],b.base[1],b.base[2]);S.add(sp);}
document.getElementById('leg').innerHTML=D.boxes.map(b=>`<span><i class="sw" style="background:${b.colour}"></i>잎 ${b.leaf}</span>`).join('')+`<span><i class="sw" style="background:#808080"></i>경합 ${D.counts.contested}</span><span><i class="sw" style="background:#373737"></i>줄기 ${D.counts.stem}</span><span><i class="sw" style="background:#afafaf"></i>못 칠한 엽축 ${D.counts.rachis_unpainted}</span><span><i class="sw" style="background:#ff3030"></i>잎자루 밑</span><span><i class="sw" style="background:#ffd400"></i>엽축 선</span>`;
document.getElementById('op').oninput=e=>meshes.forEach(m=>m.material.opacity=+e.target.value);document.getElementById('showBox').onchange=e=>meshes.forEach(m=>{m.visible=e.target.checked;m.userData.edges.visible=e.target.checked;});document.getElementById('showPts').onchange=e=>{pts.visible=e.target.checked;};
const rc=new THREE.Raycaster();let down=null;cv.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});cv.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;const r=cv.getBoundingClientRect();
rc.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),cam);const hs=rc.intersectObjects(meshes.filter(m=>m.visible));if(!hs.length)return;const c=hs[0].object.userData;
document.getElementById('info').innerHTML=`<b>잎 ${c.leaf}</b> · 서펠 ${c.n.toLocaleString()} · 면적(b) ${c.area[0]}–${c.area[2]} mm² · 겹침: ${c.overlap.map(o=>'잎 '+o.leaf_id+' '+(o.volume_share*100).toFixed(0)+'%').join(', ')||'없음'}${c.flags.length?' · 표시: '+c.flags.join(', '):''}`;});
const still=window.matchMedia('(prefers-reduced-motion: reduce)').matches;size();(function loop(){ctl.update();R.render(S,cam);requestAnimationFrame(loop);})();})();</script>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--individual", default="R3-1b"); a = ap.parse_args(argv)
    d = LP.ROOT / a.individual; data = LP.page_data(a.individual); bj = json.load(open(d / "boxes.json", encoding="utf-8")); res = json.load(open(d / "results.json", encoding="utf-8"))
    rows = []
    for lf in bj["leaves"]:
        k = str(lf["leaf_id"]); h2 = res["H2"][k]; h4 = res["H4"][k]; big = max(lf["overlap_with"], key=lambda o: o["volume_share"]) if lf["overlap_with"] else None
        pin = h2["polyline_inside_share"]; ur = h2["u_extent_over_rachis_length"]; big_txt = f"잎 {big['leaf_id']} ({big['volume_share']:.2f})" if big else "—"
        rows.append(f"<tr><td>잎 {k}</td><td>{lf['n_surfels']:,}</td><td>{lf['area_b_mm2_range'][0]:,.1f} / {lf['area_b_mm2_range'][2]:,.1f}</td><td>{' · '.join(f'{x:.1f}' for x in lf['box']['half_extent_mm'])}</td>"
                    f"<td>{lf['petiole_base_sigma_mm']:.1f}</td><td class=\"{'bad' if pin < 0.95 else ''}\">{pin:.2f}</td><td class=\"{'bad' if (ur or 0) < 0.9 else ''}\">{ur:.2f}</td>"
                    f"<td>{h4['contested_area_share'] * 100:.1f} % · {h4['contested_light_share'] * 100:.1f} %</td><td>{big_txt}</td></tr>")
    tags = "".join(f'<span class="tag{" warn" if t.startswith("leafbox_weak") else ""}">{html.escape(t)}</span>' for t in res["labels"] + [res["tag"]])
    summ = (f"대상(R8_pc 에서 못 칠한 엽신) {res['targets']:,} 중 배정 {res['assigned']:,}, 경합 {res['contested']:,}, 상자 밖 {res['outside']}. 경합 중 817 은 같은 층으로 이어진 후보가 없는 경우입니다. "
            f"빨간 글씨 = H2 기준 미달.")
    out = PAGE.replace("__DATA__", json.dumps(data)).replace("__ROWS__", "".join(rows)).replace("__TAGS__", tags).replace("__SUMMARY__", summ)
    (d / "box_artifact.html").write_text(out, encoding="utf-8"); print(d / "box_artifact.html", len(out))


if __name__ == "__main__":
    main()
