"""Labelling page for ONE individual of the all-individuals run (contract build/p1_all_compound_20261026_01, Amendment 1 item 6).

Same page as the R3-1b one (`userlabel_page`): leaflet masks from the UNION, click = leaf number, SAM point for a missed leaflet, split, undo, "모름".
Additions: leaf-number legend; faint numbered outlines of the leaves labelled so far on the OTHER training views of the same individual (V-H, hints only,
the user's click decides; the held-out view never feeds a hint); labels saved append-only in <individual>/labels/leaf_labels.json with an `individual` field.
No rachis hints (no traces exist for these plants). The session never edits records.

usage: python -m tomato3d.organs.all_label_page --individual n_04 [--port 8830] [--no-sam]   (127.0.0.1 only)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from collections import Counter
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.organs import all_compound as AC
from tomato3d.organs import userlabel as UL
from tomato3d.organs import userlabel_page as UP
from tomato3d.organs import vote2d as V2

H2 = UL.H2
STATE = UP.STATE
STATE.update(iid=None, dir=None, hint_cache={})


# ---------------------------------------------------------------- records (append-only, per individual)
def labels_file():
    return STATE["dir"] / "labels" / "leaf_labels.json"


def load_labels():
    f = labels_file(); f.parent.mkdir(exist_ok=True)
    return json.load(open(f, encoding="utf-8")) if f.exists() else {}


def save_record(rec):
    with STATE["lock"]:
        recs = load_labels(); rid = f"{rec['view']}_{int(time.time() * 1000):x}_{len(recs)}"
        rec = dict(rec, id=rid, at=time.strftime("%Y-%m-%dT%H:%M:%S"), individual=STATE["iid"], pipeline="R3-1b-style", status=AC.LABELS["status"]); recs[rid] = rec
        tmp = labels_file().with_suffix(".tmp"); json.dump(recs, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=0); tmp.replace(labels_file())
    return rec


def detect_doc(vid):
    return json.load(open(STATE["dir"] / "detect" / f"{vid}.json", encoding="utf-8"))


def label_image(vid, recs):
    det = detect_doc(vid); by_id = {m["id"]: m for m in det["masks"]}; active = {}
    for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
        if r.get("removed"):
            active.pop(r["removed"], None); continue
        active[r.get("mask_id") or r["id"]] = r
    img = np.zeros((H2, H2), np.int16)
    for r in active.values():
        if r["leaf"] in ("모름", "unknown", None):
            continue
        m = UL.record_rle(r, recs, by_id)
        if m is not None:
            img[m] = int(r["leaf"])
    return img, len(active)


# ---------------------------------------------------------------- numbering aid
def surfel_visibility(view):
    pv = STATE["proj"].get(view)
    if pv is None:
        pl = STATE["pl"]; pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, 1512); STATE["proj"][view] = pv
    return pv


def vh_labels(sources):
    """Leaf number per surfel from finished (non held-out) views: mode over the views that see it (V_min 1)."""
    recs = load_labels(); n = len(STATE["pl"]["xyz"]); votes = np.zeros((n, 13), np.int16)
    for v in sources:
        stem = Path(v).stem; img, cnt = label_image(stem, recs)
        if cnt == 0:
            continue
        pv = surfel_visibility(v); vis = pv["vis"]; lab = img[np.clip(pv["w"][vis] * H2 // 1512, 0, H2 - 1), np.clip(pv["u"][vis] * H2 // 1512, 0, H2 - 1)]
        idx = np.nonzero(vis)[0]; ok = (lab > 0) & (lab <= 12); votes[idx[ok], lab[ok]] += 1
    best = votes.argmax(1); best[votes.max(1) == 0] = 0; return best


def hints_for(vid, view, page_scale_grid):
    import cv2
    sources = [v for v in STATE["views"] if v != view and v != STATE["held_view"]]
    lab = vh_labels(sources)
    if not (lab > 0).any():
        return []
    pv = surfel_visibility(view); vis = pv["vis"]; sx = 1512; out = []
    for L in sorted(set(lab[lab > 0].tolist())):
        sel = vis & (lab == L)
        if sel.sum() < 15:
            continue
        ras = np.zeros((1512, 1512), np.uint8); ras[np.clip(pv["w"][sel], 0, 1511), np.clip(pv["u"][sel], 0, 1511)] = 255
        ras = cv2.morphologyEx(cv2.dilate(ras, np.ones((9, 9), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(ras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:2]:
            if cv2.contourArea(c) < 150:
                continue
            a = cv2.approxPolyDP(c, 3, True).reshape(-1, 2).astype(float) * page_scale_grid * (H2 / 1512); pts = np.vstack([a, a[:1]])
            out.append(dict(leaf=int(L), pts=pts.round(1).tolist()))
    return out


# ---------------------------------------------------------------- payload / sam
def view_payload(vid):
    det = detect_doc(vid); view = det["view"]
    img = Image.open(STATE["dir"] / "undist_3024" / view).convert("RGB"); W0 = img.size[0]; img.thumbnail((UP.PAGE_PX, UP.PAGE_PX)); W, Hh = img.size; sc = W / H2
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=80)
    masks = [dict(id=m["id"], prop=m["prop"], score=m["score"], polys=UP.polys_of(UL.unrle(m["rle"]), sc)) for m in det["masks"]]
    recs = [r for r in load_labels().values() if r["view"] == vid]
    for r in recs:
        if "rle" in r and "polys" not in r:
            r["polys"] = UP.polys_of(UL.unrle(r["rle"]), sc)
    return dict(vid=vid, view=view, jpg=base64.b64encode(buf.getvalue()).decode(), W=W, H=Hh, scale_page_per_full=W / W0, masks=masks, hints=hints_for(vid, view, sc),
                records=recs, held_out=(vid == STATE["held"]), individual=STATE["iid"], ring=STATE["rep"]["geometry"][view]["ring"], azimuth=STATE["rep"]["geometry"][view]["azimuth"])


def sam_set_view(vid):
    if STATE["sam"] is None:
        return None
    if STATE["cur_view"] != vid:
        rgb = STATE["rgb"].get(vid)
        if rgb is None:
            rgb = np.asarray(Image.open(STATE["dir"] / "undist_3024" / (vid + ".jpg")).convert("RGB")); STATE["rgb"][vid] = rgb
        STATE["sam"].predictor.set_image(rgb); STATE["cur_view"] = vid
    return STATE["rgb"][vid]


def api_split(body):
    vid = body["vid"]; k = body["full_per_page"]; by_id = {m["id"]: m for m in detect_doc(vid)["masks"]}
    parent = UL.unrle(by_id[body["mask_id"]]["rle"]) if body["mask_id"] in by_id else UL.unrle(body["rle"])
    pts = np.array([[p[0] * k, p[1] * k] for p in body["points"]], np.float32); ys, xs = np.nonzero(parent); gf = 3024 / H2 if parent.any() else 1.0
    box = [xs.min() * gf, ys.min() * gf, (xs.max() + 1) * gf, (ys.max() + 1) * gf] if parent.any() else None; out = []
    with STATE["lock"]:
        if sam_set_view(vid) is None:
            return dict(error="SAM unavailable on this server")
        for i in range(len(pts)):
            g, s, q = UP.sam_mask(pts[i:i + 1], np.delete(pts, i, 0), box); g = g & parent
            if g.sum() >= 30:
                out.append(dict(rle=UL.rle(g), polys=UP.polys_of(g, body["page_per_grid"]), score=round(q, 3), area_px=int(g.sum())))
    return dict(parts=out)


def api_point(body):
    vid = body["vid"]; k = body["full_per_page"]; pos = [[body["x"] * k, body["y"] * k]]; neg = [[p[0] * k, p[1] * k] for p in body.get("negatives", [])]
    with STATE["lock"]:
        if sam_set_view(vid) is None:
            return dict(error="SAM unavailable on this server")
        g, s, q = UP.sam_mask(pos, neg)
    return dict(rle=UL.rle(g), polys=UP.polys_of(g, body["page_per_grid"]), score=round(q, 3), area_px=int(g.sum()))


# ---------------------------------------------------------------- page text (the R3-1b page with the R3-1b-only parts replaced)
def page_html():
    h = UP.HTML
    h = h.replace("<title>복엽 라벨</title>", "<title>복엽 라벨 (개체별)</title>")
    h = h.replace("R3-1b_unaccepted · 사람 라벨 (잎 번호는 사람만 정함) · 아래 잎부터 1, 2, … (엽축 추적·자 측정과 같은 번호) · 보류 사진은 채점에만 쓰임",
                  "R3-1b_unaccepted · R3-1b-style · conditional_on_unvalidated_scale · 사람 라벨 (잎 번호는 사람만 정함) · 번호는 아래 잎부터 1, 2, … (자 측정표와 같은 번호) · 같은 잎은 사진이 바뀌어도 같은 번호 · 보류 사진은 채점에만 쓰임 · 연한 점선 번호 = 앞서 칠한 사진에서 투영한 힌트일 뿐, 클릭이 정답")
    h = h.replace("엽축 힌트", "번호 힌트 (앞 사진에서)")
    h = h.replace("'엽축 '+h.leaf", "'힌트 '+h.leaf").replace("rgba(255,212,0,.55)", "rgba(255,255,255,.45)").replace("rgba(255,212,0,.8)", "rgba(255,255,255,.7)")
    h = h.replace("사진 ${i+1}/${VIDS.length} · ${V.vid}${V.held_out?' (보류: 채점용)':''}", "${V.individual} · 사진 ${i+1}/${VIDS.length} · ${V.vid} · ${V.ring} 고리 · 방위 ${V.azimuth}°${V.held_out?' (보류: 채점용)':''}")
    return h


def make_handler():
    Base = UP.H

    class Hh(Base):
        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                b = page_html().encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            if self.path == "/api/state":
                return self._json(dict(vids=STATE["vids"], held_out=STATE["held"], sam=STATE["sam"] is not None, individual=STATE["iid"]))
            if self.path.startswith("/api/view/"):
                return self._json(view_payload(self.path.rsplit("/", 1)[1]))
            self._json(dict(error="not found"), 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/api/label":
                    return self._json(save_record(body))
                if self.path == "/api/sam_point":
                    return self._json(api_point(body))
                if self.path == "/api/split":
                    return self._json(api_split(body))
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
            self._json(dict(error="not found"), 404)
    return Hh


def warm(vids):
    for vid in vids:
        with STATE["lock"]:
            try:
                sam_set_view(vid)
            except Exception as e:
                print("warm failed", vid, e, flush=True)
    print("sam warm done", flush=True)


def setup(iid, sam=True):
    d = AC.OUT / iid; rep = json.load(open(d / "views_chosen.json", encoding="utf-8")); STATE.update(iid=iid, dir=d, rep=rep)
    STATE["views"] = rep["train"] + [rep["held_out"]]; STATE["vids"] = [Path(v).stem for v in STATE["views"]]; STATE["held"] = Path(rep["held_out"]).stem; STATE["held_view"] = rep["held_out"]
    STATE["pl"] = AC.load_individual(iid); STATE["proj"] = {}
    if sam:
        from tomato3d.leafseg import run as r2
        from tomato3d.topview.measure2d import load_sam
        STATE["sam"] = load_sam(r2.SAMCFG); threading.Thread(target=warm, args=(STATE["vids"],), daemon=True).start()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--individual", required=True); ap.add_argument("--port", type=int, default=8830); ap.add_argument("--no-sam", action="store_true"); a = ap.parse_args(argv)
    setup(a.individual, not a.no_sam); srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler())
    print(f"serving http://127.0.0.1:{a.port}/  {a.individual} views {STATE['vids']} sam={'on' if STATE['sam'] else 'off'}", flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()
