"""Disc viewer of the light-organs run: surfels coloured by compound leaf (or by leaflet), with the normal and height colour
modes of build/p1_recon_r3-1_contam_20261022_01/viewer/r3-1b_viewer_height.html (0 = pot-top plane). The template is that
file; only its data block, labels and banner are replaced.

usage: python -m tomato3d.organs.light_viewer --run <run dir> [--surfels ply] [--template html]
Writes <run>/viewer/light_organs_viewer.html (by compound leaf) and light_organs_viewer_leaflets.html (by leaflet).
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import csv
import json
import re
from pathlib import Path

import numpy as np

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs.light_organs import DEFAULTS, POT_TOP_Z

TEMPLATE = DEFAULTS["surfels"].parents[1] / "viewer" / "r3-1b_viewer_height.html"
GREY = (118, 124, 128)      # axis / unclustered blade surfels
UNASSIGNED = (235, 235, 235)


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def palette(n):
    """n distinct colours (golden-angle hues, two lightness bands)."""
    out = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0; l = 0.55 if i % 2 == 0 else 0.42; r, g, b = colorsys.hls_to_rgb(h, l, 0.85)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def read_csv(p):
    lines = Path(p).read_text(encoding="utf-8").splitlines()
    return list(csv.DictReader(lines[1:]))


def build(run: Path, surfels: Path, template: Path, by: str, status: str):
    sf = sio.read_surfels_full(surfels); n = len(sf["xyz"])
    lab = np.load(run / "surfel_labels.npz", allow_pickle=False); rows = lab["rows"]; cl = lab["cl"]; bucket = lab["bucket"]
    leaflet_of_cl = lab["leaflet_id"]; leaf_of_cl = lab["leaf_id"]
    leaves = read_csv(run / "compound_leaves.csv"); leaflets = read_csv(run / "leaflets.csv")
    keys = [L["leaf_id"] for L in leaves] if by == "leaf" else [x["leaflet_id"] for x in leaflets]
    pal = dict(zip(keys, palette(len(keys))))
    col = np.tile(np.array(GREY, np.uint8), (n, 1)); cls = np.zeros(n, np.uint8); cls[rows] = 1
    for i, (r, c) in enumerate(zip(rows, cl)):
        if c < 0:
            continue
        k = leaf_of_cl[c] if by == "leaf" else leaflet_of_cl[c]
        col[r] = pal.get(k, UNASSIGNED) if k != "unassigned" else UNASSIGNED
    html = template.read_text(encoding="utf-8")
    m = re.search(r'<script id="D" type="application/json">(.*?)</script>', html, re.S); M = json.loads(m.group(1))
    lo = sf["xyz"].min(0); rng = np.maximum(sf["xyz"].max(0) - lo, 1e-9)
    M["discs"] = dict(n=n, n_plant=int(cls.sum()), H=float(rng[2]), lo=lo.tolist(), rng=rng.tolist(), pos=_b64(np.round((sf["xyz"] - lo) / rng * 65535).astype(np.uint16)),
                      tu=_b64(np.round(np.clip(sf["tu"], -1, 1) * 32767).astype(np.int16)), tv=_b64(np.round(np.clip(sf["tv"], -1, 1) * 32767).astype(np.int16)),
                      s=_b64(sf["scale"].astype(np.float16)), op=_b64(np.round(np.clip(sf["opacity"], 0, 1) * 255).astype(np.uint8)), col=_b64(col), cls=_b64(cls))
    M["ctr"] = (lo + rng / 2).tolist()
    what = "compound leaf" if by == "leaf" else "leaflet"
    M["note"] = (f"{status}: leaflets and compound leaves only (grouping automatic, unvalidated); colour = {what}; grey = axis / unclustered surfels, "
                 "white = unassigned leaflets; height mode 0 = pot-top plane; shape only, scale not validated, no stem topology")
    html = html[:m.start(1)] + json.dumps(M, separators=(",", ":")) + html[m.end(1):]
    label0 = "복엽 색" if by == "leaf" else "소엽 색"
    html = re.sub(r'(<button data-cm="0" class="on">)[^<]*(</button>)', r"\g<1>" + label0 + r"\g<2>", html)
    title = f"{status} · 소엽·복엽 속성 (광 관련) · {what} 색"
    html = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", html); html = re.sub(r'<h1 id="ttl">[^<]*</h1>', f'<h1 id="ttl">{title}</h1>', html)
    banner = (f"{status} · 소엽·복엽만 (기관 그룹화 자동, 검증 안 됨) · 축·마디·화방 = unavailable · 형태만, 척도 미검증 (conditional_on_unvalidated_scale) · 계측 아님")
    html = re.sub(r'(z-index:50;background:#7a1f1f;[^>]*>)[^<]*(</div>)', lambda mm: mm.group(1) + banner + mm.group(2), html, count=1)
    # legend of colours
    items = leaves if by == "leaf" else leaflets
    chips = "".join(f'<span style="display:inline-flex;align-items:center;gap:4px;margin:0 6px 4px 0;font-size:11px"><i style="width:10px;height:10px;border-radius:2px;background:rgb{pal[k]}"></i>'
                    f'{k}{(" ·" + it["n_leaflets"] + "枚") if by == "leaf" else ""}</span>' for k, it in zip(keys, items))
    legend = (f'<label>{"복엽" if by == "leaf" else "소엽"} 범례 ({len(keys)}) · 회색 = 축/미군집 · 흰색 = 미배정</label><div style="max-height:160px;overflow:auto">{chips}</div>')
    html = html.replace('<label>불투명도 배율', legend + '<label>불투명도 배율', 1)
    return html


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, default=DEFAULTS["out"]); ap.add_argument("--surfels", type=Path, default=DEFAULTS["surfels"])
    ap.add_argument("--template", type=Path, default=TEMPLATE); ap.add_argument("--input-status", default="R3-1b_unaccepted"); a = ap.parse_args(argv)
    out = a.run / "viewer"; out.mkdir(parents=True, exist_ok=True)
    for by, name in (("leaf", "light_organs_viewer.html"), ("leaflet", "light_organs_viewer_leaflets.html")):
        html = build(a.run, a.surfels, a.template, by, a.input_status); (out / name).write_text(html, encoding="utf-8"); print(out / name, len(html))


if __name__ == "__main__":
    main()
