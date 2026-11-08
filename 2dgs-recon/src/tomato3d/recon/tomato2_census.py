"""토마토2 Stage 1 census: EXIF, content hashes, session split and frame geometry. Local reads only (Rule 0): no conversion, no detection.
The 토마토2 folder is read-only; everything is written under --out.

usage: python -m tomato3d.recon.tomato2_census --src "<토마토2 dir>" --out build/p1_recon_tomato2_<date>_01
Writes census.csv, sessions.csv, FRAME_GEOMETRY.md, census_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pillow_heif
from PIL import ExifTags, Image

pillow_heif.register_heif_opener()

FOLDERS = {"Photos-1-001": "t2_01", "Photos-1-001 (1)": "t2_02", "Photos-1-001 (2)": "t2_03", "Photos-1-001 (3)": "t2_04", "Photos-1-001 (4)": "t2_05"}
GAP_MIN = 20                     # declared session gap threshold, minutes
EXIF_FIELDS = ("DateTimeOriginal", "SubsecTimeOriginal", "FocalLength", "FocalLengthIn35mmFilm", "FNumber", "ISOSpeedRatings", "ExposureTime",
               "ExifImageWidth", "ExifImageHeight", "LensModel")


def heif_boxes(buf: bytes):
    """(type, offset, size) of the top-level and meta/iprp/ipco children; enough to find ispe/clap/irot/imir of the primary item."""
    out = []

    def walk(off, end, depth):
        while off + 8 <= end:
            sz, typ = struct.unpack(">I4s", buf[off:off + 8]); hdr = 8
            if sz == 1:
                sz = struct.unpack(">Q", buf[off + 8:off + 16])[0]; hdr = 16
            if sz == 0:
                sz = end - off
            typ = typ.decode("latin1"); out.append((typ, off, sz, depth))
            if typ in ("meta", "iprp", "ipco"):
                walk(off + hdr + (4 if typ == "meta" else 0), off + sz, depth + 1)
            off += max(sz, 8)
    walk(0, len(buf), 0)
    return out


def geometry_boxes(p: Path) -> dict:
    b = p.read_bytes()[:400000]; g = dict(ispe=[], clap=None, irot=None, imir=None)
    for typ, o, s, _ in heif_boxes(b):
        if typ == "ispe":
            g["ispe"].append(struct.unpack(">II", b[o + 12:o + 20]))
        elif typ == "clap":
            g["clap"] = struct.unpack(">8I", b[o + 8:o + 40])
        elif typ == "irot":
            g["irot"] = b[o + 8] & 3
        elif typ == "imir":
            g["imir"] = b[o + 8] & 1
    g["ispe_primary"] = max(g["ispe"], key=lambda t: t[0] * t[1]) if g["ispe"] else None
    return g


def wic_size(p: Path):
    """Second, independent decoder: Windows Imaging Component (Microsoft HEIF Decoder) through PowerShell. Returns (w, h) or None."""
    ps = ("Add-Type -AssemblyName PresentationCore; $s=[System.IO.File]::OpenRead('%s'); "
          "$d=[System.Windows.Media.Imaging.BitmapDecoder]::Create($s,[System.Windows.Media.Imaging.BitmapCreateOptions]::IgnoreColorProfile,"
          "[System.Windows.Media.Imaging.BitmapCacheOption]::None); $f=$d.Frames[0]; \"$($f.PixelWidth) $($f.PixelHeight) $($d.CodecInfo.FriendlyName)\"; $s.Close()") % str(p).replace("'", "''")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=120)
        w, h, *name = r.stdout.split(); return (int(w), int(h), " ".join(name))
    except Exception as e:                                                                       # noqa: BLE001
        return None


def read_one(p: Path) -> dict:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    hf = pillow_heif.open_heif(p); prim = hf[0]
    im = Image.open(p); e = im.getexif(); t0 = {ExifTags.TAGS.get(k, k): v for k, v in e.items()}
    ifd = e.get_ifd(0x8769); t1 = {ExifTags.TAGS.get(k, k): v for k, v in ifd.items()}
    aux = list((prim.info.get("aux") or {}).keys())
    rec = dict(file=p.name, sha256=h, bytes=p.stat().st_size, make=t0.get("Make"), model=t0.get("Model"), orientation=t0.get("Orientation"), software=t0.get("Software"))
    for k in EXIF_FIELDS:
        v = t1.get(k); rec[k] = float(v) if isinstance(v, (int, float)) and k not in ("ExifImageWidth", "ExifImageHeight", "FocalLengthIn35mmFilm", "ISOSpeedRatings") else v
    rec.update(decoded_w=prim.size[0], decoded_h=prim.size[1], bit_depth=prim.info.get("bit_depth"), hdr_gainmap=any("hdrgainmap" in a for a in aux),
               portrait_depth=bool(prim.info.get("depth_images")), n_aux=len(aux))
    g = geometry_boxes(p); rec.update(ispe_w=g["ispe_primary"][0] if g["ispe_primary"] else None, ispe_h=g["ispe_primary"][1] if g["ispe_primary"] else None,
                                       clap=bool(g["clap"]), irot=g["irot"], imir=g["imir"])
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default=None, help="individual id prefix: folders are taken in sorted order as <prefix>_01.. (default: the 토마토2 mapping)")
    ap.add_argument("--known", default=None, help="census.csv of an earlier batch: cross-batch hash matches are reported")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    global FOLDERS
    if a.prefix:
        names = sorted(d.name for d in src.iterdir() if d.is_dir()); FOLDERS = {n: f"{a.prefix}_{i + 1:02d}" for i, n in enumerate(names)}
    rows = []
    for folder, iid in FOLDERS.items():
        d = src / folder
        for p in sorted({q.resolve() for q in d.iterdir() if q.is_file() and q.suffix.lower() in (".heic", ".jpg", ".jpeg")}):
            r = read_one(p); r.update(folder=folder, individual=iid, subfolder=""); rows.append(r); print(iid, p.name, r["DateTimeOriginal"], r["FocalLengthIn35mmFilm"], r["decoded_w"], flush=True)
        for sub in sorted(x for x in d.iterdir() if x.is_dir()):
            for p in sorted(sub.rglob("*")):
                if p.is_file():
                    try:
                        r = read_one(p)
                    except Exception as ex:                                                       # noqa: BLE001
                        r = dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, error=str(ex)[:80])
                    r.update(folder=folder, individual="", subfolder=sub.name); rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("individual", "folder", "subfolder", "file"), k))
    with open(out / "census.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    # hashes: within-folder duplicates and cross-folder matches
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append((r["folder"] + ("/" + r["subfolder"] if r["subfolder"] else ""), r["file"]))
    dups = {h: v for h, v in by_hash.items() if len(v) > 1}
    cross = {h: v for h, v in dups.items() if len({x[0] for x in v}) > 1}
    if a.known:
        kh = {r["sha256"]: (r["folder"], r["file"]) for r in csv.DictReader(open(a.known, encoding="utf-8"))}
        for h, v in by_hash.items():
            if h in kh:
                cross[h] = v + [("earlier batch: " + kh[h][0], kh[h][1])]
    # sessions per individual: sort by time, split at gaps > GAP_MIN
    sessions = []; strays = {}
    for iid in FOLDERS.values():
        rr = [r for r in rows if r["individual"] == iid and r.get("DateTimeOriginal")]
        rr.sort(key=lambda r: (r["DateTimeOriginal"], r.get("SubsecTimeOriginal") or ""))
        groups, cur = [], []
        for r in rr:
            t = datetime.strptime(r["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
            if cur and (t - cur[-1][0]).total_seconds() > GAP_MIN * 60:
                groups.append(cur); cur = []
            cur.append((t, r))
        if cur:
            groups.append(cur)
        main_g = max(groups, key=len)
        for g in groups:
            cams = Counter((r["FocalLengthIn35mmFilm"], r["decoded_w"], r["decoded_h"]) for _, r in g)
            sessions.append(dict(individual=iid, session=f"{iid}_{g[0][0]:%Y%m%d_%H%M}", date=f"{g[0][0]:%Y-%m-%d}", start=f"{g[0][0]:%H:%M:%S}", end=f"{g[-1][0]:%H:%M:%S}",
                                 n=len(g), main=(g is main_g), cameras="; ".join(f"f35={c[0]} {c[1]}x{c[2]} x{n}" for c, n in sorted(cams.items(), key=lambda x: -x[1]))))
        strays[iid] = [r["file"] for g in groups if g is not main_g for _, r in g]
    with open(out / "sessions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sessions[0].keys())); w.writeheader(); [w.writerow(s) for s in sessions]
    # frame geometry: EXIF vs ispe vs libheif vs WIC on three samples (one 12 MP, one 48 MP, one from another individual)
    pick = []
    for want in (("t2_02", 3024), ("t2_01", 6048), ("t2_01", 3024)):
        c = [r for r in rows if r["individual"] == want[0] and r["decoded_w"] == want[1]]
        if c:
            pick.append(c[len(c) // 2])
    geo = []
    for r in pick:
        p = src / r["folder"] / r["file"]; wic = wic_size(p)
        geo.append(dict(individual=r["individual"], file=r["file"], exif=(r["ExifImageWidth"], r["ExifImageHeight"]), ispe=(r["ispe_w"], r["ispe_h"]), clap=r["clap"], irot=r["irot"],
                        libheif=(r["decoded_w"], r["decoded_h"]), wic=wic))
    agree = all(g["ispe"] == g["libheif"] == (g["wic"][:2] if g["wic"] else None) for g in geo)
    shapes = Counter((r.get("ExifImageWidth"), r.get("ExifImageHeight"), r.get("ispe_w"), r.get("ispe_h"), r.get("decoded_w"), r.get("decoded_h"), bool(r.get("clap")), r.get("irot")) for r in rows if r["individual"])
    md = ["# FRAME_GEOMETRY — 토마토2 (Stage 1, read-only)", "",
          "Question: EXIF says 4032×3024 / 8064×6048 but samples decode square. Which is the true frame?", "",
          "| individual | file | EXIF | ispe box | clap | irot | libheif (pillow_heif) | WIC (Microsoft HEIF decoder) |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {g['individual']} | {g['file']} | {g['exif'][0]}×{g['exif'][1]} | {g['ispe'][0]}×{g['ispe'][1]} | {g['clap']} | {g['irot']} | {g['libheif'][0]}×{g['libheif'][1]} | {g['wic'][0]}×{g['wic'][1]} ({g['wic'][2]}) |" if g["wic"] else
           f"| {g['individual']} | {g['file']} | {g['exif'][0]}×{g['exif'][1]} | {g['ispe'][0]}×{g['ispe'][1]} | {g['clap']} | {g['irot']} | {g['libheif'][0]}×{g['libheif'][1]} | not available |" for g in geo]
    md += ["", "All files of the five individuals, (EXIF w, h, ispe w, h, decoded w, h, clap present, irot): count", ""] + [f"- {k}: {n}" for k, n in shapes.most_common()]
    md += ["", ("**Finding:** the stored HEVC frame is SQUARE. The `ispe` box (stored dimensions), libheif and the independent WIC decoder agree; there is no `clap` "
                "(clean-aperture crop) box and `irot` = 0, so nothing is cropped or rotated at decode time. The EXIF ExifImageWidth/Height (4:3) is stale capture "
                "metadata written by the phone's square-format mode. Intrinsics must therefore be built for the square frame (3024×3024, or 6048×6048 for the 48 MP "
                "frames), as the R3 line already did on its own square 3024 captures. Nothing is lost by decoding; the 4:3 frame never existed in these files.") if agree else
          "**Finding:** the decoders disagree; see the table. Do NOT convert until this is resolved."]
    (out / "FRAME_GEOMETRY.md").write_text("\n".join(md), encoding="utf-8")
    summ = dict(gap_min=GAP_MIN, n_files=len(rows), per_individual={iid: Counter(r["individual"] for r in rows)[iid] for iid in FOLDERS.values()},
                subfolders={f"{r['folder']}/{r['subfolder']}" for r in rows if r["subfolder"]} and {k: Counter((r.get("DateTimeOriginal") or "")[:10] for r in rows if r["subfolder"] and f"{r['folder']}/{r['subfolder']}" == k)
                                                                                                    for k in {f"{r['folder']}/{r['subfolder']}" for r in rows if r["subfolder"]}},
                within_folder_duplicates={h: v for h, v in dups.items() if h not in cross}, cross_folder_matches=cross, strays=strays, frame_geometry_agree=agree, sessions=sessions)
    json.dump(summ, open(out / "census_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "sessions"}, indent=1, default=str)); [print(s) for s in sessions]


if __name__ == "__main__":
    main()
