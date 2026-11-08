"""HEIC capture -> JPG with EXIF orientation applied; exact duplicates (same bytes) dropped.

usage: python -m tomato3d.recon.charuco_photos --src <heic dir> --out <dir>
Writes <out>/jpg/<stem>.jpg and <out>/photos.csv (name, sha256, width, height, kept, duplicate_of).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


def _convert(args):
    src, dst = args
    import pillow_heif
    pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if not dst.exists():
        im.save(dst, "JPEG", quality=93)
    return im.size


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); out = Path(a.out); (out / "jpg").mkdir(parents=True, exist_ok=True)
    # originals before their " 1" copies, so the copy is the one dropped
    files = sorted((p for p in Path(a.src).iterdir() if p.suffix.lower() == ".heic"), key=lambda p: (" " in p.stem, p.name))
    seen, rows, todo = {}, [], []
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        dup = seen.get(h)
        if dup is None:
            seen[h] = p.name; todo.append((p, out / "jpg" / (p.stem.replace(" ", "_") + ".jpg")))
        rows.append(dict(name=p.name, sha256=h, kept=int(dup is None), duplicate_of=dup or ""))
    with ProcessPoolExecutor(6) as ex:
        sizes = dict(zip((t[0].name for t in todo), ex.map(_convert, todo)))
    for r in rows:
        w, hh = sizes.get(r["name"], (0, 0)); r["width"], r["height"] = w, hh
        r["jpg"] = (Path(r["name"]).stem.replace(" ", "_") + ".jpg") if r["kept"] else ""
    with open(out / "photos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "jpg", "sha256", "width", "height", "kept", "duplicate_of"]); w.writeheader(); w.writerows(rows)
    print(len(files), "heic,", len(todo), "kept")


if __name__ == "__main__":
    main()
