"""2DGS surfel files: read the full export, convert to 3D Gaussians (3DGS ply / antimatter15 .splat), read them back.

The full export (`<set>_surfels_full.ply`, kernel dolgfish/tomato-dwt03-2dgs-full) keeps the rotation: R = quat(qw, qx,
qy, qz), tangent axes tu, tv = columns 0, 1 of R, normal = column 2; s0, s1 are the standard deviations of the 2D
Gaussian along tu, tv; opacity is post-sigmoid; f_dc are the trained colour logits (rgb = sigmoid(f_dc), not SH).

A 3D Gaussian with scales (s0, s1, eps) and the same rotation approximates the surfel: its local z is the normal, so
it is a thin ellipsoid rather than a flat disc. Geometry measurements stay on the 2DGS surfels.
Contract: build/p1_2dgs_full_20261018_01/CONTRACT.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
FULL_KEYS = ("x", "y", "z", "qw", "qx", "qy", "qz", "tux", "tuy", "tuz", "tvx", "tvy", "tvz", "nx", "ny", "nz", "s0", "s1", "opacity")
GS_KEYS = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
           "rot_0", "rot_1", "rot_2", "rot_3")
SPLAT_DTYPE = np.dtype([("xyz", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
assert SPLAT_DTYPE.itemsize == 32

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "uchar": "u1", "uint8": "u1", "double": "<f8", "int": "<i4", "uint": "<u4"}


def read_binary_ply(path) -> np.ndarray:
    """Structured array of the vertex element of a binary little-endian PLY (scalar properties only)."""
    b = Path(path).read_bytes(); i = b.index(b"end_header\n") + 11
    hdr = b[:i].decode("ascii").split("\n"); n = None; dt = []
    for line in hdr:
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            n = int(t[2])
        elif t[:1] == ["property"] and n is not None:
            dt.append((t[2], _PLY_TYPES[t[1]]))
    return np.frombuffer(b[i:], dtype=dt, count=n)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N, 4) w x y z (normalised here) -> (N, 3, 3); same formula as the trainer's `quat_to_rot`."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def read_surfels_full(path) -> dict:
    r = read_binary_ply(path)
    return dict(xyz=np.c_[r["x"], r["y"], r["z"]].astype(np.float64), quat=np.c_[r["qw"], r["qx"], r["qy"], r["qz"]].astype(np.float64),
                tu=np.c_[r["tux"], r["tuy"], r["tuz"]].astype(np.float64), tv=np.c_[r["tvx"], r["tvy"], r["tvz"]].astype(np.float64),
                normal=np.c_[r["nx"], r["ny"], r["nz"]].astype(np.float64), scale=np.c_[r["s0"], r["s1"]].astype(np.float64),
                opacity=r["opacity"].astype(np.float64), rgb=np.c_[r["red"], r["green"], r["blue"]],
                f_dc=np.c_[r["f_dc_0"], r["f_dc_1"], r["f_dc_2"]].astype(np.float64))


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def to_3dgs(sf: dict, eps) -> dict:
    """Surfel dict -> 3DGS parameters. `eps`: per-surfel array or scalar thickness (std along the normal), same units."""
    eps = np.broadcast_to(np.asarray(eps, np.float64), (len(sf["xyz"]),))
    rgb = 1.0 / (1.0 + np.exp(-sf["f_dc"]))                                   # trained colour (sigmoid of the logits)
    q = sf["quat"] / np.linalg.norm(sf["quat"], axis=1, keepdims=True)
    return dict(xyz=sf["xyz"], f_dc=(rgb - 0.5) / SH_C0, opacity_logit=_logit(sf["opacity"]),
                scale_log=np.log(np.c_[sf["scale"], eps]), rot=q, rgb=rgb, opacity=sf["opacity"])


def write_3dgs_ply(path, g: dict) -> None:
    n = len(g["xyz"])
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           "comment 3DGS layout from 2DGS surfels (DC colour only); thickness eps is an approximation of a flat surfel\n"
           f"element vertex {n}\n" + "".join(f"property float {k}\n" for k in GS_KEYS) + "end_header\n").encode()
    rec = np.zeros(n, dtype=[(k, "<f4") for k in GS_KEYS])
    for k, a in zip(("x", "y", "z"), g["xyz"].T): rec[k] = a
    for j in range(3):
        rec[f"f_dc_{j}"] = g["f_dc"][:, j]; rec[f"scale_{j}"] = g["scale_log"][:, j]
    rec["opacity"] = g["opacity_logit"]
    for j in range(4): rec[f"rot_{j}"] = g["rot"][:, j]
    Path(path).write_bytes(hdr + rec.tobytes())


def read_3dgs_ply(path) -> dict:
    r = read_binary_ply(path)
    return dict(xyz=np.c_[r["x"], r["y"], r["z"]].astype(np.float64), f_dc=np.c_[r["f_dc_0"], r["f_dc_1"], r["f_dc_2"]].astype(np.float64),
                opacity_logit=r["opacity"].astype(np.float64), scale_log=np.c_[r["scale_0"], r["scale_1"], r["scale_2"]].astype(np.float64),
                rot=np.c_[r["rot_0"], r["rot_1"], r["rot_2"], r["rot_3"]].astype(np.float64))


def splat_order(g: dict) -> np.ndarray:
    """antimatter15 ordering: descending exp(sum log scale) * sigmoid(opacity)."""
    imp = np.exp(g["scale_log"].sum(1)) / (1.0 + np.exp(-g["opacity_logit"]))
    return np.argsort(-imp, kind="stable")


def write_splat(path, g: dict) -> np.ndarray:
    """32 B per Gaussian. Returns the order used (index into g) so the round trip can be compared."""
    o = splat_order(g); rec = np.zeros(len(o), SPLAT_DTYPE)
    rec["xyz"] = g["xyz"][o]; rec["scale"] = np.exp(g["scale_log"][o])
    rgb = np.clip(SH_C0 * g["f_dc"][o] + 0.5, 0, 1); a = 1.0 / (1.0 + np.exp(-g["opacity_logit"][o]))
    rec["rgba"] = np.clip(np.round(np.c_[rgb, a] * 255), 0, 255).astype(np.uint8)
    q = g["rot"][o] / np.linalg.norm(g["rot"][o], axis=1, keepdims=True)
    rec["rot"] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
    Path(path).write_bytes(rec.tobytes())
    return o


def read_splat(path) -> dict:
    rec = np.frombuffer(Path(path).read_bytes(), SPLAT_DTYPE)
    q = (rec["rot"].astype(np.float64) - 128) / 128; q /= np.linalg.norm(q, axis=1, keepdims=True)
    return dict(xyz=rec["xyz"].astype(np.float64), scale=rec["scale"].astype(np.float64), rgb=rec["rgba"][:, :3] / 255.0,
                alpha=rec["rgba"][:, 3] / 255.0, rot=q)


def round_trip(g: dict, ply_path, splat_path, order: np.ndarray) -> dict:
    """Max abs error of every field after reading the two files back (rotation compared as matrices, sign-free)."""
    P = read_3dgs_ply(ply_path); Sp = read_splat(splat_path)
    rm = lambda q: quat_to_rot(q)  # noqa: E731
    err_ply = dict(xyz=np.abs(P["xyz"] - g["xyz"]).max(), f_dc=np.abs(P["f_dc"] - g["f_dc"]).max(),
                   opacity_logit=np.abs(P["opacity_logit"] - g["opacity_logit"]).max(), scale_log=np.abs(P["scale_log"] - g["scale_log"]).max(),
                   rotation_matrix=np.abs(rm(P["rot"]) - rm(g["rot"])).max())
    go = {k: v[order] for k, v in g.items()}
    err_splat = dict(xyz=np.abs(Sp["xyz"] - go["xyz"]).max(),
                     scale_rel=(np.abs(Sp["scale"] - np.exp(go["scale_log"])) / np.exp(go["scale_log"])).max(),
                     rgb=np.abs(Sp["rgb"] - np.clip(SH_C0 * go["f_dc"] + 0.5, 0, 1)).max(),
                     alpha=np.abs(Sp["alpha"] - 1.0 / (1.0 + np.exp(-go["opacity_logit"]))).max(),
                     rotation_matrix=np.abs(rm(Sp["rot"]) - rm(go["rot"])).max())
    return dict(ply={k: float(v) for k, v in err_ply.items()}, splat={k: float(v) for k, v in err_splat.items()}, n=int(len(order)))
