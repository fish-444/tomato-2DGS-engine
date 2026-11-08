"""CONTRACT A2: diagnose the A1 FAIL by measuring the bent cylinder axes directly (local, already downloaded geometry; nothing re-run or changed).
Leaf-only faces -> connected components -> cylinders (PCA axis) -> chain from the model leaf base -> petiole elevation, turns, lengths.
python diag_dtheta.py <geometry dir>"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from roundtrip_bmodel import leaf_rows, spec_of  # noqa: E402
from roundtrip_dtheta import SEG, BLADES, BX, predicted_chord  # noqa: E402

PLANTS = ("mtg_template_mean_v2draft_sd0", "mtg_R3-1b", "mtg_n_04", "P01", "P02")


def obj_mesh(p):
    V, F = [], []
    for ln in open(p, encoding="utf-8", errors="replace"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) for t in ln.split()[1:]]; idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for j in range(1, len(idx) - 1):
                F.append((idx[0], idx[j], idx[j + 1]))
    V = np.asarray(V, float)[:, [0, 2, 1]]                     # GroIMP y-up OBJ -> (x, y, z)
    return V, np.asarray(F, int)


def components(V, F, stem_keys):
    """The exporter writes a triangle soup (3 private vertices per triangle), so neither index nor position connectivity separates cylinders
    (A2 tool fix, before any diagnostic number). A cylinder's side triangles have their longest edge along the axis with the cylinder's length:
    group leaf-only sliver triangles by (longest-edge length 0.05 mm, sign-normalised direction 1 deg), then PCA over the group's vertices."""
    keys = [tuple(k) for k in np.round(V, 7)]
    groups = {}
    for f in F:
        if all(keys[i] in stem_keys for i in f):
            continue
        T = V[list(f)]; E = [T[1] - T[0], T[2] - T[1], T[0] - T[2]]; L = [float(np.linalg.norm(e)) for e in E]; j = int(np.argmax(L))
        if L[j] < 0.0005 or L[j] < 3 * min(L):
            continue
        d = E[j] / L[j]
        if d[2] < 0 or (d[2] == 0 and (d[1] < 0 or (d[1] == 0 and d[0] < 0))):
            d = -d
        key = (round(L[j] / 5e-5), round(math.degrees(math.atan2(d[1], d[0])) % 360), round(elev(d)))
        groups.setdefault(key, []).append(T)
    out = []
    for g in groups.values():
        if len(g) < 4:                                                    # a cylinder side has >= 4 sliver triangles
            continue
        P = np.vstack(g); c = P.mean(0); u, s, vt = np.linalg.svd(P - c, full_matrices=False); d = vt[0]; t = (P - c) @ d
        out.append(dict(a=c + d * t.min(), b=c + d * t.max(), length=float(t.max() - t.min()), n=len(g)))
    return out


def chain(cyls, base, h_dir):
    """From the leaf base: next = unused cylinder with an end within 1.5 mm of the current end, most aligned with the current direction."""
    cur = np.asarray(base, float); d_cur = None; used = set(); out = []
    for _ in range(12):
        best = None
        for i, c in enumerate(cyls):
            if i in used:
                continue
            for s, e in ((c["a"], c["b"]), (c["b"], c["a"])):
                if np.linalg.norm(s - cur) <= 0.0015:
                    v = (e - s) / np.linalg.norm(e - s); score = float(v @ d_cur) if d_cur is not None else float(v @ h_dir)
                    if d_cur is not None and score < 0.5:            # lateral petiolules leave at about 90 deg
                        continue
                    if best is None or score > best[0]:
                        best = (score, i, s, e, v)
        if best is None:
            break
        _, i, s, e, v = best; used.add(i); out.append(dict(start=s, end=e, dir=v, length=float(np.linalg.norm(e - s)))); cur = e; d_cur = v
    return out


def elev(v):
    return math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))


def run(g):
    rows_out = []
    for name in PLANTS:
        spec = spec_of(name.replace("_sd0", "")); mask = name.replace("_sd0", "")
        stem_keys = {tuple(k) for k in np.round(obj_mesh(g / f"{mask}_stem.csv.obj")[0], 7)}
        for lf in sorted(spec["leaves"], key=lambda l: l["k"]):
            f = g / f"{mask}_leaf{lf['k']:02d}nb.csv.obj"; t = g / f"{mask}_leaf{lf['k']:02d}nb.csv_leaves.csv"
            if not f.exists() or not t.exists():
                continue
            r = next((x for x in leaf_rows(t) if int(float(x["order"])) == 0 and int(float(x["rank"])) == lf["k"]), None)
            if r is None:
                continue
            cls = BX.leaf_class(lf["k"]); n_inner = max(0, BLADES[cls] - 2); base = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
            az = math.radians((float(r["orientation_deg"]) - 90) % 360); h_dir = np.array([math.cos(az), math.sin(az), 0.0])
            V, F = obj_mesh(f); C = chain(components(V, F, stem_keys), base, h_dir)
            expect = 1 + len(SEG[cls]) + (1 if cls != "first" else 0)        # petiole + rachis segments + last petiolule
            rec = dict(plant=name, rank=lf["k"], cls=cls, n_inner=n_inner, dtheta_in=lf["dtheta_deg"], theta0_in=lf["theta0_deg"], chain_n=len(C), chain_expected=expect)
            if len(C) < 2 or (cls != "first" and len(C) < 1 + n_inner + 1):
                rec["status"] = "chain_incomplete"; rows_out.append(rec); continue
            turns = []
            for a, b in zip(C, C[1:]):
                e_a, e_b = elev(a["dir"]), elev(b["dir"]); turns.append(e_a - e_b)                 # down = positive
            pet_m = float(r["petiole_m"]); rach = float(r["rachis_total_m"]); assumed = [pet_m] + [rach * s for s in SEG[cls]] + [0.01]
            meas = [c["length"] for c in C]
            # chord recomputed from measured lengths + measured turns (2-D walk in the vertical plane)
            x = z = 0.0; e = elev(C[0]["dir"])
            for i, c in enumerate(C):
                if i > 0:
                    e -= turns[i - 1]
                x += c["length"] * math.cos(math.radians(e)); z += c["length"] * math.sin(math.radians(e))
            tip = C[-1]["end"]; v = tip - base; chord_model = elev(v / np.linalg.norm(v)); chord_recomp = math.degrees(math.atan2(z, x))
            inner_sum = float(sum(turns[:n_inner])) if n_inner else 0.0
            p_bent, _ = predicted_chord(90 - float(r["inclination_deg"]), pet_m, rach, cls, lf["dtheta_deg"])
            rec.update(petiole_elev=elev(C[0]["dir"]), d_petiole=elev(C[0]["dir"]) - lf["theta0_deg"], turns=";".join(f"{x:.2f}" for x in turns), inner_turn_sum=inner_sum,
                       d_inner=inner_sum - lf["dtheta_deg"] if n_inner else None, last_turn=turns[-1] if len(turns) > n_inner else None,
                       lengths_mm=";".join(f"{x * 1000:.2f}" for x in meas), assumed_mm=";".join(f"{x * 1000:.2f}" for x in assumed),
                       chord_model=chord_model, chord_recomputed=chord_recomp, d_recomp=chord_recomp - chord_model, chord_pred_bent=p_bent, status="ok")
            rows_out.append(rec)
    ok = [x for x in rows_out if x["status"] == "ok" and x["n_inner"] > 0]
    n_adapter = sum(abs(x["d_inner"]) <= 1 for x in ok); n_recomp = sum(abs(x["d_recomp"]) <= 0.5 for x in ok)
    if ok and n_adapter >= 0.9 * len(ok):
        verdict = "adapter_confirmed;" + ("predictor_is_cause" if n_recomp >= 0.9 * len(ok) else "recompute_does_not_reproduce(not_separated)")
    elif ok and (len(ok) - n_adapter) > 0.1 * len(ok):
        verdict = "adapter_at_fault"
    else:
        verdict = "not_separated"
    S = dict(leaves_with_inner=len(ok), chain_incomplete=sum(x["status"] == "chain_incomplete" for x in rows_out), inner_sum_within_1deg=n_adapter, recomputed_chord_within_0_5deg=n_recomp, verdict=verdict,
             d_inner_median=float(np.median([x["d_inner"] for x in ok])) if ok else None, d_petiole_median=float(np.median([x["d_petiole"] for x in ok])) if ok else None,
             last_turn_median=float(np.median([x["last_turn"] for x in ok if x["last_turn"] is not None])) if ok else None)
    (HERE / "diag_dtheta.json").write_text(json.dumps(dict(summary=S, leaves=rows_out), indent=1, default=float), encoding="utf-8")
    keys = []
    for x in rows_out:
        keys += [k for k in x if k not in keys]
    with open(HERE / "diag_dtheta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows_out)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
