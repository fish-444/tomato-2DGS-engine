"""Compound-leaf span / L_r from the _03 split (CONTRACT 'Addition'). python -X utf8 span.py"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE))
import run_split as RS, split_lib as SL  # noqa: E402

S = {(r["plant"], r["uid"]): r for r in csv.DictReader(open(HERE / "leaflet_split_summary.csv", encoding="utf-8")) if r["status"] == "split"}
rows = []
for pid in RS.PLANTS:
    D = RS.FP.load_real(pid); spec = json.load(open(RS.V02 / f"static/mtg_{pid}_spec.json", encoding="utf-8")); X, A, N, prow, _ = RS.surfel_data(pid, D)
    Z = np.load(HERE / f"split_{pid}.npz", allow_pickle=True); assert np.array_equal(np.asarray(Z["rows"]), prow)
    leaf = Z["leaf"].astype(str); lt = Z["leaflet"].astype(str)
    for lf in spec["leaves"]:
        u = lf["uid"]
        if (pid, u) not in S:
            continue
        m = np.nonzero((leaf == u) & (lt != ""))[0]; c, _ = RS.curve_of(lf)
        if len(m) < 30:
            print("skipped (surfels in split file):", pid, u, len(m)); continue
        fr = SL.frame(X[m], c)
        Y = X[m] - X[m].mean(0); w, v = np.linalg.eigh(np.cov(Y.T)); nc = v[:, 0] * (1 if v[2, 0] >= 0 else -1)
        Q, sQ = fr["Q"], fr["sQ"]; T = SL.tangents(Q); j = np.clip(np.searchsorted(sQ, fr["s"]), 0, len(Q) - 1)
        foot = np.stack([np.interp(fr["s"], sQ, Q[:, i]) for i in range(3)], 1); b = np.cross(T[j], nc); b /= np.maximum(np.linalg.norm(b, axis=1)[:, None], 1e-12)
        lat = ((X[m] - foot) * b).sum(1); Lr = float(S[(pid, u)]["L_mm"])
        rows.append(dict(plant=pid, uid=u, k=lf["k"], leaflets=int(S[(pid, u)]["n_leaflets"]), L_r_mm=Lr, span_mm=float(lat.max() - lat.min()), span_p1_p99_mm=float(np.percentile(lat, 99) - np.percentile(lat, 1)),
                         span_over_L=float((lat.max() - lat.min()) / Lr), span_p1_p99_over_L=float((np.percentile(lat, 99) - np.percentile(lat, 1)) / Lr)))
with open(HERE / "span_per_leaf.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def st(v):
    v = np.asarray(v, float); p = np.percentile(v, [10, 25, 50, 75, 90])
    return dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1), median=p[2], p10=p[0], p25=p[1], p75=p[3], p90=p[4])


out = []
for sub, sel in (("all", lambda r: True), ("leaflets_5", lambda r: r["leaflets"] == 5), ("leaflets_7", lambda r: r["leaflets"] == 7), ("pilot_only", lambda r: r["plant"] == "R3-1b")):
    g = [r for r in rows if sel(r)]
    for q, desc in (("span_over_L", "max - min lateral offset (compound plane, perpendicular to the rachis) / L_r (primary)"), ("span_p1_p99_over_L", "p1-p99 lateral range / L_r (robust variant)"),
                    ("span_mm", "max - min lateral offset, mm"), ("L_r_mm", "rachis length node -> end-leaflet base, mm")):
        out.append(dict(subset=sub, rank_class="main", quantity=q, **{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in st([r[q] for r in g]).items()}, description=desc, source="reconstruction_ungraded"))
TR = ROOT / "build/traits_recon_20261101_01"; f_ = TR / "MTG_LEAFLET_SPAN_v1.csv"; assert not f_.exists()
with open(f_, "w", newline="", encoding="utf-8") as f:
    f.write("# MTG_LEAFLET_SPAN_v1; compound-leaf span / rachis length from the _03 split (build/mtg_build_v2_clicks_leaflets_20261105_01, CONTRACT 'Addition'); companion of MTG_LEAFLET_ARRANGEMENT_v1 (unchanged); labels leaflet_arrangement_from_reconstruction;skeleton_labels_ungraded;conditional_on_unvalidated_scale;not_validation\n")
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
h = hashlib.sha256(f_.read_bytes()).hexdigest(); (TR / "MTG_LEAFLET_SPAN_v1.sha256").write_text(f"{h}  MTG_LEAFLET_SPAN_v1.csv\n", encoding="utf-8")
print(h); [print(r["subset"], r["quantity"], r["n"], r["median"], r["p10"], r["p90"]) for r in out if r["quantity"] in ("span_over_L", "span_p1_p99_over_L")]
