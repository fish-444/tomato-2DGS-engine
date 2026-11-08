"""CONTRACT A1 (A-dtheta): new scenario set with column 32 = dtheta (deg) on every xL row (main leaves from the spec by rank order; side leaves
in file order), plus `nb` masks (blade area 1e-9) for the geometric check plants. Earlier files untouched. Packs kaggle/import_ds_dtheta/."""
import hashlib, json, shutil, tarfile, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
SRC = HERE / "kaggle/import_ds/scn"; DST = HERE / "kaggle/import_ds_dtheta"; SCN = DST / "scn"; SCN.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CHECK = ("mtg_template_mean_v2draft_sd0", "mtg_R3-1b", "mtg_n_04", "P01", "P02")


def spec_for(name):
    base = name.replace("_sd0", "")
    for d in (HERE / "static", HERE / "dynamic/check_plants"):
        for cand in (base, base.rsplit("_leaf", 1)[0], base.rsplit("_stem", 1)[0]):
            p = d / f"{cand}_spec.json"
            if p.exists():
                return json.load(open(p, encoding="utf-8"))
    return None


def fill(name):
    src = SRC / f"{name}.csv"; lines = src.read_bytes().decode("utf-8").split("\n"); head, body = lines[:77], lines[77:]
    spec = spec_for(name)
    if name == "one_row_test":
        spec = None
    rows = [(ln.split(",") + [""] * 33)[:33] if ln.strip() else None for ln in body]
    main = [lf["dtheta_deg"] for lf in sorted(spec["leaves"], key=lambda l: l["k"])] if spec else []
    side = [lf["dtheta_deg"] for sa in (spec or {}).get("side_axes", []) for lf in sa["leaves"]]
    mi = si = 0; out = []; filled = 0
    for i, c in enumerate(rows):
        if c is None:
            out.append(body[i]); continue
        c = list(c)
        if c[2].startswith("xL"):
            if mi < len(main):
                c[32] = f"{max(0.0, main[mi]):.4f}"; filled += 1
            mi += 1
        elif c[3].startswith("xL"):
            if si < len(side):
                c[32] = f"{max(0.0, side[si]):.4f}"; filled += 1
            si += 1
        out.append(",".join(c))
    (SCN / f"{name}.csv").write_bytes(("\n".join(head + out)).encode("utf-8"))
    return filled, mi, len(main)


def nb_masks(name):
    """<name>_leafKK.csv -> <name>_leafKKnb.csv with blade area 1e-9 on the kept leaf (as make_organ_scenarios)."""
    n = 0
    for p in sorted(SCN.glob(f"{name}_leaf??.csv")):
        lines = p.read_bytes().decode("utf-8").split("\n"); head, body = lines[:77], lines[77:]; out = []
        for ln in body:
            if not ln.strip():
                out.append(ln); continue
            c = (ln.split(",") + [""] * 33)[:33]
            if (c[2].startswith("xL") or c[3].startswith("xL")) and c[9] == "1":
                c[17] = "0.000000001"
            out.append(",".join(c))
        (SCN / f"{p.stem}nb.csv").write_bytes(("\n".join(head + out)).encode("utf-8")); n += 1
    return n


report = {}
for p in sorted(SRC.glob("*.csv")):
    report[p.stem] = fill(p.stem)
for name in CHECK:
    report[name + "_nb_masks"] = nb_masks(name)
shutil.copy2(HERE / "kaggle/import_ds/work_model.tar.bin", DST / "work_model_geo7_unpatched.tar.bin")
with tarfile.open(DST / "work_model.tar.bin", "w") as t:
    t.add(HERE / "work_dtheta/model", arcname="model")
with tarfile.open(DST / "scenarios.tar.bin", "w") as t:
    for p in sorted(SCN.glob("*.csv")):
        t.add(p, arcname=p.name)
man = dict(inputs={"work_model.tar.bin": sha(DST / "work_model.tar.bin"), "scenarios.tar.bin": sha(DST / "scenarios.tar.bin")}, core_sha256="35c605d65d9b54fe64cdf93bf28ae44ae7a1b6ded03b461dedcb0156a5c22147",
           adapter="A-dtheta (CONTRACT A1): work_dtheta/model = geo7 work_model + organs.rgg patch (A_dtheta_organs.rgg.diff)", organs_rgg_sha256=sha(HERE / "work_dtheta/model/organs.rgg"),
           unpatched_work_model_sha256=sha(DST / "work_model_geo7_unpatched.tar.bin"), n_scenarios=len(list(SCN.glob("*.csv"))), fill_report=report)
(DST / "MTGB_IMPORT_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
(DST / "dataset-metadata.json").write_text(json.dumps(dict(title="mtgb-import-dtheta-20261102", id="dolgfish/mtgb-import-dtheta-20261102", licenses=[dict(name="other")], isPrivate=True)), encoding="utf-8")
k = HERE / "kaggle/kernels/mtgb-bm-import-dtheta-20261102"; k.mkdir(parents=True, exist_ok=True); shutil.copy2(HERE / "kaggle/kernel_import.py", k / "kernel.py")
(k / "kernel-metadata.json").write_text(json.dumps(dict(id="dolgfish/mtgb-bm-import-dtheta-20261102", title="mtgb-bm-import-dtheta-20261102", code_file="kernel.py", language="python", kernel_type="script", is_private=True,
                                                         enable_gpu=False, enable_tpu=False, enable_internet=False, dataset_sources=["dolgfish/mtgb-import-dtheta-20261102", "dolgfish/tomato-h-input"], competition_sources=[], kernel_sources=[])), encoding="utf-8")
print(man["n_scenarios"], "scenarios;", sum(v[0] for v in report.values() if isinstance(v, tuple)), "xL rows filled;", {k: v for k, v in report.items() if k.endswith("_nb_masks")})
