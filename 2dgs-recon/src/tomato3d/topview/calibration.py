"""T0 - generator bending calibration from side close-ups (gate for the paired dataset, T2).

A person records, per compound seen in a side close-up, its bend type and - where the A1.5 points P0/P1/P3 are
visible - the angle theta (and the per-node alpha_i of MTG 12.3). Close-ups carry no scale: ANGLES ONLY, never a
length. This module reads that human table (read-only), derives ``BENT_FRACTION`` and the theta distribution with
``source_kind = measured`` and ``n``, and audits the generator's current bending/shape parameters against measured
facts. An empty or absent table gives ``unavailable``; :func:`calibration_gate` then refuses T2.

The 2026-10-10 bend EXAMPLES (``records/compound_bend_types_20261010``) are illustrative picks of each type, not a
sample, so they are never counted into a fraction - they are reported as such.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path(__file__).with_name("topview_config.json")
UNAVAILABLE = "unavailable"
BEND_TYPES = ("flat", "bent", "intermediate", "unknown")
COLUMNS = ("record_id", "plant_id", "capture_datetime", "photo_file", "photo_sha256", "axis", "rank",
           "bend_type", "theta_deg", "alpha_deg", "curl_note", "observer", "note")
CALIBRATION_SCHEMA = "topview-bend-calibration-1"


class BendRecordError(ValueError):
    """A human bend record is malformed. Reported, never repaired."""


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _float_or_none(text: str, what: str, where: str) -> float | None:
    text = (text or "").strip()
    if text in ("", UNAVAILABLE):
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise BendRecordError(f"{where}: {what} {text!r} is not a number") from exc


def read_bend_records(path: Path) -> list[dict[str, Any]]:
    """Validate and parse the human table. Raises :class:`BendRecordError` on the first malformed row."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise BendRecordError(f"{path.name}: missing columns {missing}")
        rows = []
        for i, raw in enumerate(reader, start=2):
            where = f"{path.name}:{i}"
            if not any((v or "").strip() for v in raw.values()):
                continue
            bend_type = (raw["bend_type"] or "").strip()
            if bend_type not in BEND_TYPES:
                raise BendRecordError(f"{where}: bend_type {bend_type!r} not in {BEND_TYPES}")
            theta = _float_or_none(raw["theta_deg"], "theta_deg", where)
            if theta is not None and not 0.0 <= theta <= 180.0:
                raise BendRecordError(f"{where}: theta_deg {theta} outside 0-180")
            alphas = [_float_or_none(a, "alpha_deg", where) for a in (raw["alpha_deg"] or "").split(";") if a.strip()]
            if not (raw["plant_id"] or "").strip() or not (raw["photo_sha256"] or "").strip():
                raise BendRecordError(f"{where}: plant_id and photo_sha256 are required")
            rows.append(dict(record_id=raw["record_id"].strip(), plant_id=raw["plant_id"].strip(),
                             photo_sha256=raw["photo_sha256"].strip(), axis=(raw["axis"] or "").strip() or UNAVAILABLE,
                             rank=(raw["rank"] or "").strip() or UNAVAILABLE, bend_type=bend_type,
                             theta_deg=theta, alpha_deg=alphas, curl_note=(raw["curl_note"] or "").strip()))
    return rows


def derive_bending(rows: list[dict[str, Any]], source_ref: str) -> dict[str, Any]:
    """BENT_FRACTION and theta distribution from validated rows. Every number carries n and its source."""
    if not rows:
        return dict(status=UNAVAILABLE, reason=f"no bend records in {source_ref}",
                    BENT_FRACTION=dict(value=UNAVAILABLE), theta_deg=dict(value=UNAVAILABLE))
    counts = {t: sum(r["bend_type"] == t for r in rows) for t in BEND_TYPES}
    typed = counts["flat"] + counts["bent"] + counts["intermediate"]
    out: dict[str, Any] = dict(status="derived", source_ref=source_ref, n_records=len(rows), counts=counts,
                               n_plants=len({r["plant_id"] for r in rows}))
    if typed:
        # 'intermediate' stays in the denominator and is reported on its own: whether it should count as bent is
        # not decided (MTG 8.4 on hold), so BENT_FRACTION is the bent share of typed compounds, nothing more.
        out["BENT_FRACTION"] = dict(value=round(counts["bent"] / typed, 4), n=typed, source_kind="measured",
                                    definition="n_bent / (n_flat + n_bent + n_intermediate); unknown excluded")
        out["INTERMEDIATE_FRACTION"] = dict(value=round(counts["intermediate"] / typed, 4), n=typed,
                                            source_kind="measured")
    else:
        out["BENT_FRACTION"] = dict(value=UNAVAILABLE, reason="every record is bend_type unknown")
    for t in ("flat", "bent", "intermediate", "all"):
        th = np.array([r["theta_deg"] for r in rows if r["theta_deg"] is not None and t in (r["bend_type"], "all")])
        key = f"theta_deg_{t}"
        out[key] = (dict(value=UNAVAILABLE, n=0) if len(th) == 0 else
                    dict(n=int(len(th)), source_kind="measured", min=float(th.min()), max=float(th.max()),
                         p10_p50_p90=[round(float(x), 2) for x in np.percentile(th, [10, 50, 90])]))
    out["curl_notes"] = sum(bool(r["curl_note"]) for r in rows)
    return out


def generator_audit(run: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compare the generator's current shape/bend parameters (one run) with measured facts. Read-only."""
    from tomato3d.synth.contours import load_library

    run = Path(run)
    gcfg = json.loads((run / "generator_config.json").read_text(encoding="utf-8"))
    contours, _ = load_library(run / "leaflet_contours.json")
    aspects, n_comp = [], []
    for f in sorted(run.glob("scenes/*/plant_graph.json")):
        g = json.loads(f.read_text(encoding="utf-8"))
        n_comp.append(len(g["compounds"]))
        for it in g["interiors"].values():
            for leaf in it["leaflets"]:
                c = np.asarray(contours[leaf["contour_index"]], float)
                uv_aspect = (c[:, 0].max() - c[:, 0].min()) / (c[:, 1].max() - c[:, 1].min())
                aspects.append(uv_aspect / float(leaf["width_scale"]))
    lo, hi = cfg["leaflet_true_aspect_range"]["min"], cfg["leaflet_true_aspect_range"]["max"]
    a = np.array(aspects)
    ranges = gcfg.get("ranges", {})
    ref = cfg["generator_audit_reference"]["compounds_per_plant_real"]
    pick = lambda k: {kk: ranges.get(k, {}).get(kk) for kk in ("min", "max", "source_kind")}  # noqa: E731
    return dict(
        run=run.as_posix(),
        leaflet_true_aspect=dict(generator_p10_p50_p90=[round(float(x), 3) for x in np.percentile(a, [10, 50, 90])],
                                 n=int(len(a)), measured_range=[lo, hi],
                                 share_inside_measured_range=round(float(((a >= lo) & (a <= hi)).mean()), 4),
                                 verdict="CONFLICT" if not lo <= float(np.median(a)) <= hi else "consistent",
                                 cause="library contours are top-view masks (already foreshortened) and "
                                       "leaflet_width_scale widens them further"),
        compounds_per_plant=dict(generator=sorted(set(n_comp)), real=[ref["min"], ref["max"]],
                                 real_source_kind=ref["source_kind"],
                                 verdict="CONFLICT" if min(n_comp) > ref["max"] else "consistent"),
        leaflet_width_scale=pick("leaflet_width_scale"), leaflet_bend_deg=pick("leaflet_bend_deg"),
        leaflet_roll_deg=pick("leaflet_roll_deg"),
        rachis_droop_deg=dict(**pick("rachis_droop_deg"),
                              note="one continuous droop spread evenly over the rachis segments; synth-stage1-0.2.0 "
                                   "has no flat/bent type mixture, so BENT_FRACTION has nowhere to enter yet "
                                   "(the tools/ V3 PARAMS BENT_FRACTION is 0 = unmeasured)"),
        blade_curl=dict(value="on_hold", note="MTG 8.3: no definition, axis or sign; the generator's roll is "
                                              "user_assumed, not a curl measurement"),
    )


def calibration_gate(calibration: dict[str, Any]) -> tuple[bool, str]:
    """T2 may run only with a measured BENT_FRACTION and a measured theta distribution."""
    bend = calibration.get("bending", {})
    bf = bend.get("BENT_FRACTION", {})
    th = bend.get("theta_deg_all", {})
    if bf.get("source_kind") != "measured":
        return False, f"BENT_FRACTION is {bf.get('value', UNAVAILABLE)} (no measured bend records)"
    if th.get("source_kind") != "measured":
        return False, "theta distribution unavailable (no theta measured with the A1.5 points)"
    return True, "measured"


def run_calibration(data_root: Path, out: Path, *, audit_run: Path | None = None) -> dict[str, Any]:
    cfg = load_config()
    rec = Path(data_root) / cfg["records"]["leaf_bend"]
    if rec.is_file():
        rows = read_bend_records(rec)
        source = dict(path=rec.as_posix(), sha256=hashlib.sha256(rec.read_bytes()).hexdigest())
    else:
        rows, source = [], dict(path=rec.as_posix(), status="absent")
    examples = Path(data_root) / cfg["records"]["bend_examples_illustrative"]
    result = dict(schema=CALIBRATION_SCHEMA, records=source, bending=derive_bending(rows, rec.as_posix()),
                  illustrative_examples=dict(path=examples.as_posix(), present=examples.is_file(),
                                             use="NOT counted: picked to illustrate each type, not a sample"),
                  generator_audit=generator_audit(audit_run, cfg) if audit_run else None,
                  angles_only="close-ups carry no scale; no length is ever taken from them")
    ok, why = calibration_gate(result)
    result["t2_gate"] = dict(open=ok, reason=why)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def write_template(folder: Path) -> list[Path]:
    """Template + README for the human table (written under build/, never into data/records/)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tpl = folder / "leaf_bend_records.csv"
    with tpl.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerow(["EXAMPLE-1", "dwt_101", "2026-10-12T11:00:00+09:00", "IMG_0000.HEIC", "<sha256 of the photo>",
                    "M", "3", "bent", "95", "10;40;35", "blade curls upward along midrib", "<initials>",
                    "delete this example row"])
    readme = folder / "README.md"
    readme.write_text(
        "# leaf_bend_records.csv - side close-up bend records (human-owned)\n\n"
        "Copy to `data/records/leaf_bend/leaf_bend_records.csv` and fill one row per compound leaf seen from the side.\n"
        "The agents only read it. Rules:\n\n"
        "- **Angles only.** Close-ups have no scale; never write a length here.\n"
        "- `bend_type`: flat | bent | intermediate | unknown (definitions: docs/정의집, 휨 타입). The flat/bent theta\n"
        "  boundary is on hold, so the type is your judgement, not a threshold.\n"
        "- `theta_deg`: A1.5 angle between P0->P1 (petiole) and P2->P3 (rachis middle -> last pair node), 0 = straight.\n"
        "  Leave empty when P0, P1 or P3 is not visible.\n"
        "- `alpha_deg`: optional per-node angles alpha_1;alpha_2;... (MTG 12.3, + = bends down), empty if not visible.\n"
        "- `curl_note`: free text about leaflet blade curl (no number - curl is undefined, MTG 8.3).\n"
        "- `photo_sha256`: sha256 of the original photo file; `axis` M / S1 ...; `rank` as in the plant record.\n"
        "- Record EVERY compound you can judge in a photo, not just clear examples - the fraction of bent leaves is\n"
        "  computed from these rows, so picking only typical ones biases it.\n",
        encoding="utf-8")
    return [tpl, readme]


def main(argv: list[str] | None = None) -> None:
    import argparse

    from tomato3d.config import get_settings

    ap = argparse.ArgumentParser(description="T0 bend calibration intake")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-run", help="synthetic run folder whose generator parameters are audited")
    ap.add_argument("--template", help="write the human template + README into this folder (under build/)")
    args = ap.parse_args(argv)
    if args.template:
        write_template(Path(args.template))
    r = run_calibration(get_settings().data_root, Path(args.out),
                        audit_run=Path(args.audit_run) if args.audit_run else None)
    print(json.dumps(dict(records=r["records"], bent_fraction=r["bending"]["BENT_FRACTION"], t2_gate=r["t2_gate"]),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
