"""Compare individual/canopy leaf PPFD using the existing direct-light engine.

Inputs are explicit scene/geometry cases, not PPFD values copied between plants.
Only light-based feasibility is assessed; growth and agronomic optima stay unavailable.
Run: python -m tools.canopy_light --help
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
import sys

import numpy as np

from tomato3d.io import schemas
from tomato3d.light.geometry import load_geometry, triangle_areas_mm2
from tomato3d.light.pipeline import run_light
from tools.leaf_light.common import (ROOT, StudyError, file_hash, number, output_dir,
                                     read_json, rows, write_json, write_rows)

SUMMARY_COLUMNS = ("case_id", "scenario_id", "geometry_variant_id", "data_kind",
    "n_plants", "plot_area_m2", "plants_per_m2", "lower_area_mm2", "upper_area_mm2",
    "lower_ppfd_umol_m2_s", "upper_ppfd_umol_m2_s", "lower_ppfd_min_umol_m2_s",
    "upper_ppfd_max_umol_m2_s", "absorbed_mol_day", "electricity_kwh_day",
    "light_criteria_status", "growth_prediction", "agronomic_optimum")


def identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise StudyError(f"{name}: use 1-80 ASCII letters/digits/underscore/dot/hyphen")
    return value


def validate_plan(plan: dict) -> None:
    if plan.get("schema_version") != "canopy-light-1":
        raise StudyError("Unsupported canopy plan")
    if not plan.get("cases"):
        raise StudyError("Supply explicit lighting/placement cases")
    seen = set()
    for case in plan["cases"]:
        cid = identifier(case.get("case_id"), "case_id")
        if cid in seen:
            raise StudyError("Duplicate case_id")
        seen.add(cid)
        if case.get("data_kind") not in ("observed_geometry", "hypothetical_layout", "synthetic"):
            raise StudyError("Each case needs an explicit data_kind")
        if number(case.get("plot_area_m2"), "plot_area_m2") <= 0:
            raise StudyError("plot_area_m2 must be a positive actual floor footprint")
        number(case.get("height_split_board_mm"), "height_split_board_mm")
        if not case.get("instances"):
            raise StudyError("Each case needs explicit plant instances")
        ids = [identifier(item.get("plant_id"), "plant_id") for item in case["instances"]]
        if len(ids) != len(set(ids)):
            raise StudyError("Instance plant IDs must be unique within a case")
        if case["data_kind"] == "observed_geometry" and any(item["plant_id"] != item["source_plant_id"] for item in case["instances"]):
            raise StudyError("Copied template plants must be hypothetical_layout, not observed_geometry")
    criteria = plan.get("light_criteria", {})
    for key in ("minimum_lower_leaf_ppfd_umol_m2_s", "maximum_upper_leaf_ppfd_umol_m2_s"):
        value = number(criteria.get(key), key, missing=True)
        if value is not None and value < 0:
            raise StudyError("PPFD criteria cannot be negative")


def materialize_case(case: dict, base: Path, folder: Path) -> tuple[schemas.LightScene, Path, dict]:
    geometry = (base / case["geometry"]).resolve()
    scene_path = (base / case["scene"]).resolve()
    source = rows(geometry, schemas.LEAF_SURFACES_CSV_COLUMNS)
    selected = [row for row in source if row["geometry_variant_id"] == case["geometry_variant_id"]]
    if not selected:
        raise StudyError(f"{case['case_id']}: geometry variant has no triangles")
    # The loader rejects invalid units, zero-area faces and repeated triangle identities.
    load_geometry([geometry])
    output, placements = [], []
    scene = deepcopy(read_json(scene_path))
    original_placements = {item["plant_id"]: item for item in scene["plants"]}
    origins = []
    provenance = []
    for item in case["instances"]:
        pid, source_id = item["plant_id"], item["source_plant_id"]
        records = [r for r in selected if r["plant_id"] == source_id]
        if not records:
            raise StudyError(f"Missing source plant geometry: {source_id}")
        if "placement" in item:
            placement = dict(item["placement"], plant_id=pid)
        else:
            if source_id not in original_placements:
                raise StudyError(f"No measured/explicit placement for {source_id}")
            placement = dict(original_placements[source_id], plant_id=pid)
        parsed = schemas.PlantPlacement.model_validate(placement)
        origin = tuple(parsed.board_origin_grow_mm.value)
        if origin in origins:
            raise StudyError("Two plant instances share the same board origin; check the layout")
        origins.append(origin)
        placements.append(parsed.model_dump(mode="json"))
        for row in records:
            output.append(dict(row, plant_id=pid,
                geometry_source="assumed" if case["data_kind"] in ("synthetic", "hypothetical_layout") else row["geometry_source"]))
        provenance.append({"instance_plant_id":pid, "source_plant_id":source_id,
                           "copied_template":pid != source_id})
    scene["plants"] = placements
    # All lamp angle/height overrides use the existing validated scenario schema.
    # Cases can replace the scenario list explicitly; otherwise retain the source list.
    if "scenarios" in case:
        scene["scenarios"] = case["scenarios"]
    model = schemas.LightScene.model_validate(scene)
    material = folder / "leaf_surfaces.csv"
    write_rows(material, output, schemas.LEAF_SURFACES_CSV_COLUMNS)
    write_json(folder / "scene.json", model.model_dump(mode="json"))
    return model, material, {"geometry_path":str(geometry), "geometry_sha256":file_hash(geometry),
        "scene_path":str(scene_path), "scene_sha256":file_hash(scene_path), "instances":provenance}


def height_groups(geometry: Path, variant: str, split_mm: float) -> dict[tuple[str, str], str]:
    surfaces = load_geometry([geometry])[variant]
    groups = defaultdict(list)
    areas = triangle_areas_mm2(surfaces.vertices_board_mm)
    for i in range(surfaces.n_triangles):
        if surfaces.is_receiver[i]:
            groups[(surfaces.plant_ids[i], surfaces.organ_ids[i])].append(i)
    return {key: "lower" if np.average(surfaces.vertices_board_mm[ids,:,2].mean(axis=1),
        weights=areas[ids]) < split_mm else "upper" for key, ids in groups.items()}


def finite(value: object) -> float | None:
    return number(value, "light output", missing=True)


def summarize(case: dict, result, groups: dict, criteria: dict) -> list[dict]:
    records = []
    variant = case["geometry_variant_id"]
    for headline in result.scenario_rows:
        if headline["geometry_variant_id"] != variant:
            continue
        leaves = [r for r in result.leaf_rows if r["geometry_variant_id"] == variant and r["scenario_id"] == headline["scenario_id"]]
        parts = {band:[r for r in leaves if groups[(r["plant_id"],r["organ_id"])] == band] for band in ("lower","upper")}
        def values(band: str) -> list[float | None]:
            return [finite(r["ppfd_mean_full_output_umol_m2_s"]) for r in parts[band]]
        def mean(band: str) -> float | None:
            v = values(band)
            return float(np.average(v,weights=[float(r["area_mm2"]) for r in parts[band]])) if v and all(x is not None for x in v) else None
        low, high = values("lower"), values("upper")
        low_min = min(low) if low and all(x is not None for x in low) else None
        high_max = max(high) if high and all(x is not None for x in high) else None
        minimum = number(criteria.get("minimum_lower_leaf_ppfd_umol_m2_s"), "minimum",missing=True)
        maximum = number(criteria.get("maximum_upper_leaf_ppfd_umol_m2_s"), "maximum",missing=True)
        status = "unavailable"
        if minimum is not None and maximum is not None and low_min is not None and high_max is not None:
            status = "meets_light_criteria" if low_min >= minimum and high_max <= maximum else "outside_light_criteria"
        records.append(dict(case_id=case["case_id"], scenario_id=headline["scenario_id"], geometry_variant_id=variant,
            data_kind=case["data_kind"], n_plants=len(case["instances"]), plot_area_m2=case["plot_area_m2"],
            plants_per_m2=len(case["instances"])/float(case["plot_area_m2"]),
            lower_area_mm2=sum(float(r["area_mm2"]) for r in parts["lower"]),
            upper_area_mm2=sum(float(r["area_mm2"]) for r in parts["upper"]),
            lower_ppfd_umol_m2_s=mean("lower"), upper_ppfd_umol_m2_s=mean("upper"),
            lower_ppfd_min_umol_m2_s=low_min, upper_ppfd_max_umol_m2_s=high_max,
            absorbed_mol_day=finite(headline["absorbed_mol_day"]), electricity_kwh_day=finite(headline["electricity_kwh_day"]),
            light_criteria_status=status, growth_prediction=None, agronomic_optimum=None))
    return records


def run(plan_path: Path, out: Path) -> dict:
    plan = read_json(plan_path); validate_plan(plan)
    folder = output_dir(out)
    summaries, cases = [], []
    table_specs = (
        ("triangle_rows","leaf_triangle_light.csv",schemas.LEAF_TRIANGLE_LIGHT_CSV_COLUMNS),
        ("leaf_rows","leaf_light.csv",schemas.LEAF_LIGHT_CSV_COLUMNS),
        ("plant_rows","plant_light.csv",schemas.PLANT_LIGHT_CSV_COLUMNS),
        ("scenario_rows","scenario_summary.csv",schemas.SCENARIO_SUMMARY_CSV_COLUMNS))
    for case in plan["cases"]:
        case_folder = folder / case["case_id"]; case_folder.mkdir()
        scene, material, provenance = materialize_case(case,plan_path.parent,case_folder)
        result = run_light(scene=scene,scene_path=case_folder/"scene.json",geometry_paths=[material],
            readings_path=None,reference_variant_id=case["geometry_variant_id"],calibration_variant_id="no_plants",
            run_id=case["case_id"],created_at=datetime.now(timezone.utc))
        for attr, filename, columns in table_specs:
            write_rows(case_folder/filename,getattr(result,attr),columns)
        write_json(case_folder/"run_summary.json",result.summary.model_dump(mode="json"))
        bands = height_groups(material,case["geometry_variant_id"],float(case["height_split_board_mm"]))
        write_rows(case_folder/"leaf_height_groups.csv",[{"plant_id":p,"organ_id":o,"height_group":b} for (p,o),b in bands.items()],
                   ("plant_id","organ_id","height_group"))
        summaries.extend(summarize(case,result,bands,plan.get("light_criteria",{})))
        cases.append({"case_id":case["case_id"],"provenance":provenance,"warnings":result.summary.model_dump(mode="json").get("warnings",[])})
    write_rows(folder/"comparison.csv",summaries,SUMMARY_COLUMNS)
    report = {"schema_version":"canopy-comparison-1","plan_sha256":file_hash(plan_path),"cases":cases,
        "growth_prediction":"unavailable","agronomic_optimum":"unavailable",
        "limits":["Raw direct-light predictions; no quantum-sensor validation in this comparison command.",
            "Geometry is static. Crowding-induced changes of leaf posture and growth are not simulated.",
            "Area-weighted leaf-centroid height groups; upper/lower means canopy height, not leaf face.",
            "No reflection, spectral growth recipe, temperature/nutrient model or trained vision model.",
            "Template copies are hypothetical plants. Light-feasible cases are not agronomic recommendations."]}
    write_json(folder/"comparison.json",report)
    headers = ("case_id","scenario_id","plants_per_m2","lower_ppfd_umol_m2_s","upper_ppfd_umol_m2_s","light_criteria_status")
    table = ''.join('<tr>'+''.join('<td>'+escape(str(row[key] if row[key] is not None else 'unavailable'))+'</td>' for key in headers)+'</tr>' for row in summaries)
    (folder/"comparison.html").write_text('<!doctype html><meta charset="utf-8"><title>Canopy PPFD comparison</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;background:#101713;color:#e2eee6}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #46594b}a{color:#8ddfa9}</style>'
        '<h1>Canopy PPFD comparison</h1><p>Static direct-light estimates. Growth and optimal planting density are unavailable.</p>'
        '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+table+'</table><ul>'
        +''.join('<li>'+escape(line)+'</li>' for line in report['limits'])+'</ul><a href="comparison.csv">CSV</a>',encoding='utf-8')
    return report


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    args = cli.parse_args(argv)
    try:
        run(args.plan,args.out)
    except (ValueError,OSError,KeyError) as exc:
        print(f'Canopy input error: {exc}',file=sys.stderr); return 2
    print(args.out.resolve()/'comparison.html'); return 0


if __name__ == '__main__':
    raise SystemExit(main())
