"""Explicit user-parameter 30-day growth scenarios; no fitted physiology."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import date, datetime, timedelta
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

HORIZON_DAYS = 30
MAX_COMPOUNDS = 300  # Computational limit, not a biological threshold.
MAX_REQUEST_BYTES = 1_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PROJECT_ROOT / "build"
LEAF_FIELDS = ("attachment_height_mm", "azimuth_deg", "elevation_deg", "axis_length_mm",
               "blade_width_mm", "area_cm2", "unfolded_fraction")
GROWTH_FIELDS = ("stem_mm_per_dev_day", "stem_max_mm", "axis_mm_per_dev_day", "axis_max_mm",
                 "width_mm_per_dev_day", "width_max_mm", "area_cm2_per_dev_day", "area_max_cm2",
                 "unfolding_per_dev_day", "elevation_deg_per_dev_day", "elevation_target_deg")
ENV_FIELDS = ("temperature_c", "relative_humidity_pct", "ppfd_umol_m2_s", "photoperiod_h",
              "ec_ms_cm", "ph", "co2_ppm")
ASSUMPTIONS = [
    "User-entered parameters are assumptions, not fitted cultivar coefficients.",
    "Dimensions expand linearly per development day until the supplied caps; no senescence.",
    "Existing attachment heights remain fixed; stem elongation occurs above existing attachments.",
    "New compounds attach at apical stem height at their exact emergence time.",
    "Elevation approaches the supplied target at the supplied rate; azimuth stays fixed after birth.",
    "Displayed width is blade_width_mm times unfolded_fraction; area is an independent input trajectory.",
    "Daily multipliers and optional piecewise-linear environment response tables are user assumptions.",
    "Manual mode records environment only; response-table mode multiplies the explicitly enabled factors.",
    "Axes and width bars are schematic envelopes, not measured tissue or optical surfaces.",
    "No carbon balance, phototropism, nutrient uptake, mortality, flowering or yield calculation.",
    "No calibrated prediction intervals or validated real-plant forecast accuracy.",
]


def number(value: Any, name: str, minimum: float | None = 0,
           maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: supply a finite number (missing is not zero)")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name}: outside allowed range [{minimum}, {maximum}]")
    return float(value)


def blank_input() -> dict:
    return {"schema": "monthly-growth-input-1", "parameter_source": "user_assumptions",
            "metadata": {k: None for k in ("plant_id", "cultivar", "start_date", "sowing_date", "transplant_date")},
            "initial_stem_height_mm": None,
            "initial_compounds": [{"compound_id": "entered_001", **{k: None for k in LEAF_FIELDS}}],
            "growth": {k: None for k in GROWTH_FIELDS},
            "response": {"mode": "manual", "curves": {}},
            "emergence": {"enabled": None, "next_dev_day": None, "interval_dev_days": None,
                          "first_azimuth_deg": None, "divergence_deg": None,
                          "birth": {k: None for k in LEAF_FIELDS if k not in ("attachment_height_mm", "azimuth_deg")}},
            "days": [{"day": i, "development_multiplier": None,
                      **{k: None for k in ENV_FIELDS}} for i in range(1, HORIZON_DAYS + 1)]}


def validate(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object")
    p = copy.deepcopy(payload)
    if p.get("schema") != "monthly-growth-input-1" or p.get("parameter_source") != "user_assumptions":
        raise ValueError("Use monthly-growth-input-1 and parameter_source=user_assumptions")
    meta = p["metadata"]
    for key in ("plant_id", "cultivar"):
        if meta.get(key) is not None and not isinstance(meta[key], str):
            raise ValueError(f"metadata.{key}: string or null required")
    dates = {}
    for key in ("sowing_date", "transplant_date", "start_date"):
        if meta.get(key) is not None:
            if not isinstance(meta[key], str):
                raise ValueError(f"{key}: YYYY-MM-DD or null required")
            dates[key] = date.fromisoformat(meta[key])
            if dates[key].isoformat() != meta[key]:
                raise ValueError(f"{key}: YYYY-MM-DD required")
    ordered = [dates[k] for k in ("sowing_date", "transplant_date", "start_date") if k in dates]
    if ordered != sorted(ordered):
        raise ValueError("Dates must follow sowing <= transplant <= forecast start")
    height = number(p["initial_stem_height_mm"], "initial_stem_height_mm")
    growth = p["growth"]
    for key in GROWTH_FIELDS:
        number(growth[key], f"growth.{key}", -90 if key == "elevation_target_deg" else 0,
               90 if key == "elevation_target_deg" else None)
    if height > growth["stem_max_mm"]:
        raise ValueError("Initial stem height exceeds its supplied cap")

    def check_leaf(leaf: dict, name: str, initial: bool) -> None:
        for key in LEAF_FIELDS:
            if not initial and key in ("attachment_height_mm", "azimuth_deg"):
                continue
            number(leaf[key], f"{name}.{key}", None if key == "azimuth_deg" else -90 if key == "elevation_deg" else 0,
                   90 if key == "elevation_deg" else 1 if key == "unfolded_fraction" else None)
        if initial and leaf["attachment_height_mm"] > height:
            raise ValueError(f"{name}: attachment lies above initial stem")
        for field, cap in (("axis_length_mm", "axis_max_mm"), ("blade_width_mm", "width_max_mm"), ("area_cm2", "area_max_cm2")):
            if leaf[field] > growth[cap]:
                raise ValueError(f"{name}.{field}: exceeds its supplied cap")

    leaves = p["initial_compounds"]
    if not isinstance(leaves, list) or len(leaves) > MAX_COMPOUNDS:
        raise ValueError(f"initial_compounds: a list of at most {MAX_COMPOUNDS} entries required")
    ids = set()
    for i, leaf in enumerate(leaves):
        key = leaf["compound_id"]
        if not isinstance(key, str) or not key.strip() or key.startswith("scenario_new_") or key in ids:
            raise ValueError("Compound IDs must be unique, nonempty, and outside scenario_new_ namespace")
        ids.add(key)
        check_leaf(leaf, f"compound[{i}]", True)
    emergence = p["emergence"]
    if type(emergence["enabled"]) is not bool:
        raise ValueError("Choose emergence.enabled=true or false explicitly")
    if emergence["enabled"]:
        for key in ("next_dev_day", "interval_dev_days"):
            if number(emergence[key], f"emergence.{key}") <= 0:
                raise ValueError(f"emergence.{key}: must be positive")
        for key in ("first_azimuth_deg", "divergence_deg"):
            number(emergence[key], f"emergence.{key}", None)
        check_leaf(emergence["birth"], "emergence.birth", False)
    response = p["response"]
    if response["mode"] not in ("manual", "curves") or not isinstance(response["curves"], dict):
        raise ValueError("response requires mode=manual or curves and a curves mapping")
    if response["mode"] == "curves":
        if not response["curves"]:
            raise ValueError("Enable at least one response curve, or select manual mode")
        for key, knots in response["curves"].items():
            if key not in ENV_FIELDS or not isinstance(knots, list) or len(knots) < 2:
                raise ValueError(f"response.{key}: known environment field and at least two knots required")
            for knot in knots:
                number(knot["value"], f"response.{key}.value", -273.15 if key == "temperature_c" else 0,
                       {"relative_humidity_pct": 100, "photoperiod_h": 24, "ph": 14}.get(key))
                number(knot["multiplier"], f"response.{key}.multiplier")
            if any(a["value"] >= b["value"] for a, b in zip(knots, knots[1:])):
                raise ValueError(f"response.{key}: knot values must be strictly increasing")
    days = p["days"]
    if not isinstance(days, list) or len(days) != HORIZON_DAYS:
        raise ValueError("Supply exactly 30 ordered daily environment/multiplier records")
    for i, day in enumerate(days, 1):
        if type(day["day"]) is not int or day["day"] != i:
            raise ValueError("Daily rows must have consecutive integer day values 1..30")
        number(day["development_multiplier"], f"day {i}.development_multiplier")
        for key in ENV_FIELDS:
            if day.get(key) is not None:
                number(day[key], f"day {i}.{key}", -273.15 if key == "temperature_c" else 0,
                       {"relative_humidity_pct": 100, "photoperiod_h": 24, "ph": 14}.get(key))
    total_dev = sum(effective_step(p, d)[0] for d in days)
    if not math.isfinite(total_dev):
        raise ValueError("Development accumulation overflow")
    if emergence["enabled"]:
        ratio = max(0, total_dev - emergence["next_dev_day"]) / emergence["interval_dev_days"]
        if not math.isfinite(ratio) or (total_dev >= emergence["next_dev_day"] and math.floor(ratio) + 1 + len(leaves) > MAX_COMPOUNDS):
            raise ValueError(f"Scenario exceeds computational limit of {MAX_COMPOUNDS} compounds")
    return p


def effective_step(payload: dict, day: dict) -> tuple[float, dict]:
    """Interpolate explicit response knots without extrapolation or cultivar defaults."""
    factors = {}
    if payload["response"]["mode"] == "curves":
        for key, knots in payload["response"]["curves"].items():
            value = day.get(key)
            if value is None:
                raise ValueError(f"day {day['day']}.{key}: required by enabled response curve")
            if not knots[0]["value"] <= value <= knots[-1]["value"]:
                raise ValueError(f"day {day['day']}.{key}: outside supplied response domain; no extrapolation")
            for a, b in zip(knots, knots[1:]):
                if a["value"] <= value <= b["value"]:
                    fraction = (value - a["value"]) / (b["value"] - a["value"])
                    factors[key] = a["multiplier"] * (1 - fraction) + b["multiplier"] * fraction
                    break
    effective = day["development_multiplier"] * math.prod(factors.values())
    if not math.isfinite(effective):
        raise ValueError("Environment response multiplication overflow")
    return effective, factors


def simulate(payload: dict) -> dict:
    p = validate(payload)
    g, e = p["growth"], p["emergence"]
    initial_height = p["initial_stem_height_mm"]
    born = [{**v, "birth_dev_day": 0., "birth_calendar_day": None,
             "origin": "user_entered_initial_state"} for v in p["initial_compounds"]]
    frames, cumulative = [], 0.
    next_event = e["next_dev_day"] if e["enabled"] else math.inf
    new_count = 0
    start = p["metadata"].get("start_date")

    def stem(dev_day: float) -> float:
        return min(g["stem_max_mm"], initial_height + g["stem_mm_per_dev_day"] * dev_day)

    def snapshot(day: int, environment: dict | None) -> None:
        compounds = []
        for leaf in born:
            elapsed = max(0., cumulative - leaf["birth_dev_day"])
            state = dict(leaf)
            for field, rate, cap in (("axis_length_mm", "axis_mm_per_dev_day", "axis_max_mm"),
                                     ("blade_width_mm", "width_mm_per_dev_day", "width_max_mm"),
                                     ("area_cm2", "area_cm2_per_dev_day", "area_max_cm2")):
                state[field] = min(g[cap], leaf[field] + g[rate] * elapsed)
            state["unfolded_fraction"] = min(1., leaf["unfolded_fraction"] + g["unfolding_per_dev_day"] * elapsed)
            delta = g["elevation_target_deg"] - leaf["elevation_deg"]
            state["elevation_deg"] = leaf["elevation_deg"] + math.copysign(min(abs(delta), g["elevation_deg_per_dev_day"] * elapsed), delta)
            state["spread_width_mm"] = state["blade_width_mm"] * state["unfolded_fraction"]
            az, el = math.radians(state["azimuth_deg"] % 360), math.radians(state["elevation_deg"])
            length = state["axis_length_mm"]
            state["attachment_xyz_mm"] = [0., 0., leaf["attachment_height_mm"]]
            state["tip_xyz_mm"] = [length * math.cos(el) * math.cos(az), length * math.cos(el) * math.sin(az),
                                     leaf["attachment_height_mm"] + length * math.sin(el)]
            compounds.append(state)
        area = sum(v["area_cm2"] for v in compounds)
        if not math.isfinite(area):
            raise ValueError("Area accumulation overflow")
        frames.append({"day": day, "date": (date.fromisoformat(start) + timedelta(days=day)).isoformat() if start else None,
                       "development_days": cumulative, "stem_height_mm": stem(cumulative),
                       "compound_count": len(compounds), "area_cm2": area,
                       "environment": environment, "compounds": compounds})

    snapshot(0, None)
    for env in p["days"]:
        previous = cumulative
        increment, factors = effective_step(p, env)
        cumulative += increment
        while next_event <= cumulative:
            new_count += 1
            # A positive next_event and chronological advancement guarantee a positive multiplier here.
            fraction = (next_event - previous) / increment
            azimuth = e["first_azimuth_deg"] + (new_count - 1) * e["divergence_deg"]
            if not math.isfinite(azimuth):
                raise ValueError("Azimuth accumulation overflow")
            born.append({**e["birth"], "compound_id": f"scenario_new_{new_count:03d}",
                         "attachment_height_mm": stem(next_event), "azimuth_deg": azimuth % 360,
                         "birth_dev_day": next_event, "birth_calendar_day": env["day"] - 1 + fraction,
                         "origin": "scenario_generated_event"})
            next_event = e["next_dev_day"] + new_count * e["interval_dev_days"]
            if len(born) > MAX_COMPOUNDS:
                raise ValueError("Compound computational limit exceeded")
        snapshot(env["day"], {**env, "response_factors": factors, "effective_development_increment": increment})
    return {"schema": "monthly-growth-result-1", "model_status": "user_parameter_scenario_not_calibrated",
            "validated_growth": False, "suitable_for_ppfd": False, "uncertainty": "unavailable",
            "environment_response": p["response"]["mode"] + "; user assumptions, not fitted",
            "coordinate_frame": "schematic_local_plant_z_up_mm", "horizon_days": HORIZON_DAYS,
            "assumptions": ASSUMPTIONS, "input": p, "frames": frames}


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def page(payload: dict, result: dict | None = None, live: bool = False) -> str:
    boot = encode({"input": payload, "result": result, "live": live}).replace("<", "\\u003c")
    return Path(__file__).with_suffix(".html").read_text(encoding="utf-8").replace("__MONTHLY_BOOT__", boot)


def export_result(payload: dict, result: dict, out: Path) -> None:
    if not out.resolve().is_relative_to(BUILD_ROOT.resolve()) or out.resolve() == BUILD_ROOT.resolve():
        raise ValueError("Output must be a new directory within this project's build/")
    out.mkdir(parents=True, exist_ok=False)
    (out / "input.json").write_text(encode(payload), encoding="utf-8")
    (out / "result.json").write_text(encode(result), encoding="utf-8")
    (out / "viewer.html").write_text(page(payload, result), encoding="utf-8")
    daily_fields = ["day", "date", "development_days", "stem_height_mm", "compound_count", "area_cm2"]
    compound_fields = ["day", "date", "compound_id", "origin", "birth_calendar_day", "birth_dev_day", *LEAF_FIELDS, "spread_width_mm"]
    for name, fields, rows in (
        ("daily.csv", daily_fields, result["frames"]),
        ("compounds.csv", compound_fields, [dict(v, day=f["day"], date=f["date"]) for f in result["frames"] for v in f["compounds"]]),
        ("environment.csv", ["day", "development_multiplier", *ENV_FIELDS, "effective_development_increment"],
         [f["environment"] for f in result["frames"][1:]]),
    ):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    hashes = {v.name: hashlib.sha256(v.read_bytes()).hexdigest() for v in out.iterdir() if v.is_file()}
    code = {v.name: hashlib.sha256(v.read_bytes()).hexdigest() for v in (Path(__file__), Path(__file__).with_suffix(".html"))}
    (out / "manifest.json").write_text(encode({"schema": "monthly-growth-run-1", "files": hashes, "code": code,
                                               "validated_growth": False, "parameter_source": "user_assumptions"}), encoding="utf-8")


def serve(port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, content: str, mime: str = "application/json") -> None:
            data = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/":
                self.respond(200, page(blank_input(), live=True), "text/html")
            else:
                self.respond(404, encode({"error": "Not found"}))

        def do_POST(self) -> None:
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != origin.removeprefix("http://") or self.headers.get("Origin") not in (None, origin):
                self.respond(403, encode({"error": "Local same-origin requests only"})); return
            if self.path not in ("/simulate", "/save"):
                self.respond(404, encode({"error": "Not found"})); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Supply a JSON body under 1 MB")
                payload = json.loads(self.rfile.read(length))
                result = simulate(payload)
                if self.path == "/save":
                    out = BUILD_ROOT / f"monthly_growth_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
                    export_result(payload, result, out)
                    self.respond(200, encode({"path": str(out.resolve())}))
                else:
                    self.respond(200, encode(result))
            except (ValueError, KeyError, TypeError, OverflowError, OSError) as exc:
                self.respond(400, encode({"error": str(exc)}))

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="Write blank parameters in a new build directory")
    template.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run", help="Calculate and export 30 days from explicit user parameters")
    run.add_argument("--input", type=Path, required=True); run.add_argument("--out", type=Path, required=True)
    server = sub.add_parser("serve", help="Open local parameter editor/API")
    server.add_argument("--port", type=int, default=8855)
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            serve(args.port)
        elif args.command == "template":
            if not args.out.resolve().is_relative_to(BUILD_ROOT.resolve()) or args.out.resolve() == BUILD_ROOT.resolve():
                raise ValueError("Output must be a new build directory")
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "input.json").write_text(encode(blank_input()), encoding="utf-8")
            (args.out / "viewer.html").write_text(page(blank_input()), encoding="utf-8")
        else:
            payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
            result = simulate(payload)
            export_result(payload, result, args.out)
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as exc:
        parser.exit(2, f"Monthly scenario: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
