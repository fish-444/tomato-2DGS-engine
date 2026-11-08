"""Leaf-age-aware leaflet sizes (session E3; Session G rule, user 2026-10-14 19:55).

Rule (G1): all leaflets of a compound exist from birth; they ENLARGE in order terminal -> pair nearest the tip ->
... -> pair nearest the main stem; the rachis lengthens in proportion; FINAL ratios are unchanged. Offsets in thermal
time after the compound's birth: terminal 0, pair m (tip side) 1 x stagger, ..., pair 1 (stem side) m x stagger.

Growth form = the engine's (``tools/vtc_development_native.rgg::targetMass``): mass fraction
``(s(t) - s0) / (1 - s0)`` with ``s(t) = 1 / (1 + exp(-K (t - MID_DD)))``, ``t = max(0, age - offset)``; a leaflet's
length scales with sqrt(mass) (engine ``kind == 9``), a rachis segment with cbrt(mass) (other organs).

Single source: ``LEAFLET_STAGGER_DD`` / ``LEAFLET_GROWTH_ORDER`` / ``PHYLLOCHRON`` are read from
``tools/microtom_parameters.csv`` when registered there; until Session G registers the stagger, the local copy in
the priors file is used and marked ``pending_registry``. Leaflet ``MID_DD`` / ``K`` come from the engine's parameter
table (``tools/vtc_development.py``, V3 hypothesis, user_assumed). No other value is invented.

Compound age (in this order): registry / capture history if any -> node position counted from the apex on its axis
(age = (position + 1) x PHYLLOCHRON) -> ``unavailable`` (caller falls back to the mature prior and flags it).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.recon.stage1.priors import PROJECT_ROOT, Priors

REGISTRY = PROJECT_ROOT / "tools" / "microtom_parameters.csv"


def _registry() -> dict[str, dict]:
    if not REGISTRY.exists():
        return {}
    with REGISTRY.open(encoding="utf-8", newline="") as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def parameters(priors: Priors) -> dict:
    """Stagger scenarios, growth order, phyllochron, leaflet MID_DD / K - with where each came from."""
    reg = _registry()
    local = priors.raw["leaf_age"]
    out = {}
    if "LEAFLET_STAGGER_DD" in reg:
        r = reg["LEAFLET_STAGGER_DD"]
        cands = [float(x) for x in r["candidates"].strip("[]").split(",")] if r.get("candidates") else []
        # the registry lists 0 (= simultaneous, the regression case) beside the three declared scenarios
        positive = sorted(c for c in cands if c > 0)
        out["stagger_dd"] = dict(zip(("low", "base", "high"), positive)) if len(positive) == 3 else \
            {"base": float(r["value"])}
        out["stagger_source"] = f"registry tools/microtom_parameters.csv ({r['source_kind']})"
    else:
        out["stagger_dd"] = dict(local["LEAFLET_STAGGER_DD"]["scenarios"])
        out["stagger_source"] = "pending_registry: local copy of the Session G G3 values (priors file)"
    if "LEAFLET_GROWTH_ORDER" in reg:   # the registry stores the option as a JSON string ('"basipetal"')
        v = reg["LEAFLET_GROWTH_ORDER"]["value"].strip()
        out["growth_order"] = json.loads(v) if v.startswith('"') else v
    else:
        out["growth_order"] = local["LEAFLET_GROWTH_ORDER"]["value"]
    ph = reg.get("PHYLLOCHRON")
    out["phyllochron_dd"] = float(ph["value"]) if ph else float(local["PHYLLOCHRON_DD"]["value"])
    out["phyllochron_source"] = (f"registry PHYLLOCHRON ({ph['source_kind']}, {ph['cultivar']})" if ph else
                                 local["PHYLLOCHRON_DD"]["source_ref"])
    out["mid_dd"] = float(local["LEAFLET_MID_DD"]["value"])
    out["k"] = float(local["LEAFLET_K"]["value"])
    return out


def mass_fraction(t: float, mid: float, k: float) -> float:
    if t <= 0:
        return 0.0
    s0 = 1 / (1 + math.exp(k * mid))
    s = 1 / (1 + math.exp(-k * (t - mid)))
    return max(0.0, min(1.0, (s - s0) / (1 - s0)))


def leaflet_length_fraction(t: float, par: Mapping) -> float:
    return math.sqrt(mass_fraction(t, par["mid_dd"], par["k"]))


def rachis_length_fraction(t: float, par: Mapping) -> float:
    return mass_fraction(t, par["mid_dd"], par["k"]) ** (1 / 3)


def offset_dd(position: str, pair_index: int | None, n_pairs: int, stagger: float) -> float:
    """Terminal 0; pair p (1 = stem side, n = tip side) -> (n - p + 1) x stagger (basipetal order)."""
    if position == "terminal" or pair_index is None:
        return 0.0
    return (n_pairs - pair_index + 1) * stagger


def compound_ages(registry: Mapping[str, Mapping], par: Mapping) -> dict[str, dict]:
    """Age (degree days) per compound from its node position counted from the apex on its axis."""
    by_axis: dict[str, list[tuple[int, str]]] = {}
    for cid, reg in registry.items():
        if reg.get("birth_dd") is not None:
            continue
        if reg.get("rank") is None:
            continue
        # position along the axis = node index (U5 ranks share rank 1 on the first two nodes; A7); rank is the fallback
        by_axis.setdefault(reg.get("axis", "M"), []).append((int(reg.get("node_index", reg["rank"])), cid))
    out: dict[str, dict] = {}
    for cid, reg in registry.items():
        if reg.get("birth_dd") is not None:
            out[cid] = dict(age_dd=float(reg["age_dd"]), age_source="registry_capture_history")
    for axis, items in by_axis.items():
        items.sort(reverse=True)  # highest rank = youngest = position 0 from the apex
        for pos, (_, cid) in enumerate(items):
            out[cid] = dict(age_dd=(pos + 1) * par["phyllochron_dd"], age_source="node_position_from_apex",
                            node_position_from_apex=pos)
    for cid in registry:
        out.setdefault(cid, dict(age_dd=None, age_source=UNAVAILABLE))
    return out
