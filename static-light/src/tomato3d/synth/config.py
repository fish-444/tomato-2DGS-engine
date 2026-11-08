"""``generator_config.json``: every generator range with its evidence.

Each range carries ``source_kind`` in :data:`SOURCE_KINDS` and a non-empty ``source_ref``.
No range here is a default of any real model; the file lives under the run's ``build/``
folder. The initial ranges ship as ``generator_config_template.json`` next to this module
(``user_assumed`` unless stated) and are replaced by ``measured_range`` values only through
:func:`apply_measured_range`, which keeps the old value in ``history``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tomato3d.synth import GENERATOR_VERSION, ORIGIN

CONFIG_SCHEMA = "synth-generator-config-1"
SOURCE_KINDS = frozenset({"literature_prior", "user_assumed", "measured_range"})
CAPTURE_STATUSES = frozenset({"pending", "integrated", "partial"})
TEMPLATE_PATH = Path(__file__).with_name("generator_config_template.json")


class GeneratorConfigError(ValueError):
    """Raised when ``generator_config.json`` breaks its contract."""


def _check_evidence(name: str, entry: dict[str, Any]) -> None:
    if entry.get("source_kind") not in SOURCE_KINDS:
        raise GeneratorConfigError(f"{name}: source_kind must be one of {sorted(SOURCE_KINDS)}")
    ref = entry.get("source_ref")
    if not isinstance(ref, (str, list, dict)) or not ref:
        raise GeneratorConfigError(f"{name}: source_ref is required")
    if not isinstance(entry.get("unit"), str):
        raise GeneratorConfigError(f"{name}: unit is required")


def validate_generator_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate in place and return ``cfg``; raise :class:`GeneratorConfigError`."""
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise GeneratorConfigError(f"schema must be {CONFIG_SCHEMA!r}")
    if cfg.get("origin") != ORIGIN:
        raise GeneratorConfigError("origin must be 'synthetic'")
    if cfg.get("compound_capture_status") not in CAPTURE_STATUSES:
        raise GeneratorConfigError(f"compound_capture_status must be one of {sorted(CAPTURE_STATUSES)}")
    for name, entry in cfg.get("ranges", {}).items():
        _check_evidence(name, entry)
        low, high = entry.get("min"), entry.get("max")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or low > high:
            raise GeneratorConfigError(f"{name}: need numeric min <= max")
    for name, entry in cfg.get("constants", {}).items():
        _check_evidence(name, entry)
        if "value" not in entry:
            raise GeneratorConfigError(f"{name}: constant needs a value")
    if not isinstance(cfg.get("history", []), list):
        raise GeneratorConfigError("history must be a list")
    return cfg


def load_generator_config(path: Path | None = None) -> dict[str, Any]:
    """Load and validate a generator config (default: the shipped template)."""
    source = Path(path) if path is not None else TEMPLATE_PATH
    cfg = json.loads(source.read_text(encoding="utf-8"))
    cfg.setdefault("generator_version", GENERATOR_VERSION)
    return validate_generator_config(cfg)


def write_generator_config(cfg: dict[str, Any], path: Path) -> Path:
    validate_generator_config(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def config_hash(cfg: dict[str, Any]) -> str:
    """SHA256 of the canonical JSON of ``ranges`` + ``constants``."""
    body = {"ranges": cfg.get("ranges", {}), "constants": cfg.get("constants", {})}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def constant(cfg: dict[str, Any], name: str) -> Any:
    try:
        return cfg["constants"][name]["value"]
    except KeyError as exc:
        raise GeneratorConfigError(f"constant {name!r} missing from generator config") from exc


def range_of(cfg: dict[str, Any], name: str) -> tuple[float, float]:
    try:
        entry = cfg["ranges"][name]
    except KeyError as exc:
        raise GeneratorConfigError(f"range {name!r} missing from generator config") from exc
    return float(entry["min"]), float(entry["max"])


def sample_range(rng: np.random.Generator, cfg: dict[str, Any], name: str) -> float | int:
    """Uniform draw inside a configured range (inclusive integers when ``integer``)."""
    low, high = range_of(cfg, name)
    if cfg["ranges"][name].get("integer"):
        return int(rng.integers(int(round(low)), int(round(high)) + 1))
    return float(rng.uniform(low, high))


def input_path(raw: str) -> Path:
    """A file path recorded in a generator config, resolved on this machine.

    Paths are written where the config was made (e.g. an absolute Windows path to the texture bank). On another
    machine ``TOMATO3D_SYNTH_PATH_MAP`` (see :mod:`tomato3d.config`) maps the recorded prefix to the local one, so the
    config file is never edited and its pinned sha256 still holds. Unmapped paths come back unchanged.
    """
    from tomato3d.config import get_settings

    norm = str(raw).replace("\\", "/")
    for old, new in get_settings().synth_path_map:
        if norm == old or norm.startswith(old + "/"):
            return Path(new) / norm[len(old):].lstrip("/")
    return Path(raw)


def make_rng(seed: int, *keys: Any) -> np.random.Generator:
    """Independent, reproducible stream for ``(seed, *keys)``."""
    text = json.dumps([int(seed), *[str(k) for k in keys]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def apply_measured_range(
    cfg: dict[str, Any],
    name: str,
    low: float,
    high: float,
    source_ref: Any,
    reason: str,
    *,
    unit: str | None = None,
) -> dict[str, Any]:
    """Return a copy with ``name`` set to a ``measured_range``; the old entry goes to history."""
    out = copy.deepcopy(cfg)
    old = out.get("ranges", {}).get(name)
    entry = dict(old or {}, min=float(low), max=float(high), source_kind="measured_range",
                 source_ref=source_ref)
    if unit is not None:
        entry["unit"] = unit
    if "unit" not in entry:
        raise GeneratorConfigError(f"{name}: unit required for a new measured range")
    out.setdefault("ranges", {})[name] = entry
    out.setdefault("history", []).append(dict(
        changed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        name=name, old=old, new=entry, reason=reason))
    return validate_generator_config(out)
