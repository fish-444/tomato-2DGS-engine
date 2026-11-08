"""Synthetic dwarf tomato generator (Stage 1).

Renders synthetic plants in the rig's real camera geometry with ground truth that real
photos cannot give: per-leaflet visible and amodal masks, depth, organ classes and the
3D plant graph.

Every artifact carries ``origin='synthetic'``. Synthetic data never becomes an observation,
never enters light (PPFD) calculations and never fits growth coefficients.

Heavy dependencies (``openalea.lpy``, ``openalea.plantgl``, ``bpy``, ``pillow_heif``) live in
the separate ``tomato_synth`` environment and are imported lazily inside functions, so
``import tomato3d`` and ``import tomato3d.synth`` keep working without them.
"""

from __future__ import annotations

GENERATOR_VERSION = "synth-stage1-0.2.0"
ORIGIN = "synthetic"
