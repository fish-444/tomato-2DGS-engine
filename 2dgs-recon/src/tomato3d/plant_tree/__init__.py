"""plant_tree_v1: one plant-tree format shared by the synthetic generator (v2), the organ-split reconstruction and the
engine's observation route. See SCHEMA.md next to this file.

Entry points: :mod:`schema` (validate, new_tree), :mod:`obj` (to_obj, from_obj_v1), :mod:`summary` (summary, compare),
:mod:`engine` (to_engine_structure, from_engine_states).
"""

from tomato3d.plant_tree.schema import SCHEMA, ValidationError, new_tree, validate  # noqa: F401
