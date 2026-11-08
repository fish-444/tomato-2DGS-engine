"""Leaf light-use under cultivation LEDs.

Photographs -> leaf surfaces (board frame) -> placed under the lamps (grow frame) ->
direct PPFD on every leaf triangle with mutual shading -> per-leaf and per-plant light,
daily light integral, electricity and absorbed photons per joule -> comparison with
quantum-sensor readings -> comparison of lamp arrangements and of geometry variants
(e.g. fewer photographs).

What is computed is light arriving at and absorbed by leaves. Photosynthesis, growth and
yield are not computed here and need their own measurements.

Module map:

* :mod:`~tomato3d.light.units`     unit conversions (the only place they are written)
* :mod:`~tomato3d.light.frames`    board frame -> grow frame, fixture axes
* :mod:`~tomato3d.light.scene`     the lighting environment: loading, scenarios, emitters
* :mod:`~tomato3d.light.geometry`  plant surfaces: CSV loading, simplified leaves, sampling
* :mod:`~tomato3d.light.raytrace`  direct PPFD with occlusion
* :mod:`~tomato3d.light.budget`    per triangle / leaf / plant / scenario light and energy
* :mod:`~tomato3d.light.validate`  predictions against quantum-sensor readings
* :mod:`~tomato3d.light.compare`   geometry variants against a reference variant
* :mod:`~tomato3d.light.pipeline`  the whole flow for one run
"""
