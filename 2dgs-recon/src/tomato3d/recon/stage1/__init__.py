"""Stage-1 reconstruction from a top + side image pair (session E, 2026-10-14): prior structure -> point cloud ->
observation-based refinement. Design + synthetic test only; real photos need the user's approval after the
synthetic report (prompt E0.3).

Modules: ``priors`` (Micro-Tom ratio / shape priors, file-based), ``views`` (synthetic observations + truth),
``height_order`` (E1), ``attach`` (E2), ``tilt`` (E3), ``complete`` (E4), ``assemble`` (E5), ``refine`` (E6),
``evaluate`` (E8-1). No new schema, renderer or id system: MTG ids (``<plant>/C0001``), the project's top rasteriser
and the generator's leaflet surface are reused.
"""
