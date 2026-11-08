"""One real top-view photo -> measured 2-D -> inferred 3-D (handoff/CLAUDE_CODE_CLI_PROMPT_TOP_TO_3D_2026-10-12.md).

Why a new package and not ``twoview`` / ``recon`` / ``fuse3d`` (the prompt asked to say so): those were scaffolded
for different inputs. ``twoview`` (skeleton section 6.9) takes a top AND a side image on the board and is explicitly
"not a full 3D reconstruction"; ``recon`` carves a visual hull from many board poses and aligns monocular depth;
``fuse3d`` lifts masks onto that reconstructed geometry. A single board-less top photo has none of those inputs, so
filling those stubs would bend their contracts. They stay stubs.

Modules:
  calibration  T0 - side close-up bend records -> generator bending parameters (+ generator audit)
  measure2d    T1 - real top photo -> detector boxes -> SAM masks -> per-leaflet projected 2-D
  geometry     T3 - projected 2-D -> tilt / true length / true area, aspect-prior baseline (no learned model yet)

Every value carries a status. Nothing inferred is ever written as ``measured``.
"""
