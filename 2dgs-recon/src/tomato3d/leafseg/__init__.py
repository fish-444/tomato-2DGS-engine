"""Leaflet instance segmentation with a separate occlusion output (Option A label engine).

Contract: build/p1_leafseg_20261019_01/CONTRACT.md. Uses only project-tuned models (OBB v5 leaflet detector, SAM 2.1
tiny leaflet-tuned v5); T1 occlusion features are imported from tomato3d.topview.occlusion, never modified.
Everything this package writes is a model proposal; human answers live in separate files written by the user.
"""
