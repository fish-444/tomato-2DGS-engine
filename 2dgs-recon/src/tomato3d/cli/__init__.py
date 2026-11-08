"""Command line entry points.

Both entry points exit non-zero on a hard failure (missing config, unreadable manifest,
too few calibration images). A per-frame pose failure is **not** a hard failure: it is
recorded in ``poses.csv`` and the run continues.
"""
