"""Root folder for every run folder the modules read or write (``build/...``, ``data/...``).

Default: the repository root (two levels above ``src/tomato3d``). Set the environment variable ``MICROTOM_ROOT`` to use another folder.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ["MICROTOM_ROOT"]).resolve() if os.environ.get("MICROTOM_ROOT") else Path(__file__).resolve().parents[2]
