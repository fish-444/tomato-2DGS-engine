"""Unit conversions used by the light calculation.

These are definitions, not tunable values. Lengths are millimetres everywhere in this
project; PPFD is per square metre, so the area conversion is applied exactly once, where
irradiance is formed.
"""

from __future__ import annotations

from typing import Final

MM2_PER_M2: Final[float] = 1.0e6
UMOL_PER_MOL: Final[float] = 1.0e6
SECONDS_PER_HOUR: Final[float] = 3600.0
WATT_HOURS_PER_KWH: Final[float] = 1000.0
JOULES_PER_KWH: Final[float] = 3.6e6
