"""java.util.Random and String.hashCode in Python, so the keyed draws of the dynamic rule file are the ones the engine hook
(tools/fsp_random_parameters_native.rgg line 25) and the B-model adapter (CONTRACT §5) make: Random(seed + (id + ":" + key).hashCode()),
first nextDouble() skipped, then nextDouble() (u) or nextGaussian() (z). CONTRACT §4.2."""
from __future__ import annotations

import math

MASK = (1 << 48) - 1
MULT = 0x5DEECE66D


def java_hash(s: str) -> int:
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h


class JRandom:
    def __init__(self, seed: int):
        self.seed = (seed ^ MULT) & MASK
        self._next_gauss = None

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * MULT + 0xB) & MASK
        r = self.seed >> (48 - bits)
        if bits == 32 and r >= (1 << 31):
            r -= 1 << 32
        return r

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) * (1.0 / (1 << 53))

    def next_gaussian(self) -> float:
        if self._next_gauss is not None:
            g, self._next_gauss = self._next_gauss, None
            return g
        while True:
            v1 = 2 * self.next_double() - 1; v2 = 2 * self.next_double() - 1; s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        m = math.sqrt(-2 * math.log(s) / s)
        self._next_gauss = v2 * m
        return v1 * m


def keyed(seed: int, id_: str, key: str) -> JRandom:
    r = JRandom(seed + java_hash(f"{id_}:{key}")); r.next_double()
    return r


def u_of(seed: int, id_: str, key: str) -> float:
    return keyed(seed, id_, key).next_double()


def z_of(seed: int, id_: str, key: str) -> float:
    return keyed(seed, id_, key).next_gaussian()


def self_test() -> dict:
    """Known Java values: new Random(42).nextDouble() = 0.7275636800328681; new Random(42).nextGaussian() = 1.1419053154730547;
    "hello".hashCode() = 99162322; "".hashCode() = 0."""
    r = JRandom(42); d = r.next_double(); g = JRandom(42).next_gaussian(); h = java_hash("hello")
    out = dict(next_double_seed42=d, expect_double=0.7275636800328681, next_gaussian_seed42=g, expect_gaussian=1.1419053154730547,
               hash_hello=h, expect_hash=99162322, hash_empty=java_hash(""))
    out["pass"] = abs(d - out["expect_double"]) < 1e-15 and abs(g - out["expect_gaussian"]) < 1e-12 and h == 99162322 and out["hash_empty"] == 0
    return out


if __name__ == "__main__":
    print(self_test())
