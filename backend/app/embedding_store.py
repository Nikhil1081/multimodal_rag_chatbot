from __future__ import annotations

import math
from array import array
from typing import Iterable, List, Tuple


def normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum((x * x) for x in vec))
    if norm <= 0.0:
        return vec
    return [x / norm for x in vec]


def pack_f32(vec: Iterable[float]) -> bytes:
    a = array("f", vec)
    return a.tobytes()


def unpack_f32(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return float(sum((x * y) for x, y in zip(a, b)))


def top_k_cosine(
    *,
    query_unit: List[float],
    rows: Iterable[Tuple[int, bytes]],
    k: int,
) -> List[Tuple[float, int]]:
    """Return [(score, id)] top-k for unit-normalized query and stored unit vectors."""

    scored: list[tuple[float, int]] = []
    for row_id, blob in rows:
        v = unpack_f32(blob)
        score = dot(query_unit, v)
        scored.append((score, row_id))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:k]
