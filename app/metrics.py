from __future__ import annotations

import numpy as np


def availability_series(routes: list[list[str] | None]) -> np.ndarray:
    """Булев массив: есть ли маршрут на каждом шаге."""
    return np.array([r is not None for r in routes], dtype=bool)


def availability(routes: list[list[str] | None]) -> float:
    if not routes:
        return 0.0
    return float(availability_series(routes).mean())


def max_gap_steps(routes: list[list[str] | None]) -> int:
    """Максимальная длина непрерывного отсутствия маршрута (в шагах)."""
    best = cur = 0
    for r in routes:
        if r is None:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def hop_counts(routes: list[list[str] | None]) -> list[int | None]:
    """Число рёбер маршрута (включая наземные линии)."""
    return [len(r) - 1 if r else None for r in routes]
