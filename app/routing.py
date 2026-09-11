from __future__ import annotations

from collections import deque
from typing import Optional


def build_adjacency(snapshot: dict) -> dict[str, list[tuple[str, float]]]:
    """Строит граф из snapshot: узел -> [(сосед, вес), ...]."""
    adj: dict[str, list[tuple[str, float]]] = {}
    for a, b, w in snapshot["edges"]:
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    return adj


def bfs_route(adj: dict, src: str, dst: str) -> Optional[list[str]]:
    """Кратчайший по числу хопов путь. None если недостижим."""
    if src == dst:
        return [src]
    if src not in adj or dst not in adj:
        return None
    prev: dict[str, Optional[str]] = {src: None}
    q = deque([src])
    while q:
        v = q.popleft()
        for w, _ in adj.get(v, ()):
            if w not in prev:
                prev[w] = v
                if w == dst:
                    path = [w]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(w)
    return None


def dijkstra_route(adj: dict, src: str, dst: str) -> Optional[list[str]]:
    """Путь с минимальной суммой весов (км)."""
    import heapq
    if src == dst:
        return [src]
    if src not in adj or dst not in adj:
        return None
    dist = {src: 0.0}
    prev: dict[str, Optional[str]] = {src: None}
    pq = [(0.0, src)]
    while pq:
        d, v = heapq.heappop(pq)
        if v == dst:
            break
        if d > dist.get(v, float("inf")):
            continue
        for w, weight in adj.get(v, ()):
            nd = d + weight
            if nd < dist.get(w, float("inf")):
                dist[w] = nd
                prev[w] = v
                heapq.heappush(pq, (nd, w))
    if dst not in prev:
        return None
    path = [dst]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]
