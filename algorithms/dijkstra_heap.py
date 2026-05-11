"""
Dijkstra's Algorithm — Adjacency List + Min-Heap variant
=========================================================
Uses Python's `heapq` (a binary min-heap) as the priority queue.
Instead of scanning all V vertices to find the next minimum, the heap
always yields the globally smallest tentative distance in O(log V).

Because edges are stored in per-vertex lists, the relaxation loop only
visits real neighbours (O(degree)), not all V vertices.

Time complexity  : O((V + E) log V)
Space complexity : O(V + E)  — adjacency lists + heap entries

Why this dominates on *sparse* graphs
--------------------------------------
When E << V², O((V + E) log V) ≈ O(E log V) << O(V²).
For real-world road networks or communication graphs, which are
typically sparse, this is the go-to implementation.
"""

import math
import heapq
from graphs.representations import AdjacencyList


def dijkstra_heap(graph: AdjacencyList, source: int) -> list[float]:
    """
    Single-source shortest paths from `source` using adjacency list + heap.

    Parameters
    ----------
    graph  : AdjacencyList
    source : starting vertex

    Returns
    -------
    dist[v] = shortest distance from source to v  (math.inf if unreachable)
    """
    V = graph.V
    dist = [math.inf] * V
    dist[source] = 0.0

    # Heap entries: (tentative_distance, vertex)
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)

        # Lazy deletion: skip stale entries (a vertex may appear multiple
        # times in the heap if its distance was updated after insertion).
        if d > dist[u]:
            continue

        # ── Relax only the real neighbours — O(degree(u)) ──
        for v, w in graph.neighbors(u):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist