"""
Dijkstra's Algorithm — Adjacency Matrix variant
================================================
Uses a plain linear scan (no heap) to find the next minimum-distance
vertex.  Because every relaxation step scans the full matrix row, and
every "find min" step scans the full distance array, both inner loops
are O(V).  The overall complexity is therefore O(V²).

Why this is competitive on *dense* graphs
-----------------------------------------
When E ≈ V², the heap variant runs in O((V + E) log V) ≈ O(V² log V),
which is actually *worse* than this simple O(V²) implementation.
Additionally, the matrix allows O(1) edge-weight lookup, which keeps
cache behaviour predictable.

Time complexity  : O(V²)
Space complexity : O(V)   — only the dist and visited arrays
"""

import math
from graphs.representations import AdjacencyMatrix


def dijkstra_matrix(graph: AdjacencyMatrix, source: int) -> list[float]:
    """
    Single-source shortest paths from `source` using adjacency matrix.

    Parameters
    ----------
    graph  : AdjacencyMatrix
    source : starting vertex

    Returns
    -------
    dist[v] = shortest distance from source to v  (math.inf if unreachable)
    """
    V = graph.V
    dist = [math.inf] * V
    dist[source] = 0.0
    visited = [False] * V

    for _ in range(V):
        # ── Find the unvisited vertex with the smallest tentative distance ──
        # This linear scan is the reason for the O(V²) complexity.
        u = -1
        for v in range(V):
            if not visited[v]:
                if u == -1 or dist[v] < dist[u]:
                    u = v

        if u == -1 or dist[u] == math.inf:
            break  # all remaining vertices are unreachable

        visited[u] = True

        # ── Relax every edge out of u using the matrix row ──
        # Another O(V) scan — mandatory because the matrix stores all V slots.
        for v in range(V):
            w = graph.matrix[u][v]
            if w != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    return dist