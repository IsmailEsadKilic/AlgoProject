"""
Floyd-Warshall — All-Pairs Shortest Paths with Preprocessing
=============================================================
Unlike Dijkstra (which answers one source at a time, on demand),
Floyd-Warshall pays a large upfront cost to build a complete V×V
distance table.  After that, every query is answered in O(1) — a
simple table lookup.

The DP recurrence
-----------------
Let dist[i][j][k] = shortest path from i to j using only vertices
{0, 1, …, k} as intermediaries.

Base case  : dist[i][j][−1] = direct edge weight (or ∞)
Transition : dist[i][j][k]  = min(dist[i][j][k-1],
                                   dist[i][k][k-1] + dist[k][j][k-1])

Because we only need the previous k-layer, we collapse to a flat 2-D
table updated in-place.

Time complexity  (preprocessing) : O(V³)
Space complexity (table)         : O(V²)
Query time                       : O(1)

When does the preprocessing cost pay off?
-----------------------------------------
If Q queries will be answered on the same graph, the total cost is:
  • Repeated Dijkstra (heap) : O(Q · (V + E) log V)
  • Floyd-Warshall           : O(V³ + Q)            ← Q vanishes

The crossover point (where FW becomes cheaper) depends on V, E, and Q.
The query-volume experiment in main.py measures this empirically.
"""

import math
import copy
from graphs.representations import AdjacencyMatrix


class FloydWarshall:
    """
    Stateful wrapper that separates the preprocessing phase from queries.

    Usage
    -----
    fw = FloydWarshall()
    fw.preprocess(graph)          # O(V³) — do this once
    d = fw.query(src, dst)        # O(1)  — as many times as needed
    """

    def __init__(self):
        self._table: list[list[float]] | None = None
        self.V: int | None = None

    # ------------------------------------------------------------------ #
    #  Preprocessing                                                       #
    # ------------------------------------------------------------------ #

    def preprocess(self, graph: AdjacencyMatrix) -> None:
        """
        Build the all-pairs distance table from an AdjacencyMatrix.

        Must be called before any query.  Can be called again to
        re-preprocess a different graph.
        """
        V = graph.V
        self.V = V

        # Deep-copy the adjacency matrix as the initial distance table.
        # graph.matrix[i][i] is already 0 and off-diagonal ∞ means no edge.
        dist = [row[:] for row in graph.matrix]

        # Triple loop: k = relaxation vertex (intermediate node)
        for k in range(V):
            for i in range(V):
                # Small optimisation: skip rows where i→k is unreachable
                if dist[i][k] == math.inf:
                    continue
                for j in range(V):
                    candidate = dist[i][k] + dist[k][j]
                    if candidate < dist[i][j]:
                        dist[i][j] = candidate

        self._table = dist

    # ------------------------------------------------------------------ #
    #  Query interface                                                     #
    # ------------------------------------------------------------------ #

    def query(self, source: int, target: int) -> float:
        """
        O(1) shortest-path distance lookup.

        Returns math.inf if target is unreachable from source.
        Raises RuntimeError if preprocess() has not been called.
        """
        if self._table is None:
            raise RuntimeError("Call preprocess(graph) before querying.")
        return self._table[source][target]

    def query_all_from(self, source: int) -> list[float]:
        """
        Return the full distance vector from `source` to every vertex.
        O(V) — copies one row of the table.
        """
        if self._table is None:
            raise RuntimeError("Call preprocess(graph) before querying.")
        return self._table[source][:]

    @property
    def is_ready(self) -> bool:
        return self._table is not None