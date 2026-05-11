"""
Graph generators
================
All experiments derive their test graphs from a single source-of-truth:
a list of (u, v, w) edges.  The two helper functions then load that
edge list into whichever representation an algorithm needs.

This ensures every algorithm is tested on *exactly* the same graph,
making timing comparisons fair.
"""

import random
from graphs.representations import AdjacencyMatrix, AdjacencyList


def generate_edges(
    n: int,
    density: float = 0.10,
    weight_range: tuple[int, int] = (1, 20),
    seed: int | None = None,
) -> list[tuple[int, int, float]]:
    """
    Generate a random directed graph as a flat edge list.

    Parameters
    ----------
    n            : number of vertices  (0 … n-1)
    density      : probability that any directed pair (u, v) has an edge
    weight_range : (min_w, max_w) — both inclusive, always positive here
    seed         : fix for reproducibility across experiments

    Returns
    -------
    List of (u, v, w) tuples.  Self-loops are excluded.
    """
    if seed is not None:
        random.seed(seed)

    low, high = weight_range
    edges: list[tuple[int, int, float]] = []

    for u in range(n):
        for v in range(n):
            if u != v and random.random() < density:
                w = random.randint(low, high)
                edges.append((u, v, float(w)))

    return edges


def build_adj_matrix(n: int, edges: list[tuple[int, int, float]]) -> AdjacencyMatrix:
    """Load an edge list into an AdjacencyMatrix."""
    g = AdjacencyMatrix(n)
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


def build_adj_list(n: int, edges: list[tuple[int, int, float]]) -> AdjacencyList:
    """Load an edge list into an AdjacencyList."""
    g = AdjacencyList(n)
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g