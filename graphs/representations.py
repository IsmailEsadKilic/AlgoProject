import math


class AdjacencyMatrix:
    """
    Dense graph representation using a V×V matrix.

    Space complexity : O(V²)
    Edge lookup      : O(1)
    Neighbor scan    : O(V)  ← this is what makes Dijkstra O(V²) here

    Best suited for dense graphs (E ≈ V²), where the O(V) neighbor
    scan is acceptable because most entries are real edges anyway.
    """

    def __init__(self, num_vertices: int):
        self.V = num_vertices
        # Initialise every cell to ∞; diagonal stays 0
        self.matrix = [[math.inf] * num_vertices for _ in range(num_vertices)]
        for i in range(num_vertices):
            self.matrix[i][i] = 0.0

    def add_edge(self, u: int, v: int, w: float) -> None:
        self.matrix[u][v] = w

    def get_weight(self, u: int, v: int) -> float:
        return self.matrix[u][v]

    def neighbors(self, u: int) -> list[tuple[int, float]]:
        """Return (v, weight) for every neighbour of u."""
        return [
            (v, self.matrix[u][v])
            for v in range(self.V)
            if self.matrix[u][v] != math.inf and v != u
        ]

    def edge_count(self) -> int:
        return sum(
            1
            for u in range(self.V)
            for v in range(self.V)
            if u != v and self.matrix[u][v] != math.inf
        )


class AdjacencyList:
    """
    Sparse graph representation using per-vertex neighbour lists.

    Space complexity : O(V + E)
    Neighbor scan    : O(degree(u))  ← enables the O((V+E) log V) heap Dijkstra

    Best suited for sparse graphs (E << V²), where most pairs of
    vertices have no edge and storing a full matrix would waste memory.
    """

    def __init__(self, num_vertices: int):
        self.V = num_vertices
        self.adj: dict[int, list[tuple[int, float]]] = {
            i: [] for i in range(num_vertices)
        }

    def add_edge(self, u: int, v: int, w: float) -> None:
        self.adj[u].append((v, w))

    def neighbors(self, u: int) -> list[tuple[int, float]]:
        return self.adj[u]

    def edge_count(self) -> int:
        return sum(len(neighbours) for neighbours in self.adj.values())