"""
Experiment scenarios
====================
Four self-contained experiments.  Each returns a plain list of dicts
that main.py can both print and plot.

Scenario 1 — Fair comparison
    Same graph, same single query, all three approaches.
    Establishes baseline relative costs.

Scenario 2 — Density comparison
    Dijkstra-matrix vs Dijkstra-heap across densities 0.05 → 1.0.
    Shows the crossover where the matrix variant stops being slower.

Scenario 3 — Scalability
    All three approaches on n = 10 … 500 (sparse graph).
    Illustrates the V², (V+E)logV, and V³ growth curves.

Scenario 4 — Query-volume trade-off  ← the unique angle
    Repeated Dijkstra-heap (Q runs) vs Floyd-Warshall (preprocess once,
    then Q O(1) lookups).  Finds the empirical crossover point.
"""

import random
from graphs.generators import generate_edges, build_adj_matrix, build_adj_list
from algorithms.dijkstra_matrix import dijkstra_matrix
from algorithms.dijkstra_heap import dijkstra_heap
from algorithms.floyd_warshall import FloydWarshall
from experiments.benchmark import time_once, time_average


# ── Scenario 1 ────────────────────────────────────────────────────────── #

def run_fair_comparison(
    sizes: list[int] = [50, 100, 200],
    density: float = 0.10,
    seed: int = 42,
) -> list[dict]:
    """
    All three approaches on the same sparse graph, one query each.
    Floyd-Warshall time includes preprocessing + one query_all_from call.
    """
    results = []
    for n in sizes:
        edges = generate_edges(n, density=density, seed=seed)
        mat = build_adj_matrix(n, edges)
        lst = build_adj_list(n, edges)

        t_dm, dist_dm = time_average(dijkstra_matrix, mat, 0)
        t_dh, dist_dh = time_average(dijkstra_heap, lst, 0)

        fw = FloydWarshall()
        t_pre, _ = time_once(fw.preprocess, mat)
        t_q, _   = time_once(fw.query_all_from, 0)

        # Sanity check: all three should agree on every reachable distance
        import math
        mismatches = 0
        for v in range(n):
            fw_d = fw.query(0, v)
            dm_d = dist_dm[v]
            dh_d = dist_dh[v]
            if not (math.isinf(dm_d) and math.isinf(fw_d) and math.isinf(dh_d)):
                if abs(dm_d - dh_d) > 1e-9 or abs(dm_d - fw_d) > 1e-9:
                    mismatches += 1

        results.append({
            "n":               n,
            "edges":           len(edges),
            "dijkstra_matrix": t_dm,
            "dijkstra_heap":   t_dh,
            "fw_preprocess":   t_pre,
            "fw_query":        t_q,
            "fw_total":        t_pre + t_q,
            "mismatches":      mismatches,
        })
    return results


# ── Scenario 2 ────────────────────────────────────────────────────────── #

def run_density_comparison(
    n: int = 100,
    densities: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00),
    seed: int = 42,
) -> list[dict]:
    """
    Compare Dijkstra-matrix vs Dijkstra-heap as graph density increases.
    At low density, heap wins.  At high density, matrix catches up.
    """
    results = []
    for d in densities:
        edges = generate_edges(n, density=d, seed=seed)
        mat = build_adj_matrix(n, edges)
        lst = build_adj_list(n, edges)

        t_dm, _ = time_average(dijkstra_matrix, mat, 0)
        t_dh, _ = time_average(dijkstra_heap, lst, 0)

        results.append({
            "density":         d,
            "edges":           len(edges),
            "dijkstra_matrix": t_dm,
            "dijkstra_heap":   t_dh,
        })
    return results


# ── Scenario 3 ────────────────────────────────────────────────────────── #

def run_scalability(
    sizes: list[int] = [10, 50, 100, 200, 500],
    density: float = 0.10,
    seed: int = 42,
) -> list[dict]:
    """
    Runtime vs graph size for all three approaches on a sparse graph.
    Floyd-Warshall time is preprocessing only (the expensive part).
    """
    results = []
    for n in sizes:
        edges = generate_edges(n, density=density, seed=seed)
        mat = build_adj_matrix(n, edges)
        lst = build_adj_list(n, edges)

        t_dm, _ = time_average(dijkstra_matrix, mat, 0)
        t_dh, _ = time_average(dijkstra_heap, lst, 0)

        fw = FloydWarshall()
        t_fw, _ = time_once(fw.preprocess, mat)

        results.append({
            "n":                        n,
            "edges":                    len(edges),
            "dijkstra_matrix":          t_dm,
            "dijkstra_heap":            t_dh,
            "floyd_warshall_preprocess": t_fw,
        })
    return results


# ── Scenario 4 ────────────────────────────────────────────────────────── #

def run_query_volume_tradeoff(
    n: int = 100,
    density: float = 0.10,
    max_queries: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    The unique experiment: at what query volume does Floyd-Warshall
    become cheaper than running Dijkstra-heap once per query?

    Total cost model
    ----------------
    Dijkstra-heap  : Q × T_dijkstra
    Floyd-Warshall : T_preprocess + Q × T_lookup  (T_lookup ≈ 0)

    Crossover ≈ T_preprocess / (T_dijkstra - T_lookup)
    """
    random.seed(seed)
    edges = generate_edges(n, density=density, seed=seed)
    mat = build_adj_matrix(n, edges)
    lst = build_adj_list(n, edges)

    # Preprocess Floyd-Warshall once — this cost is amortised over all queries
    fw = FloydWarshall()
    t_fw_pre, _ = time_once(fw.preprocess, mat)

    # Random source→target pairs
    all_queries = [
        (random.randint(0, n - 1), random.randint(0, n - 1))
        for _ in range(max_queries)
    ]

    query_counts = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    results = []

    for q in query_counts:
        batch = all_queries[:q]

        # Dijkstra-heap: one full run per (unique) source
        t_dh_total = 0.0
        for src, _ in batch:
            t, _ = time_once(dijkstra_heap, lst, src)
            t_dh_total += t

        # Floyd-Warshall: preprocessing already paid; measure lookup cost
        t_fw_q = 0.0
        for src, dst in batch:
            t, _ = time_once(fw.query, src, dst)
            t_fw_q += t

        results.append({
            "queries":              q,
            "dijkstra_heap_total":  t_dh_total,
            "fw_preprocess":        t_fw_pre,
            "fw_queries_only":      t_fw_q,
            "fw_total":             t_fw_pre + t_fw_q,
        })

    return results