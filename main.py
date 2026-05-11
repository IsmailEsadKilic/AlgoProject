"""
main.py — entry point
=====================
Runs all four experiments, prints a formatted console summary, and
saves four plots to ./plots/.

Run with:
    uv run main.py
    (or: python main.py)
"""

import os
import math
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from experiments.scenarios import (
    run_fair_comparison,
    run_density_comparison,
    run_scalability,
    run_query_volume_tradeoff,
)

PLOTS_DIR = "plots"
COL = 22   # column width for console table


# ── Console helpers ───────────────────────────────────────────────────── #

def banner(title: str) -> None:
    bar = "█" * 65
    print(f"\n{bar}")
    for line in title.strip().splitlines():
        print(f"  {line}")
    print(f"{bar}\n")


def table(headers: list[str], rows: list[list[str]]) -> None:
    print("  " + "".join(h.ljust(COL) for h in headers))
    print("  " + "─" * (COL * len(headers)))
    for row in rows:
        print("  " + "".join(str(c).ljust(COL) for c in row))
    print()


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.4f} ms"


# ── Plot helpers ──────────────────────────────────────────────────────── #

COLOURS = {
    "dijkstra_matrix": "#3A86FF",
    "dijkstra_heap":   "#06D6A0",
    "floyd_warshall":  "#EF476F",
}


def save(fig: Figure, name: str) -> None:
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ── Plot 1: Fair comparison bar chart ─────────────────────────────────── #

def plot_fair(results: list[dict]) -> None:
    labels = [f"n={r['n']}" for r in results]
    dm = [r["dijkstra_matrix"] * 1000 for r in results]
    dh = [r["dijkstra_heap"]   * 1000 for r in results]
    fw = [r["fw_total"]        * 1000 for r in results]

    x = range(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w for i in x], dm, w, label="Dijkstra Matrix  O(V²)",           color=COLOURS["dijkstra_matrix"])
    ax.bar([i     for i in x], dh, w, label="Dijkstra Heap  O((V+E) log V)",    color=COLOURS["dijkstra_heap"])
    ax.bar([i + w for i in x], fw, w, label="Floyd-Warshall  O(V³) + O(1) qry", color=COLOURS["floyd_warshall"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Fair Comparison — Single Query, Sparse Graph (density=0.10)")
    ax.legend()
    save(fig, "1_fair_comparison.png")


# ── Plot 2: Density comparison line chart ─────────────────────────────── #

def plot_density(results: list[dict]) -> None:
    densities = [r["density"] for r in results]
    dm = [r["dijkstra_matrix"] * 1000 for r in results]
    dh = [r["dijkstra_heap"]   * 1000 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(densities, dm, marker="o", label="Dijkstra Matrix  O(V²)",        color=COLOURS["dijkstra_matrix"])
    ax.plot(densities, dh, marker="s", label="Dijkstra Heap  O((V+E) log V)", color=COLOURS["dijkstra_heap"])
    ax.set_xlabel("Graph Density  (fraction of V² edges present)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Density Comparison — n=100, density 0.05 → 1.0")
    ax.legend()
    save(fig, "2_density_comparison.png")


# ── Plot 3: Scalability line chart ────────────────────────────────────── #

def plot_scalability(results: list[dict]) -> None:
    ns  = [r["n"] for r in results]
    dm  = [r["dijkstra_matrix"]           * 1000 for r in results]
    dh  = [r["dijkstra_heap"]             * 1000 for r in results]
    fw  = [r["floyd_warshall_preprocess"] * 1000 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, dm, marker="o", label="Dijkstra Matrix  O(V²)",        color=COLOURS["dijkstra_matrix"])
    ax.plot(ns, dh, marker="s", label="Dijkstra Heap  O((V+E) log V)", color=COLOURS["dijkstra_heap"])
    ax.plot(ns, fw, marker="^", label="Floyd-Warshall preprocess  O(V³)", color=COLOURS["floyd_warshall"])
    ax.set_xlabel("Number of Vertices (n)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Scalability — Runtime vs Graph Size  (density=0.10)")
    ax.legend()
    save(fig, "3_scalability.png")


# ── Plot 4: Query-volume crossover ────────────────────────────────────── #

def plot_query_volume(results: list[dict]) -> None:
    qs  = [r["queries"]             for r in results]
    dh  = [r["dijkstra_heap_total"] * 1000 for r in results]
    fw  = [r["fw_total"]            * 1000 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(qs, dh, marker="o", label="Dijkstra Heap × Q  (no preprocessing)",       color=COLOURS["dijkstra_heap"])
    ax.plot(qs, fw, marker="^", label="Floyd-Warshall  (preprocess once + Q lookups)", color=COLOURS["floyd_warshall"])
    ax.set_xlabel("Number of Queries (Q)")
    ax.set_ylabel("Total Time (ms)")
    ax.set_title("Query-Volume Trade-off — When Does Preprocessing Pay Off?  (n=100)")
    ax.legend()
    save(fig, "4_query_volume_tradeoff.png")


# ── Main ──────────────────────────────────────────────────────────────── #

def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Step 1: Fair comparison ──────────────────────────────────────── #
    banner(
        "STEP 1 — FAIR COMPARISON\n"
        "Same sparse graph (density=0.10), single query, all 3 approaches.\n"
        "Floyd-Warshall time = preprocessing + one query_all_from call."
    )
    fair = run_fair_comparison(sizes=[50, 100, 200])
    table(
        ["Size", "Edges", "Dijkstra Matrix", "Dijkstra Heap", "FW total"],
        [[f"n={r['n']}", r["edges"], ms(r["dijkstra_matrix"]),
          ms(r["dijkstra_heap"]), ms(r["fw_total"])] for r in fair],
    )
    # Correctness check
    for r in fair:
        status = "✓ all agree" if r["mismatches"] == 0 else f"✗ {r['mismatches']} mismatches"
        print(f"  n={r['n']}  correctness: {status}")
    print()
    plot_fair(fair)

    # ── Step 2: Density comparison ───────────────────────────────────── #
    banner(
        "STEP 2 — DENSITY COMPARISON\n"
        "Dijkstra Matrix vs Dijkstra Heap as graph density increases.\n"
        "n=100  |  density: 0.05 → 1.0"
    )
    density = run_density_comparison(n=100)
    table(
        ["Density", "Edges", "Dijkstra Matrix", "Dijkstra Heap"],
        [[f"{r['density']:.2f}", r["edges"],
          ms(r["dijkstra_matrix"]), ms(r["dijkstra_heap"])] for r in density],
    )
    plot_density(density)

    # ── Step 3: Scalability ──────────────────────────────────────────── #
    banner(
        "STEP 3 — SCALABILITY\n"
        "All 3 approaches, sparse graph (density=0.10), n = 10 → 500.\n"
        "Floyd-Warshall column = preprocessing time only (the expensive part)."
    )
    scale = run_scalability(sizes=[10, 50, 100, 200, 500])
    table(
        ["n", "Edges", "Dijkstra Matrix", "Dijkstra Heap", "FW preprocess"],
        [[r["n"], r["edges"], ms(r["dijkstra_matrix"]),
          ms(r["dijkstra_heap"]), ms(r["floyd_warshall_preprocess"])] for r in scale],
    )
    plot_scalability(scale)

    # ── Step 4: Query-volume trade-off ───────────────────────────────── #
    banner(
        "STEP 4 — QUERY-VOLUME TRADE-OFF\n"
        "How many queries until Floyd-Warshall total cost < repeated Dijkstra?\n"
        "n=100  |  density=0.10  |  Q = 1 → 1000"
    )
    qv = run_query_volume_tradeoff(n=100, max_queries=1000)
    table(
        ["Queries (Q)", "Dijkstra Heap total", "FW total", "FW cheaper?"],
        [[r["queries"], ms(r["dijkstra_heap_total"]), ms(r["fw_total"]),
          "YES ✓" if r["fw_total"] < r["dijkstra_heap_total"] else "no"]
         for r in qv],
    )
    plot_query_volume(qv)

    print("\n  All experiments complete.  Plots saved to ./plots/\n")


if __name__ == "__main__":
    main()