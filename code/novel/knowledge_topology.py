"""
Analysis I: Temporal Knowledge Flow Topology.

Is citation flow becoming more hierarchical or more democratic? Builds decade-
sliced citation subgraphs and computes structural metrics: in-degree Gini,
clustering coefficient, giant component fraction, power-law exponent.

Output → results/novel/
  knowledge_topology_by_decade.csv
  indegree_distributions.csv

Usage:
    python -m src.novel.knowledge_topology --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/novel"

DECADES = [
    (1970, 1979),
    (1980, 1989),
    (1990, 1999),
    (2000, 2009),
    (2010, 2019),
]


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def gini_coefficient(values):
    """Compute Gini coefficient from array of non-negative values."""
    values = np.array(values, dtype=float)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def analyze_decade(con, start_year: int, end_year: int) -> tuple:
    """Build citation subgraph for a decade and compute topology metrics."""
    import igraph as ig

    label = f"{start_year}s"
    print(f"\n  Decade {start_year}–{end_year}:", flush=True)

    # Fetch edges
    t0 = time.time()
    edges_df = con.execute(f"""
        SELECT cit.citing_work_id, cit.cited_work_id
        FROM citations cit
        JOIN works_clean w1 ON cit.citing_work_id = w1.work_id
        JOIN works_clean w2 ON cit.cited_work_id = w2.work_id
        WHERE w1.year BETWEEN {start_year} AND {end_year}
          AND w2.year BETWEEN {start_year} AND {end_year}
    """).df()
    n_edges = len(edges_df)
    print(f"    Edges: {n_edges:,} ({time.time()-t0:.0f}s)", flush=True)

    if n_edges == 0:
        return None, None

    # Build igraph
    t1 = time.time()
    all_nodes = set(edges_df["citing_work_id"]) | set(edges_df["cited_work_id"])
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(all_nodes)

    edge_list = list(zip(
        edges_df["citing_work_id"].map(node_to_idx),
        edges_df["cited_work_id"].map(node_to_idx),
    ))
    del edges_df
    gc.collect()

    g = ig.Graph(n=n_nodes, edges=edge_list, directed=True)
    del edge_list
    gc.collect()
    print(f"    Graph: {n_nodes:,} nodes ({time.time()-t1:.0f}s)", flush=True)

    # In-degree distribution
    in_degrees = np.array(g.indegree())

    # Gini
    gini = gini_coefficient(in_degrees)

    # Clustering (on undirected version, sampled if too large)
    t2 = time.time()
    if n_nodes > 2_000_000:
        # Sample 500K nodes for clustering
        sample_idx = np.random.choice(n_nodes, min(500_000, n_nodes), replace=False)
        clustering = np.mean([g.transitivity_local_undirected(int(i))
                              for i in sample_idx[:10000]
                              if g.transitivity_local_undirected(int(i)) is not None
                              and not np.isnan(g.transitivity_local_undirected(int(i)))])
    else:
        clustering = g.transitivity_undirected()
    print(f"    Clustering: {clustering:.4f} ({time.time()-t2:.0f}s)", flush=True)

    # Giant component fraction
    components = g.connected_components(mode="weak")
    giant_frac = max(components.sizes()) / n_nodes

    # Mean and max in-degree
    mean_indeg = float(in_degrees.mean())
    max_indeg = int(in_degrees.max())

    # Power-law exponent (simple MLE on in-degree >= 5)
    high_deg = in_degrees[in_degrees >= 5]
    if len(high_deg) > 100:
        alpha = 1.0 + len(high_deg) / np.sum(np.log(high_deg / 4.5))
    else:
        alpha = np.nan

    metrics = {
        "decade": label,
        "start_year": start_year,
        "end_year": end_year,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "gini_indegree": round(gini, 4),
        "clustering_coeff": round(clustering, 4) if not np.isnan(clustering) else None,
        "giant_component_frac": round(giant_frac, 4),
        "mean_indegree": round(mean_indeg, 2),
        "max_indegree": max_indeg,
        "powerlaw_alpha": round(alpha, 3) if not np.isnan(alpha) else None,
        "democratization_index": round(1 - gini, 4),
    }

    # In-degree distribution (binned)
    max_bin = min(int(in_degrees.max()), 1000)
    bins = np.arange(0, max_bin + 2)
    hist, _ = np.histogram(in_degrees, bins=bins)
    dist_df = pd.DataFrame({
        "decade": label,
        "degree": bins[:-1],
        "count": hist,
    })
    dist_df = dist_df[dist_df["count"] > 0]

    del g
    gc.collect()

    print(f"    Gini={gini:.4f}, Giant={giant_frac:.4f}, "
          f"α={alpha:.3f}" if not np.isnan(alpha) else "", flush=True)

    return metrics, dist_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    t0 = time.time()

    con = create_database(args.db_path, read_only=True)
    con.execute("SET memory_limit='60GB'")
    con.execute("SET threads=8")

    print("Computing knowledge flow topology per decade…", flush=True)

    all_metrics = []
    all_dists = []

    for start, end in DECADES:
        metrics, dist_df = analyze_decade(con, start, end)
        if metrics is not None:
            all_metrics.append(metrics)
        if dist_df is not None:
            all_dists.append(dist_df)

    con.close()

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"{OUT_DIR}/knowledge_topology_by_decade.csv", index=False)

    if all_dists:
        dist_df = pd.concat(all_dists, ignore_index=True)
        dist_df.to_csv(f"{OUT_DIR}/indegree_distributions.csv", index=False)

    # Summary
    print("\n=== Knowledge Flow Topology Summary ===")
    if all_metrics:
        mdf = pd.DataFrame(all_metrics)
        print(f"{'Decade':<10} {'Nodes':>12} {'Edges':>14} {'Gini':>8} "
              f"{'Cluster':>8} {'Giant':>8} {'α':>6}")
        print("-" * 72)
        for _, r in mdf.iterrows():
            cl = f"{r['clustering_coeff']:.4f}" if r['clustering_coeff'] else "N/A"
            al = f"{r['powerlaw_alpha']:.3f}" if r['powerlaw_alpha'] else "N/A"
            print(f"  {r['decade']:<8} {r['n_nodes']:>12,} {r['n_edges']:>14,} "
                  f"{r['gini_indegree']:>8.4f} {cl:>8} "
                  f"{r['giant_component_frac']:>8.4f} {al:>6}")

    elapsed = int(time.time() - t0)
    print(f"\nAnalysis I complete ({elapsed}s)", flush=True)


if __name__ == "__main__":
    main()
