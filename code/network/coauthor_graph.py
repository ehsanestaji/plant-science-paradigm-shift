"""
Theme 4: Co-authorship Network Analysis.

Produces outputs in results/networks/ for:
  - Q22: Co-authorship network structure (giant component, communities)
  - Q23: Bridge authors (betweenness centrality)

For 7.4M authors, we sample the most productive authors to keep the graph
manageable (top-N by paper count).

Usage:
    python -m src.network.coauthor_graph --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/networks"
TOP_AUTHORS = 50000  # Build graph from top-N most productive authors


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def build_coauthor_graph(con):
    """Build co-authorship graph from top productive authors."""
    print(f"Building co-authorship graph (top {TOP_AUTHORS:,} authors)...", flush=True)

    try:
        import networkx as nx
    except ImportError:
        print("  ERROR: networkx not installed, skipping network analysis", flush=True)
        return

    # Get top productive authors
    print("  Fetching top authors...", flush=True)
    top_auth = con.execute(f"""
        SELECT author_id, COUNT(*) AS n_papers
        FROM work_authors
        WHERE author_id != 'A9999999999'
        GROUP BY author_id
        ORDER BY n_papers DESC
        LIMIT {TOP_AUTHORS}
    """).df()

    author_set = set(top_auth["author_id"])
    print(f"  Selected {len(author_set):,} authors", flush=True)

    # Get co-authorship edges: pairs of top-authors on the same paper
    print("  Computing co-authorship edges...", flush=True)
    edges_df = con.execute(f"""
        WITH top_wa AS (
            SELECT work_id, author_id
            FROM work_authors
            WHERE author_id IN (
                SELECT author_id FROM (
                    SELECT author_id, COUNT(*) AS n
                    FROM work_authors
                    WHERE author_id != 'A9999999999'
                    GROUP BY author_id
                    ORDER BY n DESC
                    LIMIT {TOP_AUTHORS}
                )
            )
        )
        SELECT a.author_id AS author_a, b.author_id AS author_b, COUNT(*) AS weight
        FROM top_wa a
        JOIN top_wa b ON a.work_id = b.work_id AND a.author_id < b.author_id
        GROUP BY a.author_id, b.author_id
        HAVING weight >= 2
    """).df()
    print(f"  {len(edges_df):,} edges (weight >= 2)", flush=True)

    # Build graph
    G = nx.Graph()
    for _, row in top_auth.iterrows():
        G.add_node(row["author_id"], n_papers=int(row["n_papers"]))
    for _, row in edges_df.iterrows():
        G.add_edge(row["author_a"], row["author_b"], weight=int(row["weight"]))

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges", flush=True)

    # Giant component
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gcc = G.subgraph(components[0]).copy()
    print(f"  Giant component: {gcc.number_of_nodes():,} nodes "
          f"({100*gcc.number_of_nodes()/G.number_of_nodes():.1f}%)", flush=True)

    # Network stats
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": len(components),
        "gcc_nodes": gcc.number_of_nodes(),
        "gcc_edges": gcc.number_of_edges(),
        "density": nx.density(G),
    }

    # Clustering coefficient (sample for speed)
    if gcc.number_of_nodes() > 10000:
        import random
        sample_nodes = random.sample(list(gcc.nodes()), 10000)
        stats["avg_clustering"] = nx.average_clustering(gcc, nodes=sample_nodes)
    else:
        stats["avg_clustering"] = nx.average_clustering(gcc)
    print(f"  Avg clustering: {stats['avg_clustering']:.4f}", flush=True)

    pd.DataFrame([stats]).to_csv(f"{OUT_DIR}/coauthor_network_stats.csv", index=False)

    # Community detection (Louvain)
    try:
        import community as community_louvain
        print("  Running Louvain community detection...", flush=True)
        partition = community_louvain.best_partition(gcc, random_state=42)
        n_communities = len(set(partition.values()))
        print(f"  Found {n_communities} communities", flush=True)

        modularity = community_louvain.modularity(partition, gcc)
        stats["modularity"] = modularity
        stats["n_communities"] = n_communities
        print(f"  Modularity: {modularity:.4f}", flush=True)

        # Community sizes
        comm_sizes = defaultdict(int)
        for _, c in partition.items():
            comm_sizes[c] += 1
        comm_df = pd.DataFrame(
            sorted(comm_sizes.items(), key=lambda x: -x[1]),
            columns=["community_id", "size"]
        )
        comm_df.to_csv(f"{OUT_DIR}/coauthor_communities.csv", index=False)
    except ImportError:
        print("  python-louvain not installed, skipping community detection", flush=True)

    # Betweenness centrality (top-N bridge authors) — use GCC, sample if large
    print("  Computing betweenness centrality (sampled)...", flush=True)
    if gcc.number_of_nodes() > 5000:
        bc = nx.betweenness_centrality(gcc, k=min(1000, gcc.number_of_nodes()))
    else:
        bc = nx.betweenness_centrality(gcc)

    bc_df = pd.DataFrame([
        {"author_id": k, "betweenness": v, "n_papers": gcc.nodes[k].get("n_papers", 0)}
        for k, v in sorted(bc.items(), key=lambda x: -x[1])[:100]
    ])
    bc_df.to_csv(f"{OUT_DIR}/bridge_authors_top100.csv", index=False)
    print(f"  Top bridge author: {bc_df.iloc[0]['author_id']} "
          f"(betweenness={bc_df.iloc[0]['betweenness']:.6f})", flush=True)

    # Save edge list for visualization
    edges_df.to_csv(f"{OUT_DIR}/coauthor_edges.csv.gz", index=False, compression="gzip")
    print(f"  Saved edge list ({len(edges_df):,} edges)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    ap.add_argument("--read-only", action="store_true")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    con = create_database(args.db_path, read_only=args.read_only)

    t0 = time.time()
    build_coauthor_graph(con)

    elapsed = int(time.time() - t0)
    print(f"\nCo-authorship network complete ({elapsed}s)", flush=True)
    con.close()


if __name__ == "__main__":
    main()
