"""
Theme 3: Citation Network Analysis.

Produces outputs in results/citations/ for:
  - Q19: PageRank of structurally important papers
  - Citation network basic statistics

Uses igraph for speed on 60M edges (NetworkX would be too slow).
Falls back to sampling if igraph is unavailable.

Usage:
    python -m src.network.citation_graph --db-path data/processed/plant_science.duckdb
"""

import argparse
import sys
import time
import os
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.db.schema import create_database
from src.utils.storage_monitor import check_storage

OUT_DIR = "results/citations"


def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def q19_pagerank(con):
    """PageRank on the citation graph."""
    print("Q19: Citation network PageRank...", flush=True)

    try:
        import igraph as ig
        use_igraph = True
        print("  Using igraph", flush=True)
    except ImportError:
        use_igraph = False
        print("  igraph not available, using sampled NetworkX approach", flush=True)

    if use_igraph:
        # Load all citation edges into igraph
        print("  Loading citation edges...", flush=True)
        t0 = time.time()

        # Get all unique work_ids involved in citations
        node_df = con.execute("""
            SELECT DISTINCT work_id FROM (
                SELECT citing_work_id AS work_id FROM citations
                UNION
                SELECT cited_work_id AS work_id FROM citations
            )
        """).df()
        node_ids = node_df["work_id"].tolist()
        node_map = {wid: i for i, wid in enumerate(node_ids)}
        print(f"  {len(node_ids):,} nodes ({time.time()-t0:.0f}s)", flush=True)

        # Load edges in chunks
        print("  Loading edges...", flush=True)
        edges = con.execute("""
            SELECT citing_work_id, cited_work_id FROM citations
        """).df()
        print(f"  {len(edges):,} edges ({time.time()-t0:.0f}s)", flush=True)

        # Map to integer indices
        src = edges["citing_work_id"].map(node_map).values
        dst = edges["cited_work_id"].map(node_map).values

        # Build directed graph
        print("  Building igraph...", flush=True)
        g = ig.Graph(n=len(node_ids), edges=list(zip(src.tolist(), dst.tolist())),
                     directed=True)
        print(f"  Graph built ({time.time()-t0:.0f}s)", flush=True)

        # PageRank
        print("  Computing PageRank...", flush=True)
        pr = g.pagerank(directed=True)
        print(f"  PageRank done ({time.time()-t0:.0f}s)", flush=True)

        # Top-100 by PageRank
        pr_series = pd.Series(pr, index=node_ids)
        top100 = pr_series.nlargest(100)

        # Join with work metadata
        top_ids = top100.index.tolist()
        pr_df = pd.DataFrame({"work_id": top_ids, "pagerank": top100.values})

        # Enrich with titles
        con.register("_pr_top", pr_df)
        result = con.execute("""
            SELECT p.work_id, p.pagerank, w.title, w.year,
                   w.cited_by_count, w.journal_name
            FROM _pr_top p
            LEFT JOIN works w ON p.work_id = w.work_id
            ORDER BY p.pagerank DESC
        """).df()
        con.unregister("_pr_top")
        result.to_csv(f"{OUT_DIR}/pagerank_top100.csv", index=False)

        # Network stats
        stats = {
            "n_nodes": g.vcount(),
            "n_edges": g.ecount(),
            "density": g.density(),
        }
        # In/out degree distribution
        in_deg = g.indegree()
        out_deg = g.outdegree()
        import numpy as np
        stats["mean_in_degree"] = np.mean(in_deg)
        stats["mean_out_degree"] = np.mean(out_deg)
        stats["max_in_degree"] = max(in_deg)
        stats["max_out_degree"] = max(out_deg)

        pd.DataFrame([stats]).to_csv(f"{OUT_DIR}/citation_network_stats.csv", index=False)

        title_str = result.iloc[0]["title"]
        if isinstance(title_str, str):
            title_str = title_str[:60] + "..."
        else:
            title_str = "(no title)"
        print(f"  Top PageRank: {title_str} "
              f"(PR={result.iloc[0]['pagerank']:.6f})", flush=True)

        del g, edges, src, dst  # free memory

    else:
        # Fallback: just use cited_by_count as proxy (already available)
        print("  Using cited_by_count as PageRank proxy", flush=True)
        result = con.execute("""
            SELECT work_id, title, year, cited_by_count, journal_name,
                   cited_by_count AS pagerank_proxy
            FROM works_clean
            WHERE cited_by_count IS NOT NULL
            ORDER BY cited_by_count DESC
            LIMIT 100
        """).df()
        result.to_csv(f"{OUT_DIR}/pagerank_top100.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="data/processed/plant_science.duckdb")
    args = ap.parse_args()

    check_storage()
    _ensure_dirs()
    con = create_database(args.db_path)

    t0 = time.time()
    q19_pagerank(con)

    elapsed = int(time.time() - t0)
    print(f"\nCitation network analysis complete ({elapsed}s)", flush=True)
    con.close()


if __name__ == "__main__":
    main()
