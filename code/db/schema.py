"""DuckDB schema for the plant science metascience database."""

import duckdb
import os

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "plant_science.duckdb"
)


SCHEMA_SQL = """
-- ═══════════════════════════════════════════════════════════
-- Plant Science Metascience — DuckDB Schema
-- ═══════════════════════════════════════════════════════════

-- Core works table: one row per unique scholarly work
CREATE TABLE IF NOT EXISTS works (
    work_id        VARCHAR PRIMARY KEY,   -- OpenAlex ID (W...) or internal
    doi            VARCHAR,
    pmid           VARCHAR,
    pmcid          VARCHAR,
    title          TEXT,
    abstract       TEXT,
    year           SMALLINT,
    publication_date DATE,
    journal_id     VARCHAR,
    journal_name   VARCHAR,
    journal_issn   VARCHAR,
    volume         VARCHAR,
    issue          VARCHAR,
    pages          VARCHAR,
    oa_status      VARCHAR,               -- gold, green, hybrid, bronze, closed
    cited_by_count INTEGER DEFAULT 0,
    reference_count INTEGER DEFAULT 0,
    source         VARCHAR,               -- openalex, pubmed, europepmc, crossref
    has_fulltext   BOOLEAN DEFAULT FALSE,
    language       VARCHAR,
    type           VARCHAR,               -- journal-article, review, preprint, book-chapter
    is_canonical   BOOLEAN DEFAULT TRUE,  -- FALSE if duplicate of another record
    canonical_id   VARCHAR,               -- points to canonical work_id if duplicate
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Authors
CREATE TABLE IF NOT EXISTS authors (
    author_id      VARCHAR PRIMARY KEY,   -- OpenAlex ID (A...) or generated
    display_name   VARCHAR,
    orcid          VARCHAR,
    works_count    INTEGER DEFAULT 0,
    cited_by_count INTEGER DEFAULT 0
);

-- Work-author relationship (many-to-many)
CREATE TABLE IF NOT EXISTS work_authors (
    work_id          VARCHAR NOT NULL,
    author_id        VARCHAR NOT NULL,
    author_position  SMALLINT,            -- 1-based order
    is_corresponding BOOLEAN DEFAULT FALSE,
    institution_id   VARCHAR,
    institution_name VARCHAR,
    country_code     VARCHAR(2),          -- ISO 3166-1 alpha-2
    PRIMARY KEY (work_id, author_id)
);

-- Citation edges
CREATE TABLE IF NOT EXISTS citations (
    citing_work_id VARCHAR NOT NULL,
    cited_work_id  VARCHAR NOT NULL,
    PRIMARY KEY (citing_work_id, cited_work_id)
);

-- Concepts/topics assigned to works
CREATE TABLE IF NOT EXISTS concepts (
    work_id      VARCHAR NOT NULL,
    concept_id   VARCHAR NOT NULL,
    concept_name VARCHAR,
    level        SMALLINT,                -- 0=broad, 5=narrow
    score        FLOAT,                   -- confidence score [0,1]
    PRIMARY KEY (work_id, concept_id)
);

-- Funding acknowledgments
CREATE TABLE IF NOT EXISTS funders (
    work_id    VARCHAR NOT NULL,
    funder_id  VARCHAR,
    funder_name VARCHAR,
    award_id   VARCHAR
);

-- Sequence for ingestion_log auto-increment id
CREATE SEQUENCE IF NOT EXISTS seq_ingestion_log START 1;

-- Data lineage: track what was ingested from where
CREATE TABLE IF NOT EXISTS ingestion_log (
    id           INTEGER DEFAULT nextval('seq_ingestion_log') PRIMARY KEY,
    source       VARCHAR,                 -- openalex, pubmed, etc.
    file_path    VARCHAR,
    records_total INTEGER,
    records_matched INTEGER,
    ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checksum     VARCHAR
);

-- ═══════════════════════════════════════════════════════════
-- Indexes for common query patterns
-- ═══════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_works_doi ON works(doi);
CREATE INDEX IF NOT EXISTS idx_works_pmid ON works(pmid);
CREATE INDEX IF NOT EXISTS idx_works_year ON works(year);
CREATE INDEX IF NOT EXISTS idx_works_journal ON works(journal_id);
CREATE INDEX IF NOT EXISTS idx_works_source ON works(source);
CREATE INDEX IF NOT EXISTS idx_work_authors_author ON work_authors(author_id);
CREATE INDEX IF NOT EXISTS idx_work_authors_country ON work_authors(country_code);
CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(citing_work_id);
CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(cited_work_id);
CREATE INDEX IF NOT EXISTS idx_concepts_work ON concepts(work_id);
CREATE INDEX IF NOT EXISTS idx_concepts_concept ON concepts(concept_id);
"""


def create_database(db_path: str = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Create or open the DuckDB database with full schema."""
    if db_path is None:
        db_path = os.path.abspath(DEFAULT_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path, read_only=read_only)
    if not read_only:
        con.execute(SCHEMA_SQL)
    print(f"Database ready: {db_path}" + (" (read-only)" if read_only else ""))
    return con


def get_stats(con: duckdb.DuckDBPyConnection) -> dict:
    """Get summary statistics from the database."""
    stats = {}
    for table in ["works", "authors", "work_authors", "citations", "concepts", "funders"]:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        except Exception:
            stats[table] = 0
    # Year range
    try:
        row = con.execute(
            "SELECT MIN(year), MAX(year) FROM works WHERE year IS NOT NULL"
        ).fetchone()
        stats["year_range"] = (row[0], row[1])
    except Exception:
        stats["year_range"] = (None, None)
    return stats


if __name__ == "__main__":
    con = create_database()
    stats = get_stats(con)
    print("Database statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    con.close()
