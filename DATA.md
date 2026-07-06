# Data Availability

This study draws on ~4.9 million bibliographic records (1900–2024) from
[OpenAlex](https://openalex.org) and [PubMed](https://pubmed.ncbi.nlm.nih.gov),
both openly available. This document describes what ships in the git repository,
what is archived on Zenodo, and what is re-derivable rather than distributed.

## In this repository (git)

- `manuscript/` — LaTeX source, figures, and compiled PDF.
- `code/` — full analysis pipeline.
- `results/paper_a/` — derived result tables and figure outputs backing every
  display item in the paper (small enough for version control).

## Archived on Zenodo (DOI: 10.5281/zenodo.21199139 (reserved; registers on publish))

Large derived artifacts, too big for git, are deposited as a single Zenodo
record. **Planned contents:**

| Artifact | Approx. size | Description |
|---|---|---|
| `plant_science.duckdb` | ~24 GB | Relational knowledge base of the cleaned corpus — works, authors, affiliations, concepts, citations, journals. This is the graph/relational database underlying all analyses. |
| `coauthor_edges_5yr.duckdb` | ~1.1 GB | Pre-aggregated co-author edge lists per 5-year window (for the co-author network analyses). |
| `abstracts_for_embedding.parquet` | (see record) | Abstract texts used for SPECTER2 embedding and BERTopic. |
| `works_clean.csv.gz` | (see record) | The analysed publication list: one row per deduplicated work, sorted by year and organism, with DOI/OpenAlex ID, year, organism label + confidence, paradigm label, and BERTopic assignment. |
| `summary_tables/` | small | Roll-up tables — papers per year; papers per organism per year; method-diffusion parameters; organism × paradigm counts; attention-gap rankings. |
| `figure_source_tables/` | small | The exact CSVs behind each main, Extended Data, and Supplementary figure (mirror of `results/paper_a/`). |

> The DOI `10.5281/zenodo.21199139` is reserved on a Zenodo draft
> (`https://zenodo.org/uploads/21199139`) and is already cited in the manuscript
> Data Availability section. It becomes active once the draft's files are
> uploaded and the record is published — no further edit to the manuscript is
> needed.

## Not distributed (re-derivable)

The ~41 GB of **raw** OpenAlex/PubMed harvest (`data/raw/`) is not archived,
because it is fully reconstructible from the public OpenAlex and PubMed APIs
using `code/collect/` (harvest) and `code/db/clean.py` (dedup + cleaning:
year capped at 2024, pre-1900 removed). PubMed harvesting uses an NCBI API key
(`config/api_keys.env`, see `config/api_keys.env.template`).

## Provenance notes

- Corpus window: full cleaned corpus spans **1900–2024**; the modern window
  **1990–2024** (≈ 3.98 million records) is used for field-growth and
  organism-share analyses, while historical analyses (temporal placebo,
  concept-marriage epochs, orphan-crop gap) use the full back-catalogue.
- PubMed contributes 1990–2024; OpenAlex contributes 1900–2024.
