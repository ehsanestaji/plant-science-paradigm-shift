# Plant Science is Reorganizing Around Climate-Relevant Crops

Code, manuscript, and derived data for a bibliometric analysis of ~4.9 million
plant-science publications (1900–2024), showing that the field is undergoing a
structural reorganization away from model-organism-centred discovery science and
toward crop-centred, climate-relevant research.

> **Estaji, E. & Mao, J.-F.** *Plant Science is Reorganizing Around
> Climate-Relevant Crops: Evidence from 4.9 Million Publications.* (2026).

## Key findings

- Annual output over the modern era (1990–2024) follows a logistic trajectory
  (carrying capacity *K* ≈ 217,781 papers/year, inflection 2006), ending
  exponential expansion.
- *Arabidopsis thaliana*'s relative share peaked in 2012 and declined as rice,
  wheat, and maize rose — but citation-flow analysis reframes this as
  **consolidation, not retreat**: *Arabidopsis* has become crop science's
  foundational methodological substrate.
- 40 sleeping-beauty papers on minor crops awoke in 2019–2022; CRISPR reached
  mainstream adoption roughly twice as fast as prior genomic tools.
- A persistent orphan-crop attention gap shows the realignment is incomplete.

## Repository layout

```
manuscript/     LaTeX source + compiled PDF (self-contained; figures in fig/)
code/           Analysis pipeline (Python)
  collect/      OpenAlex + PubMed harvesting
  db/           corpus cleaning / schema (DuckDB)
  nlp/          SPECTER2 embeddings, zero-shot organism classification, BERTopic
  novel/        the paper's core analyses (Analyses A–J: disruption index,
                sleeping beauties, method diffusion, concept marriages,
                orphan organisms, ...)
  biblio/       growth dynamics, productivity / power-law fits
  dynamics/     technology-adoption S-curves, authorship dynamics
  network/      citation and co-author graphs
  analysis/paper_a/   figure-ready result tables and hardening/robustness checks
  viz/          figure generation
results/paper_a/  derived result tables and figure outputs backing the manuscript
config/         controlled vocabularies, MeSH/OpenAlex concept filters, templates
reproducibility/  environment pinning
DATA.md         data availability + Zenodo manifest
requirements.txt
```

## Reproducing the analysis

1. Create the environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Provide an NCBI API key (for PubMed harvesting) via `config/api_keys.env`
   (copy `config/api_keys.env.template`). OpenAlex needs no key.
3. Harvest and build the corpus (`code/collect/`, `code/db/`), then run the
   analyses in `code/`. The full 41 GB raw corpus is **not** in this repository
   (see `DATA.md`); it is re-derivable from the public OpenAlex and PubMed APIs
   with the code here, or downloadable from the Zenodo archive.

## Building the manuscript

```bash
cd manuscript && make      # -> main.pdf (pdflatex + bibtex; naturemag.bst)
```

## Data availability

Bibliographic records are from [OpenAlex](https://openalex.org) and
[PubMed](https://pubmed.ncbi.nlm.nih.gov). Derived datasets — the processed
corpus, organism/paradigm/topic labels, summary tables, and the knowledge-graph
database — are archived on Zenodo; see [`DATA.md`](DATA.md).

## License

- Code: [MIT](LICENSE).
- Manuscript text, figures, and derived data tables: CC BY 4.0.

## Contact

Ehsan Estaji — ehsan.estaji@umu.se · Umeå Plant Science Centre (UPSC), Department of Plant Physiology, Umeå University.
