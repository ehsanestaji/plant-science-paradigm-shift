# Provenance

## Bibliographic harvest

- OpenAlex: 1900–2024 plant-science concept filter (`config/openalex_concepts.json`).
  Harvest date: not recovered from git; re-derive via `code/collect/openalex_*.py`.
- PubMed: 1990–2024 MeSH filter (`config/mesh_terms.json`).
  Harvest date: not recovered from git; re-derive via `code/collect/pubmed_*.py`.

## FAOSTAT

- Metrics used in the attention-gap regression: gross production value (USD,
  five-year average 2019–2023), harvested area (hectares), and number of
  countries where the crop is ≥5% of national caloric supply.
- Download the three official extracts into
  `results/paper_a/external_snapshots/faostat_production_value.csv`,
  `faostat_harvested_area.csv`, and `faostat_staple_countries.csv`.
- Record the FAOSTAT download date on the day you fetch them.

## Climate-event catalogue

Record IDs from `results/paper_a/insights/climate_shocks/table_events.csv`:

| event_id | name | date | source |
|---|---|---|---|
| E01 | European heatwave 2003 | 2003-07-01 | EM-DAT |
| E02 | Russian drought 2010 | 2010-07-01 | EM-DAT |
| E03 | US Midwest drought 2012 | 2012-07-01 | NOAA-BDD |
| E04 | East African drought 2011 | 2011-05-01 | EM-DAT |
| E05 | Horn of Africa drought 2017 | 2017-03-01 | EM-DAT |
| E06 | Indian heatwave 2015 | 2015-05-01 | EM-DAT |
| E07 | Australian bushfires 2020 | 2020-01-01 | NOAA-BDD |
| E08 | Pakistan floods 2022 | 2022-08-01 | EM-DAT |
| E09 | California drought 2014 | 2014-01-01 | NOAA-BDD |
| E10 | European heatwave 2018 | 2018-07-01 | EM-DAT |
| E11 | Argentine drought 2018 | 2018-01-01 | EM-DAT |
| E12 | European heatwave 2022 | 2022-07-01 | EM-DAT |
| E13 | IPCC AR5 WGII 2014 | 2014-03-01 | IPCC |
| E14 | Paris COP21 2015 | 2015-12-01 | UNFCCC |
| E15 | IPCC AR6 WGII 2022 | 2022-02-01 | IPCC |

Copy `table_events.csv` to `results/paper_a/external_snapshots/climate_event_catalogue.csv`.
Download the IPCC AR6 WGII bibliographic landing page / citation snapshot to
`results/paper_a/external_snapshots/ipcc_ar6_wgii.txt` (DOI `10.1017/9781009325844`).

## Not deposited

The raw ~41 GB OpenAlex/PubMed harvest stays out of git and off Zenodo.
