"""
PubMed collector using NCBI E-utilities.

Downloads plant science papers (with abstracts) into JSONL.gz files,
one record per line. Designed to complement the OpenAlex collection
by capturing papers missed by subfield filtering.

Search scope:
    MeSH: Plants, Plant Diseases, Plant Physiology, Crops, Forestry, Ecology
    Years: 1990-2024
    Filter: hasabstract

Usage:
    python -m src.collect.pubmed_api \
        --output-dir data/raw/pubmed_api \
        --api-key YOUR_KEY

    # Resume after interruption (uses checkpoint):
    python -m src.collect.pubmed_api --output-dir data/raw/pubmed_api

Output per record (JSONL):
    {pmid, doi, title, abstract, year, publication_date,
     journal_name, journal_issn, authors:[{name,affiliation}],
     mesh_terms:[str], language}
"""

import os
import sys
import gzip
import json
import time
import argparse
import socket
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.checkpointing import checkpoint_save, checkpoint_load
from src.utils.storage_monitor import check_storage

# ── Search parameters ─────────────────────────────────────────────────
SEARCH_QUERY = (
    '("Plants"[MeSH] OR "Plant Physiological Phenomena"[MeSH] '
    'OR "Plant Diseases"[MeSH] OR "Crops, Agricultural"[MeSH] '
    'OR "Forestry"[MeSH] OR "Ecology"[MeSH]) '
    'AND ("1990"[PDAT] : "2024"[PDAT]) '
    'AND (hasabstract[text])'
)

BASE_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BATCH_SIZE = 500       # NCBI max per efetch request
FILE_SIZE  = 50_000    # records per output file
MIN_DELAY  = 0.11      # 10 req/sec with API key (0.33 without)

socket.setdefaulttimeout(60)


# ── HTTP helpers ──────────────────────────────────────────────────────

def _get(session: requests.Session, url: str, params: dict,
         retries: int = 6) -> requests.Response:
    """GET with exponential back-off on 429 / 5xx."""
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = delay * (2 ** attempt)
                print(f"  [warn] HTTP {resp.status_code} — retrying in {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = delay * (2 ** attempt)
            print(f"  [warn] {e.__class__.__name__} — retrying in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


# ── E-utilities calls ─────────────────────────────────────────────────

ESEARCH_MAX = 9999   # safe per-call limit without usehistory

# Per-year query template (avoids NCBI's paging issues for large result sets)
YEAR_QUERY = (
    '("Plants"[MeSH] OR "Plant Physiological Phenomena"[MeSH] '
    'OR "Plant Diseases"[MeSH] OR "Crops, Agricultural"[MeSH] '
    'OR "Forestry"[MeSH] OR "Ecology"[MeSH]) '
    'AND (hasabstract[text]) '
    'AND ("{y}"[PDAT] : "{y}"[PDAT])'
)
YEARS = list(range(1990, 2025))


def _esearch_json(session, api_key: str, params: dict) -> dict:
    """Call esearch, return parsed JSON — handles NCBI control-char quirks."""
    params = dict(db="pubmed", retmode="json", api_key=api_key, **params)
    resp = _get(session, BASE_URL + "esearch.fcgi", params)
    try:
        return json.loads(resp.text, strict=False)
    except json.JSONDecodeError:
        clean = "".join(c for c in resp.text if c >= " " or c in "\n\r\t")
        return json.loads(clean)


def esearch_year_pmids(session, api_key: str, year: int) -> list[str]:
    """Fetch all PMIDs for one year. Sub-partitions by month if >9999."""
    query = YEAR_QUERY.replace("{y}", str(year))
    # Check count first
    data  = _esearch_json(session, api_key, {"term": query, "retmax": 0})
    total = int(data["esearchresult"]["count"])
    time.sleep(MIN_DELAY)
    if total == 0:
        return []

    if total <= ESEARCH_MAX:
        data  = _esearch_json(session, api_key,
                              {"term": query, "retmax": total, "retstart": 0})
        pmids = data["esearchresult"].get("idlist", [])
        time.sleep(MIN_DELAY)
        return pmids

    # Busy year (>9999): partition by month
    pmids = []
    for month in range(1, 13):
        mq = (
            '("Plants"[MeSH] OR "Plant Physiological Phenomena"[MeSH] '
            'OR "Plant Diseases"[MeSH] OR "Crops, Agricultural"[MeSH] '
            'OR "Forestry"[MeSH] OR "Ecology"[MeSH]) '
            'AND (hasabstract[text]) '
            f'AND ("{year}/{month:02d}"[PDAT] : "{year}/{month:02d}"[PDAT])'
        )
        d = _esearch_json(session, api_key, {"term": mq, "retmax": ESEARCH_MAX})
        pmids.extend(d["esearchresult"].get("idlist", []))
        time.sleep(MIN_DELAY)
    return pmids


def efetch_pmids(session, api_key: str, pmids: list[str]) -> str:
    """Fetch full XML records for a list of PMIDs using POST (avoids 414)."""
    data = dict(
        db="pubmed", id=",".join(pmids),
        rettype="xml", retmode="xml",
        api_key=api_key,
    )
    for attempt in range(6):
        try:
            resp = session.post(BASE_URL + "efetch.fcgi", data=data, timeout=60)
            if resp.status_code == 200:
                return resp.text
            wait = 1.0 * (2 ** attempt)
            print(f"  [warn] efetch HTTP {resp.status_code} — retrying in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = 1.0 * (2 ** attempt)
            print(f"  [warn] {e.__class__.__name__} — retrying in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"efetch failed after 6 retries")


# ── XML parsing ───────────────────────────────────────────────────────

def _text(el, path: str, default=None):
    node = el.find(path)
    return node.text if node is not None and node.text else default


def _pub_date(article_el) -> tuple[int | None, str | None]:
    """Extract (year, date_str) from PubDate element."""
    ji = article_el.find(".//JournalIssue/PubDate")
    if ji is None:
        return None, None
    year_el = ji.find("Year")
    month_el = ji.find("Month")
    day_el   = ji.find("Day")
    # Fall back to MedlineDate
    if year_el is None:
        med = ji.findtext("MedlineDate", "")
        try:
            year = int(med[:4])
        except (ValueError, IndexError):
            year = None
        return year, med or None
    year = int(year_el.text) if year_el.text else None
    parts = [year_el.text]
    if month_el is not None and month_el.text:
        parts.append(month_el.text)
        if day_el is not None and day_el.text:
            parts.append(day_el.text)
    return year, "-".join(str(p) for p in parts)


def parse_article(pubmed_article: ET.Element) -> dict | None:
    """Parse a single <PubmedArticle> element into a dict."""
    mc = pubmed_article.find("MedlineCitation")
    if mc is None:
        return None

    pmid = _text(mc, "PMID")
    if not pmid:
        return None

    article = mc.find("Article")
    if article is None:
        return None

    # Abstract (may have multiple AbstractText with labels)
    abstract_parts = []
    for at in article.findall(".//Abstract/AbstractText"):
        label = at.get("Label")
        text  = at.text or ""
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts).strip() or None
    if not abstract:
        return None   # skip no-abstract records

    # Title
    title_el = article.find("ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else None

    # DOI
    doi = None
    for loc in article.findall("ELocationID"):
        if loc.get("EIdType") == "doi":
            doi = loc.text
            break
    if doi is None:
        for aid in pubmed_article.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text
                break

    # Journal
    journal_name = _text(article, ".//Journal/Title")
    journal_issn = (
        _text(article, ".//Journal/ISSN") or
        _text(article, ".//MedlineJournalInfo/ISSNLinking")
    )

    # Publication date
    year, pub_date = _pub_date(article)

    # Authors
    authors = []
    for auth in article.findall(".//AuthorList/Author"):
        last  = _text(auth, "LastName", "")
        fore  = _text(auth, "ForeName", "")
        coll  = _text(auth, "CollectiveName")
        name  = coll if coll else f"{last} {fore}".strip()
        aff   = _text(auth, ".//AffiliationInfo/Affiliation")
        if name:
            authors.append({"name": name, "affiliation": aff})

    # Language
    language = _text(article, "Language")

    # MeSH terms
    mesh_terms = [
        desc.text for desc in mc.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        if desc.text
    ]

    return {
        "pmid":             pmid,
        "doi":              doi.lower().strip() if doi else None,
        "title":            title,
        "abstract":         abstract,
        "year":             year,
        "publication_date": pub_date,
        "journal_name":     journal_name,
        "journal_issn":     journal_issn,
        "authors":          authors,
        "mesh_terms":       mesh_terms,
        "language":         language,
    }


def parse_batch_xml(xml_text: str) -> list[dict]:
    """Parse an efetch XML response; return list of article dicts."""
    records = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  [warn] XML parse error: {e}", flush=True)
        return records
    for pa in root.findall("PubmedArticle"):
        rec = parse_article(pa)
        if rec:
            records.append(rec)
    return records


# ── Writer ────────────────────────────────────────────────────────────

class FileWriter:
    """Rotates gzip JSONL output files every FILE_SIZE records."""
    def __init__(self, out_dir: Path):
        self.out_dir   = out_dir
        self.file_idx  = 0
        self.file_recs = 0
        self.total     = 0
        self._fh       = None
        self._open_next()

    def _open_next(self):
        if self._fh:
            self._fh.close()
        name = self.out_dir / f"pubmed_plant_{self.file_idx:04d}.jsonl.gz"
        self._fh = gzip.open(name, "wb")
        self.file_recs = 0

    def write(self, record: dict):
        self._fh.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
        self.file_recs += 1
        self.total     += 1
        if self.file_recs >= FILE_SIZE:
            self.file_idx += 1
            self._open_next()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    @property
    def current_file(self) -> str:
        return f"pubmed_plant_{self.file_idx:04d}.jsonl.gz"


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/raw/pubmed_api")
    ap.add_argument("--api-key",    default=os.environ.get("NCBI_API_KEY", ""))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.api_key.strip()

    if not api_key:
        print("[warn] No NCBI API key — rate limited to 3 req/sec", flush=True)
        global MIN_DELAY
        MIN_DELAY = 0.34

    check_storage()

    session = requests.Session()
    session.headers["User-Agent"] = "PlantSciMeta/1.0 (ehes0002@umu.se)"

    # ── Resume from checkpoint ────────────────────────────────────────
    ckpt         = checkpoint_load("pubmed_collect") or {}
    done_years   = set(ckpt.get("done_years",  []))   # completed years
    pmid_queue   = ckpt.get("pmid_queue",  [])        # PMIDs from current year not yet fetched
    cur_year     = ckpt.get("cur_year",    YEARS[0])
    file_idx     = ckpt.get("file_idx",   0)
    written      = ckpt.get("written",    0)

    if done_years or written:
        print(f"Resuming: {written:,} written, "
              f"{len(done_years)}/{len(YEARS)} years done, "
              f"cur_year={cur_year}, queue={len(pmid_queue):,}",
              flush=True)
    else:
        print(f"Starting fresh — {len(YEARS)} years to collect "
              f"({YEARS[0]}–{YEARS[-1]})", flush=True)

    # ── Writer ────────────────────────────────────────────────────────
    writer = FileWriter(out_dir)
    writer.file_idx = file_idx
    writer.total    = written
    writer._open_next()

    t_start    = time.time()
    t_last_log = t_start
    total_expected = 406_875  # approximate; updated as years complete

    def _log(year, year_pmids_total):
        now  = time.time()
        pct  = writer.total * 100 // max(1, total_expected)
        rate = writer.total / max(1, now - t_start)
        eta  = int((total_expected - writer.total) / max(1, rate))
        print(f"  [{pct:3d}%] written={writer.total:,}/{total_expected:,}  "
              f"year={year}  file={writer.current_file}  "
              f"rate={rate:.0f}/s  "
              f"ETA={eta//3600}h{(eta%3600)//60:02d}m",
              flush=True)

    # ── Main loop: year by year ───────────────────────────────────────
    for year in YEARS:
        if year in done_years:
            continue

        # Get PMIDs for this year (use cached queue if resuming mid-year)
        if not pmid_queue or cur_year != year:
            print(f"  Collecting PMIDs for {year}...", flush=True)
            pmid_queue = esearch_year_pmids(session, api_key, year)
            cur_year   = year
            print(f"    {year}: {len(pmid_queue):,} PMIDs", flush=True)

        year_start_written = writer.total

        # Efetch in batches of 500
        while pmid_queue:
            batch      = pmid_queue[:BATCH_SIZE]
            pmid_queue = pmid_queue[BATCH_SIZE:]

            t0       = time.time()
            xml_text = efetch_pmids(session, api_key, batch)
            records  = parse_batch_xml(xml_text)
            for rec in records:
                writer.write(rec)

            elapsed = time.time() - t0
            if elapsed < MIN_DELAY:
                time.sleep(MIN_DELAY - elapsed)

            # Checkpoint every 5K written
            if writer.total % 5_000 < BATCH_SIZE:
                checkpoint_save("pubmed_collect", dict(
                    done_years=list(done_years), pmid_queue=pmid_queue,
                    cur_year=cur_year, file_idx=writer.file_idx,
                    written=writer.total,
                ))

            # Log every 30s
            now = time.time()
            if now - t_last_log >= 30:
                _log(year, len(pmid_queue))
                t_last_log = now

        year_written = writer.total - year_start_written
        print(f"  {year} done — {year_written:,} records written", flush=True)
        done_years.add(year)
        pmid_queue = []
        checkpoint_save("pubmed_collect", dict(
            done_years=list(done_years), pmid_queue=[],
            cur_year=year, file_idx=writer.file_idx,
            written=writer.total,
        ))

    writer.close()
    check_storage()

    checkpoint_save("pubmed_collect", dict(
        done_years=list(done_years), pmid_queue=[],
        cur_year=YEARS[-1], file_idx=writer.file_idx,
        written=writer.total, completed=True,
    ))

    print(f"\nCollection complete!", flush=True)
    print(f"  Years collected: {YEARS[0]}–{YEARS[-1]}", flush=True)
    print(f"  Total written:   {writer.total:,}  (after abstract filter)", flush=True)
    print(f"  Output files:    {writer.file_idx + 1}", flush=True)
    print(f"  Output dir:      {out_dir}", flush=True)


if __name__ == "__main__":
    main()
