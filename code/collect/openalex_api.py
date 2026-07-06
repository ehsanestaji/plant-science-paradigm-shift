"""
Collect plant science works from OpenAlex via cursor-paginated API.

Two-phase approach:
  Phase 1 (this script): Download API pages to compressed JSONL files.
    - Runs on LOGIN node (needs internet)
    - Pure network I/O, no database overhead
    - ~200 works/page, ~2-3s/page → ~7-8 hours for 5.4M works
    - Saves to data/raw/openalex_api/ as .jsonl.gz files (one per 1000 pages)
    - Fully resumable via checkpoint

  Phase 2 (openalex_ingest.py): Bulk-load JSONL into DuckDB.
    - Runs as SLURM job (compute-bound)
    - Uses DuckDB native bulk loading for speed

Usage (on login node, in tmux/screen):
    python -u -m src.collect.openalex_api \
        --output-dir data/raw/openalex_api \
        --email your@email.com
"""

import os
import sys
import json
import gzip
import time
import socket
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

# Hard OS-level socket timeout — catches half-open TCP sockets that
# requests' application-level timeout misses (e.g. after TCP handshake
# but before data arrives, or mid-stream stalls on the login node).
socket.setdefaulttimeout(45)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.storage_monitor import check_storage
from src.utils.checkpointing import checkpoint_save, checkpoint_load, checkpoint_clear

# OpenAlex API base
API_BASE = "https://api.openalex.org/works"

# Plant science subfield IDs (OpenAlex topics/subfields system)
PLANT_SUBFIELDS = [
    "1110",   # Plant Science
    "1102",   # Agronomy and Crop Science
    "1108",   # Horticulture
    "1107",   # Forestry
]

SUBFIELD_FILTER = "|".join(PLANT_SUBFIELDS)

BATCH_SIZE = 200       # max per API request
PAGES_PER_FILE = 1000  # ~200K works per file, ~300-500MB compressed
CHECKPOINT_EVERY = 50  # save cursor checkpoint every N pages
RATE_LIMIT_DELAY = 0.12


def make_session(email: str) -> requests.Session:
    """Create a requests session with retry logic and strict timeouts."""
    session = requests.Session()
    session.headers["User-Agent"] = f"PlantScienceMetascience/0.1 (mailto:{email})"
    retry = Retry(
        total=5,
        backoff_factor=2,          # 2, 4, 8, 16, 32s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_page(cursor: str, email: str, session: requests.Session) -> dict:
    """Fetch one page of results from OpenAlex API."""
    params = {
        "filter": f"primary_topic.subfield.id:{SUBFIELD_FILTER}",
        "per_page": BATCH_SIZE,
        "cursor": cursor,
        "mailto": email,
    }
    # Tuple: (connect_timeout, read_timeout) — read timeout is per-chunk,
    # so a stalled response is caught within 40s regardless of total size.
    resp = session.get(API_BASE, params=params, timeout=(10, 40))
    resp.raise_for_status()
    return resp.json()


def collect_all(output_dir: str, email: str):
    """Download all plant science works to compressed JSONL files."""
    check_storage()

    os.makedirs(output_dir, exist_ok=True)

    session = make_session(email)

    # Resume from checkpoint
    ckpt = checkpoint_load("openalex_api")
    if ckpt:
        cursor = ckpt["cursor"]
        total_collected = ckpt["total_collected"]
        page = ckpt["page"]
        file_index = ckpt.get("file_index", page // PAGES_PER_FILE)
        print(f"Resuming from checkpoint: page {page}, {total_collected:,} works, "
              f"file {file_index}", flush=True)
    else:
        cursor = "*"
        total_collected = 0
        page = 0
        file_index = 0

    print(f"Downloading plant science works from OpenAlex API...", flush=True)
    print(f"  Subfields: {PLANT_SUBFIELDS}", flush=True)
    print(f"  Filter: primary_topic.subfield.id:{SUBFIELD_FILTER}", flush=True)
    print(f"  Output: {output_dir}/", flush=True)
    print(f"  Pages per file: {PAGES_PER_FILE} (~{PAGES_PER_FILE * BATCH_SIZE:,} works)", flush=True)
    t0 = time.time()

    # Open current output file
    current_file = None
    pages_in_file = page % PAGES_PER_FILE if page > 0 else 0

    def open_new_file():
        nonlocal current_file, file_index, pages_in_file
        if current_file:
            current_file.close()
        filepath = os.path.join(output_dir, f"openalex_plant_{file_index:04d}.jsonl.gz")
        # Append mode if resuming into same file
        mode = "ab" if os.path.exists(filepath) and pages_in_file > 0 else "wb"
        current_file = gzip.open(filepath, mode)
        print(f"  Writing to {filepath} (mode={mode[0]})", flush=True)

    open_new_file()

    try:
        while cursor:
            page += 1
            try:
                data = fetch_page(cursor, email, session)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"  Rate limited at page {page}, sleeping 60s...", flush=True)
                    time.sleep(60)
                    continue
                elif e.response.status_code == 400:
                    print(f"  400 error at page {page}: {e.response.text[:200]}", flush=True)
                    break
                raise
            except requests.exceptions.RequestException as e:
                print(f"  Network error at page {page}: {e}. Retrying in 10s...", flush=True)
                time.sleep(10)
                continue

            results = data.get("results", [])
            if not results:
                break

            # Write each work as a JSON line
            for work in results:
                line = json.dumps(work, ensure_ascii=False) + "\n"
                current_file.write(line.encode("utf-8"))
            total_collected += len(results)
            pages_in_file += 1

            cursor = data.get("meta", {}).get("next_cursor")

            # Progress
            total_count = data.get("meta", {}).get("count", "?")
            elapsed = time.time() - t0
            rate = total_collected / elapsed if elapsed > 0 else 0
            pct = (total_collected / total_count * 100) if isinstance(total_count, int) and total_count > 0 else 0
            eta_hours = ((total_count - total_collected) / rate / 3600) if rate > 0 and isinstance(total_count, int) else 0
            print(f"  Page {page}: {total_collected:,} / {total_count:,} "
                  f"({pct:.2f}%, {rate:.0f} w/s, ETA {eta_hours:.1f}h)",
                  flush=True)

            # New file every PAGES_PER_FILE pages
            if pages_in_file >= PAGES_PER_FILE:
                current_file.close()
                file_index += 1
                pages_in_file = 0
                open_new_file()

            # Checkpoint
            if page % CHECKPOINT_EVERY == 0:
                current_file.flush()
                checkpoint_save("openalex_api", {
                    "cursor": cursor,
                    "total_collected": total_collected,
                    "page": page,
                    "file_index": file_index,
                })

            # Storage check every 500 pages
            if page % 500 == 0:
                check_storage()

            time.sleep(RATE_LIMIT_DELAY)

    except KeyboardInterrupt:
        print(f"\nInterrupted! Saving checkpoint...", flush=True)
        current_file.flush()
        checkpoint_save("openalex_api", {
            "cursor": cursor,
            "total_collected": total_collected,
            "page": page,
            "file_index": file_index,
        })

    if current_file:
        current_file.close()

    elapsed = time.time() - t0
    checkpoint_clear("openalex_api")

    # Summary
    files = sorted(Path(output_dir).glob("*.jsonl.gz"))
    total_size = sum(f.stat().st_size for f in files)
    print(f"\nDownload complete!", flush=True)
    print(f"  Total works: {total_collected:,}", flush=True)
    print(f"  Files: {len(files)} ({total_size / 1e9:.1f} GB)", flush=True)
    print(f"  Time: {elapsed/3600:.1f} hours", flush=True)
    if elapsed > 0:
        print(f"  Rate: {total_collected/elapsed:.0f} works/sec", flush=True)
    print(f"\nNext step: run openalex_ingest.py to load into DuckDB", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Download plant science works from OpenAlex API")
    parser.add_argument("--output-dir", default="data/raw/openalex_api",
                        help="Directory for JSONL output files")
    parser.add_argument("--email", default=None, help="Email for polite API pool")
    args = parser.parse_args()

    email = args.email
    if not email:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("OPENALEX_EMAIL="):
                        email = line.strip().split("=", 1)[1]
        if not email:
            email = "ehes0002@umu.se"
            print(f"  No email configured, using: {email}")

    collect_all(args.output_dir, email)


if __name__ == "__main__":
    main()
