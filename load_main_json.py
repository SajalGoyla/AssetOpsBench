"""Load main.json into a running CouchDB instance.

main.json is a CouchDB bulk-docs export: {"docs": [doc1, doc2, ...]}.
At ~1.1 GB it's too large for a single _bulk_docs POST, so this script
streams the file, batches documents, and uploads them in chunks.

Usage:
    uv run python load_main_json.py                         # defaults
    uv run python load_main_json.py --file main.json        # explicit
    uv run python load_main_json.py --batch-size 2000       # larger batches
    uv run python load_main_json.py --db chiller            # target DB name
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_FILE = Path(__file__).parent / "main.json"


def _connect_info() -> tuple[str, str, str, str]:
    url = os.environ.get("COUCHDB_URL", "http://localhost:5984")
    db = os.environ.get("IOT_DBNAME", "chiller")
    user = os.environ.get("COUCHDB_USERNAME", "admin")
    pw = os.environ.get("COUCHDB_PASSWORD", "password")
    return url, db, user, pw


def _ensure_db(url: str, db: str, auth: tuple[str, str]) -> None:
    """Create the database if it doesn't exist."""
    r = requests.get(f"{url}/{db}", auth=auth)
    if r.status_code == 404:
        print(f"  Creating database '{db}'...")
        r2 = requests.put(f"{url}/{db}", auth=auth)
        r2.raise_for_status()
        print(f"  Database '{db}' created.")
    elif r.status_code == 200:
        print(f"  Database '{db}' already exists.")
    else:
        r.raise_for_status()


def _ensure_index(url: str, db: str, auth: tuple[str, str]) -> None:
    """Create an index on (asset_id, timestamp) for efficient queries."""
    idx = {"type": "json", "index": {"fields": ["asset_id", "timestamp"]}}
    r = requests.post(
        f"{url}/{db}/_index", json=idx, auth=auth,
        headers={"Content-Type": "application/json"},
    )
    if r.status_code in (200, 201):
        result = r.json().get("result", "unknown")
        print(f"  Index: {result}")
    else:
        print(f"  Index creation returned {r.status_code}: {r.text[:200]}")


def _load_docs_streaming(filepath: Path) -> list[dict]:
    """Load all docs from main.json. Expects {"docs": [...]}."""
    print(f"  Loading {filepath} ({filepath.stat().st_size / 1e9:.2f} GB)...")
    t0 = time.perf_counter()
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    docs = data.get("docs", data if isinstance(data, list) else [])
    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(docs):,} documents in {elapsed:.1f}s")
    return docs


def _strip_rev(doc: dict) -> dict:
    """Remove _rev so CouchDB treats these as new inserts (avoids conflicts)."""
    d = dict(doc)
    d.pop("_rev", None)
    return d


def _upload_batch(
    url: str, db: str, auth: tuple[str, str],
    batch: list[dict], batch_num: int,
) -> tuple[int, int]:
    """Upload a batch via _bulk_docs. Returns (ok_count, err_count)."""
    r = requests.post(
        f"{url}/{db}/_bulk_docs",
        json={"docs": batch},
        auth=auth,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    if r.status_code not in (200, 201):
        print(f"    Batch {batch_num} FAILED ({r.status_code}): {r.text[:200]}")
        return 0, len(batch)

    results = r.json()
    ok = sum(1 for d in results if "ok" in d or "id" in d)
    err = sum(1 for d in results if "error" in d)
    return ok, err


def main() -> None:
    parser = argparse.ArgumentParser(description="Load main.json into CouchDB")
    parser.add_argument("--file", default=str(_DEFAULT_FILE), help="Path to JSON file")
    parser.add_argument("--batch-size", type=int, default=1000, help="Docs per batch (default 1000)")
    parser.add_argument("--db", default=None, help="Target DB name (default: IOT_DBNAME from .env)")
    parser.add_argument("--skip-existing", action="store_true", help="Strip _id to avoid conflicts with existing docs")
    args = parser.parse_args()

    url, db_name, user, pw = _connect_info()
    if args.db:
        db_name = args.db
    auth = (user, pw)

    print(f"\n{'=' * 60}")
    print(f"  Loading main.json → CouchDB")
    print(f"{'=' * 60}")
    print(f"  CouchDB:    {url}")
    print(f"  Database:   {db_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"{'=' * 60}\n")

    # 1. Ensure DB exists
    _ensure_db(url, db_name, auth)

    # 2. Load documents
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"ERROR: {filepath} not found", file=sys.stderr)
        sys.exit(1)
    docs = _load_docs_streaming(filepath)

    # 3. Strip _rev (and optionally _id) to avoid conflicts
    cleaned = []
    for doc in docs:
        d = _strip_rev(doc)
        if args.skip_existing:
            d.pop("_id", None)
        cleaned.append(d)
    docs = cleaned

    # 4. Upload in batches
    total = len(docs)
    batch_size = args.batch_size
    total_ok = 0
    total_err = 0
    t0 = time.perf_counter()

    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        ok, err = _upload_batch(url, db_name, auth, batch, batch_num)
        total_ok += ok
        total_err += err

        pct = min(100, (i + len(batch)) / total * 100)
        print(f"    Batch {batch_num}/{total_batches}: {ok} ok, {err} err  [{pct:.0f}%]")

    elapsed = time.perf_counter() - t0

    # 5. Ensure index
    _ensure_index(url, db_name, auth)

    print(f"\n{'=' * 60}")
    print(f"  Done! {total_ok:,} inserted, {total_err:,} errors in {elapsed:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
