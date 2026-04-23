"""extract/extract_sales_detail.py — Sales Detail cube extract (SL / SL-I / OPSL).

Connects to AutoCube_DTR_23160 / Sales Detail and pulls 2 years of
transactions filtered to tran codes SL, SL-I, and OPSL.  Writes to
the sales_detail_transactions table in Supabase.

CUBE FIELDS
-----------
    [Sales Date].[Invoice Date].[Inv Date]    -> tran_date
    [Location].[Loc].[Loc]                    -> location_id  (LOC-NNN)
    [Tran Code].[Tran Code].[Tran Code]       -> tran_code
    [Sales Detail].[Ship To].[Ship To]        -> ship_to
    [Product].[Prod Line PN].[Prod Line PN]   -> prod_line_pn
    [Measures].[Qty Ship]                     -> qty_ship
    [Measures].[Sales]                        -> sales
    [Measures].[Gross Profit]                 -> gross_profit
    [Measures].[Backorder]                    -> backorder_qty

MDX APPROACH
------------
7-day date chunks over the ROWS axis (same pattern as MDX_MONTHLY_RANGE in
autocube_pull.py).  Tran Code is fetched via .MEMBERS and filtered Python-side
to _ALLOWED_TRAN_CODES — avoids SSAS "member not found" errors if any code has
no data in this deployment.

MEASURE / HIERARCHY NAME NOTES
-------------------------------
If the extract returns a SOAP fault about unknown members or measures, verify
the exact names against the cube with:
    python -m extract.autocube_pull --test
Known alternatives:
    [Measures].[Sales]       -> try [Measures].[Ext Price] if not found
    [Measures].[Backorder]   -> try [Measures].[Backorder Qty] if not found
    [Product].[Prod Line PN] -> try [Product].[Prod Code] if not found

MODES
-----
    python -m extract.extract_sales_detail --dry-run
        Pull last 2 years, print first 20 rows to stdout, no Supabase write.
        If zero rows are returned on the first chunk, prints raw XML.

    python -m extract.extract_sales_detail
        Full 2-year pull and write to sales_detail_transactions.

    python -m extract.extract_sales_detail --lookback-days N
        Pull last N days instead of the default 730.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extract.autocube_pull import (
    AutocubeClient,
    SecurityError,
    clean_date,
    clean_numeric,
    _extract_location_code,
    get_client,
    query_validator,
)
from extract.autocube_product_pull import _extract_tran_code
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOOKBACK_DAYS     = 730   # Default: 2 years of history
_BATCH_SIZE        = 1000  # Supabase upsert batch size
_LOG_EVERY         = 5000  # Log loaded-row progress every N rows
_MAX_RETRIES       = 3     # Retry attempts on 57014 statement timeout
_RETRY_BASE_SLEEP  = 5     # Base seconds per retry (multiplied by attempt number)
_CHUNK_DAYS        = 7     # Days per MDX request

# Python-side tran-code filter.  .MEMBERS is used in MDX so SSAS does not error
# on codes that may have no activity in this deployment.
_ALLOWED_TRAN_CODES = frozenset({"SL", "SL-I", "OPSL"})

# ---------------------------------------------------------------------------
# MDX — Sales Detail cube, 7-day date range chunk.
#
# Tran Code stays on ROWS (not in a WHERE slicer) so the bare code value is
# visible for Python-side filtering.  The date range is expressed as a member
# set between keyed start/end (YYYYMMDD) rather than .MEMBERS to avoid
# materialising 3+ years of date members.
#
# [Product].[Prod Line PN] is the combined product-line+part-number hierarchy
# in the Sales Detail cube.  See HIERARCHY NAME NOTES in the module docstring
# if the query raises a SOAP fault.
# ---------------------------------------------------------------------------
MDX_SALES_DETAIL_RANGE = """\
SELECT
  NON EMPTY {{
    [Measures].[Qty Ship],
    [Measures].[Ext Price],
    [Measures].[Gross Profit]
  }} ON COLUMNS,
  NON EMPTY CROSSJOIN(
    {{
      [Sales Date].[Invoice Date].[Inv Date].&[{start_key}]
      : [Sales Date].[Invoice Date].[Inv Date].&[{end_key}]
    }},
    CROSSJOIN(
      [Location].[Loc].[Loc].MEMBERS,
      CROSSJOIN(
        [Tran Code].[Tran Code].[Tran Code].MEMBERS,
        CROSSJOIN(
          [Sales Detail].[Ship To].[Ship To].MEMBERS,
          [Product].[Prod Code].[Prod Code].MEMBERS
        )
      )
    )
  ) ON ROWS
FROM [Sales Detail]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_chunk_ranges(
    start: date, end: date, chunk_days: int = _CHUNK_DAYS,
) -> list[tuple[str, str, str]]:
    """Return (label, start_key, end_key) tuples for non-overlapping windows."""
    chunks: list[tuple[str, str, str]] = []
    cur = start
    while cur <= end:
        ce = min(cur + timedelta(days=chunk_days - 1), end)
        chunks.append((
            f"{cur.isoformat()}..{ce.isoformat()}",
            cur.strftime("%Y%m%d"),
            ce.strftime("%Y%m%d"),
        ))
        cur = ce + timedelta(days=1)
    return chunks


def _sdt_id(
    tran_date: str,
    location_id: str,
    tran_code: str,
    ship_to: str,
    prod_line_pn: str,
) -> str:
    """Deterministic transaction ID — SHA-1 of natural key, 20 hex chars."""
    key = f"{tran_date}|{location_id}|{tran_code}|{ship_to}|{prod_line_pn}"
    return "SDT-" + hashlib.sha1(key.encode()).hexdigest()[:20]


def _clean_row(r: dict[str, Any]) -> dict[str, Any] | None:
    """Map one raw SSAS result row to a sales_detail_transactions record.

    Returns None if the row should be skipped: tran_code not in
    _ALLOWED_TRAN_CODES, missing date, or missing location.
    """
    tran_raw  = r.get("[Tran Code].[Tran Code].[Tran Code]", "")
    tran_code = _extract_tran_code(tran_raw)
    if tran_code not in _ALLOWED_TRAN_CODES:
        return None

    tran_date = clean_date(r.get("[Sales Date].[Invoice Date].[Inv Date]", ""))
    if not tran_date:
        return None

    loc_raw     = r.get("[Location].[Loc].[Loc]", "")
    location_id = _extract_location_code(loc_raw) if loc_raw else None
    if not location_id:
        return None

    ship_to      = (r.get("[Sales Detail].[Ship To].[Ship To]") or "").strip()
    prod_line_pn = (r.get("[Product].[Prod Code].[Prod Code]")  or "").strip()

    qty_ship      = clean_numeric(r.get("[Measures].[Qty Ship]"))
    sales         = clean_numeric(r.get("[Measures].[Ext Price]"))
    gross_profit  = clean_numeric(r.get("[Measures].[Gross Profit]"))
    backorder_qty = clean_numeric(r.get("[Measures].[Backorder]"))

    return {
        "transaction_id": _sdt_id(tran_date, location_id, tran_code, ship_to, prod_line_pn),
        "tran_date":      tran_date,
        "location_id":    location_id,
        "tran_code":      tran_code,
        "ship_to":        ship_to      or None,
        "prod_line_pn":   prod_line_pn or None,
        "qty_ship":       round(float(qty_ship),      4) if qty_ship      is not None else None,
        "sales":          round(float(sales),          4) if sales         is not None else None,
        "gross_profit":   round(float(gross_profit),   4) if gross_profit  is not None else None,
        "backorder_qty":  round(float(backorder_qty),  4) if backorder_qty is not None else None,
    }


def _upsert_with_retry(db: Any, batch: list[dict]) -> None:
    """Upsert a batch into sales_detail_transactions, retrying on 57014 timeouts."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            db.table("sales_detail_transactions").upsert(
                batch, on_conflict="transaction_id",
            ).execute()
            return
        except Exception as exc:
            if "57014" in str(exc) and attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_SLEEP * attempt
                log.warning(
                    "57014 statement timeout (attempt %d/%d) — retrying in %ds.",
                    attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            raise


# ---------------------------------------------------------------------------
# Core extract
# ---------------------------------------------------------------------------

def run_sales_detail_extract(
    dry_run: bool = False,
    lookback_days: int = _LOOKBACK_DAYS,
) -> int:
    """Pull SL / SL-I / OPSL transactions from the Sales Detail cube.

    Iterates the lookback window in 7-day chunks.  Each chunk's rows are
    filtered Python-side to _ALLOWED_TRAN_CODES before writing to Supabase.

    On the first chunk: if zero rows are returned, raw XML is printed to
    stdout for diagnosis (wrong measure name, wrong hierarchy, etc.).

    Returns 0 on success, 1 if any chunks failed.
    """
    t0 = time.perf_counter()

    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days - 1)
    chunks     = _generate_chunk_ranges(start_date, end_date)

    log.info("=" * 60)
    log.info("  SALES DETAIL EXTRACT%s", " [DRY RUN]" if dry_run else "")
    log.info("  Cube:       Sales Detail  (AutoCube_DTR_23160)")
    log.info("  Tran codes: %s", sorted(_ALLOWED_TRAN_CODES))
    log.info("  Window:     %s to %s (%d days, %d chunks)",
             start_date, end_date, lookback_days, len(chunks))
    log.info("=" * 60)

    try:
        client = get_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    if not dry_run:
        try:
            from db.connection import get_client as get_db_client
            db = get_db_client()
        except Exception:
            log.exception("Supabase connection failed.")
            return 1
    else:
        db = None

    all_dry_rows:   list[dict[str, Any]] = []
    failed_chunks:  list[str]            = []
    total_loaded    = 0
    total_skipped   = 0
    first_chunk_done = False

    for idx, (label, start_key, end_key) in enumerate(chunks, 1):
        chunk_t0 = time.perf_counter()
        mdx = MDX_SALES_DETAIL_RANGE.format(start_key=start_key, end_key=end_key)

        try:
            if not first_chunk_done:
                # First chunk: capture raw XML bytes so we can print them if
                # zero rows come back (diagnoses wrong measure / hierarchy names).
                query_validator(mdx)
                envelope = client._build_execute_envelope(mdx)
                resp     = client._post(envelope, action="Execute")
                raw_xml  = resp.content
                rows     = client._parse_rowset(raw_xml)
                first_chunk_done = True

                if not rows:
                    print("=" * 60)
                    print("ZERO ROWS on first chunk — raw XML (first 8000 chars):")
                    print("=" * 60)
                    print(raw_xml[:8000].decode("utf-8", errors="replace"))
                    print("=" * 60)
            else:
                rows = client.execute_mdx(mdx)

        except SecurityError:
            log.exception("Security violation in chunk %s — aborting.", label)
            return 1
        except Exception:
            log.exception("MDX failed for chunk %s — skipping.", label)
            failed_chunks.append(label)
            continue

        chunk_cleaned: list[dict[str, Any]] = []
        for r in rows:
            cleaned = _clean_row(r)
            if cleaned is None:
                total_skipped += 1
                continue
            chunk_cleaned.append(cleaned)

        if dry_run:
            all_dry_rows.extend(chunk_cleaned)
        elif chunk_cleaned:
            for i in range(0, len(chunk_cleaned), _BATCH_SIZE):
                batch = chunk_cleaned[i:i + _BATCH_SIZE]
                try:
                    _upsert_with_retry(db, batch)
                    total_loaded += len(batch)
                    if total_loaded % _LOG_EVERY < _BATCH_SIZE:
                        log.info("  %d rows loaded so far ...", total_loaded)
                except Exception:
                    log.exception(
                        "Upsert failed in chunk %s at offset %d — skipping batch.",
                        label, i,
                    )
                    failed_chunks.append(f"{label}@{i}")
                    break

        chunk_elapsed = time.perf_counter() - chunk_t0
        log.info(
            "[%3d/%d] %s  raw=%d  kept=%d  (%.1fs)",
            idx, len(chunks), label, len(rows), len(chunk_cleaned), chunk_elapsed,
        )

    elapsed = time.perf_counter() - t0

    log.info("=" * 60)
    log.info("  SALES DETAIL EXTRACT COMPLETE")
    log.info("  Chunks:   %d succeeded / %d total",
             len(chunks) - len(failed_chunks), len(chunks))
    log.info("  Skipped:  %d (tran code filtered or missing fields)", total_skipped)
    if dry_run:
        log.info("  Ready:    %d rows (dry run — not written)", len(all_dry_rows))
    else:
        log.info("  Loaded:   %d rows", total_loaded)
    if failed_chunks:
        log.warning("  Failed:   %s", failed_chunks)
    log.info("  Elapsed:  %.1fs", elapsed)
    log.info("=" * 60)

    if dry_run:
        print("\n=== FIRST 20 ROWS ===")
        for i, r in enumerate(all_dry_rows[:20]):
            print(f"  {i + 1:2d}. {r}")
        if not all_dry_rows:
            print("  (no rows — check raw XML output above)")
        print(f"\nTotal rows ready: {len(all_dry_rows)}")

    return 1 if failed_chunks else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sales Detail cube extract — SL / SL-I / OPSL tran codes",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pull and clean data but do not write to Supabase",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=_LOOKBACK_DAYS, metavar="N",
        help=f"Days of history to pull (default: {_LOOKBACK_DAYS})",
    )
    args = parser.parse_args()
    return run_sales_detail_extract(dry_run=args.dry_run, lookback_days=args.lookback_days)


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
