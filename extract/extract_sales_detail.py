"""extract/extract_sales_detail.py — Sales Detail cube extract (SL / SL-I).

Connects to AutoCube_DTR_23160 / Sales Detail and pulls 2 years of
transactions filtered to tran codes SL and SL-I.  Writes to the
sales_detail_transactions table in Supabase.

NOTE: OPSL is not a tran code in this Autocube catalog.  Outside purchases
are identified by [Sales Detail].[Stock Flag] = 'N' on SL lines.

CUBE FIELDS
-----------
    [Sales Date].[Invoice Date].[Inv Date]        -> tran_date
    [Location].[Loc].[Loc]                        -> location_id  (LOC-NNN)
    [Tran Code].[Tran Code].[Tran Code]           -> tran_code
    [Sales Detail].[Ship To].[Ship To]            -> ship_to
    [Sales Detail].[Stock Flag].[Stock Flag]      -> stock_flag  ('Y'=stocked, 'N'=outside)
    [Product].[Prod Line PN].[Prod Line PN]       -> prod_line_pn
    [Counterman].[Counterman].[Counterman]        -> salesman_id
    [Measures].[Qty Ship]                         -> qty_ship
    [Measures].[Sales]                            -> sales
    [Measures].[Gross Profit]                     -> gross_profit
    [Measures].[Backorder]                        -> backorder_qty

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

    python -m extract.extract_sales_detail --start-date YYYY-MM-DD
        Pull from a fixed start date to yesterday (overrides --lookback-days).
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
# OPSL is not a tran code in AutoCube_DTR_23160; outside purchases are
# identified by stock_flag = 'N' on SL lines.
_ALLOWED_TRAN_CODES = frozenset({"SL", "SL-I"})

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
          CROSSJOIN(
            [Sales Detail].[Stock Flag].[Stock Flag].MEMBERS,
            CROSSJOIN(
              [Product].[Prod Code].[Prod Code].MEMBERS,
              [Counterman].[Counterman].[Counterman].MEMBERS
            )
          )
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

    ship_to      = (r.get("[Sales Detail].[Ship To].[Ship To]")       or "").strip()
    stock_flag   = (r.get("[Sales Detail].[Stock Flag].[Stock Flag]") or "").strip().upper()
    prod_line_pn = (r.get("[Product].[Prod Code].[Prod Code]")        or "").strip()
    salesman_raw = (r.get("[Counterman].[Counterman].[Counterman]")   or "").strip()
    salesman_id  = salesman_raw[:20] if salesman_raw else None

    qty_ship      = clean_numeric(r.get("[Measures].[Qty Ship]"))
    sales         = clean_numeric(r.get("[Measures].[Ext Price]"))
    gross_profit  = clean_numeric(r.get("[Measures].[Gross Profit]"))
    backorder_qty = clean_numeric(r.get("[Measures].[Backorder]"))

    return {
        "transaction_id": _sdt_id(tran_date, location_id, tran_code, ship_to, prod_line_pn),
        "tran_date":      tran_date,
        "location_id":    location_id,
        "tran_code":      tran_code,
        "ship_to":        ship_to             or None,
        "stock_flag":     stock_flag[:1]      if stock_flag else None,
        "prod_line_pn":   prod_line_pn        or None,
        "salesman_id":    salesman_id,
        "qty_ship":       round(float(qty_ship),      4) if qty_ship      is not None else None,
        "sales":          round(float(sales),          4) if sales         is not None else None,
        "gross_profit":   round(float(gross_profit),   4) if gross_profit  is not None else None,
        "backorder_qty":  round(float(backorder_qty),  4) if backorder_qty is not None else None,
    }


def _dedupe_chunk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate transaction_ids within a single chunk, keeping first occurrence.

    The SSAS cube can return multiple rows that collapse to the same natural key
    (e.g. the same part sold to the same customer on the same day appearing under
    two dimension member aliases).  Postgres ON CONFLICT DO UPDATE raises 21000
    ("cannot affect a row a second time") when a batch contains duplicates, so
    we must deduplicate before each upsert.
    """
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        tid = r["transaction_id"]
        if tid not in seen:
            seen[tid] = r
    removed = len(rows) - len(seen)
    if removed:
        log.debug("Deduped %d duplicate transaction_ids within chunk.", removed)
    return list(seen.values())


def _detect_resume_date(db: Any) -> str | None:
    """Return the latest tran_date already in sales_detail_transactions, or None."""
    try:
        resp = (
            db.table("sales_detail_transactions")
            .select("tran_date")
            .order("tran_date", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]["tran_date"][:10]
    except Exception as exc:
        log.warning("Could not detect resume date: %s", exc)
    return None


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
    start_date_override: date | None = None,
) -> int:
    """Pull SL / SL-I transactions from the Sales Detail cube.

    Iterates the lookback window in 7-day chunks.  Each chunk's rows are
    filtered Python-side to _ALLOWED_TRAN_CODES before writing to Supabase.
    stock_flag ('Y'/'N') is captured from [Sales Detail].[Stock Flag] so
    outside purchases (stock_flag='N') can be identified without a separate
    OPSL tran code (which does not exist in this Autocube catalog).

    On the first chunk: if zero rows are returned, raw XML is printed to
    stdout for diagnosis (wrong measure name, wrong hierarchy, etc.).

    Returns 0 on success, 1 if any chunks failed.
    """
    t0 = time.perf_counter()

    end_date   = date.today() - timedelta(days=1)
    if start_date_override is not None:
        start_date = start_date_override
    else:
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

    # Resume: find the latest date already in Supabase and skip chunks whose
    # end date is strictly before it.  Re-processes the chunk containing the
    # latest date to catch any partial writes on that day.
    resume_date: str | None = None
    if not dry_run:
        resume_date = _detect_resume_date(db)
        if resume_date:
            log.info("[RESUME] Latest tran_date in Supabase: %s — skipping earlier chunks.",
                     resume_date)

    all_dry_rows:    list[dict[str, Any]] = []
    failed_chunks:   list[str]            = []
    skipped_chunks   = 0
    total_loaded     = 0
    total_skipped    = 0
    first_chunk_done = False

    for idx, (label, start_key, end_key) in enumerate(chunks, 1):
        # Skip chunks whose entire window is before the resume date.
        # Re-process the chunk that contains resume_date (end_key >= resume_date).
        if resume_date and not dry_run:
            chunk_end_iso = f"{end_key[:4]}-{end_key[4:6]}-{end_key[6:]}"
            if chunk_end_iso < resume_date:
                skipped_chunks += 1
                continue

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

        chunk_cleaned = _dedupe_chunk(chunk_cleaned)

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
    log.info("  Chunks:   %d succeeded / %d processed / %d total (%d skipped via resume)",
             len(chunks) - len(failed_chunks) - skipped_chunks,
             len(chunks) - skipped_chunks,
             len(chunks), skipped_chunks)
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
        description="Sales Detail cube extract — SL / SL-I tran codes + stock_flag",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pull and clean data but do not write to Supabase",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=_LOOKBACK_DAYS, metavar="N",
        help=f"Days of history to pull (default: {_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--start-date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Fixed start date (overrides --lookback-days); pulls from this date to yesterday",
    )
    args = parser.parse_args()

    start_date_override: date | None = None
    if args.start_date:
        try:
            start_date_override = date.fromisoformat(args.start_date)
        except ValueError:
            print(f"ERROR: --start-date must be YYYY-MM-DD, got {args.start_date!r}")
            return 1

    return run_sales_detail_extract(
        dry_run=args.dry_run,
        lookback_days=args.lookback_days,
        start_date_override=start_date_override,
    )


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
