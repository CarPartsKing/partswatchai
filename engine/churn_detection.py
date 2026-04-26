"""engine/churn_detection.py — Customer churn detection for PartsWatch AI.

Reads sales_detail_transactions, computes per-customer per-location spend
patterns, and writes churn flags to customer_churn_flags.

METHODOLOGY
-----------
2-year lookback window = today - 730 days  ..  today

Baseline period  = months 1-18 of the window
                 = window_start  ..  window_start + BASELINE_MONTHS*30 days
Comparison period = last COMPARISON_DAYS days  (default 90)

FLAG logic
----------
AT_RISK   — zero spend in comparison period; last purchase 90–180 days ago
CHURNED   — zero spend in comparison period; last purchase 181–365 days ago
LOST      — zero spend in comparison period; last purchase > 365 days ago
DECLINING — comparison spend < expected by DECLINING_THRESHOLD
STABLE    — everything else

Only customers with at least MIN_BASELINE_PURCHASES transactions,
MIN_BASELINE_SPEND/month average, and MIN_ACTIVE_MONTHS distinct calendar
months active in the baseline window are written to the output table.

Performance
-----------
Processes one location at a time instead of paginating across all rows.
Each location query is small enough to stay within Supabase's statement
timeout.  Locations are discovered from location_names (fast) with a
fallback scan of the source table.

Usage
-----
    python -m engine.churn_detection            # live run
    python -m engine.churn_detection --dry-run  # compute, log, no DB writes
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from typing import Any

from db.connection import get_client, get_new_client
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
LOOKBACK_DAYS: int = 730
BASELINE_MONTHS: int = 18
COMPARISON_DAYS: int = 90

MIN_BASELINE_PURCHASES: int = 3
MIN_BASELINE_SPEND: float = 333.0   # $/month; ~$1,000/quarter
MIN_ACTIVE_MONTHS: int = 3          # distinct calendar months in baseline window

DECLINING_THRESHOLD: float = 0.30

WRITE_BATCH_SIZE: int = 1_000
LOG_EVERY: int = 5_000

_LOCATION_SCAN_PAGE: int = 2_000  # used only in fallback location scan
_MAX_RETRIES: int = 5
_RETRY_DELAY: float = 5.0
_RETRYABLE_TOKENS: tuple[str, ...] = (
    "57014",
    "statement timeout",
    "canceling statement",
    "ConnectionTerminated",
    "RemoteProtocolError",
    "ReadTimeout",
    "ConnectTimeout",
    "PoolTimeout",
    "ConnectionError",
)

_TARGET_TABLE = "customer_churn_flags"
_SOURCE_TABLE = "sales_detail_transactions"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    blob = type(exc).__name__ + " " + str(exc)
    return any(tok in blob for tok in _RETRYABLE_TOKENS)


def _get_fresh_client() -> Any:
    try:
        return get_new_client()
    except Exception:
        return get_client()


def _fetch_page(client: Any, query_fn, attempt_label: str) -> list[dict]:
    """Execute a single Supabase query page with retry."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return query_fn(client).execute().data
        except Exception as exc:
            if attempt < _MAX_RETRIES and _is_retryable(exc):
                delay = _RETRY_DELAY * attempt
                log.warning(
                    "%s attempt %d/%d failed (%s), retrying in %.0fs",
                    attempt_label, attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                client = _get_fresh_client()
            else:
                raise
    return []  # unreachable


def _upsert_batch(client: Any, rows: list[dict]) -> None:
    """Upsert a batch with retry on timeout/connection errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client.table(_TARGET_TABLE).upsert(
                rows,
                on_conflict="customer_id,location_id",
            ).execute()
            return
        except Exception as exc:
            if attempt < _MAX_RETRIES and _is_retryable(exc):
                delay = _RETRY_DELAY * attempt
                log.warning(
                    "upsert attempt %d/%d failed (%s), retrying in %.0fs",
                    attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                client = _get_fresh_client()
            else:
                raise


# ---------------------------------------------------------------------------
# Date window helpers
# ---------------------------------------------------------------------------

def _detect_effective_today(client: Any) -> date:
    """Return the max tran_date in the source table as the reference date.

    Using the actual data ceiling rather than calendar today prevents mass
    false-positive CHURNED flags when the cube's latest data lags behind the
    current date by days or weeks.
    """
    try:
        resp = (
            client.table(_SOURCE_TABLE)
            .select("tran_date")
            .order("tran_date", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            effective = date.fromisoformat(resp.data[0]["tran_date"])
            log.info("Effective today (max tran_date in data): %s", effective)
            return effective
    except Exception as exc:
        log.warning("Could not detect max tran_date (%s); falling back to calendar today.", exc)
    return date.today()


def _compute_date_windows(effective_today: date) -> tuple[date, date, date]:
    """Return (window_start, baseline_end, comparison_start).

    Windows are anchored to the max date in the data, not calendar today,
    so the comparison period always aligns with actual transactions.
    """
    window_start = effective_today - timedelta(days=LOOKBACK_DAYS)
    baseline_end = window_start + timedelta(days=BASELINE_MONTHS * 30)
    comparison_start = effective_today - timedelta(days=COMPARISON_DAYS)
    return window_start, baseline_end, comparison_start


# ---------------------------------------------------------------------------
# Location discovery
# ---------------------------------------------------------------------------

def _fetch_location_ids(
    client: Any,
    window_start_str: str,
    today_str: str,
) -> list[str]:
    """Return sorted list of distinct location_ids to process.

    Primary: queries the locations table for active locations (fast, one round-trip).
    Fallback: scans source table exhaustively, stopping only when data is
    fully consumed (no premature exit on page with no new locations).
    """
    try:
        rows = _fetch_page(
            client,
            lambda c: (
                c.table("locations")
                .select("location_id")
                .eq("is_active", True)
            ),
            "locations table lookup",
        )
        locs = sorted({r["location_id"] for r in rows if r.get("location_id")})
        if locs:
            log.info("Found %d active locations from locations table.", len(locs))
            return locs
    except Exception as exc:
        log.warning("locations table query failed (%s); scanning source table.", exc)

    # Fallback: page through source table until exhausted, collecting unique
    # location_ids.  Does NOT stop early on "no new locations" — that caused
    # early exit when one location dominated the first page.
    seen: set[str] = set()
    offset = 0
    while True:
        _offset = offset  # close over current value in lambda
        rows = _fetch_page(
            client,
            lambda c: (
                c.table(_SOURCE_TABLE)
                .select("location_id")
                .gte("tran_date", window_start_str)
                .lte("tran_date", today_str)
                .not_.is_("location_id", "null")
                .order("location_id")
                .range(_offset, _offset + _LOCATION_SCAN_PAGE - 1)
            ),
            f"location scan offset={_offset}",
        )
        if not rows:
            break
        for r in rows:
            loc = r.get("location_id")
            if loc:
                seen.add(loc)
        if len(rows) < _LOCATION_SCAN_PAGE:
            break
        offset += _LOCATION_SCAN_PAGE

    locs = sorted(seen)
    log.info("Found %d distinct locations in source table.", len(locs))
    return locs


# ---------------------------------------------------------------------------
# Per-location aggregation via RPC
# ---------------------------------------------------------------------------

def _rpc_churn_buckets(
    client: Any,
    location_id: str,
    window_start: date,
    baseline_end: date,
    comparison_start: date,
) -> list[dict]:
    """Call get_churn_buckets() RPC and return one pre-aggregated row per customer.

    All heavy GROUP-BY work runs inside Postgres against the
    idx_sdt_loc_date composite index — no raw row transfer to Python.
    Returns list of dicts with keys:
        customer_id, is_commercial, baseline_sales, baseline_tx, baseline_months,
        comparison_sales, last_purchase_date
    """
    params = {
        "p_location_id":      location_id,
        "p_window_start":     window_start.isoformat(),
        "p_baseline_end":     baseline_end.isoformat(),
        "p_comparison_start": comparison_start.isoformat(),
    }
    return _fetch_page(
        client,
        lambda c: c.rpc("get_churn_buckets", params),
        f"get_churn_buckets({location_id})",
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(
    baseline_monthly: float,
    comparison_sales: float,
    baseline_tx: int,
) -> tuple[str, float]:
    """Return (flag, pct_change).

    Returns CHURNED for any customer with zero comparison-period spend —
    _build_row segments CHURNED into AT_RISK / CHURNED / LOST based on
    days_since_last_purchase.
    """
    if baseline_tx < MIN_BASELINE_PURCHASES or baseline_monthly < MIN_BASELINE_SPEND:
        return "STABLE", 0.0

    if comparison_sales == 0.0:
        return "CHURNED", -100.0

    expected_90d = baseline_monthly * (COMPARISON_DAYS / 30.0)
    if expected_90d > 0:
        pct_change = ((comparison_sales - expected_90d) / expected_90d) * 100.0
    else:
        pct_change = 0.0

    if pct_change <= -(DECLINING_THRESHOLD * 100):
        return "DECLINING", round(pct_change, 2)

    return "STABLE", round(pct_change, 2)


def _build_row(
    rpc_row: dict,
    loc: str,
    effective_today: date,
) -> dict | None:
    """Classify one RPC result row and return an output row, or None to skip.

    rpc_row keys: customer_id, is_commercial, baseline_sales, baseline_tx,
                  baseline_months, comparison_sales, last_purchase_date

    salesman_id is populated via sales_detail_transactions.salesman_id
    (migration 041 + extract update) and will be non-NULL once that column
    is backfilled.  For now it is always None.
    """
    cust             = (rpc_row.get("customer_id") or "").strip()
    baseline_sales   = float(rpc_row.get("baseline_sales") or 0.0)
    baseline_tx      = int(rpc_row.get("baseline_tx") or 0)
    baseline_months  = int(rpc_row.get("baseline_months") or 0)
    comp_sales       = float(rpc_row.get("comparison_sales") or 0.0)
    lpd_raw          = rpc_row.get("last_purchase_date")
    last_purchase    = date.fromisoformat(lpd_raw) if lpd_raw else None
    is_commercial    = bool(rpc_row.get("is_commercial") or False)
    salesman_id: str | None = None   # populated after migration 041 backfill

    if not cust:
        return None

    baseline_monthly = baseline_sales / BASELINE_MONTHS

    # Quality gates: skip unless all three thresholds are met
    if (
        baseline_tx < MIN_BASELINE_PURCHASES
        or baseline_monthly < MIN_BASELINE_SPEND
        or baseline_months < MIN_ACTIVE_MONTHS
    ):
        return None

    flag, pct_change = _classify(baseline_monthly, comp_sales, baseline_tx)

    # Segment zero-spend customers by recency of last purchase
    days_since: int | None = (
        (effective_today - last_purchase).days if last_purchase else None
    )
    risk_segment: str | None = None
    if flag == "CHURNED":
        if days_since is None or days_since > 365:
            flag, risk_segment = "LOST", "LOST"
        elif days_since > 180:
            flag, risk_segment = "CHURNED", "CHURNED"
        else:
            flag, risk_segment = "AT_RISK", "AT_RISK"

    return {
        "customer_id":              cust,
        "location_id":              loc,
        "is_commercial":            is_commercial,
        "salesman_id":              salesman_id,
        "baseline_monthly_spend":   round(baseline_monthly, 4),
        "last_90_days_spend":       round(comp_sales, 4),
        "pct_change":               pct_change,
        "flag":                     flag,
        "risk_segment":             risk_segment,
        "days_since_last_purchase": days_since,
        "last_purchase_date":       last_purchase.isoformat() if last_purchase else None,
        "run_date":                 effective_today.isoformat(),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_churn_detection(dry_run: bool = False) -> int:
    """Compute churn flags and write to customer_churn_flags.

    Processes one location at a time to stay within Supabase's statement
    timeout.  Returns the count of CHURNED + DECLINING customers written.
    """
    setup_logging()
    client = get_client()

    effective_today = _detect_effective_today(client)
    window_start, baseline_end, comparison_start = _compute_date_windows(effective_today)
    window_start_str = window_start.isoformat()
    today_str = effective_today.isoformat()

    log.info("=" * 60)
    log.info("  CHURN DETECTION%s", " [DRY RUN]" if dry_run else "")
    log.info("  Effective today:  %s  (max tran_date in data)", effective_today)
    log.info("  Window start:     %s", window_start)
    log.info("  Baseline end:     %s  (%d months)", baseline_end, BASELINE_MONTHS)
    log.info("  Comparison start: %s  (last %d days)", comparison_start, COMPARISON_DAYS)
    log.info("  Declining threshold: %d%%", int(DECLINING_THRESHOLD * 100))
    log.info("=" * 60)

    location_ids = _fetch_location_ids(client, window_start_str, today_str)
    if not location_ids:
        log.warning("No locations found — nothing to process.")
        return 0

    counts = {"AT_RISK": 0, "CHURNED": 0, "LOST": 0, "DECLINING": 0, "STABLE": 0, "skipped": 0}
    total_written = 0
    write_buffer: list[dict] = []

    def _flush(force: bool = False) -> None:
        nonlocal total_written, client
        if not write_buffer:
            return
        if not force and len(write_buffer) < WRITE_BATCH_SIZE:
            return
        while write_buffer:
            batch = write_buffer[:WRITE_BATCH_SIZE]
            del write_buffer[:WRITE_BATCH_SIZE]
            if not dry_run:
                _upsert_batch(client, batch)
            total_written += len(batch)
            if total_written % LOG_EVERY < WRITE_BATCH_SIZE or force:
                log.info("  wrote %d rows so far", total_written)

    for loc_idx, loc in enumerate(location_ids, 1):
        log.info("[%d/%d] Processing location %s ...", loc_idx, len(location_ids), loc)

        rpc_rows = _rpc_churn_buckets(
            client, loc, window_start, baseline_end, comparison_start
        )
        log.info("  %d customers returned by RPC", len(rpc_rows))

        for rpc_row in rpc_rows:
            row = _build_row(rpc_row, loc, effective_today)
            if row is None:
                counts["skipped"] += 1
                continue
            counts[row["flag"]] += 1
            write_buffer.append(row)

        _flush()

    _flush(force=True)

    log.info(
        "Classified: AT_RISK=%d  CHURNED=%d  LOST=%d  DECLINING=%d  STABLE=%d  skipped=%d",
        counts["AT_RISK"], counts["CHURNED"], counts["LOST"],
        counts["DECLINING"], counts["STABLE"], counts["skipped"],
    )
    if dry_run:
        log.info("[DRY RUN] %d rows computed, no DB writes.", total_written)
    else:
        log.info("Churn detection complete. %d rows written to %s.", total_written, _TARGET_TABLE)

    return counts["AT_RISK"] + counts["CHURNED"] + counts["LOST"] + counts["DECLINING"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Customer churn detection engine")
    p.add_argument("--dry-run", action="store_true", help="Compute but skip DB writes")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    flagged = run_churn_detection(dry_run=args.dry_run)
    sys.exit(0 if flagged >= 0 else 1)
