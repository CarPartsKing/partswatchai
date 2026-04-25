"""engine/opsl_analysis.py — Outside-purchase flag engine for PartsWatch AI.

An outside purchase = a store sourced a part from an outside vendor to fill a
customer order because it wasn't in stock locally.  Every such event represents
lost margin vs. stocking the part.

In AutoCube_DTR_23160, outside purchases are NOT a separate tran code.  They
are SL (sale) lines where [Sales Detail].[Stock Flag] = 'N'.  This engine
filters sales_detail_transactions WHERE stock_flag = 'N' AND tran_code = 'SL'.

METHODOLOGY
-----------
90-day rolling window of tran_code = 'OPSL' rows from sales_detail_transactions.
Aggregates per (prod_line_pn, location_id):

  opsl_count               — number of outside-purchase events
  total_opsl_sales         — sum of sales dollars
  total_opsl_gp            — sum of gross_profit dollars
  avg_gp_pct               — gross_profit / sales (0.0 when sales == 0)
  estimated_margin_recovery — (NORMAL_GP_PCT - avg_gp_pct) × total_opsl_sales
                              = recoverable margin if the part were stocked locally

FLAG THRESHOLDS
---------------
  HIGH   — opsl_count >= HIGH_THRESHOLD   (5+ events)
  MEDIUM — opsl_count >= MEDIUM_THRESHOLD (2–4 events)
  LOW    — opsl_count == 1

REORDER QUEUE CROSS-REFERENCE
------------------------------
Checks reorder_recommendations for an active rec with the same
(sku_id=prod_line_pn, location_id).  Note: prod_line_pn uses a compound
format (e.g. "AC 12345") that may differ from sku_master.sku_id bare
part numbers — in_reorder_queue will be False for unmatched formats.

PERFORMANCE
-----------
Uses a single get_opsl_summary() RPC call (SECURITY DEFINER, 60s timeout)
so Postgres does the GROUP BY server-side against the idx_sdt_opsl_covering
partial index.  Reorder cross-reference is loaded once into memory.

Usage
-----
    python -m engine.opsl_analysis            # live run
    python -m engine.opsl_analysis --dry-run  # compute, log, no DB writes
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
LOOKBACK_DAYS: int = 90

NORMAL_GP_PCT: float = 0.35        # assumed GP% if part were stocked locally

HIGH_THRESHOLD: int = 5            # opsl_count >= this → HIGH
MEDIUM_THRESHOLD: int = 2          # opsl_count >= this → MEDIUM (else LOW)

WRITE_BATCH_SIZE: int = 1_000

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

_SOURCE_TABLE  = "sales_detail_transactions"
_REORDER_TABLE = "reorder_recommendations"
_TARGET_TABLE  = "opsl_flags"

_RPC_NAME = "get_opsl_summary"


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


def _fetch_with_retry(client: Any, query_fn, label: str) -> list[dict]:
    """Execute query_fn(client).execute().data with exponential-backoff retry."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = query_fn(client).execute().data
            return result if result is not None else []
        except Exception as exc:
            if attempt < _MAX_RETRIES and _is_retryable(exc):
                delay = _RETRY_DELAY * attempt
                log.warning(
                    "%s attempt %d/%d failed (%s), retrying in %.0fs",
                    label, attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                client = _get_fresh_client()
            else:
                raise
    return []  # unreachable


def _upsert_batch(client: Any, rows: list[dict]) -> None:
    """Upsert a batch into opsl_flags with retry."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client.table(_TARGET_TABLE).upsert(
                rows,
                on_conflict="prod_line_pn,location_id",
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
# Data window
# ---------------------------------------------------------------------------

def _detect_effective_today(client: Any) -> date:
    """Return max tran_date from outside-purchase rows (stock_flag='N').

    Anchors the lookback window to actual data rather than calendar today
    so the analysis stays valid when the nightly extract lags by a day.
    Falls back to global max tran_date if no outside-purchase rows exist yet
    (e.g. before the first re-extract with stock_flag populated).
    """
    for filters in [
        {"tran_code": "SL", "stock_flag": "N"},   # preferred: outside purchases only
        {},                                         # fallback: any row
    ]:
        try:
            q = client.table(_SOURCE_TABLE).select("tran_date").order("tran_date", desc=True).limit(1)
            for col, val in filters.items():
                q = q.eq(col, val)
            resp = q.execute()
            if resp.data:
                effective = date.fromisoformat(resp.data[0]["tran_date"])
                label = "outside-purchase" if filters else "global"
                log.info("Effective today (max %s tran_date): %s", label, effective)
                return effective
        except Exception as exc:
            log.warning("Could not detect max tran_date (%s); trying fallback.", exc)
    return date.today()


# ---------------------------------------------------------------------------
# RPC call
# ---------------------------------------------------------------------------

def _fetch_opsl_rows(client: Any, start_date: date) -> list[dict]:
    """Call get_opsl_summary RPC; returns one pre-aggregated row per
    (prod_line_pn, location_id).  All GROUP BY runs inside Postgres."""
    params = {"p_start_date": start_date.isoformat()}
    rows = _fetch_with_retry(
        client,
        lambda c: c.rpc(_RPC_NAME, params),
        f"{_RPC_NAME}(start={start_date})",
    )
    log.info("RPC returned %d (prod_line_pn, location_id) pairs.", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Reorder queue cross-reference
# ---------------------------------------------------------------------------

def _load_reorder_set(client: Any) -> set[tuple[str, str]]:
    """Return a set of (sku_id, location_id) for the most recent reorder run.

    Note: reorder_recommendations.sku_id uses bare part numbers; prod_line_pn
    uses a compound format.  Exact matches will only occur where formats align.
    """
    try:
        # Find the latest recommendation_date in one cheap query
        latest_resp = (
            client.table(_REORDER_TABLE)
            .select("recommendation_date")
            .order("recommendation_date", desc=True)
            .limit(1)
            .execute()
        )
        if not latest_resp.data:
            log.info("No reorder recommendations found; in_reorder_queue will be False.")
            return set()
        latest_date = latest_resp.data[0]["recommendation_date"]

        # Page through all recs for that date
        seen: set[tuple[str, str]] = set()
        offset = 0
        page = 1_000
        while True:
            rows = (
                client.table(_REORDER_TABLE)
                .select("sku_id,location_id")
                .eq("recommendation_date", latest_date)
                .range(offset, offset + page - 1)
                .execute()
                .data or []
            )
            for r in rows:
                sku = (r.get("sku_id") or "").strip()
                loc = (r.get("location_id") or "").strip()
                if sku and loc:
                    seen.add((sku, loc))
            if len(rows) < page:
                break
            offset += page

        log.info(
            "Loaded %d (sku_id, location_id) pairs from reorder_recommendations (%s).",
            len(seen), latest_date,
        )
        return seen

    except Exception as exc:
        log.warning("Could not load reorder set (%s); in_reorder_queue will be False.", exc)
        return set()


# ---------------------------------------------------------------------------
# Classification + row building
# ---------------------------------------------------------------------------

def _flag(opsl_count: int) -> str:
    if opsl_count >= HIGH_THRESHOLD:
        return "HIGH"
    if opsl_count >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _build_row(
    rpc_row: dict,
    reorder_set: set[tuple[str, str]],
    run_date: date,
) -> dict | None:
    """Classify one RPC result row and return an opsl_flags output row.

    Returns None if the row lacks required fields.
    """
    pn  = (rpc_row.get("prod_line_pn") or "").strip()
    loc = (rpc_row.get("location_id")  or "").strip()
    if not pn or not loc:
        return None

    opsl_count  = int(rpc_row.get("opsl_count")       or 0)
    total_sales = float(rpc_row.get("total_opsl_sales") or 0.0)
    total_gp    = float(rpc_row.get("total_opsl_gp")    or 0.0)

    avg_gp_pct = round(total_gp / total_sales, 4) if total_sales > 0 else 0.0

    # Recoverable margin: how much more GP we'd capture if stocked locally
    margin_recovery = round(
        max(0.0, NORMAL_GP_PCT - avg_gp_pct) * total_sales, 4
    )

    return {
        "prod_line_pn":              pn,
        "location_id":               loc,
        "opsl_count":                opsl_count,
        "total_opsl_sales":          round(total_sales, 4),
        "total_opsl_gp":             round(total_gp, 4),
        "avg_gp_pct":                avg_gp_pct,
        "estimated_margin_recovery": margin_recovery,
        "flag":                      _flag(opsl_count),
        "last_opsl_date":            rpc_row.get("last_opsl_date"),
        "in_reorder_queue":          (pn, loc) in reorder_set,
        "run_date":                  run_date.isoformat(),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_opsl_analysis(dry_run: bool = False) -> int:
    """Compute OPSL flags and write to opsl_flags.

    Returns the total number of rows written (or that would have been
    written in dry-run mode).
    """
    setup_logging()
    client = get_client()

    effective_today = _detect_effective_today(client)
    start_date = effective_today - timedelta(days=LOOKBACK_DAYS)

    log.info("=" * 60)
    log.info("  OPSL ANALYSIS%s", " [DRY RUN]" if dry_run else "")
    log.info("  Effective today : %s", effective_today)
    log.info("  Lookback window : %s → %s (%d days)", start_date, effective_today, LOOKBACK_DAYS)
    log.info("  Normal GP%%      : %.0f%%", NORMAL_GP_PCT * 100)
    log.info("  Flag thresholds : HIGH >= %d, MEDIUM >= %d, else LOW",
             HIGH_THRESHOLD, MEDIUM_THRESHOLD)
    log.info("=" * 60)

    rpc_rows = _fetch_opsl_rows(client, start_date)
    if not rpc_rows:
        log.warning("No OPSL rows returned by RPC — nothing to write.")
        return 0

    reorder_set = _load_reorder_set(client)

    flag_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    in_queue_count = 0
    rows_to_write: list[dict] = []

    for rpc_row in rpc_rows:
        row = _build_row(rpc_row, reorder_set, effective_today)
        if row is None:
            continue
        flag_counts[row["flag"]] += 1
        if row["in_reorder_queue"]:
            in_queue_count += 1
        rows_to_write.append(row)

    log.info(
        "Classified: HIGH=%d  MEDIUM=%d  LOW=%d  in_reorder_queue=%d",
        flag_counts["HIGH"], flag_counts["MEDIUM"], flag_counts["LOW"], in_queue_count,
    )

    if not rows_to_write:
        log.info("No rows to write.")
        return 0

    total_written = 0
    for i in range(0, len(rows_to_write), WRITE_BATCH_SIZE):
        batch = rows_to_write[i : i + WRITE_BATCH_SIZE]
        if dry_run:
            log.info("[DRY RUN] Would upsert batch %d–%d (%d rows)",
                     i + 1, i + len(batch), len(batch))
        else:
            _upsert_batch(client, batch)
            log.info("Upserted batch %d–%d (%d rows)", i + 1, i + len(batch), len(batch))
        total_written += len(batch)

    if dry_run:
        log.info("[DRY RUN] %d rows computed, no DB writes.", total_written)
    else:
        log.info("OPSL analysis complete. %d rows written to %s.", total_written, _TARGET_TABLE)

    # Log top 5 by margin recovery for visibility
    top5 = sorted(rows_to_write, key=lambda r: r["estimated_margin_recovery"], reverse=True)[:5]
    if top5:
        log.info("Top 5 by estimated margin recovery:")
        for r in top5:
            log.info(
                "  %-30s  %-10s  flag=%-6s  count=%d  recovery=$%.2f",
                r["prod_line_pn"], r["location_id"], r["flag"],
                r["opsl_count"], r["estimated_margin_recovery"],
            )

    return total_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPSL outside-purchase flag engine")
    p.add_argument("--dry-run", action="store_true", help="Compute but skip DB writes")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    written = run_opsl_analysis(dry_run=args.dry_run)
    sys.exit(0 if written >= 0 else 1)
