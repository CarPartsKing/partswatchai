"""engine/understocking.py — Chronic Understocking Report.

Answers the owner's question:
    "What should each store be stocking more of instead of waiting
     for transfers?"

The actual computation lives in the Postgres function
    fn_run_understocking_report(...)
which is created by db/migrations/029_understocking_report.sql.  Doing
the aggregation server-side collapses what would otherwise be thousands
of PostgREST round-trips into a single .rpc() call that runs in
seconds, even across millions of inventory_snapshots and
sales_transactions rows.

This module is a thin orchestrator:
  1. Call the SQL function with the pipeline's policy parameters.
  2. Parse the single-row summary it returns.
  3. Log a per-location breakdown (queried back from understocking_report).
  4. Return non-zero exit code if no rows were built or persistence
     fell short of what the function claims to have built.

Pipeline position: WEEKLY_STAGES (after dead_stock / churn).

CLI:
    python -m engine.understocking [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Policy parameters — passed straight into the Postgres function.  These
# duplicate the function's own DEFAULTs so the engine, the dashboard
# section, and the SQL all agree on the numbers being applied this run.
# ---------------------------------------------------------------------------

LOOKBACK_DAYS: int = 90
"""Target lookback window for snapshot + sales aggregation."""

STOCKOUT_PCT_THRESHOLD: float = 0.30
"""SKU is chronically understocked if below reorder_point this fraction of days."""

LEAD_TIME_DAYS: float = 7.0
"""Default lead time.  Mirrors engine/reorder.py DEFAULT_LEAD_TIME_DAYS."""

BUFFER_DAYS: float = 14.0
"""Safety buffer added on top of lead time when computing suggested_min_qty."""

TOP_N_PER_LOCATION: int = 20
"""How many chronic SKUs to persist per location."""

TRANSFER_REC_LOOKBACK_DAYS: int = 30
"""Window for counting transfer-IN recommendations as confirmation signal."""

# Same set as dashboard/server.py.EXCLUDED_DISPLAY_LOCATIONS plus
# LOC-025 MAIN DC (warehouse, not a retail branch the buyer adjusts
# minimum stocks on).
EXCLUDED_LOCATIONS: list[str] = [
    "LOC-014", "LOC-019", "LOC-022", "LOC-023", "LOC-030", "LOC-031",  # retired
    "LOC-021",  # INTERNET (virtual)
    "LOC-025",  # MAIN DC (not retail)
]

LOCATION_NAMES: dict[str, str] = {
    "LOC-001": "BROOKPARK",        "LOC-002": "NOLMSTEAD",
    "LOC-003": "S.EUCLID",         "LOC-004": "CLARK AUTO",
    "LOC-005": "PARMA",            "LOC-006": "MEDINA",
    "LOC-007": "BOARDMAN",         "LOC-008": "ELYRIA",
    "LOC-009": "AKRON-GRANT",      "LOC-010": "MIDWAY CROSSINGS",
    "LOC-011": "ERIE ST",          "LOC-012": "MAYFIELD",
    "LOC-013": "CANTON",           "LOC-015": "JUNIATA",
    "LOC-016": "ARCHWOOD",         "LOC-017": "EUCLID",
    "LOC-018": "WARREN",           "LOC-020": "ROOTSTOWN",
    "LOC-021": "INTERNET",         "LOC-024": "MENTOR",
    "LOC-025": "MAIN DC",          "LOC-026": "COPLEY",
    "LOC-027": "CHARDON",          "LOC-028": "STRONGSVILLE",
    "LOC-029": "MIDDLEBURG",       "LOC-032": "PERRY",
    "LOC-033": "CRYSTAL",
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_understocking(dry_run: bool = False) -> int:
    """Compute and persist the Chronic Understocking Report.

    Returns 0 on success, 1 on failure (no rows built, persistence gap,
    or RPC error).  The Postgres function does the heavy lifting; this
    function just dispatches it and logs the summary.
    """
    t0 = time.monotonic()
    client = get_client()
    today = date.today()

    log.info("=" * 60)
    log.info("CHRONIC UNDERSTOCKING REPORT  (target lookback %d days, "
             "threshold %.0f%%, top %d per location)%s",
             LOOKBACK_DAYS, STOCKOUT_PCT_THRESHOLD * 100,
             TOP_N_PER_LOCATION, "  [DRY RUN]" if dry_run else "")
    log.info("=" * 60)

    # ---------- Call the Postgres aggregation function ----------
    rpc_t0 = time.monotonic()
    try:
        res = client.rpc(
            "fn_run_understocking_report",
            {
                "p_excluded_locs":    EXCLUDED_LOCATIONS,
                "p_lookback_days":    LOOKBACK_DAYS,
                "p_transfer_days":    TRANSFER_REC_LOOKBACK_DAYS,
                "p_stockout_pct":     STOCKOUT_PCT_THRESHOLD,
                "p_lead_buffer_days": LEAD_TIME_DAYS + BUFFER_DAYS,
                "p_top_n":            TOP_N_PER_LOCATION,
                "p_dry_run":          dry_run,
            },
        ).execute()
    except Exception as exc:
        msg = str(exc)
        if "fn_run_understocking_report" in msg or "schema cache" in msg \
                or "does not exist" in msg:
            log.error(
                "RPC fn_run_understocking_report not found.  "
                "Apply db/migrations/029_understocking_report.sql in Supabase."
            )
        else:
            log.exception("Understocking RPC failed.")
        return 1

    summary_rows: list[dict[str, Any]] = res.data or []
    if not summary_rows:
        log.error("RPC returned no summary row — unexpected.")
        return 1

    s = summary_rows[0]
    rpc_secs       = time.monotonic() - rpc_t0
    rows_built     = int(s.get("rows_built") or 0)
    rows_persisted = int(s.get("rows_persisted") or 0)
    locs_chronic   = int(s.get("locations_with_chronic") or 0)
    total_value    = float(s.get("total_value_at_risk") or 0)
    window_days    = int(s.get("actual_window_days") or 0)

    log.info("RPC complete in %.1fs", rpc_secs)
    log.info("  Actual snapshot window observed: %d days (target was %d)",
             window_days, LOOKBACK_DAYS)
    log.info("  Locations with chronic understocking: %d", locs_chronic)
    log.info("  Rows built: %d  persisted: %d%s",
             rows_built, rows_persisted,
             "  (DRY RUN — no writes)" if dry_run else "")
    log.info("  Total demand-at-risk: $%,.0f", total_value)

    # ---------- Per-location detail + post-write integrity check ----------
    # We always re-query the persisted table for today (even on rerun
    # when rows_built==0) so we can prove (a) old stale rows for today
    # were cleared by the function's DELETE step, and (b) the persisted
    # set exactly matches what the function claims it built.
    persisted_rows: list[dict] = []
    if not dry_run:
        try:
            res = (
                client.table("understocking_report")
                .select("location_id,inventory_value_at_risk,sku_id,priority_score,"
                        "stockout_days_pct,current_min_qty,suggested_min_qty")
                .eq("report_date", today.isoformat())
                .execute()
            )
            persisted_rows = res.data or []
        except Exception:
            log.exception("Failed post-write integrity read.")
            return 1

        if len(persisted_rows) != rows_built:
            log.error(
                "Post-write integrity check FAILED: function reports built=%d, "
                "table now has %d rows for %s.  Possible stale-row leak from a "
                "previous same-day run.",
                rows_built, len(persisted_rows), today,
            )
            return 1
        log.info("Post-write integrity OK: %d rows in table for %s "
                 "(matches rows_built).", len(persisted_rows), today)

    if rows_built > 0 and not dry_run:

        per_loc: dict[str, list[dict]] = {}
        for r in persisted_rows:
            per_loc.setdefault(r["location_id"], []).append(r)
        worst = sorted(
            per_loc.items(),
            key=lambda kv: sum(float(r["inventory_value_at_risk"] or 0)
                                for r in kv[1]),
            reverse=True,
        )
        log.info("-" * 60)
        log.info("Per-location breakdown:")
        for loc, items in worst:
            value = sum(float(r["inventory_value_at_risk"] or 0) for r in items)
            top = max(items, key=lambda r: float(r["priority_score"] or 0))
            name = LOCATION_NAMES.get(loc, "?")
            log.info(
                "  %-22s %s — %2d SKUs, $%10.0f at risk  "
                "top: %s (%.0f%% short, min %.0f→%.0f)",
                name, loc, len(items), value,
                top["sku_id"],
                float(top["stockout_days_pct"] or 0) * 100,
                float(top["current_min_qty"] or 0),
                float(top["suggested_min_qty"] or 0),
            )

    log.info("=" * 60)
    log.info("Understocking report complete  (%.1fs)", time.monotonic() - t0)
    log.info("=" * 60)

    # ---------- Health check ----------
    if not dry_run and rows_built > 0 and rows_persisted < rows_built:
        log.error(
            "Persistence integrity check FAILED: built=%d persisted=%d.",
            rows_built, rows_persisted,
        )
        return 1
    if rows_built == 0:
        # Not necessarily an error — could just mean no chronic shortages.
        # But surface as a warning so the nightly summary highlights it.
        log.warning("No chronic-understocking SKUs found this run.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chronic Understocking Report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and report without writing to understocking_report.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_understocking(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
