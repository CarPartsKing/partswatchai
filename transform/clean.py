"""
transform/clean.py — Data quality checks and flagging for sales_transactions.

Runs after every partswatch_pull. Queries the database, executes each
registered check, and writes all detected issues to data_quality_issues.

DESIGN PRINCIPLES
    - Never deletes or modifies source records. Bad data becomes visible,
      not invisible — every issue has a permanent, queryable record.
    - Each check is a completely isolated function. Adding a new check means
      writing one function and adding it to CHECKS. Nothing else changes.
    - Re-running clean.py is safe: the (source_table, source_id, issue_type)
      composite key means the same issue is upserted, never duplicated.

ADDING A NEW CHECK
    1. Write a function:
           def check_my_rule(client: Any) -> list[QualityIssue]:
               ...
    2. Append it to the CHECKS list at the bottom of this file.
    That is the complete change required.

PREREQUISITE
    Run db/migrations/003_data_quality_issues.sql in the Supabase SQL Editor
    before first use.

USAGE
    python -m transform.clean             # run all checks, write to DB
    python -m transform.clean --dry-run   # report issues, do not write to DB
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from statistics import mean, stdev
from typing import Any, Callable

from utils.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Valid location IDs for this 23-store NE Ohio network.
# Update this set when a store opens or closes — no other change is needed.
KNOWN_LOCATIONS: frozenset[str] = frozenset(f"LOC-{i:03d}" for i in range(1, 24))

# Transactions where |qty_sold − sku_mean| > OUTLIER_SIGMA * sku_stdev are flagged.
# 4σ catches ~1-in-15,000 events at normal distribution — high enough to avoid
# noise, low enough to catch genuine entry errors.
OUTLIER_SIGMA: float = 4.0

# SKUs with fewer than this many transactions are skipped by the outlier check.
# Not enough data points for meaningful statistics below this threshold.
OUTLIER_MIN_SAMPLE: int = 3

# Supabase PostgREST default page limit
_PAGE_SIZE: int = 1000


# ---------------------------------------------------------------------------
# QualityIssue — the standard output of every check function
# ---------------------------------------------------------------------------

@dataclass
class QualityIssue:
    """A single data quality problem detected in a source record.

    Attributes:
        source_table:  Table containing the affected row.
        source_id:     Natural key of the row (e.g. transaction_id).
        issue_type:    Stable machine-readable tag used as the upsert key.
                       Never change this string for an existing rule —
                       it is the identity of the issue in the database.
        issue_detail:  Human-readable description of the exact problem.
        field_name:    Column containing the bad value (empty if row-level).
        field_value:   The actual bad value as a string, for display.
        severity:      'error'   → definitively wrong data.
                       'warning' → needs review, may be legitimate.
    """
    source_table: str
    source_id:    str
    issue_type:   str
    issue_detail: str
    field_name:   str  = ""
    field_value:  str  = ""
    severity:     str  = "warning"


# Type alias — every check function has exactly this signature
CheckFn = Callable[[Any], list[QualityIssue]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_all(client: Any, table: str, select: str = "*") -> list[dict]:
    """Return every row from a Supabase table, handling the 1000-row page cap.

    Args:
        client: Active Supabase client.
        table:  Table to query.
        select: PostgREST column selector (e.g. "transaction_id,qty_sold").

    Returns:
        All matching rows as a list of dicts.
    """
    all_rows: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table(table)
            .select(select)
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        page: list[dict] = resp.data or []
        all_rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return all_rows


# ---------------------------------------------------------------------------
# Check 1 — Negative quantities
# ---------------------------------------------------------------------------

def check_negative_quantities(client: Any) -> list[QualityIssue]:
    """Flag rows where qty_sold < 0.

    Negative quantities usually mean a return or credit memo was coded as a
    negative sale rather than a proper return transaction.  They distort
    velocity calculations and ABC classification.
    """
    rows = _fetch_all(client, "sales_transactions", "transaction_id,sku_id,qty_sold")
    issues: list[QualityIssue] = []

    for row in rows:
        qty = row.get("qty_sold")
        if qty is not None and qty < 0:
            issues.append(QualityIssue(
                source_table = "sales_transactions",
                source_id    = row["transaction_id"],
                issue_type   = "negative_qty_sold",
                issue_detail = (
                    f"qty_sold is {qty}. Negative quantities indicate a return or "
                    "credit memo coded as a negative sale. Re-code as a return "
                    "transaction so forecasting models are not distorted."
                ),
                field_name   = "qty_sold",
                field_value  = str(qty),
                severity     = "error",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 2 — Future-dated transactions
# ---------------------------------------------------------------------------

def check_future_dated_transactions(client: Any) -> list[QualityIssue]:
    """Flag rows where transaction_date is after today.

    Future dates are data entry errors.  They cause forecasting models to
    see demand signals for dates that have not yet occurred, inflating
    recent trend estimates.
    """
    today = date.today()
    rows  = _fetch_all(
        client, "sales_transactions", "transaction_id,sku_id,transaction_date"
    )
    issues: list[QualityIssue] = []

    for row in rows:
        raw = row.get("transaction_date")
        if not raw:
            continue
        try:
            tx_date = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue

        if tx_date > today:
            days_ahead = (tx_date - today).days
            issues.append(QualityIssue(
                source_table = "sales_transactions",
                source_id    = row["transaction_id"],
                issue_type   = "future_transaction_date",
                issue_detail = (
                    f"transaction_date {tx_date} is {days_ahead} day(s) in the "
                    "future. This is a data entry error — correct the date in "
                    "the source system."
                ),
                field_name   = "transaction_date",
                field_value  = str(tx_date),
                severity     = "error",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 3 — Duplicate transaction IDs
# ---------------------------------------------------------------------------

def check_duplicate_transaction_ids(client: Any) -> list[QualityIssue]:
    """Flag transaction_id values that appear more than once in the database.

    The upsert pipeline deduplicates by transaction_id, so duplicates in
    the DB are rare but possible if the unique constraint is ever removed.
    This check also serves as a canary for double-posting in the source
    system.
    """
    rows   = _fetch_all(client, "sales_transactions", "transaction_id")
    counts: dict[str, int] = {}
    for row in rows:
        tid = row.get("transaction_id", "")
        counts[tid] = counts.get(tid, 0) + 1

    issues: list[QualityIssue] = []
    for tid, count in counts.items():
        if count > 1:
            issues.append(QualityIssue(
                source_table = "sales_transactions",
                source_id    = tid,
                issue_type   = "duplicate_transaction_id",
                issue_detail = (
                    f"transaction_id '{tid}' appears {count} times. Should be "
                    "unique. Investigate the source system for double-posting."
                ),
                field_name   = "transaction_id",
                field_value  = tid,
                severity     = "error",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 4 — SKUs with no sku_master record
# ---------------------------------------------------------------------------

def check_orphaned_skus(client: Any) -> list[QualityIssue]:
    """Flag sales referencing a sku_id not present in sku_master.

    The extract pipeline auto-registers skeleton SKU records, so orphaned
    SKUs here mean either the record bypassed the normal load path or a
    sku_master row was manually deleted after the transaction was loaded.
    """
    sales_rows  = _fetch_all(client, "sales_transactions", "transaction_id,sku_id")
    master_rows = _fetch_all(client, "sku_master", "sku_id")
    known_skus  = {r["sku_id"] for r in master_rows}

    issues: list[QualityIssue] = []
    for row in sales_rows:
        sku = row.get("sku_id", "")
        if sku and sku not in known_skus:
            issues.append(QualityIssue(
                source_table = "sales_transactions",
                source_id    = row["transaction_id"],
                issue_type   = "orphaned_sku_id",
                issue_detail = (
                    f"sku_id '{sku}' has no record in sku_master. The part may "
                    "have been deleted from the master catalogue or was never "
                    "registered. Re-register via the extract pipeline."
                ),
                field_name   = "sku_id",
                field_value  = sku,
                severity     = "error",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 5 — Zero-price sales on non-zero quantity
# ---------------------------------------------------------------------------

def check_zero_price_nonzero_qty(client: Any) -> list[QualityIssue]:
    """Flag rows where qty_sold > 0 but unit_price is 0 or NULL.

    Likely causes: scan error, unrecorded price override, or an internal
    transfer mis-coded as a sale.  These understate revenue and skew
    margin analysis.
    """
    rows = _fetch_all(
        client, "sales_transactions", "transaction_id,sku_id,qty_sold,unit_price"
    )
    issues: list[QualityIssue] = []

    for row in rows:
        qty   = row.get("qty_sold")
        price = row.get("unit_price")
        if qty is not None and qty > 0 and (price is None or price == 0):
            issues.append(QualityIssue(
                source_table = "sales_transactions",
                source_id    = row["transaction_id"],
                issue_type   = "zero_price_nonzero_qty",
                issue_detail = (
                    f"unit_price is {price!r} but qty_sold is {qty}. "
                    "Possible causes: barcode scan error, price override not "
                    "recorded, or internal transfer coded as a sale."
                ),
                field_name   = "unit_price",
                field_value  = str(price),
                severity     = "warning",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 6 — Location IDs outside the known 23-store network
# ---------------------------------------------------------------------------

def check_invalid_locations(client: Any) -> list[QualityIssue]:
    """Flag transactions referencing a location_id not in KNOWN_LOCATIONS.

    Either a typo in the source system, or a store that opened or closed
    without KNOWN_LOCATIONS being updated.  When adding a new store,
    update KNOWN_LOCATIONS at the top of this file — no other change needed.
    """
    rows = _fetch_all(client, "sales_transactions", "transaction_id,location_id")
    issues: list[QualityIssue] = []

    for row in rows:
        loc = row.get("location_id", "")
        if loc and loc not in KNOWN_LOCATIONS:
            issues.append(QualityIssue(
                source_table = "sales_transactions",
                source_id    = row["transaction_id"],
                issue_type   = "invalid_location_id",
                issue_detail = (
                    f"location_id '{loc}' is not in the 23-store known list. "
                    "Either correct the source record or add this location to "
                    "KNOWN_LOCATIONS in transform/clean.py."
                ),
                field_name   = "location_id",
                field_value  = loc,
                severity     = "error",
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 7 — qty_sold statistical outliers (per-SKU, 4σ threshold)
# ---------------------------------------------------------------------------

def check_qty_sold_outliers(client: Any) -> list[QualityIssue]:
    """Flag transactions where qty_sold is more than OUTLIER_SIGMA std deviations
    from that SKU's historical mean.

    Applied only to SKUs with at least OUTLIER_MIN_SAMPLE transactions — smaller
    samples don't provide enough data for reliable statistics.

    Records are flagged for human review and never deleted.  Legitimate bulk
    orders (fleet purchases, contractor orders) will also trigger this check;
    context matters when resolving.
    """
    rows = _fetch_all(
        client, "sales_transactions", "transaction_id,sku_id,qty_sold"
    )

    # Group (transaction_id, qty_sold) tuples by SKU
    by_sku: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        sku = row.get("sku_id", "")
        qty = row.get("qty_sold")
        tid = row.get("transaction_id", "")
        if sku and qty is not None:
            by_sku.setdefault(sku, []).append((tid, float(qty)))

    issues: list[QualityIssue] = []

    for sku, txns in by_sku.items():
        if len(txns) < OUTLIER_MIN_SAMPLE:
            continue

        qtys     = [q for _, q in txns]
        sku_mean = mean(qtys)

        if len(qtys) < 2:
            continue
        try:
            sku_std = stdev(qtys)
        except Exception:
            continue
        if sku_std == 0:
            continue  # All identical values — no outliers possible

        threshold = OUTLIER_SIGMA * sku_std

        for tid, qty in txns:
            deviation = abs(qty - sku_mean)
            if deviation > threshold:
                sigmas = deviation / sku_std
                issues.append(QualityIssue(
                    source_table = "sales_transactions",
                    source_id    = tid,
                    issue_type   = "qty_sold_outlier",
                    issue_detail = (
                        f"qty_sold {qty:.0f} is {sigmas:.1f}σ from this SKU's mean "
                        f"({sku_mean:.1f}, stdev {sku_std:.1f}, n={len(txns)}). "
                        "Could be a bulk/fleet order or a data entry error. "
                        "Review before resolving."
                    ),
                    field_name   = "qty_sold",
                    field_value  = str(qty),
                    severity     = "warning",
                ))

    return issues


# ---------------------------------------------------------------------------
# Check registry
# New checks: write one function above, add one line here. That is all.
# ---------------------------------------------------------------------------

CHECKS: list[CheckFn] = [
    check_negative_quantities,
    check_future_dated_transactions,
    check_duplicate_transaction_ids,
    check_orphaned_skus,
    check_zero_price_nonzero_qty,
    check_invalid_locations,
    check_qty_sold_outliers,
]


# ---------------------------------------------------------------------------
# Issue writer
# ---------------------------------------------------------------------------

def write_issues(client: Any, issues: list[QualityIssue]) -> int:
    """Upsert detected issues into data_quality_issues.

    The unique key (source_table, source_id, issue_type) ensures repeat runs
    update existing rows rather than accumulating duplicates.
    checked_at is refreshed on every run so you always know when each issue
    was last confirmed present.

    Args:
        client: Active Supabase client.
        issues: Issues to persist.

    Returns:
        Number of rows upserted.

    Raises:
        Exception: Propagates Supabase errors after logging.
    """
    if not issues:
        return 0

    now  = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "checked_at":   now,
            "source_table": iss.source_table,
            "source_id":    iss.source_id,
            "issue_type":   iss.issue_type,
            "issue_detail": iss.issue_detail,
            "field_name":   iss.field_name,
            "field_value":  iss.field_value,
            "severity":     iss.severity,
            "is_resolved":  False,
        }
        for iss in issues
    ]

    try:
        client.table("data_quality_issues").upsert(
            rows,
            on_conflict="source_table,source_id,issue_type",
        ).execute()
        return len(rows)
    except Exception as exc:
        log.error("Failed to write to data_quality_issues: %s", exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_checks(dry_run: bool = False) -> int:
    """Run every registered quality check and persist results to the database.

    Processing:
        1. Execute each function in CHECKS in order.
        2. Collect all QualityIssue objects returned.
        3. Log a per-check summary.
        4. Unless dry_run, upsert all issues to data_quality_issues.

    Args:
        dry_run: When True, logs findings without writing to the database.

    Returns:
        Exit code: 0 on success, 1 if any check raised an unhandled exception.
    """
    from db.connection import get_client
    client = get_client()

    all_issues: list[QualityIssue] = []
    failed_checks: list[str]       = []

    log.info("Running %d quality check(s) against sales_transactions …", len(CHECKS))

    for check_fn in CHECKS:
        name = check_fn.__name__
        try:
            found = check_fn(client)
            all_issues.extend(found)
            if found:
                log.warning("  %-42s %d issue(s)", name, len(found))
            else:
                log.info(   "  %-42s OK", name)
        except Exception as exc:
            log.error("  %-42s FAILED: %s", name, exc, exc_info=True)
            failed_checks.append(name)

    # ── Summary ──────────────────────────────────────────────────────────────
    errors_count   = sum(1 for i in all_issues if i.severity == "error")
    warnings_count = sum(1 for i in all_issues if i.severity == "warning")

    log.info("=" * 60)
    log.info(
        "Checks complete — %d issue(s) found  [%d error(s), %d warning(s)]",
        len(all_issues), errors_count, warnings_count,
    )

    if all_issues:
        by_type: dict[str, int] = {}
        for iss in all_issues:
            by_type[iss.issue_type] = by_type.get(iss.issue_type, 0) + 1
        for issue_type, count in sorted(by_type.items()):
            log.info("    %-38s %d", issue_type, count)

    if failed_checks:
        log.error("Checks that raised exceptions: %s", ", ".join(failed_checks))

    # ── Persist ──────────────────────────────────────────────────────────────
    if dry_run:
        log.info("DRY RUN — nothing written to database.")
        log.info("=" * 60)
        return 0

    if all_issues:
        written = write_issues(client, all_issues)
        log.info("Wrote %d issue record(s) to data_quality_issues.", written)
    else:
        log.info("No issues to write — data is clean.")

    log.info("=" * 60)
    return 1 if failed_checks else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse arguments and run the quality check pipeline."""
    parser = argparse.ArgumentParser(
        description="partswatch-ai: data quality checks for sales_transactions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prerequisite: run db/migrations/003_data_quality_issues.sql\n"
            "in the Supabase SQL Editor before first use.\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks and print results without writing to the database.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("partswatch-ai — clean  dry_run=%s", args.dry_run)
    log.info("=" * 60)

    return run_checks(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
