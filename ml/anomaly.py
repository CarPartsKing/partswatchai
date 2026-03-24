"""
ml/anomaly.py — Isolation Forest anomaly detection for sales_transactions.

Runs before any forecasting model touches the training data.  Identifies
days where a SKU's sales pattern is statistically anomalous — most likely
a bulk sale, data-entry error, or barcode-scan mistake that would corrupt
a demand forecast if included in the training set.

ALGORITHM
    1.  Fetch all sales_transactions from Supabase.
    2.  Aggregate to daily resolution per SKU:
            daily_qty    = SUM(qty_sold)         per (sku_id, date)
            avg_price    = MEAN(unit_price)       per (sku_id, date)
            tx_count     = COUNT(transactions)    per (sku_id, date)
    3.  For each SKU with >= MIN_TRANSACTIONS daily observations, fit an
        IsolationForest(contamination=CONTAMINATION) on those three features.
    4.  Predictions of -1 mark the day as anomalous.  The anomaly score is
        also logged (lower / more-negative = more anomalous).
    5.  For every anomalous day, ALL transactions on that (sku_id, date)
        are updated: is_anomaly = TRUE.
    6.  One data_quality_issues record is written per flagged transaction.

SKIPPING
    SKUs with fewer than MIN_TRANSACTIONS daily data points are skipped.
    With insufficient history the model cannot distinguish true outliers
    from normal demand variance.

IDEMPOTENCY
    Each run first resets is_anomaly = FALSE for every SKU that it
    processes, then re-flags current anomalies.  SKUs that are below the
    minimum threshold are never touched, so their existing flags (always
    FALSE) remain unchanged.

PREREQUISITE
    Run db/migrations/005_sales_transactions_is_anomaly.sql in the
    Supabase SQL Editor before first use.

USAGE
    python -m ml.anomaly            # full run
    python -m ml.anomaly --dry-run  # compute detections without writing
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from utils.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum number of *daily* data points required to fit a model for a SKU.
# Below this threshold the model cannot reliably separate signal from noise.
MIN_TRANSACTIONS: int = 30

# Fraction of observations the model treats as anomalies.
# 0.05 = top 5 % most anomalous days per SKU are flagged.
CONTAMINATION: float = 0.05

# Fixed random seed so model decisions are reproducible across runs.
RANDOM_STATE: int = 42

# Number of trees in the Isolation Forest ensemble.
N_ESTIMATORS: int = 100

# Supabase pagination page size
_PAGE_SIZE: int = 1000

# Write batch size for upserts / updates
BATCH_SIZE: int = 500

# Severity assigned to anomaly records in data_quality_issues
ISSUE_SEVERITY: str = "warning"

# issue_type label — distinct from clean.py's statistical outlier labels
ISSUE_TYPE: str = "isolation_forest_anomaly"


# ---------------------------------------------------------------------------
# Shared fetch helper
# ---------------------------------------------------------------------------

def _fetch_all(client: Any, table: str, select: str = "*") -> list[dict]:
    """Return every row from a Supabase table, paging through the 1000-row cap.

    Args:
        client: Active Supabase client.
        table:  Table name to query.
        select: PostgREST column selector string.

    Returns:
        All matching rows as a list of dicts.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page = (
            client.table(table)
            .select(select)
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def _aggregate_daily(tx_rows: list[dict]) -> dict[str, list[dict]]:
    """Aggregate raw transaction rows to daily (sku_id, date) observations.

    Returns a mapping: sku_id → list of daily dicts sorted by date ascending.

    Each daily dict contains:
        date        (str)   YYYY-MM-DD
        daily_qty   (float) sum of qty_sold
        avg_price   (float) mean of unit_price
        tx_count    (int)   number of transactions on that day
        tx_ids      (list)  transaction_id strings for all rows on that day
    """
    # Two-level accumulator: sku → date → running totals
    accum: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "qty_sum": 0.0,
        "price_sum": 0.0,
        "tx_count": 0,
        "tx_ids": [],
    }))

    for r in tx_rows:
        sku     = r.get("sku_id", "")
        raw_dt  = str(r.get("transaction_date", ""))[:10]
        qty     = float(r.get("qty_sold") or 0)
        price   = float(r.get("unit_price") or 0)
        tx_id   = r.get("transaction_id", "")

        if not sku or not raw_dt or not tx_id:
            continue

        bucket = accum[sku][raw_dt]
        bucket["qty_sum"]   += qty
        bucket["price_sum"] += price
        bucket["tx_count"]  += 1
        bucket["tx_ids"].append(tx_id)

    # Flatten into per-SKU sorted lists
    result: dict[str, list[dict]] = {}
    for sku, by_date in accum.items():
        daily = []
        for date_str, b in sorted(by_date.items()):
            daily.append({
                "date":       date_str,
                "daily_qty":  b["qty_sum"],
                "avg_price":  b["price_sum"] / b["tx_count"],
                "tx_count":   b["tx_count"],
                "tx_ids":     b["tx_ids"],
            })
        result[sku] = daily

    return result


# ---------------------------------------------------------------------------
# Isolation Forest per SKU
# ---------------------------------------------------------------------------

def _fit_and_detect(sku_id: str, daily: list[dict]) -> list[dict]:
    """Fit an Isolation Forest for one SKU and return its anomalous days.

    Skips the SKU if it has fewer than MIN_TRANSACTIONS daily observations.

    Args:
        sku_id: SKU identifier (used only for logging).
        daily:  Sorted list of daily aggregation dicts from _aggregate_daily.

    Returns:
        List of daily dicts (subset of ``daily``) where the model predicted
        an anomaly.  Empty list if the SKU was skipped or no anomalies found.
        Returns None if the SKU was skipped (enables caller to count skips).
    """
    n = len(daily)
    if n < MIN_TRANSACTIONS:
        return None  # type: ignore[return-value]   # caller checks for None

    # Feature matrix: [daily_qty, avg_price, tx_count]
    X = np.array(
        [[d["daily_qty"], d["avg_price"], d["tx_count"]] for d in daily],
        dtype=float,
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    predictions = model.fit_predict(X)   # -1 = anomaly, +1 = normal
    scores      = model.decision_function(X)  # lower = more anomalous

    flagged = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        if pred == -1:
            day = daily[i]
            flagged.append({**day, "anomaly_score": round(float(score), 4)})
            log.info(
                "    ANOMALY  %-12s  %s  qty=%.0f  price=%.2f  "
                "tx=%d  score=%.4f",
                sku_id,
                day["date"],
                day["daily_qty"],
                day["avg_price"],
                day["tx_count"],
                score,
            )

    return flagged


# ---------------------------------------------------------------------------
# Database write helpers
# ---------------------------------------------------------------------------

def _reset_is_anomaly_for_sku(client: Any, sku_id: str, dry_run: bool) -> None:
    """Set is_anomaly=FALSE for all transactions belonging to this SKU.

    Called before re-flagging so that days that were previously anomalous
    but are no longer (after a model re-fit with more data) get cleared.
    """
    if dry_run:
        return
    client.table("sales_transactions").update(
        {"is_anomaly": False}
    ).eq("sku_id", sku_id).execute()


def _write_anomaly_flags(
    client: Any,
    tx_ids: list[str],
    dry_run: bool,
) -> None:
    """Mark a list of transaction_ids as is_anomaly=TRUE."""
    if dry_run:
        return
    for tx_id in tx_ids:
        client.table("sales_transactions").update(
            {"is_anomaly": True}
        ).eq("transaction_id", tx_id).execute()


def _write_quality_issues(
    client: Any,
    flagged_days: list[dict],   # list of enriched daily dicts with tx metadata
    daily_lookup: dict[str, dict],  # tx_id → original row data
    dry_run: bool,
) -> int:
    """Write one data_quality_issues row per flagged transaction.

    Args:
        client:       Supabase client.
        flagged_days: Output of _fit_and_detect (anomalous daily dicts).
        daily_lookup: Mapping of transaction_id → raw transaction dict for
                      detail fields (qty_sold, unit_price, sku_id).
        dry_run:      When True, skips all writes.

    Returns:
        Number of data_quality_issues rows written.
    """
    if dry_run:
        return sum(len(d["tx_ids"]) for d in flagged_days)

    records: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for day in flagged_days:
        for tx_id in day["tx_ids"]:
            raw = daily_lookup.get(tx_id, {})
            sku = raw.get("sku_id", "")
            qty = raw.get("qty_sold", "")
            records.append({
                "source_table": "sales_transactions",
                "source_id":    tx_id,
                "issue_type":   ISSUE_TYPE,
                "issue_detail": (
                    f"Isolation Forest flagged {day['date']} as anomalous for SKU {sku}. "
                    f"daily_qty={day['daily_qty']:.0f}  "
                    f"avg_price={day['avg_price']:.2f}  "
                    f"tx_count={day['tx_count']}  "
                    f"anomaly_score={day['anomaly_score']:.4f}"
                ),
                "field_name":  "qty_sold",
                "field_value": str(qty),
                "severity":    ISSUE_SEVERITY,
                "detected_at": now,
            })

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        client.table("data_quality_issues").upsert(
            batch,
            on_conflict="source_table,source_id,issue_type",
        ).execute()

    return len(records)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_anomaly_detection(dry_run: bool = False) -> int:
    """Run the full anomaly-detection pipeline.

    Args:
        dry_run: When True, detects anomalies and logs results but does not
                 write any changes to the database.

    Returns:
        Exit code: 0 on success, 1 if an unrecoverable error occurred.
    """
    from db.connection import get_client
    client = get_client()

    t_start = time.perf_counter()

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")

    # ── 1. Fetch all transactions ──────────────────────────────────────────
    log.info("Fetching sales_transactions …")
    tx_rows = _fetch_all(
        client,
        "sales_transactions",
        "transaction_id,sku_id,transaction_date,qty_sold,unit_price",
    )
    log.info("  Fetched %d transaction(s).", len(tx_rows))

    if not tx_rows:
        log.warning("No transactions found — nothing to do.")
        return 0

    # Build a tx_id → raw row lookup for the quality-issues write
    tx_lookup: dict[str, dict] = {r["transaction_id"]: r for r in tx_rows}

    # ── 2. Aggregate to daily level ────────────────────────────────────────
    daily_by_sku = _aggregate_daily(tx_rows)
    log.info(
        "  Aggregated to daily level: %d SKU(s), %d total daily observations.",
        len(daily_by_sku),
        sum(len(v) for v in daily_by_sku.values()),
    )

    # ── 3. Fit Isolation Forest per SKU ───────────────────────────────────
    total_skus_eligible  = 0
    total_skus_skipped   = 0
    total_days_flagged   = 0
    total_tx_flagged     = 0
    total_issues_written = 0

    sku_list = sorted(daily_by_sku.keys())
    log.info("Processing %d SKU(s) …", len(sku_list))
    log.info("-" * 60)

    for sku_id in sku_list:
        daily = daily_by_sku[sku_id]

        flagged = _fit_and_detect(sku_id, daily)

        if flagged is None:
            # Not enough history
            log.info(
                "  SKIP  %-12s  %d daily obs (need >= %d)",
                sku_id, len(daily), MIN_TRANSACTIONS,
            )
            total_skus_skipped += 1
            continue

        total_skus_eligible += 1
        n_flagged_days = len(flagged)
        n_flagged_tx   = sum(len(d["tx_ids"]) for d in flagged)

        log.info(
            "  OK    %-12s  %d daily obs → %d anomalous day(s), "
            "%d transaction(s) flagged",
            sku_id, len(daily), n_flagged_days, n_flagged_tx,
        )

        if not dry_run:
            # Reset first so previously-flagged clean days are cleared
            _reset_is_anomaly_for_sku(client, sku_id, dry_run)

        if flagged:
            # Collect all tx_ids for this SKU's anomalous days
            anomalous_tx_ids = [tx for d in flagged for tx in d["tx_ids"]]
            _write_anomaly_flags(client, anomalous_tx_ids, dry_run)

            issues = _write_quality_issues(client, flagged, tx_lookup, dry_run)

            total_days_flagged   += n_flagged_days
            total_tx_flagged     += n_flagged_tx
            total_issues_written += issues

    # ── 4. Summary ────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info("Anomaly detection complete  (%.2fs)", elapsed)
    log.info("  SKUs eligible (>= %d obs):  %d", MIN_TRANSACTIONS, total_skus_eligible)
    log.info("  SKUs skipped  (< %d obs):   %d", MIN_TRANSACTIONS, total_skus_skipped)
    log.info("  Anomalous days flagged:      %d", total_days_flagged)
    log.info("  Transactions flagged:        %d", total_tx_flagged)
    log.info("  data_quality_issues written: %d", total_issues_written)
    if dry_run:
        log.info("  (DRY RUN — no writes were made)")
    log.info("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI arguments and run anomaly detection."""
    parser = argparse.ArgumentParser(
        description="partswatch-ai: Isolation Forest anomaly detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prerequisite: run db/migrations/005_sales_transactions_is_anomaly.sql\n"
            "in the Supabase SQL Editor before first use.\n\n"
            "The --dry-run flag computes detections and logs results without\n"
            "writing any changes to the database."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Detect anomalies without writing to the database.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("partswatch-ai — ml.anomaly")
    log.info(
        "  contamination=%.2f  min_obs=%d  n_estimators=%d  seed=%d",
        CONTAMINATION, MIN_TRANSACTIONS, N_ESTIMATORS, RANDOM_STATE,
    )
    log.info("=" * 60)

    return run_anomaly_detection(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
