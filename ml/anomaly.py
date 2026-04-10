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

_CLIENT_REFRESH_EVERY: int = 500
_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 5.0


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

def _fetch_chunked_by_date(
    client: Any,
    table: str,
    select: str,
    date_col: str,
    since: str,
    until: str | None = None,
    chunk_days: int = 7,
    extra_eq: dict[str, Any] | None = None,
) -> list[dict]:
    """Fetch rows in weekly date chunks to avoid Supabase statement timeouts.

    Iterates over *chunk_days*-wide date windows from *since* to *until*,
    paginating within each window.  This keeps each individual query small
    enough to complete within Supabase's statement timeout while still
    fetching all rows.

    Args:
        client:     Active Supabase client.
        table:      Table name.
        select:     PostgREST column selector.
        date_col:   Date column to chunk on.
        since:      Start date (inclusive) ISO string.
        until:      End date (inclusive) ISO string; defaults to today.
        chunk_days: Width of each date window in days.
        extra_eq:   Optional {column: value} equality filters.

    Returns:
        All matching rows as a list of dicts.
    """
    from datetime import date as _date, timedelta as _td
    start = _date.fromisoformat(since)
    end = _date.fromisoformat(until) if until else _date.today()
    all_rows: list[dict] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + _td(days=chunk_days - 1), end)
        offset = 0
        while True:
            q = client.table(table).select(select)
            q = q.gte(date_col, chunk_start.isoformat())
            q = q.lte(date_col, chunk_end.isoformat())
            if extra_eq:
                for col, val in extra_eq.items():
                    q = q.eq(col, val)
            page = q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
            all_rows.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        chunk_start = chunk_end + _td(days=1)
        if all_rows and len(all_rows) % 100_000 < (chunk_days * 7000):
            log.info("    … streamed %d rows so far from %s", len(all_rows), table)
    log.info("    … streamed %d total rows from %s", len(all_rows), table)
    return all_rows


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

def _get_fresh_client() -> Any:
    from db.connection import get_client
    return get_client()


def _db_call_with_retry(fn, client_holder: list, *args, **kwargs):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(client_holder[0], *args, **kwargs)
        except Exception as exc:
            err_name = type(exc).__name__
            err_str = str(exc)
            is_conn_err = any(k in err_name + err_str for k in (
                "ConnectionTerminated", "ConnectionError", "RemoteProtocolError",
                "ReadTimeout", "ConnectTimeout", "PoolTimeout",
            ))
            if is_conn_err and attempt < _MAX_RETRIES:
                log.warning(
                    "  DB connection error (attempt %d/%d): %s — reconnecting in %.0fs …",
                    attempt, _MAX_RETRIES, err_name, _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _get_fresh_client()
                continue
            raise


def _reset_is_anomaly_for_sku(client: Any, sku_id: str, dry_run: bool) -> None:
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
    if dry_run:
        return
    for tx_id in tx_ids:
        client.table("sales_transactions").update(
            {"is_anomaly": True}
        ).eq("transaction_id", tx_id).execute()


def _write_quality_issues(
    client: Any,
    flagged_days: list[dict],
    daily_lookup: dict[str, dict],
    dry_run: bool,
) -> int:
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
                "checked_at": now,
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
    client = _get_fresh_client()
    client_holder = [client]

    t_start = time.perf_counter()

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")

    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=365)).isoformat()
    log.info("Fetching sales_transactions since %s (chunked by week) …", cutoff)
    tx_rows = _fetch_chunked_by_date(
        client,
        "sales_transactions",
        "transaction_id,sku_id,transaction_date,qty_sold,unit_price",
        date_col="transaction_date",
        since=cutoff,
    )
    log.info("  Fetched %d transaction(s).", len(tx_rows))

    if not tx_rows:
        log.warning("No transactions found — nothing to do.")
        return 0

    tx_lookup: dict[str, dict] = {r["transaction_id"]: r for r in tx_rows}

    daily_by_sku = _aggregate_daily(tx_rows)
    log.info(
        "  Aggregated to daily level: %d SKU(s), %d total daily observations.",
        len(daily_by_sku),
        sum(len(v) for v in daily_by_sku.values()),
    )

    total_skus_eligible  = 0
    total_skus_skipped   = 0
    total_days_flagged   = 0
    total_tx_flagged     = 0
    total_issues_written = 0

    sku_list = sorted(daily_by_sku.keys())
    log.info("Processing %d SKU(s) …", len(sku_list))
    log.info("-" * 60)

    client_holder[0] = _get_fresh_client()

    for idx, sku_id in enumerate(sku_list):
        if idx > 0 and idx % _CLIENT_REFRESH_EVERY == 0:
            log.info("  [refresh] Reconnecting Supabase client after %d SKUs …", idx)
            client_holder[0] = _get_fresh_client()

        daily = daily_by_sku[sku_id]

        flagged = _fit_and_detect(sku_id, daily)

        if flagged is None:
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
            def _do_sku_writes(cl, _sku=sku_id, _flagged=flagged):
                _reset_is_anomaly_for_sku(cl, _sku, dry_run)
                if _flagged:
                    anomalous_tx_ids = [tx for d in _flagged for tx in d["tx_ids"]]
                    _write_anomaly_flags(cl, anomalous_tx_ids, dry_run)
                    return _write_quality_issues(cl, _flagged, tx_lookup, dry_run)
                return 0
            issues = _db_call_with_retry(_do_sku_writes, client_holder)
        else:
            issues = sum(len(d["tx_ids"]) for d in flagged) if flagged else 0

        if flagged:
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
