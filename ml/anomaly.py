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
from datetime import date, datetime, timedelta, timezone
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
# GP% anomaly detection constants
# ---------------------------------------------------------------------------

GP_DROP_THRESHOLD_PP: float = 0.15
"""Flag when 7-day GP% falls more than this many percentage points below baseline."""

GP_MIN_TX_7D: int = 5
"""Minimum transactions in the 7-day window — avoids noise from sparse SKUs."""

GP_WINDOW_DAYS: int = 7
"""Recent window for current GP% measurement."""

GP_BASELINE_DAYS: int = 90
"""Lookback for the per-SKU+location GP% baseline."""

# ---------------------------------------------------------------------------
# Volume anomaly alert constants
# ---------------------------------------------------------------------------

ALERT_LOOKBACK_DAYS: int = 30
"""Only write volume_anomaly alerts for anomalous days within this window."""


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

    mean_qty = float(np.mean(X[:, 0]))  # used to classify HIGH vs LOW direction

    flagged = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        if pred == -1:
            day = daily[i]
            flagged.append({**day, "anomaly_score": round(float(score), 4), "mean_qty": mean_qty})
            log.info(
                "    ANOMALY  %-12s  %s  qty=%.0f  avg=%.0f  price=%.2f  "
                "tx=%d  score=%.4f",
                sku_id,
                day["date"],
                day["daily_qty"],
                mean_qty,
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
                "ReadTimeout", "ConnectTimeout", "PoolTimeout", "57014",
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


_RESET_BATCH = 200


def _reset_is_anomaly_for_sku(client: Any, sku_id: str, tx_ids_all: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    for i in range(0, len(tx_ids_all), _RESET_BATCH):
        batch_ids = tx_ids_all[i : i + _RESET_BATCH]
        client.table("sales_transactions").update(
            {"is_anomaly": False}
        ).in_("transaction_id", batch_ids).execute()


def _write_anomaly_flags(
    client: Any,
    tx_ids: list[str],
    dry_run: bool,
) -> None:
    if dry_run:
        return
    for i in range(0, len(tx_ids), _RESET_BATCH):
        batch_ids = tx_ids[i : i + _RESET_BATCH]
        client.table("sales_transactions").update(
            {"is_anomaly": True}
        ).in_("transaction_id", batch_ids).execute()


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

    # Dedup within the batch by the unique key (source_table, source_id,
    # issue_type) before upsert.  Postgres raises 21000 ('cannot affect row
    # a second time') when a single ON CONFLICT statement targets the same
    # conflict key twice — which happens here whenever a tx_id appears in
    # more than one flagged day row for the same SKU/issue_type.  Last
    # write wins, mirroring the autocube_product_pull dedup pattern.
    seen: dict[tuple[str, str, str], dict] = {}
    for r in records:
        seen[(r["source_table"], r["source_id"], r["issue_type"])] = r
    deduped = list(seen.values())

    for i in range(0, len(deduped), BATCH_SIZE):
        batch = deduped[i : i + BATCH_SIZE]
        client.table("data_quality_issues").upsert(
            batch,
            on_conflict="source_table,source_id,issue_type",
        ).execute()

    return len(deduped)


# ---------------------------------------------------------------------------
# Volume anomaly alert helpers
# ---------------------------------------------------------------------------

def _build_volume_alert(today: date, sku_id: str, flagged_day: dict) -> dict:
    """Build a volume_anomaly alert dict for one Isolation Forest-flagged day."""
    daily_qty = flagged_day["daily_qty"]
    mean_qty  = flagged_day.get("mean_qty", 0.0)
    is_high   = daily_qty > mean_qty

    if is_high:
        direction = "HIGH"
        severity  = "warning"
        action    = "Verify demand is real — check if order was duplicated"
    else:
        direction = "LOW"
        severity  = "info"
        action    = "Check if store had supply issues or lost a key account"

    flagged_date = flagged_day["date"]
    message = (
        f"{sku_id}: {direction} volume anomaly on {flagged_date} — "
        f"daily_qty={daily_qty:.0f} vs 12-month avg={mean_qty:.0f} "
        f"(score={flagged_day['anomaly_score']:.4f}). "
        f"Recommended action: {action}"
    )
    return {
        "alert_date":      today.isoformat(),
        "alert_type":      "volume_anomaly",
        "severity":        severity,
        "sku_id":          sku_id,
        "location_id":     None,
        "message":         message,
        "alert_key":       f"volume_anomaly|{sku_id}|{flagged_date}",
        "is_acknowledged": False,
    }


def _write_alerts_to_db(client: Any, alerts: list[dict], dry_run: bool) -> int:
    """Upsert alerts rows in BATCH_SIZE chunks.  Returns count of rows passed."""
    if dry_run or not alerts:
        return len(alerts)
    written = 0
    for i in range(0, len(alerts), BATCH_SIZE):
        batch = alerts[i : i + BATCH_SIZE]
        try:
            resp = client.table("alerts").upsert(
                batch,
                on_conflict="alert_date,alert_key",
                ignore_duplicates=True,
            ).execute()
            written += len(resp.data or [])
        except Exception:
            log.exception("Failed to write alerts batch at offset %d (size=%d).", i, len(batch))
    return written


# ---------------------------------------------------------------------------
# GP% anomaly detection
# ---------------------------------------------------------------------------

def detect_gp_anomalies(client: Any, dry_run: bool = False) -> int:
    """Detect SKU+location pairs where 7-day GP% has dropped >15pp below 90-day baseline.

    Queries sales_detail_transactions for tran_code='SL' rows in the last
    GP_BASELINE_DAYS days.  For each (prod_line_pn, location_id) pair with at
    least GP_MIN_TX_7D transactions in the last GP_WINDOW_DAYS days, computes:

        baseline_gp_pct = sum(gross_profit, 90d) / sum(sales, 90d)
        current_gp_pct  = sum(gross_profit, 7d)  / sum(sales, 7d)
        drop            = baseline_gp_pct - current_gp_pct

    Flags when drop > GP_DROP_THRESHOLD_PP (default 15pp).

    Writes gp_anomaly alerts to the alerts table (one per SKU+location per day,
    idempotent via upsert on alert_date, alert_key).

    Returns the number of GP anomaly alerts generated.
    """
    today      = date.today()
    cutoff_90d = (today - timedelta(days=GP_BASELINE_DAYS)).isoformat()
    cutoff_7d  = (today - timedelta(days=GP_WINDOW_DAYS)).isoformat()

    log.info(
        "GP anomaly detection — baseline=%dd  window=%dd  threshold=%.0fpp  min_tx=%d",
        GP_BASELINE_DAYS, GP_WINDOW_DAYS,
        GP_DROP_THRESHOLD_PP * 100, GP_MIN_TX_7D,
    )

    try:
        rows = _fetch_chunked_by_date(
            client,
            "sales_detail_transactions",
            "prod_line_pn,location_id,tran_date,sales,gross_profit",
            date_col="tran_date",
            since=cutoff_90d,
            extra_eq={"tran_code": "SL"},
        )
    except Exception:
        log.exception("GP anomaly: failed to fetch sales_detail_transactions — skipping.")
        return 0

    log.info("  GP anomaly: %d SL rows fetched from sales_detail_transactions", len(rows))

    # Accumulate sums per (prod_line_pn, location_id) for 90-day baseline and 7-day current
    acc: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "s90": 0.0, "g90": 0.0, "n90": 0,
        "s7":  0.0, "g7":  0.0, "n7":  0,
    })

    for r in rows:
        pn  = r.get("prod_line_pn")
        loc = r.get("location_id")
        if not pn or not loc:
            continue
        s  = float(r.get("sales")        or 0)
        gp = float(r.get("gross_profit") or 0)
        if s <= 0:
            continue
        tran_date = str(r.get("tran_date", ""))[:10]
        key = (pn, loc)
        acc[key]["s90"] += s
        acc[key]["g90"] += gp
        acc[key]["n90"] += 1
        if tran_date >= cutoff_7d:
            acc[key]["s7"] += s
            acc[key]["g7"] += gp
            acc[key]["n7"] += 1

    alerts: list[dict] = []
    for (pn, loc), stats in acc.items():
        if stats["n7"] < GP_MIN_TX_7D:
            continue
        if stats["s7"] <= 0 or stats["s90"] <= 0:
            continue
        baseline_gp_pct = stats["g90"] / stats["s90"]
        current_gp_pct  = stats["g7"]  / stats["s7"]
        drop = baseline_gp_pct - current_gp_pct
        if drop < GP_DROP_THRESHOLD_PP:
            continue
        message = (
            f"{pn} at {loc}: GP% dropped {drop*100:.1f}pp below 90-day baseline "
            f"(baseline={baseline_gp_pct*100:.1f}%, current_7d={current_gp_pct*100:.1f}%, "
            f"txns_7d={stats['n7']}). "
            f"Recommended action: Review recent invoices for pricing errors or unauthorized discounts"
        )
        alerts.append({
            "alert_date":      today.isoformat(),
            "alert_type":      "gp_anomaly",
            "severity":        "warning",
            "sku_id":          pn,
            "location_id":     loc,
            "message":         message,
            "alert_key":       f"gp_anomaly|{pn}|{loc}",
            "is_acknowledged": False,
        })

    log.info("  GP anomaly: %d flag(s) (threshold=%.0fpp)", len(alerts), GP_DROP_THRESHOLD_PP * 100)
    if dry_run:
        for a in alerts[:5]:
            log.info("  [DRY RUN] %s", a["message"])
        if len(alerts) > 5:
            log.info("  [DRY RUN] … and %d more", len(alerts) - 5)
        return len(alerts)

    return _write_alerts_to_db(client, alerts, dry_run=False)


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
    today   = date.today()

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")
        # Show current anomaly alert counts so the user can see the baseline
        try:
            r = (
                client.table("alerts")
                .select("id", count="exact")
                .in_("alert_type", ["volume_anomaly", "gp_anomaly"])
                .limit(1)
                .execute()
            )
            log.info(
                "DRY RUN — current volume_anomaly + gp_anomaly alerts in DB: %d",
                r.count or 0,
            )
        except Exception:
            log.info("DRY RUN — could not query current anomaly alert count (migration 046 pending?)")

    cutoff = (today - timedelta(days=365)).isoformat()
    alert_cutoff = (today - timedelta(days=ALERT_LOOKBACK_DAYS)).isoformat()
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
    volume_alert_buffer: list[dict] = []

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
            all_tx_ids_for_sku = [tx for d in daily for tx in d["tx_ids"]]

            def _do_sku_writes(cl, _sku=sku_id, _flagged=flagged, _all_tx=all_tx_ids_for_sku):
                _reset_is_anomaly_for_sku(cl, _sku, _all_tx, dry_run)
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
            # Collect volume_anomaly alerts for recent flagged days
            for day in flagged:
                if day["date"] >= alert_cutoff:
                    volume_alert_buffer.append(_build_volume_alert(today, sku_id, day))

    # ── 4. Write volume_anomaly alerts ───────────────────────────────────────
    high_count = sum(1 for a in volume_alert_buffer if a["severity"] == "warning")
    low_count  = sum(1 for a in volume_alert_buffer if a["severity"] == "info")
    log.info(
        "Volume anomaly alerts (last %dd): %d total  (HIGH=%d  LOW=%d)",
        ALERT_LOOKBACK_DAYS, len(volume_alert_buffer), high_count, low_count,
    )
    if volume_alert_buffer:
        alerts_written = _write_alerts_to_db(client_holder[0], volume_alert_buffer, dry_run)
        log.info(
            "  volume_anomaly alerts written: %d%s",
            alerts_written,
            "  (DRY RUN)" if dry_run else "",
        )

    # ── 5. GP% anomaly detection ──────────────────────────────────────────────
    log.info("-" * 60)
    gp_count = detect_gp_anomalies(client_holder[0], dry_run=dry_run)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info("Anomaly detection complete  (%.2fs)", elapsed)
    log.info("  SKUs eligible (>= %d obs):  %d", MIN_TRANSACTIONS, total_skus_eligible)
    log.info("  SKUs skipped  (< %d obs):   %d", MIN_TRANSACTIONS, total_skus_skipped)
    log.info("  Anomalous days flagged:      %d", total_days_flagged)
    log.info("  Transactions flagged:        %d", total_tx_flagged)
    log.info("  data_quality_issues written: %d", total_issues_written)
    log.info("  volume_anomaly alerts:       %d (HIGH=%d  LOW=%d)",
             len(volume_alert_buffer), high_count, low_count)
    log.info("  gp_anomaly alerts:           %d", gp_count)
    log.info("  Dashboard fix: anomaly panel now reads from alerts table "
             "(alert_type IN volume_anomaly, gp_anomaly)")
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
