"""ml/accuracy.py — Forecast accuracy measurement for partswatch-ai.

Compares predicted quantities in forecast_results to actual sales, computes
MAPE / MAE / bias / hit-rate metrics, and writes weekly summary reports.

WHAT IT DOES
    1. For every forecast row where forecast_date is in the past, look up the
       actual qty_sold from sales_transactions for that SKU + location + date.
    2. Calculate per-row MAPE and back-fill actual_qty_sold + mape_error into
       forecast_results.
    3. Aggregate weekly accuracy reports by model_type × abc_class and write
       to the accuracy_reports table.
    4. Flag model+class combinations where avg MAPE > 25 % — writes
       FORECAST_ACCURACY_DROP alerts.
    5. Compute per-SKU accuracy scores and update sku_master.accuracy_score.

SCHEDULE
    Runs weekly (Sunday nights) after forecasts complete.

PREREQUISITE
    Run db/migrations/011_accuracy.sql in the Supabase SQL Editor first.

USAGE
    python -m ml.accuracy             # full run
    python -m ml.accuracy --dry-run   # compute & log, do not write
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from supabase import Client

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

_PAGE_SIZE: int = 1_000
BATCH_SIZE: int = 500
_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 5.0

MAPE_THRESHOLD_PCT: float = 25.0
HIT_RATE_TOLERANCE: float = 0.20
LOOKBACK_DAYS: int = 28
REVIEW_MAPE_THRESHOLD: float = 40.0


def _get_fresh_client() -> Client:
    get_client.cache_clear()
    return get_client()


def _db_call_with_retry(fn, client_holder: list[Client]):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(client_holder[0])
        except Exception as exc:
            err_name = type(exc).__name__
            if attempt < _MAX_RETRIES and any(
                kw in err_name or kw in str(exc)
                for kw in ("Timeout", "ConnectionTerminated", "ConnectionError",
                           "RemoteProtocolError", "502", "503")
            ):
                log.warning(
                    "  [retry %d/%d] %s — refreshing client …",
                    attempt, _MAX_RETRIES, err_name,
                )
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _get_fresh_client()
                continue
            raise


def _paginate(
    client: Client,
    table: str,
    select: str,
    filters: dict | None = None,
    gte_filters: dict | None = None,
    lte_filters: dict | None = None,
    in_filters: dict | None = None,
    is_filters: dict | None = None,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        q = client.table(table).select(select)
        for col, val in (filters or {}).items():
            q = q.eq(col, val)
        for col, val in (gte_filters or {}).items():
            q = q.gte(col, val)
        for col, val in (lte_filters or {}).items():
            q = q.lte(col, val)
        for col, vals in (in_filters or {}).items():
            q = q.in_(col, vals)
        for col, val in (is_filters or {}).items():
            q = q.is_(col, val)
        page = q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def _fetch_chunked_by_date(
    client: Client,
    table: str,
    select: str,
    date_col: str,
    since: str,
    until: str,
    chunk_days: int = 7,
    extra_eq: dict[str, Any] | None = None,
) -> list[dict]:
    start = date.fromisoformat(since)
    end = date.fromisoformat(until)
    all_rows: list[dict] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end)
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
        chunk_start = chunk_end + timedelta(days=1)
        if all_rows and len(all_rows) % 50_000 < (chunk_days * 5000):
            log.info("    … streamed %d rows so far from %s", len(all_rows), table)
    log.info("    … streamed %d total rows from %s", len(all_rows), table)
    return all_rows


def _batch_upsert(
    client_holder: list[Client],
    table: str,
    rows: list[dict],
    on_conflict: str,
    dry_run: bool,
) -> int:
    written = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        if not dry_run:
            _db_call_with_retry(
                lambda c, b=batch: c.table(table).upsert(
                    b, on_conflict=on_conflict
                ).execute(),
                client_holder,
            )
        written += len(batch)
    return written


def _batch_update(
    client_holder: list[Client],
    table: str,
    updates: list[dict],
    key_col: str,
    dry_run: bool,
) -> int:
    updated = 0
    for row in updates:
        key_val = row.pop(key_col)
        if not dry_run:
            _db_call_with_retry(
                lambda c, k=key_val, r=row: c.table(table).update(r).eq(
                    key_col, k
                ).execute(),
                client_holder,
            )
        updated += 1
    return updated


def run(dry_run: bool = False) -> int:
    t0 = time.perf_counter()
    today = date.today()
    yesterday = today - timedelta(days=1)
    window_start = today - timedelta(days=LOOKBACK_DAYS)

    log.info("=" * 70)
    log.info("FORECAST ACCURACY — %s", today.isoformat())
    log.info("  Window:   %s → %s  (%d days)", window_start, yesterday, LOOKBACK_DAYS)
    log.info("  Dry run:  %s", dry_run)
    log.info("=" * 70)

    client = get_client()
    client_holder = [client]

    log.info("Step 1: Fetch forecast rows with forecast_date in window …")
    fc_rows = _fetch_chunked_by_date(
        client, "forecast_results",
        "id,sku_id,location_id,forecast_date,model_type,predicted_qty,run_date",
        date_col="forecast_date",
        since=window_start.isoformat(),
        until=yesterday.isoformat(),
    )
    if not fc_rows:
        log.info("  No forecast rows found in window — nothing to evaluate.")
        return 0

    latest_run: dict[tuple, str] = {}
    for r in fc_rows:
        key = (r["sku_id"], r.get("location_id", "ALL"), r["forecast_date"], r["model_type"])
        run_d = r.get("run_date") or ""
        if run_d > latest_run.get(key, ""):
            latest_run[key] = run_d

    fc_latest = [
        r for r in fc_rows
        if r.get("run_date", "") == latest_run.get(
            (r["sku_id"], r.get("location_id", "ALL"), r["forecast_date"], r["model_type"]),
            "",
        )
    ]
    log.info("  %d forecast rows total, %d from latest runs", len(fc_rows), len(fc_latest))

    log.info("Step 2: Fetch actual sales for the same window …")
    sales_rows = _fetch_chunked_by_date(
        client_holder[0], "sales_transactions",
        "sku_id,location_id,transaction_date,qty_sold",
        date_col="transaction_date",
        since=window_start.isoformat(),
        until=yesterday.isoformat(),
    )

    actuals: dict[tuple[str, str, str], float] = defaultdict(float)
    for s in sales_rows:
        key = (s["sku_id"], s["location_id"], str(s["transaction_date"])[:10])
        actuals[key] += float(s.get("qty_sold") or 0)

    actuals_all: dict[tuple[str, str], float] = defaultdict(float)
    for s in sales_rows:
        key = (s["sku_id"], str(s["transaction_date"])[:10])
        actuals_all[key] += float(s.get("qty_sold") or 0)

    log.info("  %d sales rows → %d (sku,loc,date) actuals, %d (sku,date) totals",
             len(sales_rows), len(actuals), len(actuals_all))

    log.info("Step 3: Fetch SKU abc_class from sku_master …")
    sku_ids = list({r["sku_id"] for r in fc_latest})
    abc_map: dict[str, str] = {}
    for i in range(0, len(sku_ids), 500):
        chunk = sku_ids[i: i + 500]
        rows = _paginate(client_holder[0], "sku_master", "sku_id,abc_class",
                         in_filters={"sku_id": chunk})
        for r in rows:
            abc_map[r["sku_id"]] = r.get("abc_class") or "?"
    log.info("  abc_class mapped for %d SKUs", len(abc_map))

    log.info("Step 4: Compute per-row accuracy …")
    update_rows: list[dict] = []
    eval_data: list[dict] = []

    for r in fc_latest:
        sku_id = r["sku_id"]
        loc_id = r.get("location_id", "ALL")
        fc_date = str(r["forecast_date"])[:10]
        predicted = float(r.get("predicted_qty") or 0)

        if loc_id == "ALL":
            actual = actuals_all.get((sku_id, fc_date), 0.0)
        else:
            actual = actuals.get((sku_id, loc_id, fc_date), 0.0)

        if actual > 0:
            mape = abs(actual - predicted) / actual * 100
        elif predicted == 0 and actual == 0:
            mape = 0.0
        else:
            mape = 100.0

        update_rows.append({
            "id": r["id"],
            "actual_qty_sold": round(actual, 4),
            "mape_error": round(mape, 4),
        })

        eval_data.append({
            "sku_id": sku_id,
            "location_id": loc_id,
            "forecast_date": fc_date,
            "model_type": r["model_type"],
            "predicted": predicted,
            "actual": actual,
            "mape": mape,
            "mae": abs(actual - predicted),
            "error": predicted - actual,
            "abc_class": abc_map.get(sku_id, "?"),
        })

    log.info("  %d forecast rows evaluated", len(eval_data))

    log.info("Step 5: Write actual_qty_sold + mape_error back to forecast_results …")
    fr_written = 0
    for i in range(0, len(update_rows), BATCH_SIZE):
        batch = update_rows[i: i + BATCH_SIZE]
        if not dry_run:
            _db_call_with_retry(
                lambda c, b=batch: c.table("forecast_results").upsert(
                    b, on_conflict="id"
                ).execute(),
                client_holder,
            )
        fr_written += len(batch)
    log.info("  Updated %d forecast_results rows", fr_written)

    log.info("Step 6: Aggregate weekly accuracy report …")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in eval_data:
        key = (e["model_type"], e["abc_class"])
        groups[key].append(e)

    report_rows: list[dict] = []
    report_date = today.isoformat()
    alerts_to_write: list[dict] = []

    for (model_type, abc_class), items in sorted(groups.items()):
        if abc_class == "?":
            continue

        mapes = [i["mape"] for i in items]
        maes = [i["mae"] for i in items]
        errors = [i["error"] for i in items]
        n = len(items)

        avg_mape = sum(mapes) / n
        avg_mae = sum(maes) / n
        bias = sum(errors) / n
        within_20 = sum(
            1 for i in items
            if i["actual"] > 0 and abs(i["actual"] - i["predicted"]) / i["actual"] <= HIT_RATE_TOLERANCE
        )
        actuals_nonzero = sum(1 for i in items if i["actual"] > 0)
        hit_rate = (within_20 / actuals_nonzero * 100) if actuals_nonzero > 0 else 0.0
        sku_count = len({i["sku_id"] for i in items})

        report_rows.append({
            "report_date": report_date,
            "model_type": model_type,
            "abc_class": abc_class,
            "avg_mape": round(avg_mape, 2),
            "avg_mae": round(avg_mae, 4),
            "bias": round(bias, 4),
            "hit_rate_20pct": round(hit_rate, 2),
            "total_forecasts_evaluated": n,
            "sku_count": sku_count,
        })

        model_label = {
            "lightgbm": "LightGBM",
            "rolling_avg": "Rolling Avg",
            "prophet": "Prophet",
        }.get(model_type, model_type)

        log.info(
            "  %-12s %s-class  MAPE=%.1f%%  MAE=%.2f  bias=%+.2f  "
            "hit20=%.0f%%  n=%d  skus=%d",
            model_label, abc_class, avg_mape, avg_mae, bias,
            hit_rate, n, sku_count,
        )

        if avg_mape > MAPE_THRESHOLD_PCT:
            alert_key = f"FORECAST_ACCURACY_DROP|{model_type}|{abc_class}"
            alerts_to_write.append({
                "alert_date": report_date,
                "alert_type": "FORECAST_ACCURACY_DROP",
                "severity": "warning",
                "message": (
                    f"{abc_class}-class {model_label} weekly MAPE is {avg_mape:.1f}% "
                    f"(threshold: {MAPE_THRESHOLD_PCT:.0f}%, "
                    f"evaluated {n} forecasts across {sku_count} SKUs). "
                    f"Bias: {bias:+.2f} units, hit rate: {hit_rate:.0f}%."
                ),
                "alert_key": alert_key,
                "is_acknowledged": False,
            })

    log.info("  %d report rows, %d accuracy alerts", len(report_rows), len(alerts_to_write))

    if report_rows:
        _batch_upsert(
            client_holder, "accuracy_reports", report_rows,
            on_conflict="report_date,model_type,abc_class", dry_run=dry_run,
        )
        log.info("  Wrote %d accuracy_reports rows", len(report_rows))

    if alerts_to_write:
        _batch_upsert(
            client_holder, "alerts", alerts_to_write,
            on_conflict="alert_date,alert_key", dry_run=dry_run,
        )
        log.info("  Wrote %d FORECAST_ACCURACY_DROP alerts", len(alerts_to_write))

    log.info("Step 7: Compute per-SKU accuracy scores …")
    sku_mapes: dict[str, list[float]] = defaultdict(list)
    for e in eval_data:
        sku_mapes[e["sku_id"]].append(e["mape"])

    sku_updates: list[dict] = []
    flagged_count = 0
    for sku_id, mapes_list in sku_mapes.items():
        avg = sum(mapes_list) / len(mapes_list)
        sku_updates.append({
            "sku_id": sku_id,
            "accuracy_score": round(avg, 2),
        })
        if avg > REVIEW_MAPE_THRESHOLD:
            flagged_count += 1

    if sku_updates:
        for i in range(0, len(sku_updates), BATCH_SIZE):
            batch = sku_updates[i: i + BATCH_SIZE]
            if not dry_run:
                for row in batch:
                    _db_call_with_retry(
                        lambda c, sid=row["sku_id"], sc=row["accuracy_score"]: (
                            c.table("sku_master")
                            .update({"accuracy_score": sc})
                            .eq("sku_id", sid)
                            .execute()
                        ),
                        client_holder,
                    )
        log.info("  Updated accuracy_score for %d SKUs", len(sku_updates))
    log.info("  SKUs flagged for review (MAPE > %.0f%%): %d", REVIEW_MAPE_THRESHOLD, flagged_count)

    elapsed = time.perf_counter() - t0

    model_mapes: dict[str, list[float]] = defaultdict(list)
    class_mapes: dict[str, list[float]] = defaultdict(list)
    all_mapes: list[float] = []
    for e in eval_data:
        model_mapes[e["model_type"]].append(e["mape"])
        class_mapes[e["abc_class"]].append(e["mape"])
        all_mapes.append(e["mape"])

    overall_mape = (sum(all_mapes) / len(all_mapes)) if all_mapes else 0.0

    log.info("=" * 70)
    log.info("ACCURACY SUMMARY")
    log.info("  Forecasts evaluated: %d", len(eval_data))
    log.info("  Overall MAPE: %.1f%%", overall_mape)
    for m, vals in sorted(model_mapes.items()):
        log.info("  By model: %-12s %.1f%%", m, sum(vals) / len(vals))
    for c, vals in sorted(class_mapes.items()):
        if c != "?":
            log.info("  By class: %s %.1f%%", c, sum(vals) / len(vals))
    log.info("  SKUs flagged for review: %d", flagged_count)
    log.info("  Elapsed: %.1fs", elapsed)
    log.info("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast accuracy evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Compute without writing")
    args = parser.parse_args()

    sys.exit(run(dry_run=args.dry_run))
