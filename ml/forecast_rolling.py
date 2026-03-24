"""
ml/forecast_rolling.py — 13-week rolling average demand forecast for C-class SKUs.

Generates 30 days of forward-looking predictions for every (SKU, location)
pair in the C-class tier.  Runs nightly after extract → clean → derive →
anomaly.  Produces the simplest of the three forecast tiers: it requires no
ML training loop, handles any SKU with a few weeks of history, and completes
in well under 2 minutes even at full catalog scale.

MODEL
    For each (sku_id, location_id) pair:
    1.  Collect the last 13 weeks (91 days) of non-anomaly transactions.
    2.  Compute effective demand per day:
            - Normal sale     → qty_sold
            - Stockout sale   → lost_sales_imputation (avoids understating demand)
            - Day with no tx  → 0  (SKU was in stock but did not sell)
    3.  Compute rolling-average statistics on the 91-day daily series:
            mean_daily  = sum(series) / 91
            std_daily   = population std of series
    4.  Emit 30 forecast rows:
            predicted_qty  = mean_daily
            lower_bound    = max(0, mean_daily − std_daily)
            upper_bound    = mean_daily + std_daily
            confidence_pct = 0.6827  (±1σ covers ~68 % of a normal distribution)

SKIPPING
    Any (SKU, location) with fewer than MIN_SALE_DAYS distinct calendar days
    with at least one sale (anomaly-excluded) in the lookback window is
    skipped.  14 is the minimum to produce a meaningful rolling average.

PERFORMANCE
    All data is fetched in two bulk queries (C-class SKUs, then all their
    transactions).  Everything else runs in memory.  No per-SKU or per-
    location round-trips.

    At REST-API scale (Supabase pagination at 1,000 rows/page) the
    transaction fetch dominates.  For a catalog approaching 100 K+ active
    C-class SKUs switch to PARTSWATCH_SOURCE=odbc so the single query runs
    over a direct database connection instead of HTTP pages.

COUNTS
    SKU counts, location counts, and every business metric are queried live
    from the database.  Nothing is hardcoded.

PREREQUISITE
    Run db/migrations/006_forecast_results_location.sql in the Supabase SQL
    Editor before first use.

USAGE
    python -m ml.forecast_rolling            # full run
    python -m ml.forecast_rolling --dry-run  # compute, do not write
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import numpy as np

from utils.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Lookback window for the rolling average
LOOKBACK_WEEKS: int = 13
LOOKBACK_DAYS:  int = LOOKBACK_WEEKS * 7   # 91

# Minimum distinct sale dates needed to fit a meaningful average
MIN_SALE_DAYS: int = 14

# Number of future days to forecast
FORECAST_HORIZON: int = 30

# Confidence percentage for the ±1σ prediction interval
CONFIDENCE_PCT: float = 0.6827

# Model label written to forecast_results.model_type
MODEL_TYPE: str = "rolling_avg"

# Location written when there is no per-location breakdown (other model tiers)
NETWORK_LOCATION: str = "ALL"

# Supabase pagination page size (API cap)
_PAGE_SIZE: int = 1000

# Upsert batch size
BATCH_SIZE: int = 500

# Tier 3 blending — when a location is Tier 3 AND demand_quality_score is
# below LOW_QUALITY_THRESHOLD, the forecast is blended toward the Tier 1+2
# regional average to reduce third-call / residual-demand signal bias.
TIER3_BLEND_WEIGHT:    float = 0.40   # weight on regional average (0=none, 1=full regional)
LOW_QUALITY_THRESHOLD: float = 0.50   # demand_quality_score below this triggers blending


# ---------------------------------------------------------------------------
# Shared fetch helper
# ---------------------------------------------------------------------------

def _fetch_all(client: Any, table: str, select: str = "*",
               filters: dict | None = None) -> list[dict]:
    """Return every row from a Supabase table, handling the 1,000-row page cap.

    Args:
        client:  Active Supabase client.
        table:   Table name to query.
        select:  PostgREST column selector string.
        filters: Optional dict of {column: value} equality filters applied
                 before pagination.

    Returns:
        All matching rows as a list of dicts.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        q = client.table(table).select(select)
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)
        page: list[dict] = (
            q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
        )
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# Step 1 — Fetch C-class SKUs (dynamic count — never hardcoded)
# ---------------------------------------------------------------------------

def _fetch_c_class_skus(client: Any) -> list[str]:
    """Return all sku_ids currently classified as C-class in sku_master.

    The count is queried live so it adjusts automatically as SKUs are added,
    discontinued, or reclassified over time.
    """
    rows = _fetch_all(client, "sku_master", "sku_id", filters={"abc_class": "C"})
    skus = [r["sku_id"] for r in rows if r.get("sku_id")]
    log.info("  C-class SKUs found in sku_master: %d", len(skus))
    return skus


# ---------------------------------------------------------------------------
# Step 2 — Fetch transactions for C-class SKUs in the lookback window
# ---------------------------------------------------------------------------

def _fetch_transactions(client: Any, cutoff: str) -> list[dict]:
    """Return all non-anomaly transactions on or after cutoff for any SKU.

    Fetching all transactions in a single paginated sweep is far faster than
    one query per (SKU, location) pair, which would be millions of round-trips
    at production scale.  In-memory grouping handles the per-SKU/location split.

    Args:
        cutoff: ISO date string (YYYY-MM-DD); earliest transaction_date to include.

    Returns:
        List of transaction dicts with the fields needed for demand calculation.
    """
    rows: list[dict] = []
    offset = 0
    select = (
        "sku_id,location_id,transaction_date,"
        "qty_sold,lost_sales_imputation,is_stockout,is_anomaly"
    )
    while True:
        page: list[dict] = (
            client.table("sales_transactions")
            .select(select)
            .gte("transaction_date", cutoff)
            .eq("is_anomaly", False)
            .eq("is_residual_demand", False)
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
# Step 3 — Group transactions by (sku_id, location_id)
# ---------------------------------------------------------------------------

def _group_by_sku_location(
    tx_rows: list[dict],
    c_class_set: set[str],
) -> dict[tuple[str, str], list[dict]]:
    """Group transaction rows by (sku_id, location_id), C-class SKUs only.

    Args:
        tx_rows:     All fetched transaction rows.
        c_class_set: Set of sku_ids that are C-class.

    Returns:
        Dict mapping (sku_id, location_id) → list of transaction dicts.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in tx_rows:
        sku = r.get("sku_id", "")
        loc = r.get("location_id", "")
        if sku in c_class_set and sku and loc:
            groups[(sku, loc)].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Step 4 — Build 91-day demand series for one (SKU, location)
# ---------------------------------------------------------------------------

def _build_demand_series(
    rows: list[dict],
    cutoff: date,
    today: date,
) -> tuple[np.ndarray, int]:
    """Convert raw transaction rows into a 91-day daily demand array.

    Index 0 = cutoff day, index 90 = day before today.

    Demand per transaction:
        - is_stockout=False → qty_sold
        - is_stockout=True  → lost_sales_imputation if > 0, else 0

    Days with no transaction record default to 0 (in-stock but no sale).

    Args:
        rows:   Transaction rows for one (sku_id, location_id).
        cutoff: First day of the lookback window.
        today:  The run date (not included in the lookback).

    Returns:
        Tuple of (daily_series as np.ndarray shape (91,), sale_day_count).
        sale_day_count is the number of distinct calendar days with at least
        one non-zero effective demand record.
    """
    series = np.zeros(LOOKBACK_DAYS, dtype=float)
    sale_dates: set[int] = set()

    for r in rows:
        raw_dt = str(r.get("transaction_date", ""))[:10]
        if not raw_dt:
            continue
        try:
            tx_date = date.fromisoformat(raw_dt)
        except ValueError:
            continue

        day_idx = (tx_date - cutoff).days
        if not (0 <= day_idx < LOOKBACK_DAYS):
            continue

        is_stockout = r.get("is_stockout", False)
        if is_stockout:
            imputed = r.get("lost_sales_imputation")
            demand = float(imputed) if imputed is not None else 0.0
        else:
            demand = float(r.get("qty_sold") or 0)

        series[day_idx] += demand

        if demand > 0:
            sale_dates.add(day_idx)

    return series, len(sale_dates)


# ---------------------------------------------------------------------------
# Step 5 — Compute forecast rows for one (SKU, location)
# ---------------------------------------------------------------------------

def _compute_forecast(
    sku_id: str,
    location_id: str,
    series: np.ndarray,
    start_date: date,
    run_date: str,
) -> list[dict]:
    """Generate FORECAST_HORIZON rows of daily predictions.

    Args:
        sku_id:      SKU identifier.
        location_id: Location identifier.
        series:      91-day demand array (output of _build_demand_series).
        start_date:  First forecast date (usually today).
        run_date:    ISO date string for the run_date field.

    Returns:
        List of FORECAST_HORIZON dicts ready to upsert into forecast_results.
    """
    mean_demand = float(np.mean(series))
    std_demand  = float(np.std(series))   # population std

    predicted   = round(mean_demand, 4)
    lower       = round(max(0.0, mean_demand - std_demand), 4)
    upper       = round(mean_demand + std_demand, 4)

    rows: list[dict] = []
    for offset in range(FORECAST_HORIZON):
        forecast_date = (start_date + timedelta(days=offset)).isoformat()
        rows.append({
            "sku_id":        sku_id,
            "location_id":   location_id,
            "forecast_date": forecast_date,
            "model_type":    MODEL_TYPE,
            "predicted_qty": predicted,
            "lower_bound":   lower,
            "upper_bound":   upper,
            "confidence_pct": CONFIDENCE_PCT,
            "run_date":      run_date,
        })
    return rows


# ---------------------------------------------------------------------------
# Location tier and demand quality helpers
# ---------------------------------------------------------------------------

def _fetch_location_tiers(client: Any) -> dict[str, int]:
    """Return location_tier keyed by location_id from the locations table.

    Falls back gracefully — callers receive Tier 2 (neutral) for any
    location not yet classified by transform/location_classify.py.

    Returns:
        {location_id: tier (1/2/3)}
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page = (
            client.table("locations")
            .select("location_id,location_tier")
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute().data or []
        )
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return {
        r["location_id"]: int(r["location_tier"])
        for r in rows
        if r.get("location_id") and r.get("location_tier") is not None
    }


def _fetch_demand_quality(client: Any) -> dict[tuple[str, str], float]:
    """Return demand_quality_score keyed by (sku_id, location_id).

    Falls back gracefully — callers receive 1.0 (fully organic) for pairs
    not yet scored by transform/location_classify.py.

    Returns:
        {(sku_id, location_id): demand_quality_score (0.0–1.0)}
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page = (
            client.table("sku_location_demand_quality")
            .select("sku_id,location_id,demand_quality_score")
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute().data or []
        )
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return {
        (r["sku_id"], r["location_id"]): float(r["demand_quality_score"])
        for r in rows
        if r.get("sku_id") and r.get("location_id")
        and r.get("demand_quality_score") is not None
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_forecast(dry_run: bool = False) -> int:
    """Run the full rolling-average forecast pipeline.

    Args:
        dry_run: When True, computes all forecasts but writes nothing.

    Returns:
        Exit code 0 on success, 1 on unrecoverable error.
    """
    from db.connection import get_client
    client  = get_client()
    t_start = time.perf_counter()
    today   = date.today()
    run_date_str = today.isoformat()
    cutoff  = today - timedelta(days=LOOKBACK_DAYS)
    cutoff_str = cutoff.isoformat()

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")

    log.info(
        "Lookback window: %s → %s  (%d days, %d weeks)",
        cutoff_str, (today - timedelta(days=1)).isoformat(),
        LOOKBACK_DAYS, LOOKBACK_WEEKS,
    )
    log.info("Forecast horizon: %d days  (%s → %s)", FORECAST_HORIZON,
             today.isoformat(), (today + timedelta(days=FORECAST_HORIZON - 1)).isoformat())
    log.info("-" * 60)

    # ── 1. Fetch C-class SKUs (live count — never hardcoded) ──────────────
    log.info("Fetching C-class SKUs from sku_master …")
    c_class_skus = _fetch_c_class_skus(client)

    if not c_class_skus:
        log.warning("No C-class SKUs found — nothing to forecast.")
        return 0

    c_class_set = set(c_class_skus)

    # ── 2. Bulk-fetch all transactions in lookback window ─────────────────
    log.info("Fetching non-anomaly transactions since %s …", cutoff_str)
    t_fetch = time.perf_counter()
    tx_rows = _fetch_transactions(client, cutoff_str)
    log.info(
        "  Fetched %d transaction(s) in %.2fs.",
        len(tx_rows), time.perf_counter() - t_fetch,
    )

    # ── 3. Group by (sku_id, location_id) — C-class only ─────────────────
    groups = _group_by_sku_location(tx_rows, c_class_set)
    log.info(
        "  Active (SKU, location) combinations: %d  (across %d C-class SKUs)",
        len(groups),
        len({sku for sku, _ in groups}),
    )

    # Track C-class SKUs with no sales at all in the window
    skus_with_any_sales = {sku for sku, _ in groups}
    skus_no_sales = len(c_class_set) - len(skus_with_any_sales)
    if skus_no_sales:
        log.info(
            "  %d C-class SKU(s) had no sales in the lookback window — "
            "they will not appear in any (SKU, location) combination.",
            skus_no_sales,
        )

    log.info("-" * 60)

    # ── 3b. Fetch location tiers + demand quality for Tier 3 blending ─────
    log.info("Fetching location tiers and demand quality scores …")
    location_tiers = _fetch_location_tiers(client)
    demand_quality = _fetch_demand_quality(client)
    log.info(
        "  Location tiers loaded: %d  Demand quality pairs: %d",
        len(location_tiers), len(demand_quality),
    )

    # Pre-compute per-(SKU, location) mean demand so we can build regional
    # baselines for Tier 1+2 locations without a second DB round-trip.
    loc_mean_demand: dict[tuple[str, str], float] = {}
    for (sku, loc), loc_rows in groups.items():
        s, _ = _build_demand_series(loc_rows, cutoff, today)
        loc_mean_demand[(sku, loc)] = float(np.mean(s))

    # Regional mean demand per SKU: average of Tier 1+2 locations only.
    from collections import defaultdict as _dd
    sku_tier12_sums: dict[str, list[float]] = _dd(list)
    for (sku, loc), mean_d in loc_mean_demand.items():
        if location_tiers.get(loc, 2) <= 2:
            sku_tier12_sums[sku].append(mean_d)
    sku_regional_mean: dict[str, float] = {
        sku: sum(vals) / len(vals)
        for sku, vals in sku_tier12_sums.items()
        if vals
    }
    log.info(
        "  Regional baselines available for %d C-class SKU(s).",
        len(sku_regional_mean),
    )
    log.info("-" * 60)

    # ── 4. Process each (SKU, location) ──────────────────────────────────
    processed_pairs = 0
    skipped_pairs   = 0
    forecast_buffer: list[dict] = []
    total_rows_written = 0

    for (sku_id, location_id), rows in sorted(groups.items()):
        series, sale_day_count = _build_demand_series(rows, cutoff, today)

        if sale_day_count < MIN_SALE_DAYS:
            log.info(
                "  SKIP  %-12s  %-10s  %d sale day(s) (need >= %d)",
                sku_id, location_id, sale_day_count, MIN_SALE_DAYS,
            )
            skipped_pairs += 1
            continue

        fc_rows = _compute_forecast(
            sku_id, location_id, series, today, run_date_str
        )
        mean_d = float(np.mean(series))
        std_d  = float(np.std(series))

        # Tier 3 blending: weight predictions toward regional (Tier 1+2) mean
        # when demand quality is below the threshold.
        loc_tier = location_tiers.get(location_id, 2)
        dq_score = float(demand_quality.get((sku_id, location_id), 1.0))
        if loc_tier == 3 and dq_score < LOW_QUALITY_THRESHOLD:
            regional = sku_regional_mean.get(sku_id, mean_d)
            blended_pred = round(
                TIER3_BLEND_WEIGHT * regional
                + (1.0 - TIER3_BLEND_WEIGHT) * mean_d,
                4,
            )
            fc_rows = [
                {**r, "predicted_qty": blended_pred}
                for r in fc_rows
            ]
            log.debug(
                "  BLEND %-12s  %-10s  Tier 3  dq=%.2f  local=%.2f  "
                "regional=%.2f  blended=%.2f",
                sku_id, location_id, dq_score, mean_d, regional, blended_pred,
            )

        log.info(
            "  OK    %-12s  %-10s  %d sale day(s)  "
            "mean=%.2f  std=%.2f  → %d forecast row(s)",
            sku_id, location_id, sale_day_count,
            mean_d, std_d, len(fc_rows),
        )

        forecast_buffer.extend(fc_rows)
        processed_pairs += 1

        # Flush buffer in BATCH_SIZE chunks to control memory
        while len(forecast_buffer) >= BATCH_SIZE:
            batch = forecast_buffer[:BATCH_SIZE]
            forecast_buffer = forecast_buffer[BATCH_SIZE:]
            if not dry_run:
                client.table("forecast_results").upsert(
                    batch,
                    on_conflict="sku_id,location_id,forecast_date,model_type,run_date",
                ).execute()
            total_rows_written += len(batch)

    # Flush any remaining rows
    if forecast_buffer:
        if not dry_run:
            client.table("forecast_results").upsert(
                forecast_buffer,
                on_conflict="sku_id,location_id,forecast_date,model_type,run_date",
            ).execute()
        total_rows_written += len(forecast_buffer)

    # ── 5. Summary ────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info("Rolling average forecast complete  (%.2fs)", elapsed)
    log.info("  C-class SKUs queried:              %d", len(c_class_skus))
    log.info("  C-class SKUs with no sales:        %d", skus_no_sales)
    log.info("  (SKU, location) pairs processed:   %d", processed_pairs)
    log.info("  (SKU, location) pairs skipped:     %d  (< %d sale days)",
             skipped_pairs, MIN_SALE_DAYS)
    log.info("  Forecast rows written:             %d", total_rows_written)
    log.info("  Forecast horizon:                  %d days", FORECAST_HORIZON)
    if dry_run:
        log.info("  (DRY RUN — no writes were made)")
    log.info("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI arguments and run the rolling-average forecast pipeline."""
    parser = argparse.ArgumentParser(
        description="partswatch-ai: 13-week rolling average forecast for C-class SKUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prerequisite: run db/migrations/006_forecast_results_location.sql\n"
            "in the Supabase SQL Editor before first use.\n\n"
            "Nightly pipeline order:\n"
            "  extract/partswatch_pull.py  →  transform/clean.py\n"
            "  →  transform/derive.py  →  ml/anomaly.py\n"
            "  →  ml/forecast_rolling.py\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute forecasts and log results without writing to the database.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("partswatch-ai — ml.forecast_rolling")
    log.info(
        "  lookback=%dw  horizon=%dd  min_sale_days=%d",
        LOOKBACK_WEEKS, FORECAST_HORIZON, MIN_SALE_DAYS,
    )
    log.info("=" * 60)

    return run_forecast(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
