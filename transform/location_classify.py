"""transform/location_classify.py — Location tier classification and demand quality scoring.

Runs as part of the nightly pipeline, before the forecast models.  Produces
two outputs that feed directly into demand forecasting:

    1.  locations table — location_tier (1 / 2 / 3) based on fill rate,
        revenue, SKU breadth, and return rate.

    2.  sku_location_demand_quality table — demand_quality_score (0.0 – 1.0)
        per (SKU, location) pair; also flags individual sales transactions as
        is_residual_demand = TRUE when a spike correlates with simultaneous
        stockouts at peer locations (third-call demand pattern).

These outputs are consumed by:
    • ml/forecast_lgbm.py  — demand_quality_score as a training feature;
                              residual sales excluded from training data.
    • ml/forecast_rolling.py — residual sales excluded from training data;
                                Tier 3 forecasts blended toward regional mean.

DESIGN RULES
------------
- No thresholds are hardcoded in logic; all tunable values are module constants.
- Each processing stage is its own isolated function — no shared state.
- All DB reads are paginated; updates are batched to minimise round-trips.
- Never hardcodes location counts, SKU counts, or revenue totals.

PIPELINE POSITION
-----------------
    … → transform/derive.py → transform/location_classify.py
        → ml/anomaly.py → ml/forecast_rolling.py → ml/forecast_lgbm.py → …

Usage
-----
    python -m transform.location_classify            # live run
    python -m transform.location_classify --dry-run  # log only, no DB writes
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants — adjust here without touching logic
# ---------------------------------------------------------------------------

# Location tier scoring weights — must sum to 1.0
TIER_WEIGHT_FILL_RATE:   float = 0.40   # most important: reliable fulfilment
TIER_WEIGHT_REVENUE:     float = 0.30   # total demand volume
TIER_WEIGHT_SKU_BREADTH: float = 0.20   # catalog coverage
TIER_WEIGHT_RETURN_RATE: float = 0.10   # lower returns = better (score is inverted)

# Default fill-rate score for locations with no purchase-order history
DEFAULT_FILL_RATE: float = 0.50

# Residual demand detection
MIN_CONCURRENT_STOCKOUTS: int = 2        # min simultaneous stockouts across locations
                                          # on the same SKU + date to trigger detection
SPIKE_MULTIPLIER: float = 2.0            # sales must exceed baseline * this to be a spike
RESIDUAL_LOOKBACK_DAYS: int = 180        # how far back to scan for residual events
BASELINE_WINDOW_DAYS:   int = 90         # rolling window for per-(SKU, location) baseline

# Demand quality score
LOW_QUALITY_THRESHOLD: float = 0.50      # scores below this indicate primarily residual demand

_PAGE_SIZE: int = 1_000


# ---------------------------------------------------------------------------
# Shared pagination helper
# ---------------------------------------------------------------------------

def _paginate(
    client: Any,
    table: str,
    select: str,
    filters:     dict | None = None,
    gte_filters: dict | None = None,
    lte_filters: dict | None = None,
    in_filters:  dict | None = None,
    eq_bool:     dict | None = None,
) -> list[dict]:
    """Paginate through a Supabase table and return all matching rows.

    Args:
        client:      Active Supabase client.
        table:       Table name.
        select:      Column selector string.
        filters:     {col: val} for equality.
        gte_filters: {col: val} for col >= val.
        lte_filters: {col: val} for col <= val.
        in_filters:  {col: [vals]} for col IN (vals).
        eq_bool:     {col: bool} for boolean equality.

    Returns:
        All matching rows as a list of dicts.
    """
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
        for col, val in (eq_bool or {}).items():
            q = q.eq(col, val)
        page = q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
        if offset % 50_000 == 0:
            log.info("    … fetched %d rows so far from %s", offset, table)
    return rows


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

    Returns:
        All matching rows as a list of dicts.
    """
    start = date.fromisoformat(since)
    end = date.fromisoformat(until) if until else date.today()
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
        if all_rows and len(all_rows) % 100_000 < (chunk_days * 7000):
            log.info("    … streamed %d rows so far from %s", len(all_rows), table)
    log.info("    … streamed %d total rows from %s", len(all_rows), table)
    return all_rows


# ---------------------------------------------------------------------------
# Stage 1 — Location tier classification
# ---------------------------------------------------------------------------

def _classify_location_tiers(
    client: Any,
    today: date,
    dry_run: bool = False,
) -> dict[str, int]:
    """Classify all locations into Tier 1 / 2 / 3 and upsert to locations table.

    Scoring dimensions (all normalised to 0 – 1 via min-max scaling):
        fill_rate_score    — average fill_rate_pct from purchase_orders
        revenue_score      — total qty_sold from sales_transactions
        sku_breadth_score  — count of distinct SKUs sold
        return_rate_score  — inverted: 1 – normalised_return_rate

    Tier assignment (rank-based):
        Tier 1 → top    ceil(n/3) locations by composite score
        Tier 2 → middle floor(n/3) locations
        Tier 3 → bottom floor(n/3) locations

    Args:
        client:  Active Supabase client.
        today:   Classification date (written to classified_date).
        dry_run: When True, skip the DB upsert.

    Returns:
        Dict mapping location_id → location_tier (1/2/3).
    """
    log.info("Stage 1: Classifying location tiers …")

    tier_cutoff = (today - timedelta(days=180)).isoformat()
    log.info("  Fetching sales since %s for tier classification (chunked) …", tier_cutoff)
    tx_rows = _fetch_chunked_by_date(
        client, "sales_transactions",
        "location_id,sku_id,qty_sold",
        date_col="transaction_date",
        since=tier_cutoff,
    )
    revenue_by_loc:  dict[str, float] = defaultdict(float)
    skus_by_loc:     dict[str, set]   = defaultdict(set)
    returns_by_loc:  dict[str, int]   = defaultdict(int)
    total_tx_by_loc: dict[str, int]   = defaultdict(int)

    for r in tx_rows:
        loc = r.get("location_id", "")
        if not loc:
            continue
        qty = float(r.get("qty_sold") or 0)
        revenue_by_loc[loc] += max(qty, 0.0)   # negative = return; exclude from revenue
        skus_by_loc[loc].add(r.get("sku_id", ""))
        total_tx_by_loc[loc] += 1
        if qty < 0:
            returns_by_loc[loc] += 1

    all_locs = sorted(revenue_by_loc.keys())
    if not all_locs:
        log.warning("No sales_transactions found — cannot classify locations.")
        return {}

    # -- Fill rate from purchase_orders ----------------------------------------
    po_rows = _paginate(
        client, "purchase_orders",
        "location_id,fill_rate_pct",
    )
    fr_sums:  dict[str, float] = defaultdict(float)
    fr_count: dict[str, int]   = defaultdict(int)
    for r in po_rows:
        loc = r.get("location_id", "")
        fr  = r.get("fill_rate_pct")
        if loc and fr is not None:
            fr_sums[loc]  += float(fr)
            fr_count[loc] += 1

    fill_rate: dict[str, float] = {
        loc: (fr_sums[loc] / fr_count[loc]) if fr_count.get(loc) else DEFAULT_FILL_RATE
        for loc in all_locs
    }

    # -- Raw metric vectors ----------------------------------------------------
    revenue    = {loc: revenue_by_loc[loc]          for loc in all_locs}
    breadth    = {loc: len(skus_by_loc[loc])        for loc in all_locs}
    return_rate = {
        loc: returns_by_loc[loc] / max(total_tx_by_loc[loc], 1)
        for loc in all_locs
    }

    def _minmax(vals: dict[str, float]) -> dict[str, float]:
        """Normalise values to [0, 1] via min-max.  Constant → 0.5 for all."""
        mn = min(vals.values())
        mx = max(vals.values())
        rng = mx - mn
        if rng == 0:
            return {k: 0.5 for k in vals}
        return {k: (v - mn) / rng for k, v in vals.items()}

    norm_fill    = _minmax(fill_rate)
    norm_revenue = _minmax(revenue)
    norm_breadth = _minmax({k: float(v) for k, v in breadth.items()})
    norm_return  = _minmax(return_rate)

    # Composite score — return_rate inverted (lower returns = better)
    composite: dict[str, float] = {
        loc: (
            TIER_WEIGHT_FILL_RATE   * norm_fill[loc]
            + TIER_WEIGHT_REVENUE   * norm_revenue[loc]
            + TIER_WEIGHT_SKU_BREADTH * norm_breadth[loc]
            + TIER_WEIGHT_RETURN_RATE * (1.0 - norm_return[loc])
        )
        for loc in all_locs
    }

    # Rank-based tier assignment
    ranked = sorted(all_locs, key=lambda loc: composite[loc], reverse=True)
    n = len(ranked)
    t1_count = math.ceil(n / 3)
    t2_count = math.floor(n / 3)
    # bottom tier gets the remainder

    tier_map: dict[str, int] = {}
    for i, loc in enumerate(ranked):
        if i < t1_count:
            tier_map[loc] = 1
        elif i < t1_count + t2_count:
            tier_map[loc] = 2
        else:
            tier_map[loc] = 3

    log.info(
        "  Locations classified: %d  (Tier 1: %d  Tier 2: %d  Tier 3: %d)",
        n,
        sum(1 for t in tier_map.values() if t == 1),
        sum(1 for t in tier_map.values() if t == 2),
        sum(1 for t in tier_map.values() if t == 3),
    )
    for loc in ranked:
        log.info(
            "  %-10s  Tier %d  composite=%.4f  fill=%.2f  rev=%.0f  breadth=%d  returns=%.2f%%",
            loc, tier_map[loc], composite[loc],
            fill_rate[loc], revenue[loc], breadth[loc], return_rate[loc] * 100,
        )

    if dry_run:
        return tier_map

    # -- Upsert to locations table ---------------------------------------------
    rows = [
        {
            "location_id":          loc,
            "location_tier":        tier_map[loc],
            "fill_rate_score":      round(norm_fill[loc],    4),
            "revenue_score":        round(norm_revenue[loc], 4),
            "sku_breadth_score":    round(norm_breadth[loc], 4),
            "return_rate_score":    round(1.0 - norm_return[loc], 4),
            "composite_tier_score": round(composite[loc],   4),
            "classified_date":      today.isoformat(),
            "updated_at":           "NOW()",
        }
        for loc in all_locs
    ]
    client.table("locations").upsert(
        rows,
        on_conflict="location_id",
    ).execute()
    log.info("  Upserted %d row(s) to locations table.", len(rows))

    return tier_map


# ---------------------------------------------------------------------------
# Stage 2 — Residual demand detection
# ---------------------------------------------------------------------------

def _detect_residual_demand(
    client: Any,
    today: date,
    dry_run: bool = False,
) -> int:
    """Identify third-call / residual demand events and flag them.

    A sales transaction is flagged as residual when ALL of the following hold:
        1.  On the same date and SKU, >= MIN_CONCURRENT_STOCKOUTS other
            locations have is_stockout = TRUE in inventory_snapshots.
        2.  The selling location is NOT stocked out (it has available stock
            that attracts spillover demand from stocked-out peers).
        3.  The sale quantity exceeds SPIKE_MULTIPLIER × the location's
            rolling baseline demand for that SKU over BASELINE_WINDOW_DAYS.

    Flagged transactions receive is_residual_demand = TRUE.  They are then
    excluded from forecast training data by the forecast models.

    Args:
        client:  Active Supabase client.
        today:   Reference date (lookback ends yesterday).
        dry_run: When True, log events but do not update the database.

    Returns:
        Count of transactions flagged as residual.
    """
    log.info("Stage 2: Detecting residual demand events …")

    cutoff = (today - timedelta(days=RESIDUAL_LOOKBACK_DAYS)).isoformat()
    baseline_cutoff = (today - timedelta(days=BASELINE_WINDOW_DAYS)).isoformat()

    # -- Fetch inventory stockouts in the lookback window ----------------------
    inv_rows = _paginate(
        client, "inventory_snapshots",
        "sku_id,location_id,snapshot_date,is_stockout",
        gte_filters={"snapshot_date": cutoff},
        eq_bool={"is_stockout": True},
    )

    # Index: {(sku_id, date_str): {location_ids with stockout}}
    stockouts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in inv_rows:
        sku  = r.get("sku_id", "")
        loc  = r.get("location_id", "")
        dt   = str(r.get("snapshot_date", ""))[:10]
        if sku and loc and dt:
            stockouts[(sku, dt)].add(loc)

    # Only keep (sku, date) pairs with >= MIN_CONCURRENT_STOCKOUTS
    hot_keys: set[tuple[str, str]] = {
        k for k, locs in stockouts.items()
        if len(locs) >= MIN_CONCURRENT_STOCKOUTS
    }

    if not hot_keys:
        log.info("  No concurrent stockout events found — 0 residual flags.")
        return 0

    hot_skus = {k[0] for k in hot_keys}
    log.info(
        "  Concurrent stockout events: %d  (across %d SKU(s))",
        len(hot_keys), len(hot_skus),
    )

    # -- Fetch baseline demand for hot SKUs (last BASELINE_WINDOW_DAYS) --------
    baseline_rows = _paginate(
        client, "sales_transactions",
        "sku_id,location_id,qty_sold",
        in_filters={"sku_id": list(hot_skus)},
        gte_filters={"transaction_date": baseline_cutoff},
    )
    baseline_totals: dict[tuple[str, str], float] = defaultdict(float)
    baseline_counts: dict[tuple[str, str], int]   = defaultdict(int)
    for r in baseline_rows:
        sku = r.get("sku_id", "")
        loc = r.get("location_id", "")
        qty = float(r.get("qty_sold") or 0)
        if sku and loc and qty > 0:
            baseline_totals[(sku, loc)] += qty
            baseline_counts[(sku, loc)] += 1

    # baseline demand = mean daily qty_sold (days with actual sales only)
    baseline: dict[tuple[str, str], float] = {
        pair: (baseline_totals[pair] / baseline_counts[pair])
        for pair in baseline_totals
        if baseline_counts[pair] > 0
    }

    # -- Fetch all transactions for hot (sku, date) pairs ----------------------
    hot_dates = sorted({k[1] for k in hot_keys})
    tx_rows = _paginate(
        client, "sales_transactions",
        "id,sku_id,location_id,transaction_date,qty_sold",
        in_filters={"sku_id": list(hot_skus)},
        gte_filters={"transaction_date": min(hot_dates)},
        lte_filters={"transaction_date": max(hot_dates)},
    )

    # -- Identify residual transactions ----------------------------------------
    residual_ids: list[int] = []
    for r in tx_rows:
        sku  = r.get("sku_id", "")
        loc  = r.get("location_id", "")
        dt   = str(r.get("transaction_date", ""))[:10]
        qty  = float(r.get("qty_sold") or 0)
        txid = r.get("id")

        if not (sku and loc and dt and qty > 0 and txid is not None):
            continue
        if (sku, dt) not in hot_keys:
            continue

        # Selling location must NOT be stocked out itself
        if loc in stockouts.get((sku, dt), set()):
            continue

        # Must exceed baseline by SPIKE_MULTIPLIER
        base = baseline.get((sku, loc), 0.0)
        if base <= 0 or qty < base * SPIKE_MULTIPLIER:
            continue

        residual_ids.append(int(txid))
        log.debug(
            "  RESIDUAL  sku=%-12s  loc=%-8s  date=%s  qty=%.0f  base=%.1f  "
            "stockouts=%s",
            sku, loc, dt, qty, base,
            sorted(stockouts.get((sku, dt), set())),
        )

    log.info("  Residual transactions identified: %d", len(residual_ids))

    if not residual_ids or dry_run:
        if dry_run and residual_ids:
            log.info("  DRY RUN — %d flag(s) not written.", len(residual_ids))
        return len(residual_ids)

    # -- Batch-update is_residual_demand = TRUE --------------------------------
    # Supabase .in_() supports up to 1000 IDs per call; chunk to be safe
    _CHUNK = 500
    for i in range(0, len(residual_ids), _CHUNK):
        chunk = residual_ids[i : i + _CHUNK]
        (
            client.table("sales_transactions")
            .update({"is_residual_demand": True})
            .in_("id", chunk)
            .execute()
        )
    log.info("  Flagged %d transaction(s) as is_residual_demand = TRUE.", len(residual_ids))
    return len(residual_ids)


# ---------------------------------------------------------------------------
# Stage 3 — Demand quality scoring
# ---------------------------------------------------------------------------

def _compute_demand_quality(
    client: Any,
    today: date,
    dry_run: bool = False,
) -> int:
    """Compute demand_quality_score per (SKU, location) and upsert results.

    demand_quality_score = 1.0 − (residual_sale_count / total_sale_count)

    A score of 1.0 means every sale at that location for that SKU was organic
    (never flagged as residual).  A score of 0.0 means every sale was residual.

    Pairs with total_sale_count = 0 are excluded (no score to compute).

    Args:
        client:  Active Supabase client.
        today:   Classification date (written to classified_date).
        dry_run: When True, log results without upserting.

    Returns:
        Count of (SKU, location) pairs scored.
    """
    log.info("Stage 3: Computing demand quality scores …")

    cutoff = (today - timedelta(days=730)).isoformat()
    log.info("  Fetching sales_transactions since %s (chunked by week) …", cutoff)
    tx_rows = _fetch_chunked_by_date(
        client, "sales_transactions",
        "sku_id,location_id,is_residual_demand",
        date_col="transaction_date",
        since=cutoff,
    )

    totals:   dict[tuple[str, str], int] = defaultdict(int)
    residual: dict[tuple[str, str], int] = defaultdict(int)
    for r in tx_rows:
        sku = r.get("sku_id", "")
        loc = r.get("location_id", "")
        if not sku or not loc:
            continue
        totals[(sku, loc)] += 1
        if r.get("is_residual_demand"):
            residual[(sku, loc)] += 1

    if not totals:
        log.warning("  No sales_transactions found — 0 pairs scored.")
        return 0

    rows: list[dict] = []
    for (sku_id, loc_id), total in totals.items():
        res   = residual.get((sku_id, loc_id), 0)
        score = round(1.0 - (res / total), 3)
        score = max(0.0, min(1.0, score))   # clamp to [0.0, 1.0]
        rows.append({
            "sku_id":               sku_id,
            "location_id":          loc_id,
            "demand_quality_score": score,
            "organic_sale_count":   total - res,
            "residual_sale_count":  res,
            "total_sale_count":     total,
            "classified_date":      today.isoformat(),
            "updated_at":           "NOW()",
        })

    low_quality = sum(1 for r in rows if r["demand_quality_score"] < LOW_QUALITY_THRESHOLD)
    log.info(
        "  (SKU, location) pairs scored: %d  "
        "(low quality [< %.1f]: %d)",
        len(rows), LOW_QUALITY_THRESHOLD, low_quality,
    )

    if dry_run:
        log.info("  DRY RUN — %d score(s) not written.", len(rows))
        return len(rows)

    # Upsert in chunks of _PAGE_SIZE
    for i in range(0, len(rows), _PAGE_SIZE):
        chunk = rows[i : i + _PAGE_SIZE]
        client.table("sku_location_demand_quality").upsert(
            chunk,
            on_conflict="sku_id,location_id",
        ).execute()

    log.info("  Upserted %d row(s) to sku_location_demand_quality.", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_classify(dry_run: bool = False) -> int:
    """Execute all three classification stages and write results to the DB.

    Stage 1: Classify location tiers → locations table.
    Stage 2: Detect residual demand  → sales_transactions.is_residual_demand.
    Stage 3: Score demand quality    → sku_location_demand_quality table.

    Args:
        dry_run: When True, compute all classifications but skip DB writes.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    t0 = time.monotonic()
    banner = "=" * 60
    log.info(banner)
    log.info("partswatch-ai — transform.location_classify")
    log.info(
        "  tier_weights: fill=%.0f%%  rev=%.0f%%  breadth=%.0f%%  returns=%.0f%%",
        TIER_WEIGHT_FILL_RATE * 100, TIER_WEIGHT_REVENUE * 100,
        TIER_WEIGHT_SKU_BREADTH * 100, TIER_WEIGHT_RETURN_RATE * 100,
    )
    log.info(
        "  residual: min_concurrent_stockouts=%d  spike_multiplier=%.1fx  "
        "lookback=%dd",
        MIN_CONCURRENT_STOCKOUTS, SPIKE_MULTIPLIER, RESIDUAL_LOOKBACK_DAYS,
    )
    log.info(banner)

    try:
        client = get_client()
    except Exception:
        log.exception("Failed to initialise Supabase client.")
        return 1

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")

    today = date.today()
    log.info("Classification date: %s", today.isoformat())
    log.info("-" * 60)

    try:
        tier_map = _classify_location_tiers(client, today, dry_run=dry_run)
    except Exception:
        log.exception("Stage 1 (location tier classification) failed.")
        return 1

    log.info("-" * 60)

    try:
        residual_count = _detect_residual_demand(client, today, dry_run=dry_run)
    except Exception:
        log.exception("Stage 2 (residual demand detection) failed.")
        return 1

    log.info("-" * 60)

    try:
        quality_count = _compute_demand_quality(client, today, dry_run=dry_run)
    except Exception:
        log.exception("Stage 3 (demand quality scoring) failed.")
        return 1

    elapsed = time.monotonic() - t0
    log.info(banner)
    log.info("Location classification complete  (%.2fs)", elapsed)
    log.info("  Locations tiered:          %d", len(tier_map))
    log.info("  Tier 1 (first-call):       %d", sum(1 for t in tier_map.values() if t == 1))
    log.info("  Tier 2 (second-call):      %d", sum(1 for t in tier_map.values() if t == 2))
    log.info("  Tier 3 (third-call):       %d", sum(1 for t in tier_map.values() if t == 3))
    log.info("  Residual tx flagged:       %d", residual_count)
    log.info("  (SKU, location) pairs:     %d", quality_count)
    if dry_run:
        log.info("  (DRY RUN — no writes made)")
    log.info(banner)
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="partswatch-ai location classification and demand quality scoring.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute classifications without writing to the database.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    try:
        from config import LOG_LEVEL
        setup_logging(LOG_LEVEL)
    except (ImportError, EnvironmentError):
        setup_logging("INFO")

    args = _parse_args()
    return run_classify(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
