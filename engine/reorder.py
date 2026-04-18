"""engine/reorder.py — Converts ML forecasts into purchase-order and
inter-location transfer recommendations for the partswatch-ai purchasing team.

PIPELINE POSITION
-----------------
Runs nightly after ml/forecast_lgbm.py.  Consumes:
  - forecast_results        (lightgbm + rolling_avg for the next 30 days)
  - inventory_snapshots     (most recent on-hand per SKU + location)
  - supplier_scores         (most recent lead time + risk flag per supplier)
  - sku_master              (ABC class + primary supplier routing)
  - purchase_orders         (recent open POs → fallback supplier per SKU)

Writes to:
  - reorder_recommendations (one row per SKU + location, upserted with
                             ignore_duplicates so approved decisions are
                             never overwritten by a same-day re-run)

RECOMMENDATION LOGIC (per SKU × location)
------------------------------------------
1.  Compute avg_daily_forecast from the next 30 days of forecast rows,
    preferring the model appropriate to the SKU's ABC class:
      B-class → lightgbm (falls back to rolling_avg if unavailable)
      C-class → rolling_avg (falls back to lightgbm if unavailable)
      A-class → either (Prophet not yet in nightly pipeline)
    For location-less lightgbm forecasts (location_id='ALL'), the
    network-level prediction is used as-is for all locations of that SKU.

2.  days_of_supply_remaining = qty_on_hand / avg_daily_forecast
    (capped at 9 999.99 when avg_daily_forecast = 0)

3.  reorder_threshold_days = avg_lead_time_days + SAFETY_BUFFER_DAYS (7)

4.  Order needed when days_of_supply_remaining < reorder_threshold_days.

5.  qty_to_order = (avg_daily_forecast × reorder_threshold_days)
                   − qty_on_hand − qty_on_order
    Skipped when qty_to_order < MIN_ORDER_QTY (already covered by open POs).

6.  Transfer check (engine/transfer.py): if another location holds enough
    excess stock, recommend a transfer instead of an external PO.
    Transfers always take priority.

7.  Supplier routing: primary_supplier_id from sku_master.
    Fallback: most recent non-cancelled purchase_orders entry for that SKU.
    Red risk_flag suppliers are flagged in the output row.

8.  Urgency tiers based on days_of_supply_remaining:
      critical  < 3 days
      warning   3 – 6.99 days
      normal    ≥ 7 days

Usage
-----
    python -m engine.reorder            # live run
    python -m engine.reorder --dry-run  # compute and log, skip DB writes
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from engine.transfer import find_transfer_source
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Excluded locations
# ---------------------------------------------------------------------------
# These locations must never appear as a reorder destination OR as a
# transfer source.  Filtering them at the inventory-fetch layer ensures
# they're absent from every downstream computation (recommendations,
# transfer routing, basket co-purchases, urgency rollups).
#
#   LOC-021 (INTERNET) is a virtual storefront, not a physical
#     warehouse — it has no shelves to stock.
#   LOC-014, LOC-019, LOC-022, LOC-023, LOC-030, LOC-031 are retired
#     stores still present in legacy snapshots.
EXCLUDED_LOCATIONS: frozenset[str] = frozenset({
    "LOC-021",
    "LOC-014", "LOC-019", "LOC-022", "LOC-023", "LOC-030", "LOC-031",
})

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

FORECAST_HORIZON_DAYS: int = 30
"""How many calendar days of forecasts to aggregate for avg_daily_forecast."""

SAFETY_BUFFER_DAYS: int = 7
"""Days of buffer added to avg_lead_time_days when computing reorder threshold."""

DEFAULT_LEAD_TIME_DAYS: float = 7.0
"""Assumed lead time when a supplier has no avg_lead_time_days on record."""

MIN_ORDER_QTY: float = 0.01
"""qty_to_order must exceed this to generate a recommendation (open POs may
already cover the gap, so very small computed orders are silently skipped)."""

DAYS_OF_SUPPLY_CAP: float = 9_999.99
"""days_of_supply_remaining stored when avg_daily_forecast = 0 (infinite supply)."""

URGENCY_CRITICAL_DAYS: float = 3.0
"""days_of_supply_remaining < this → urgency = 'critical'."""

URGENCY_WARNING_DAYS: float = 7.0
"""days_of_supply_remaining in [CRITICAL, WARNING) → urgency = 'warning'."""

# ---------------------------------------------------------------------------
# Safety stock multipliers by XYZ class
# ---------------------------------------------------------------------------
# Applied to qty_to_order so that erratic SKUs (Z) carry more buffer than
# consistent SKUs (X).  None handles SKUs whose xyz_class is NULL —
# either UNKNOWN (insufficient history) or sku_master row not yet
# reclassified after first appearance — and uses a moderate buffer.
#
#   X    1.0x — demand is stable, rely on the base reorder formula
#   Y    1.3x — moderate variability, +30% buffer
#   Z    1.8x — highly erratic, +80% buffer
#   None 1.2x — unknown, slight conservative bump
XYZ_SAFETY_MULTIPLIERS: dict[str | None, float] = {
    "X":  1.0,
    "Y":  1.3,
    "Z":  1.8,
    None: 1.2,
}

BASKET_CONFIDENCE_THRESHOLD: float = 0.30
"""Minimum confidence for a basket rule to trigger a co-purchase recommendation."""

BASKET_LOW_STOCK_DAYS: float = 14.0
"""If a consequent SKU has fewer than this many days of supply, include it."""

_PAGE_SIZE: int = 1_000
"""Supabase PostgREST page size for paginated fetches."""

SKU_BATCH_SIZE: int = 1_000
"""How many SKUs to pass to .in_() in a single forecast / PO fetch query.
Keeps each query inside Supabase's statement timeout (error 57014)."""

_RECONCILE_THRESHOLD: float = 0.5
"""Relative-difference cutoff for AI-vs-buyer Min-Qty reconciliation.

When |ai_reorder_level - buyer_min_qty| / max(buyer_min_qty, 1) exceeds
this value, the discrepancy is logged to data_quality_issues so buyers
can review.  0.5 = 50% relative gap — chosen to surface only material
disagreements, not routine rounding noise."""

WRITE_BATCH_SIZE: int = 200
"""How many recommendation rows to upsert per network round-trip.
A single 19k-row upsert blows past Supabase's statement timeout (57014);
200-row batches finish well inside the timeout window and let us retry
individual batches on transient failure without losing prior progress."""

AT_RISK_COVERAGE_DAYS: float = 60.0
"""Pre-filter threshold: a (SKU, location) with proxy days-of-supply
(qty_on_hand / (avg_weekly_units / 7)) > this many days is clearly NOT
at risk of stockout in the 30-day forecast horizon and is pruned BEFORE
we fetch any forecasts.  Double the max reasonable reorder threshold
(lead_time + safety_buffer ≈ 14–21 days) so we never miss a real
at-risk SKU even if the true forecast spikes well above the rolling
historical average."""

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

# Preferred forecast model by ABC class.
# A-class will use whatever is available (Prophet added later).
_CLASS_MODEL_PREF: dict[str, list[str]] = {
    "A": ["lightgbm", "rolling_avg"],
    "B": ["lightgbm", "rolling_avg"],
    "C": ["rolling_avg", "lightgbm"],
}
_DEFAULT_MODEL_ORDER: list[str] = ["lightgbm", "rolling_avg"]


# ---------------------------------------------------------------------------
# Fetch helpers — all paginated, retry on transient errors
# ---------------------------------------------------------------------------

def _is_retryable_error(exc: Exception) -> bool:
    """True if exc looks like a Supabase timeout or dropped-connection error."""
    blob = type(exc).__name__ + " " + str(exc)
    return any(tok in blob for tok in _RETRYABLE_TOKENS)


def _get_fresh_client() -> Any:
    """Return a brand-new Supabase client (bypasses lru_cache when available)."""
    try:
        from db.connection import get_new_client
        return get_new_client()
    except ImportError:
        from db.connection import get_client
        return get_client()


def _paginate(client_holder: list, table: str, select: str,
              filters: dict | None = None,
              gte_filters: dict | None = None,
              lte_filters: dict | None = None,
              in_filters: dict | None = None,
              order_by: list[tuple[str, bool]] | None = None) -> list[dict]:
    """Generic paginated fetch with retry + reconnect on transient errors.

    Args:
        client_holder: Single-element list holding the active Supabase
                       client.  The reference is replaced in-place when a
                       retryable error forces a reconnect.
        table:         Table name.
        select:        PostgREST column selector.
        filters:       {column: exact_value} equality filters.
        gte_filters:   {column: value} for column >= value.
        lte_filters:   {column: value} for column <= value.
        in_filters:    {column: [values]} for column IN (values).
        order_by:      Optional list of (column, desc) tuples to apply
                       with `.order(...)`.  Pass an explicit ordering on
                       offset-paged queries that span large tables and
                       could observe concurrent writes — the inventory
                       fetch is the prime example.  Defaults to no
                       ordering: forcing ORDER BY on already-bounded
                       SKU-batched queries (e.g. forecast_results, where
                       a single batch can pull >1M rows) makes Postgres
                       perform a top-level Sort and times out (57014).
                       The downstream dedupe by run_date / snapshot_date
                       already handles read-time row shuffling in those
                       cases.

    Returns:
        All matching rows as a list of dicts.
    """
    if order_by is None:
        order_by = []
    rows: list[dict] = []
    offset = 0
    while True:
        page: list[dict] | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                q = client_holder[0].table(table).select(select)
                for col, val in (filters or {}).items():
                    q = q.eq(col, val)
                for col, val in (gte_filters or {}).items():
                    q = q.gte(col, val)
                for col, val in (lte_filters or {}).items():
                    q = q.lte(col, val)
                for col, vals in (in_filters or {}).items():
                    q = q.in_(col, vals)
                for col, desc in order_by:
                    q = q.order(col, desc=desc)
                page = q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
                break
            except Exception as exc:
                if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                    log.warning(
                        "  %s fetch retry %d/%d (offset=%d): %s — reconnecting in %.0fs …",
                        table, attempt, _MAX_RETRIES, offset,
                        type(exc).__name__, _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _get_fresh_client()
                    continue
                raise
        assert page is not None
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def _fetch_skus(client_holder: list) -> dict[str, dict]:
    """Return all active SKUs keyed by sku_id.

    Tries to fetch ``primary_supplier_id`` (added by migration 007) and
    ``avg_weekly_units`` (populated by transform/derive.py).  The latter is
    used by the at-risk pre-filter so we only fetch forecasts for SKUs that
    could plausibly trigger a reorder.

    Returns:
        {sku_id: {abc_class, primary_supplier_id, avg_weekly_units}}
    """
    from postgrest.exceptions import APIError  # local import — avoid polluting module ns

    # xyz_class drives the safety-stock multiplier (XYZ_SAFETY_MULTIPLIERS);
    # primary_supplier_id drives supplier routing.  Either column may be
    # absent if its migration has not been applied.  We try the richest
    # projection first and fall back step-by-step rather than parsing
    # PostgREST error text (which only ever names one missing column at
    # a time, so a single-pass branch on substring is unreliable when
    # both are missing).
    fallback_chain: list[tuple[str, str]] = [
        ("sku_id,abc_class,xyz_class,primary_supplier_id,avg_weekly_units", ""),
        ("sku_id,abc_class,xyz_class,avg_weekly_units",
            "Column sku_master.primary_supplier_id not found — run migration 007 "
            "to enable primary supplier routing.  Falling back to purchase_orders-"
            "derived suppliers only."),
        ("sku_id,abc_class,primary_supplier_id,avg_weekly_units",
            "Column sku_master.xyz_class not found — apply migration 010 to enable "
            "XYZ-based safety stock multipliers.  Defaulting all SKUs to the "
            "unknown-class multiplier."),
        ("sku_id,abc_class,avg_weekly_units",
            "Both xyz_class and primary_supplier_id missing — apply migrations "
            "007 and 010 for full reorder functionality.  Using minimum projection."),
    ]

    rows: list[dict] | None = None
    for select, warn_msg in fallback_chain:
        try:
            rows = _paginate(
                client_holder, "sku_master", select,
                filters={"is_active": True},
            )
            if warn_msg:
                log.warning(warn_msg)
            break
        except APIError as exc:
            # Only swallow "missing column" errors — anything else is a
            # real problem (auth, connectivity, etc.) and must surface.
            msg = str(exc).lower()
            if "column" not in msg and "does not exist" not in msg:
                raise
            continue
    if rows is None:
        raise RuntimeError(
            "sku_master is missing core columns (sku_id / abc_class / "
            "avg_weekly_units) — re-run migration 001."
        )

    return {
        r["sku_id"]: {
            "abc_class":           r.get("abc_class"),
            "xyz_class":           r.get("xyz_class"),         # None if column or value missing
            "primary_supplier_id": r.get("primary_supplier_id"),  # None if migration pending
            "avg_weekly_units":    (
                float(r["avg_weekly_units"])
                if r.get("avg_weekly_units") is not None
                else 0.0
            ),
        }
        for r in rows
        if r.get("sku_id")
    }


def _fetch_inventory(
    client_holder: list,
    sku_ids: list[str] | set[str],
    lookback_days: int = 7,
) -> dict[tuple[str, str], dict]:
    """Return the most recent inventory snapshot per (sku_id, location_id).

    Fetches snapshots from the last ``lookback_days`` days, batched in
    ``SKU_BATCH_SIZE`` chunks via ``.in_("sku_id", batch)`` to keep each
    query inside Supabase's statement timeout (error 57014).  Without
    SKU-batching, a full 6.7M-row OFFSET scan of inventory_snapshots
    hangs indefinitely (O(n²) at depth).

    Args:
        client_holder: Single-element list holding the active Supabase
                       client (replaced in-place on transient errors).
        sku_ids:       Active SKU IDs from _fetch_skus.  Snapshots for
                       SKUs not in this set are skipped at the source.
        lookback_days: Window for recent snapshot fetch.

    Returns:
        {(sku_id, location_id): {qty_on_hand, qty_on_order, reorder_point,
                                 snapshot_date}}
    """
    if not sku_ids:
        return {}

    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    sku_list = list(sku_ids)
    n_batches = math.ceil(len(sku_list) / SKU_BATCH_SIZE)

    latest: dict[tuple[str, str], dict] = {}
    skipped_excluded = 0
    total_rows = 0

    for i in range(n_batches):
        sku_batch = sku_list[i * SKU_BATCH_SIZE:(i + 1) * SKU_BATCH_SIZE]
        rows = _paginate(
            client_holder, "inventory_snapshots",
            "sku_id,location_id,snapshot_date,qty_on_hand,qty_on_order,reorder_point",
            gte_filters={"snapshot_date": cutoff},
            in_filters={"sku_id": sku_batch},
            # Newest snapshots first — first-write-wins dedupe below; id
            # is the stable tiebreaker for offset paging within the batch.
            order_by=[("snapshot_date", True), ("id", False)],
        )
        for r in rows:
            loc_id = r["location_id"]
            if loc_id in EXCLUDED_LOCATIONS:
                skipped_excluded += 1
                continue
            key = (r["sku_id"], loc_id)
            existing = latest.get(key)
            if existing is None or r["snapshot_date"] > existing["snapshot_date"]:
                # reorder_point may legitimately be NULL for SKUs the
                # buyers haven't manually set a Min Qty on — keep the
                # None so downstream reconciliation can skip those pairs.
                rp_raw = r.get("reorder_point")
                latest[key] = {
                    "qty_on_hand":   float(r.get("qty_on_hand") or 0),
                    "qty_on_order":  float(r.get("qty_on_order") or 0),
                    "reorder_point": float(rp_raw) if rp_raw is not None else None,
                    "snapshot_date": r["snapshot_date"],
                }
        total_rows += len(rows)
        log.info(
            "  [REORDER] Inventory batch %d/%d — skus=%d rows=%d "
            "(running: %d snapshots, %d unique (sku,loc) pairs)",
            i + 1, n_batches, len(sku_batch), len(rows),
            total_rows, len(latest),
        )

    if skipped_excluded:
        log.info(
            "Excluded %d snapshot rows from %d virtual/retired locations: %s",
            skipped_excluded, len(EXCLUDED_LOCATIONS), sorted(EXCLUDED_LOCATIONS),
        )
    return latest


def _fetch_suppliers(client_holder: list, lookback_days: int = 90) -> dict[str, dict]:
    """Return the most recent supplier score per supplier_id.

    Args:
        client:        Active Supabase client.
        lookback_days: Window for recent supplier score fetch.

    Returns:
        {supplier_id: {avg_lead_time_days, risk_flag, composite_score}}
    """
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    rows = _paginate(
        client_holder, "supplier_scores",
        "supplier_id,score_date,avg_lead_time_days,risk_flag,composite_score",
        gte_filters={"score_date": cutoff},
    )
    latest: dict[str, dict] = {}
    for r in rows:
        sid = r.get("supplier_id")
        if not sid:
            continue
        existing = latest.get(sid)
        if existing is None or r.get("score_date", "") > existing.get("score_date", ""):
            latest[sid] = {
                "avg_lead_time_days": (
                    float(r["avg_lead_time_days"])
                    if r.get("avg_lead_time_days") is not None
                    else None
                ),
                "risk_flag":       r.get("risk_flag"),
                "composite_score": (
                    float(r["composite_score"])
                    if r.get("composite_score") is not None
                    else None
                ),
                "score_date": r.get("score_date", ""),
            }
    return latest


def _fetch_forecasts(
    client_holder: list,
    today: date,
    horizon_end: date,
    at_risk_skus: list[str],
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Return the most recent forecast rows for the at-risk SKUs only.

    The forecast_results table is ~1.6M rows (248K C-class + 30K B-class SKUs
    × ~23 locations × 30 days); fetching all of them times out with
    Supabase error 57014.  Reorder never needs forecasts for SKUs that are
    clearly not at risk of stockout, so we restrict the fetch to the SKUs
    identified by ``_compute_at_risk_skus`` and batch those in
    ``SKU_BATCH_SIZE`` chunks.

    Args:
        client_holder: Single-element list holding the active client.
        today:         First day of the forecast horizon (inclusive).
        horizon_end:   Last day of the forecast horizon (inclusive).
        at_risk_skus:  SKU IDs that the pre-filter flagged as at-risk.

    Returns:
        {(sku_id, location_id, model_type): {date_str: predicted_qty}}
    """
    if not at_risk_skus:
        return {}

    all_rows: list[dict] = []
    n_batches = math.ceil(len(at_risk_skus) / SKU_BATCH_SIZE)
    for i in range(n_batches):
        sku_batch = at_risk_skus[i * SKU_BATCH_SIZE:(i + 1) * SKU_BATCH_SIZE]
        page = _paginate(
            client_holder, "forecast_results",
            "sku_id,location_id,forecast_date,model_type,predicted_qty,run_date",
            gte_filters={"forecast_date": today.isoformat()},
            lte_filters={"forecast_date": horizon_end.isoformat()},
            in_filters={
                "sku_id":     sku_batch,
                "model_type": ["lightgbm", "rolling_avg"],
            },
        )
        all_rows.extend(page)
        log.info(
            "  [REORDER] Forecast batch %d/%d — skus=%d rows=%d "
            "(running total %d)",
            i + 1, n_batches, len(sku_batch), len(page), len(all_rows),
        )

    # Pass 1: find the latest run_date per (sku_id, location_id, model_type)
    latest_run: dict[tuple[str, str, str], str] = {}
    for r in all_rows:
        key = (r["sku_id"], r["location_id"], r["model_type"])
        run = r.get("run_date", "")
        if run > latest_run.get(key, ""):
            latest_run[key] = run

    # Pass 2: build date → qty maps for the latest run only
    forecast_map: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in all_rows:
        key = (r["sku_id"], r["location_id"], r["model_type"])
        if r.get("run_date") == latest_run.get(key):
            d = str(r.get("forecast_date", ""))[:10]
            if d:
                forecast_map[key][d] = float(r.get("predicted_qty") or 0)

    return dict(forecast_map)


def _fetch_fallback_suppliers(
    client_holder: list,
    at_risk_skus: list[str],
    lookback_days: int = 180,
) -> dict[str, str]:
    """Return the most recent supplier_id per sku_id from purchase_orders.

    Used as a fallback when sku_master.primary_supplier_id is NULL.  Scoped
    to the at-risk SKU set and batched to stay within the statement timeout.

    Args:
        client_holder: Single-element list holding the active client.
        at_risk_skus:  SKUs that could trigger a reorder recommendation.
        lookback_days: How far back to scan POs.

    Returns:
        {sku_id: supplier_id}
    """
    if not at_risk_skus:
        return {}

    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    all_rows: list[dict] = []
    n_batches = math.ceil(len(at_risk_skus) / SKU_BATCH_SIZE)
    for i in range(n_batches):
        sku_batch = at_risk_skus[i * SKU_BATCH_SIZE:(i + 1) * SKU_BATCH_SIZE]
        page = _paginate(
            client_holder, "purchase_orders",
            "sku_id,supplier_id,po_date",
            gte_filters={"po_date": cutoff},
            in_filters={
                "sku_id": sku_batch,
                "status": ["open", "received", "partial"],
            },
        )
        all_rows.extend(page)

    latest_po: dict[str, dict] = {}
    for r in all_rows:
        sku = r.get("sku_id")
        if not sku:
            continue
        existing = latest_po.get(sku)
        if existing is None or r.get("po_date", "") > existing.get("po_date", ""):
            latest_po[sku] = r
    return {sku: info["supplier_id"] for sku, info in latest_po.items()
            if info.get("supplier_id")}


# ---------------------------------------------------------------------------
# At-risk pre-filter — identifies SKUs that could plausibly need reorder
# ---------------------------------------------------------------------------

def _compute_at_risk_skus(
    inventory: dict[tuple[str, str], dict],
    skus: dict[str, dict],
    suppliers: dict[str, dict],
) -> set[str]:
    """Return the set of SKU IDs that may need a reorder recommendation.

    Pre-filter rules (a SKU is flagged if ANY of its (SKU, location) pairs
    triggers ONE of these):

    1. **Zero on-hand** — qty_on_hand <= 0 is an actual stockout; always
       flag regardless of proxy demand.  Covers new SKUs that have no
       historical avg_weekly_units yet but do have forward forecasts.
    2. **Zero on-hand + on-order** — same rationale.
    3. **Low proxy days-of-supply** — qty_on_hand / (avg_weekly_units/7)
       below a dynamic threshold derived from the longest supplier lead
       time observed plus a 2× safety margin.  This guarantees that no
       SKU with a plausible shortfall (even accounting for the longest
       lead time in the network) is pruned.

    Args:
        inventory: Output of _fetch_inventory (latest snapshot per pair).
        skus:      Output of _fetch_skus (must include ``avg_weekly_units``).
        suppliers: Output of _fetch_suppliers (for max lead time lookup).

    Returns:
        Set of sku_ids that need forecast data fetched.
    """
    # Derive a safety-preserving upper bound from the observed lead-time
    # distribution.  threshold = max(constant floor, 2 × (max_lead + buffer)).
    max_lead = DEFAULT_LEAD_TIME_DAYS
    for info in suppliers.values():
        lt = info.get("avg_lead_time_days")
        if lt is not None and lt > max_lead:
            max_lead = lt
    dynamic_threshold = max(
        AT_RISK_COVERAGE_DAYS,
        2.0 * (max_lead + SAFETY_BUFFER_DAYS),
    )

    at_risk: set[str] = set()
    stockout_flagged = 0
    proxy_flagged = 0
    safe_count = 0

    for (sku_id, _loc_id), inv in inventory.items():
        sku_info = skus.get(sku_id)
        if not sku_info:
            continue  # inactive / unknown SKU

        qty_on_hand  = inv["qty_on_hand"]
        qty_on_order = inv["qty_on_order"]

        # Rule 1 & 2: actual stockout always wins, even with no historical demand.
        if qty_on_hand <= 0 or (qty_on_hand + qty_on_order) <= 0:
            at_risk.add(sku_id)
            stockout_flagged += 1
            continue

        # Rule 3: proxy days-of-supply check (requires historical demand).
        avg_weekly = sku_info.get("avg_weekly_units") or 0.0
        if avg_weekly <= 0:
            # Positive stock and no historical demand — rolling_avg forecast
            # will be ~0 and cannot trigger reorder; safe to skip.
            safe_count += 1
            continue

        avg_daily_proxy = avg_weekly / 7.0
        days_of_supply_proxy = qty_on_hand / avg_daily_proxy

        if days_of_supply_proxy < dynamic_threshold:
            at_risk.add(sku_id)
            proxy_flagged += 1
        else:
            safe_count += 1

    log.info(
        "  At-risk pre-filter:  %d SKU(s) flagged  "
        "(stockout=%d, proxy<%.0fd=%d, safe=%d, max_lead=%.0fd)",
        len(at_risk), stockout_flagged, dynamic_threshold,
        proxy_flagged, safe_count, max_lead,
    )
    return at_risk


# ---------------------------------------------------------------------------
# Forecast selection helpers
# ---------------------------------------------------------------------------

def _select_forecasts(
    sku_id: str,
    location_id: str,
    abc_class: str | None,
    forecast_map: dict[tuple[str, str, str], dict[str, float]],
    today: date,
    horizon_end: date,
) -> tuple[list[float], str | None]:
    """Pick the best available daily forecast series for a (SKU, location) pair.

    Tries models in preference order for the SKU's ABC class.  For lightgbm,
    also falls back from a location-specific key to the network-level
    (location_id='ALL') key if no per-location forecast exists.

    Args:
        sku_id:        SKU identifier.
        location_id:   Location identifier.
        abc_class:     'A', 'B', 'C', or None.
        forecast_map:  {(sku_id, location_id, model_type): {date_str: qty}}.
        today:         First day of the forecast horizon.
        horizon_end:   Last day of the forecast horizon.

    Returns:
        (daily_qty_list, model_type_used) where daily_qty_list is a list of
        FORECAST_HORIZON_DAYS floats (one per calendar day), or ([], None)
        when no forecast is available.
    """
    model_order = _CLASS_MODEL_PREF.get(abc_class or "", _DEFAULT_MODEL_ORDER)

    for model in model_order:
        # Try exact location match first.
        key = (sku_id, location_id, model)
        date_qty = forecast_map.get(key)

        # For lightgbm: also try the network-level 'ALL' key.
        if date_qty is None and model == "lightgbm":
            key_all = (sku_id, "ALL", model)
            date_qty = forecast_map.get(key_all)

        if date_qty is None:
            continue

        # Build an ordered list of daily quantities over the horizon.
        daily: list[float] = []
        current = today
        while current <= horizon_end:
            daily.append(date_qty.get(current.isoformat(), 0.0))
            current += timedelta(days=1)

        if not daily:
            continue

        return daily, model

    return [], None


# ---------------------------------------------------------------------------
# Basket-triggered co-purchase recommendations
# ---------------------------------------------------------------------------

def _add_basket_triggered(
    client_holder: list,
    recommendations: list[dict],
    inventory_summary: dict[tuple[str, str], dict],
    today: date,
    counts: dict[str, int],
) -> int:
    already_recommended = {
        (r["sku_id"], r["location_id"]) for r in recommendations
    }

    try:
        rules = _paginate(
            client_holder, "basket_rules",
            "antecedent_sku,consequent_sku,confidence,lift",
            gte_filters={"confidence": str(BASKET_CONFIDENCE_THRESHOLD)},
        )
    except Exception:
        log.warning("basket_rules table not available — skipping basket enhancement.")
        return 0

    if not rules:
        log.info("No basket rules found — skipping basket-triggered recommendations.")
        return 0

    trigger_skus = {r["sku_id"] for r in recommendations}

    rule_map: dict[str, list[dict]] = defaultdict(list)
    for r in rules:
        rule_map[r["antecedent_sku"]].append(r)

    rule_antecedents = set(rule_map.keys())
    matched_antecedents = rule_antecedents & trigger_skus
    log.info(
        "BASKET-DEBUG: %d rules, %d unique antecedents, %d in today's recs (overlap=%s)",
        len(rules), len(rule_antecedents), len(matched_antecedents),
        sorted(matched_antecedents) or "—",
    )
    if not matched_antecedents:
        log.info(
            "BASKET-DEBUG: no antecedents are at-risk this run; rules cannot fire. "
            "Antecedents not in recs: %s",
            sorted(rule_antecedents - trigger_skus),
        )

    added = 0
    for trigger_sku in trigger_skus:
        if trigger_sku not in rule_map:
            continue
        for rule in rule_map[trigger_sku]:
            cons_sku = rule["consequent_sku"]

            cons_positions = [
                (sk, lc, s) for (sk, lc), s in inventory_summary.items() if sk == cons_sku
            ]
            if not cons_positions:
                log.info(
                    "BASKET-DEBUG: trigger=%s -> cons=%s  conf=%.2f  SKIP: consequent absent from inventory_summary",
                    trigger_sku, cons_sku, float(rule["confidence"]),
                )
                continue

            for (sku, loc), summary in inventory_summary.items():
                if sku != cons_sku:
                    continue
                if (sku, loc) in already_recommended:
                    log.info(
                        "BASKET-DEBUG: trigger=%s -> cons=%s @ %s  SKIP: already recommended",
                        trigger_sku, cons_sku, loc,
                    )
                    continue

                days_supply = summary["days_of_supply_remaining"]
                if days_supply >= BASKET_LOW_STOCK_DAYS:
                    log.info(
                        "BASKET-DEBUG: trigger=%s -> cons=%s @ %s  conf=%.2f  days_supply=%.1f  SKIP: above %.1fd threshold",
                        trigger_sku, cons_sku, loc, float(rule["confidence"]),
                        days_supply, BASKET_LOW_STOCK_DAYS,
                    )
                    continue

                avg_daily = summary["avg_daily_forecast"]
                qty_on_hand = summary["qty_on_hand"]
                qty_on_order = summary["qty_on_order"]
                reorder_threshold = summary["reorder_threshold"]
                xyz_class = summary.get("xyz_class")
                safety_multiplier = XYZ_SAFETY_MULTIPLIERS.get(
                    xyz_class, XYZ_SAFETY_MULTIPLIERS[None]
                )

                demand_over_coverage = avg_daily * reorder_threshold * safety_multiplier
                qty_to_order = max(0.0, demand_over_coverage - qty_on_hand - qty_on_order)
                if qty_to_order < MIN_ORDER_QTY:
                    continue

                if days_supply < URGENCY_CRITICAL_DAYS:
                    urgency = "critical"
                elif days_supply < URGENCY_WARNING_DAYS:
                    urgency = "warning"
                else:
                    urgency = "normal"

                rec = {
                    "sku_id":                   sku,
                    "location_id":              loc,
                    "recommendation_date":      today.isoformat(),
                    "qty_to_order":             round(qty_to_order, 4),
                    "supplier_id":              summary.get("supplier_id"),
                    "recommendation_type":      "basket_triggered",
                    "transfer_from_location":   None,
                    "days_of_supply_remaining": round(days_supply, 2),
                    "urgency":                  urgency,
                    "forecast_model_used":      summary.get("model_used"),
                    "supplier_risk_flag":       summary.get("supplier_risk_flag"),
                    "is_approved":              False,
                    "approved_by":              None,
                    "approved_at":              None,
                }
                recommendations.append(rec)
                already_recommended.add((sku, loc))
                counts[urgency] += 1
                added += 1

                log.info(
                    "  BASKET %-12s %-10s  trigger=%-12s  conf=%.0f%%  lift=%.1fx  qty=%.2f",
                    sku, loc, trigger_sku,
                    float(rule["confidence"]) * 100,
                    float(rule["lift"]),
                    qty_to_order,
                )

    if added:
        log.info("Basket-triggered recommendations added: %d", added)
    else:
        log.info("No basket-triggered recommendations generated.")
    return added


# ---------------------------------------------------------------------------
# Core recommendation engine
# ---------------------------------------------------------------------------

def run_reorder(dry_run: bool = False) -> int:
    """Execute the full reorder recommendation pipeline.

    Args:
        dry_run: When True, all computation runs but no rows are written to
                 reorder_recommendations.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    t0 = time.monotonic()

    banner = "=" * 60
    log.info(banner)
    log.info("partswatch-ai — engine.reorder")
    log.info(
        "  horizon=%dd  safety_buffer=%dd  lead_default=%.0fd",
        FORECAST_HORIZON_DAYS, SAFETY_BUFFER_DAYS, DEFAULT_LEAD_TIME_DAYS,
    )
    log.info(banner)

    try:
        client_holder: list = [_get_fresh_client()]
    except Exception:
        log.exception("Failed to initialise Supabase client.")
        return 1

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")

    today      = date.today()
    horizon_end = today + timedelta(days=FORECAST_HORIZON_DAYS - 1)

    log.info("Recommendation date: %s", today.isoformat())
    log.info("Forecast window:     %s → %s", today.isoformat(), horizon_end.isoformat())
    log.info("-" * 60)

    # ------------------------------------------------------------------
    # 1. Bulk data fetch — SKU master, inventory, supplier scores
    # ------------------------------------------------------------------
    try:
        log.info("Fetching active SKUs from sku_master …")
        skus = _fetch_skus(client_holder)
        log.info("  Active SKUs: %d", len(skus))

        log.info("Fetching inventory snapshots (SKU-batched) …")
        inventory = _fetch_inventory(client_holder, list(skus.keys()))
        log.info("  (SKU, location) inventory pairs: %d", len(inventory))

        log.info("Fetching supplier scores …")
        suppliers = _fetch_suppliers(client_holder)
        log.info("  Suppliers with scores: %d", len(suppliers))

        # --------------------------------------------------------------
        # 1b. At-risk pre-filter — the forecast_results table has ~1.6M
        #     rows for the 30-day horizon; fetching all of them times
        #     out (Supabase error 57014).  We only need forecasts for
        #     SKUs whose proxy days-of-supply is below the threshold.
        # --------------------------------------------------------------
        log.info("Computing at-risk SKU set from inventory + avg_weekly_units …")
        at_risk_set = _compute_at_risk_skus(inventory, skus, suppliers)
        at_risk_list = sorted(at_risk_set)

        log.info(
            "Fetching forecasts (lightgbm + rolling_avg, next %dd) for %d at-risk SKUs …",
            FORECAST_HORIZON_DAYS, len(at_risk_list),
        )
        forecast_map = _fetch_forecasts(
            client_holder, today, horizon_end, at_risk_list,
        )
        unique_pairs_with_forecast = len({(k[0], k[1]) for k in forecast_map})
        log.info("  Forecast series fetched: %d (across %d SKU×location pairs)",
                 len(forecast_map), unique_pairs_with_forecast)

        log.info("Fetching fallback supplier map from purchase_orders …")
        fallback_suppliers = _fetch_fallback_suppliers(client_holder, at_risk_list)
        log.info("  SKUs with PO-derived fallback supplier: %d", len(fallback_suppliers))

    except Exception:
        log.exception("Fatal error during data fetch phase.")
        return 1

    # NOTE: after this point we must always dereference client_holder[0]
    # at the point of use.  Any retry inside _paginate (including the
    # basket-rules fetch below) may swap client_holder[0] for a fresh
    # client, so caching the reference here would risk using a dead
    # connection for the final upsert.

    # ------------------------------------------------------------------
    # 2. Build inventory_summary — compute derived fields for every pair
    # ------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Computing inventory positions …")

    inventory_summary: dict[tuple[str, str], dict] = {}

    for (sku_id, loc_id), inv in inventory.items():
        sku_info = skus.get(sku_id)
        if not sku_info:
            continue  # SKU not in sku_master (inactive or unknown)

        abc_class          = sku_info.get("abc_class")
        primary_supplier   = sku_info.get("primary_supplier_id")
        supplier_id        = primary_supplier or fallback_suppliers.get(sku_id)
        supplier_info      = suppliers.get(supplier_id or "") if supplier_id else {}
        avg_lead_time      = (
            float(supplier_info.get("avg_lead_time_days") or DEFAULT_LEAD_TIME_DAYS)
            if supplier_info and supplier_info.get("avg_lead_time_days") is not None
            else DEFAULT_LEAD_TIME_DAYS
        )

        daily_forecasts, model_used = _select_forecasts(
            sku_id, loc_id, abc_class, forecast_map, today, horizon_end,
        )
        if not daily_forecasts:
            continue  # No forecast available — cannot generate a recommendation

        avg_daily = sum(daily_forecasts) / max(len(daily_forecasts), 1)
        qty_on_hand  = inv["qty_on_hand"]
        qty_on_order = inv["qty_on_order"]

        if avg_daily > 0:
            days_of_supply = qty_on_hand / avg_daily
        else:
            days_of_supply = DAYS_OF_SUPPLY_CAP  # Effectively infinite

        inventory_summary[(sku_id, loc_id)] = {
            "qty_on_hand":             qty_on_hand,
            "qty_on_order":            qty_on_order,
            "avg_daily_forecast":      avg_daily,
            "days_of_supply_remaining": days_of_supply,
            "avg_lead_time_days":      avg_lead_time,
            "reorder_threshold":       avg_lead_time + SAFETY_BUFFER_DAYS,
            "buyer_reorder_point":     inv.get("reorder_point"),
            "supplier_id":             supplier_id,
            "supplier_risk_flag":      (
                supplier_info.get("risk_flag") if supplier_info else None
            ),
            "abc_class":               abc_class,
            "xyz_class":               sku_info.get("xyz_class"),  # may be None
            "model_used":              model_used,
        }

    log.info("  Pairs with actionable forecasts: %d", len(inventory_summary))

    # ------------------------------------------------------------------
    # 3. Generate recommendations
    # ------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Generating recommendations …")

    recommendations: list[dict] = []
    counts: dict[str, int] = {
        "critical": 0, "warning": 0, "normal": 0,
        "transfer": 0, "po": 0, "skipped_covered": 0,
    }
    # Min-Qty reconciliation accumulator.  Each entry will be written to the
    # data_quality_issues table after the recommendation loop finishes so we
    # only pay one round-trip cost for the whole catalog.  See
    # _RECONCILE_THRESHOLD for the relative-difference cutoff that flags a
    # mismatch between the AI-derived reorder level and the buyer's Min Qty.
    quality_issues: list[dict] = []

    for (sku_id, loc_id), summary in sorted(inventory_summary.items()):
        days_supply      = summary["days_of_supply_remaining"]
        reorder_threshold = summary["reorder_threshold"]

        if days_supply >= reorder_threshold:
            continue  # Sufficiently stocked — no action needed

        avg_daily    = summary["avg_daily_forecast"]
        qty_on_hand  = summary["qty_on_hand"]
        qty_on_order = summary["qty_on_order"]
        avg_lead     = summary["avg_lead_time_days"]
        xyz_class    = summary.get("xyz_class")  # X / Y / Z / None

        # Quantity needed to cover demand through the next replenishment cycle,
        # scaled by the XYZ safety multiplier so erratic SKUs (Z) carry more
        # buffer than consistent SKUs (X).
        safety_multiplier   = XYZ_SAFETY_MULTIPLIERS.get(xyz_class, XYZ_SAFETY_MULTIPLIERS[None])
        demand_over_coverage = avg_daily * reorder_threshold * safety_multiplier
        qty_to_order = max(0.0, demand_over_coverage - qty_on_hand - qty_on_order)

        # ---- Min-Qty reconciliation -----------------------------------
        # demand_over_coverage is the AI's preferred on-hand level at which
        # a replenishment cycle should kick off — directly comparable to
        # the buyer's manually-maintained Min Qty (reorder_point).  Flag
        # significant disagreements for buyer review.  Skip when:
        #   * Buyer never set a Min Qty (NULL) — nothing to compare.
        #   * Both numbers are tiny (≤ 1) — relative diff explodes on noise.
        buyer_min = summary.get("buyer_reorder_point")
        if buyer_min is not None and (buyer_min > 1 or demand_over_coverage > 1):
            denom = max(buyer_min, 1.0)
            rel_diff = abs(demand_over_coverage - buyer_min) / denom
            if rel_diff > _RECONCILE_THRESHOLD:
                quality_issues.append({
                    "source_table": "reorder_recommendations",
                    "source_id":    f"{sku_id}|{loc_id}|{today.isoformat()}",
                    "issue_type":   "reorder_threshold_mismatch",
                    "issue_detail": (
                        f"AI reorder level={demand_over_coverage:.2f} vs "
                        f"buyer Min Qty={buyer_min:.2f} "
                        f"(rel_diff={rel_diff:.2f}, "
                        f"avg_daily={avg_daily:.3f}, "
                        f"lead_time={avg_lead:.1f}d, xyz={xyz_class or 'NA'})"
                    ),
                    "field_name":   "reorder_point",
                    "field_value":  f"{buyer_min:.2f}",
                    # data_quality_issues.severity has a CHECK constraint
                    # that only permits 'warning' or 'error' (see
                    # db/migrations/003_data_quality_issues.sql).  Map our
                    # two-tier reconciliation severity onto those.
                    "severity":     "error" if rel_diff > 1.0 else "warning",
                })

        if qty_to_order < MIN_ORDER_QTY:
            counts["skipped_covered"] += 1
            log.debug(
                "SKIP (covered) %s / %s  qty_to_order=%.4f (open POs cover gap)",
                sku_id, loc_id, qty_to_order,
            )
            continue

        # Urgency tier
        if days_supply < URGENCY_CRITICAL_DAYS:
            urgency = "critical"
        elif days_supply < URGENCY_WARNING_DAYS:
            urgency = "warning"
        else:
            urgency = "normal"

        # Transfer check — always before external PO
        transfer_source = find_transfer_source(
            sku_id=sku_id,
            needing_location=loc_id,
            qty_needed=qty_to_order,
            avg_lead_time_days=avg_lead,
            inventory_summary=inventory_summary,
        )

        if transfer_source:
            rec_type            = "transfer"
            rec_supplier_id     = None
            transfer_from_loc   = transfer_source["location_id"]
            rec_risk_flag       = None
        else:
            rec_type            = "po"
            rec_supplier_id     = summary.get("supplier_id")
            transfer_from_loc   = None
            rec_risk_flag       = summary.get("supplier_risk_flag")

        days_display = (
            round(days_supply, 2) if days_supply < DAYS_OF_SUPPLY_CAP
            else DAYS_OF_SUPPLY_CAP
        )

        rec = {
            "sku_id":                   sku_id,
            "location_id":              loc_id,
            "recommendation_date":      today.isoformat(),
            "qty_to_order":             round(qty_to_order, 4),
            "supplier_id":              rec_supplier_id,
            "recommendation_type":      rec_type,
            "transfer_from_location":   transfer_from_loc,
            "days_of_supply_remaining": days_display,
            "urgency":                  urgency,
            "forecast_model_used":      summary.get("model_used"),
            "supplier_risk_flag":       rec_risk_flag,
            "is_approved":              False,
            "approved_by":              None,
            "approved_at":              None,
        }
        recommendations.append(rec)
        counts[urgency]  += 1
        counts[rec_type] += 1

        log.info(
            "  %-12s %-10s  type=%-8s  urgency=%-8s  qty=%.2f  days_supply=%.1f%s",
            sku_id, loc_id, rec_type, urgency, qty_to_order, days_supply,
            f"  [RISK: {rec_risk_flag}]" if rec_risk_flag == "red" else "",
        )

    # ------------------------------------------------------------------
    # 3b. Basket-triggered recommendations
    # ------------------------------------------------------------------
    basket_added = _add_basket_triggered(
        client_holder, recommendations, inventory_summary, today, counts,
    )

    # ------------------------------------------------------------------
    # 4. Write to database
    # ------------------------------------------------------------------
    rows_written = 0

    if recommendations and not dry_run:
        log.info("-" * 60)
        log.info("Writing %d recommendation(s) to reorder_recommendations "
                 "in batches of %d …",
                 len(recommendations), WRITE_BATCH_SIZE)

        total = len(recommendations)
        next_progress_at = 1000  # log every 1000 rows written

        for offset in range(0, total, WRITE_BATCH_SIZE):
            batch = recommendations[offset:offset + WRITE_BATCH_SIZE]

            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    # Always read client_holder[0] fresh — retry paths
                    # may have swapped in a reconnected client.
                    # ignore_duplicates=True preserves same-day approvals.
                    resp = (
                        client_holder[0].table("reorder_recommendations")
                        .upsert(
                            batch,
                            on_conflict="sku_id,location_id,recommendation_date",
                            ignore_duplicates=True,
                        )
                        .execute()
                    )
                    rows_written += len(resp.data or [])
                    break
                except Exception as exc:
                    if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                        log.warning(
                            "  reorder_recommendations write retry %d/%d "
                            "(offset=%d, batch=%d): %s — reconnecting in %.0fs …",
                            attempt, _MAX_RETRIES, offset, len(batch),
                            type(exc).__name__, _RETRY_DELAY,
                        )
                        time.sleep(_RETRY_DELAY)
                        client_holder[0] = _get_fresh_client()
                        continue
                    log.exception(
                        "Failed to write recommendations batch at offset %d "
                        "(size=%d).", offset, len(batch),
                    )
                    return 1

            written_so_far = offset + len(batch)
            if written_so_far >= next_progress_at or written_so_far == total:
                log.info("  Progress: %d / %d rows written (%.1f%%)",
                         written_so_far, total,
                         100.0 * written_so_far / total)
                while next_progress_at <= written_so_far:
                    next_progress_at += 1000

        log.info("  Rows inserted (new): %d  (existing approved rows preserved)",
                 rows_written)
    elif dry_run:
        rows_written = len(recommendations)

    # ------------------------------------------------------------------
    # 4b. Write Min-Qty reconciliation findings to data_quality_issues
    # ------------------------------------------------------------------
    # These rows give buyers a single auditable list of SKU/location pairs
    # where their hand-tuned Min Qty diverges materially from what the
    # forecast-driven engine would set.  Insert (not upsert) — each nightly
    # run produces a new audit trail dated by checked_at.
    quality_written = 0
    if quality_issues and not dry_run:
        log.info("-" * 60)
        log.info(
            "Writing %d Min-Qty reconciliation issue(s) to "
            "data_quality_issues in batches of %d …",
            len(quality_issues), WRITE_BATCH_SIZE,
        )
        for offset in range(0, len(quality_issues), WRITE_BATCH_SIZE):
            batch = quality_issues[offset:offset + WRITE_BATCH_SIZE]
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = (
                        client_holder[0].table("data_quality_issues")
                        .insert(batch)
                        .execute()
                    )
                    quality_written += len(resp.data or [])
                    break
                except Exception as exc:
                    if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                        log.warning(
                            "  data_quality_issues write retry %d/%d "
                            "(offset=%d, batch=%d): %s",
                            attempt, _MAX_RETRIES, offset, len(batch),
                            type(exc).__name__,
                        )
                        time.sleep(_RETRY_DELAY)
                        client_holder[0] = _get_fresh_client()
                        continue
                    log.exception(
                        "Failed to write data_quality_issues batch at "
                        "offset %d (size=%d).", offset, len(batch),
                    )
                    break  # don't fail the whole reorder run on audit-write
        log.info("  Reconciliation issues recorded: %d", quality_written)
    elif quality_issues and dry_run:
        log.info("[DRY RUN] %d Min-Qty reconciliation issue(s) detected "
                 "(not written).", len(quality_issues))

    # ------------------------------------------------------------------
    # 5. Summary log
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - t0
    log.info("=" * 60)
    log.info("Reorder engine complete  (%.2fs)", elapsed)
    log.info("  Inventory pairs assessed:      %d", len(inventory_summary))
    log.info("  Recommendations generated:     %d", len(recommendations))
    log.info("    ↳ transfers:                 %d", counts["transfer"])
    log.info("    ↳ purchase orders:           %d", counts["po"])
    log.info("    ↳ basket-triggered:          %d", basket_added)
    log.info("    ↳ skipped (open POs cover):  %d", counts["skipped_covered"])
    log.info("  By urgency:")
    log.info("    ↳ critical  (< %.0fd):       %d",
             URGENCY_CRITICAL_DAYS, counts["critical"])
    log.info("    ↳ warning   (< %.0fd):       %d",
             URGENCY_WARNING_DAYS, counts["warning"])
    log.info("    ↳ normal    (≥ %.0fd):       %d",
             URGENCY_WARNING_DAYS, counts["normal"])
    log.info("  Rows written to DB:            %d%s",
             rows_written, "  (DRY RUN — no writes made)" if dry_run else "")
    log.info("  Min-Qty reconciliation issues: %d (>%.0f%% relative gap)",
             len(quality_issues), _RECONCILE_THRESHOLD * 100)
    log.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="partswatch-ai reorder recommendation engine",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute recommendations but do not write to the database.",
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
    return run_reorder(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
