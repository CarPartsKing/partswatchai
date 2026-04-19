"""engine/alerts.py — Daily alert generation for the partswatch-ai dashboard.

Runs nightly after engine/reorder.py.  Generates seven categories of alerts
and writes them to the ``alerts`` table.  Re-runs on the same day are safe —
existing acknowledged alerts are never overwritten (upsert ignore_duplicates
on the (alert_date, alert_key) unique index).

ALERT TYPES
-----------
1.  CRITICAL_STOCKOUT       — SKU is at zero stock with active demand
2.  LOW_SUPPLY              — days_of_supply_remaining < LOW_SUPPLY_DAYS
3.  FREEZE_ALERT            — extreme cold forecast; battery/antifreeze SKUs
4.  SUPPLIER_RISK           — red-flag supplier has open purchase orders
5.  DEAD_STOCK              — is_dead_stock = TRUE, no sale in DEAD_STOCK_DAYS
6.  TRANSFER_OPPORTUNITY    — unapproved transfer recommendation pending
7.  FORECAST_ACCURACY_DROP  — ABC-class weekly MAPE > MAPE_THRESHOLD_PCT

DESIGN RULES
------------
- Each alert type is its own isolated function — no shared state between types.
- No thresholds are hardcoded in logic; all tunable values are module constants.
- All DB reads are paginated; no per-SKU round trips.
- Log the count of each alert type generated.

Usage
-----
    python -m engine.alerts            # live run
    python -m engine.alerts --dry-run  # compute, log, no DB writes
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging


def _get_fresh_client() -> Any:
    """Return a brand-new Supabase client (bypasses lru_cache when available).

    Mirrors engine/reorder.py — used by the batched write loop so a stale
    connection that just hit a 57014 statement timeout can be discarded
    and replaced before the next retry.
    """
    try:
        from db.connection import get_new_client
        return get_new_client()
    except ImportError:
        return get_client()


# ---------------------------------------------------------------------------
# Batched-write tunables — mirror engine/reorder.py
# ---------------------------------------------------------------------------
WRITE_BATCH_SIZE: int = 200
"""How many alert rows to upsert per network round-trip.

A single 125k-row upsert blows past Supabase's statement timeout (57014);
200-row batches finish well inside the timeout window and let us retry
individual batches on transient failure without losing prior progress."""

WRITE_PROGRESS_INTERVAL: int = 5_000
"""Log a progress line every N alerts written so long runs stay observable."""

_WRITE_MAX_RETRIES: int = 5
_WRITE_RETRY_DELAY: float = 5.0
_WRITE_RETRYABLE_TOKENS: tuple[str, ...] = (
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


def _is_write_retryable(exc: Exception) -> bool:
    """True if exc looks like a Supabase timeout or dropped-connection error."""
    blob = type(exc).__name__ + " " + str(exc)
    return any(tok in blob for tok in _WRITE_RETRYABLE_TOKENS)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants — easy to adjust without touching logic
# ---------------------------------------------------------------------------

# CRITICAL_STOCKOUT: how many days back to look for "active demand" (recent sales)
ACTIVE_DEMAND_LOOKBACK_DAYS: int = 30

# LOW_SUPPLY: threshold from reorder_recommendations.days_of_supply_remaining
LOW_SUPPLY_DAYS: float = 3.0

# FREEZE_ALERT
FREEZE_TEMP_THRESHOLD_F: float = 20.0        # Flag when temp_min_f drops below this
FREEZE_LOOKAHEAD_DAYS: int = 7               # How many days ahead to scan weather

# Part categories and sub-categories considered freeze-sensitive.
# Add entries here as the SKU catalog grows — no code changes needed.
FREEZE_SENSITIVE_PART_CATEGORIES: frozenset[str] = frozenset({
    "electrical",   # batteries, starters, alternators
    "cooling",      # coolant system — radiator hoses, water pumps
})
FREEZE_SENSITIVE_SUBCATEGORIES: frozenset[str] = frozenset({
    "batteries",
    "antifreeze",
    "coolant",
    "water pumps",
    "radiator hoses",
})

# DEAD_STOCK: minimum days since last sale to fire the alert
DEAD_STOCK_DAYS: int = 180

# FORECAST_ACCURACY_DROP
MAPE_THRESHOLD_PCT: float = 25.0     # Alert when weekly MAPE exceeds this %
MAPE_LOOKBACK_DAYS: int = 7          # Window for "weekly" MAPE calculation

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
        select:      PostgREST column selector string.
        filters:     {col: value} exact equality.
        gte_filters: {col: value} for col >= value.
        lte_filters: {col: value} for col <= value.
        in_filters:  {col: [values]} for col IN (values).
        eq_bool:     {col: bool} for boolean equality (avoids string coercion).

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
    return rows


_LOCATION_NAMES: dict[str, str] = {
    "LOC-001": "BROOKPARK",
    "LOC-002": "NOLMSTEAD",
    "LOC-003": "S.EUCLID",
    "LOC-004": "CLARK AUTO",
    "LOC-005": "PARMA",
    "LOC-006": "MEDINA",
    "LOC-007": "BOARDMAN",
    "LOC-008": "ELYRIA",
    "LOC-009": "AKRON-GRANT",
    "LOC-010": "MIDWAY CROSSINGS",
    "LOC-011": "ERIE ST",
    "LOC-012": "MAYFIELD",
    "LOC-013": "CANTON",
    "LOC-015": "JUNIATA",
    "LOC-016": "ARCHWOOD",
    "LOC-017": "EUCLID",
    "LOC-018": "WARREN",
    "LOC-020": "ROOTSTOWN",
    "LOC-021": "INTERNET",
    "LOC-024": "MENTOR",
    "LOC-025": "MAIN DC",
    "LOC-026": "COPLEY",
    "LOC-027": "CHARDON",
    "LOC-028": "STRONGSVILLE",
    "LOC-029": "MIDDLEBURG",
    "LOC-032": "PERRY",
    "LOC-033": "CRYSTAL",
}


def _make_alert(
    alert_date: date,
    alert_type: str,
    severity:   str,
    message:    str,
    alert_key:  str,
    sku_id:     str | None = None,
    location_id: str | None = None,
    supplier_id: str | None = None,
) -> dict:
    """Build a single alert dict ready for DB insertion.

    Args:
        alert_date:  Date the alert was generated (always today's date).
        alert_type:  One of the seven ALERT TYPE constants.
        severity:    'critical', 'warning', or 'info'.
        message:     Human-readable text shown on the dashboard.
        alert_key:   Pipe-delimited dedup key (must be unique per alert_date).
        sku_id:      Optional SKU context.
        location_id: Optional location context.
        supplier_id: Optional supplier context.

    Returns:
        Dict ready to insert into the alerts table.
    """
    return {
        "alert_date":      alert_date.isoformat(),
        "alert_type":      alert_type,
        "severity":        severity,
        "sku_id":          sku_id,
        "location_id":     location_id,
        "location_name":   _LOCATION_NAMES.get(location_id or "") or None,
        "supplier_id":     supplier_id,
        "message":         message,
        "alert_key":       alert_key,
        "is_acknowledged": False,
        "acknowledged_by": None,
        "acknowledged_at": None,
    }


# ---------------------------------------------------------------------------
# Alert generators — each is fully isolated (own DB reads, own return value)
# ---------------------------------------------------------------------------

def _alert_critical_stockout(client: Any, today: date) -> list[dict]:
    """CRITICAL_STOCKOUT — SKU at zero stock with active recent demand.

    Pulls the most recent inventory snapshot per (sku_id, location_id),
    flags rows where is_stockout = TRUE, then cross-references with
    sales_transactions from the last ACTIVE_DEMAND_LOOKBACK_DAYS days
    to confirm the SKU has had recent demand (not a permanently dead line).

    Returns one alert per (sku_id, location_id) stockout with active demand.
    """
    cutoff_inv = (today - timedelta(days=7)).isoformat()
    inv_rows = _paginate(
        client, "inventory_snapshots",
        "sku_id,location_id,snapshot_date,qty_on_hand,is_stockout",
        gte_filters={"snapshot_date": cutoff_inv},
    )

    # Most recent snapshot per pair
    latest: dict[tuple[str, str], dict] = {}
    for r in inv_rows:
        key = (r["sku_id"], r["location_id"])
        if key not in latest or r["snapshot_date"] > latest[key]["snapshot_date"]:
            latest[key] = r

    stockouts = {k: v for k, v in latest.items() if v.get("is_stockout")}
    if not stockouts:
        return []

    # SKUs with recent sales — active demand proxy
    demand_cutoff = (today - timedelta(days=ACTIVE_DEMAND_LOOKBACK_DAYS)).isoformat()
    sales_rows = _paginate(
        client, "sales_transactions",
        "sku_id,location_id",
        gte_filters={"transaction_date": demand_cutoff},
    )
    active_pairs: set[tuple[str, str]] = {
        (r["sku_id"], r["location_id"]) for r in sales_rows
    }
    # Also count at the SKU level (any location had sales → SKU has demand)
    active_skus: set[str] = {r["sku_id"] for r in sales_rows}

    alerts: list[dict] = []
    for (sku_id, loc_id), inv in stockouts.items():
        if sku_id not in active_skus and (sku_id, loc_id) not in active_pairs:
            continue  # No recent demand — skip to avoid noise
        alerts.append(_make_alert(
            alert_date=today,
            alert_type="CRITICAL_STOCKOUT",
            severity="critical",
            sku_id=sku_id,
            location_id=loc_id,
            message=(
                f"{sku_id} is completely out of stock at {loc_id} "
                f"and has had sales activity in the past {ACTIVE_DEMAND_LOOKBACK_DAYS} days."
            ),
            alert_key=f"CRITICAL_STOCKOUT|{sku_id}|{loc_id}",
        ))
    return alerts


def _alert_low_supply(client: Any, today: date) -> list[dict]:
    """LOW_SUPPLY — days_of_supply_remaining below LOW_SUPPLY_DAYS.

    Reads from today's reorder_recommendations (generated by engine/reorder.py
    just before this module runs).  Excludes full stockouts (qty == 0) since
    those are captured by CRITICAL_STOCKOUT.

    Returns one alert per (sku_id, location_id) with critically low supply
    that is not yet a stockout.
    """
    recs = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,days_of_supply_remaining,urgency,forecast_model_used",
        filters={"recommendation_date": today.isoformat()},
    )

    alerts: list[dict] = []
    for r in recs:
        days = float(r.get("days_of_supply_remaining") or 0)
        # Exclude zero days (stockout — already in CRITICAL_STOCKOUT)
        if days <= 0 or days >= LOW_SUPPLY_DAYS:
            continue
        sku_id  = r["sku_id"]
        loc_id  = r["location_id"]
        alerts.append(_make_alert(
            alert_date=today,
            alert_type="LOW_SUPPLY",
            severity="critical",
            sku_id=sku_id,
            location_id=loc_id,
            message=(
                f"{sku_id} at {loc_id} has only {days:.1f} days of supply remaining "
                f"(threshold: {LOW_SUPPLY_DAYS:.0f} days).  "
                f"Forecast model: {r.get('forecast_model_used', 'unknown')}."
            ),
            alert_key=f"LOW_SUPPLY|{sku_id}|{loc_id}",
        ))
    return alerts


def _alert_freeze(client: Any, today: date) -> list[dict]:
    """FREEZE_ALERT — extreme cold forecast; battery and antifreeze SKUs flagged.

    Scans weather_log for the next FREEZE_LOOKAHEAD_DAYS days.  If any day
    has temp_min_f < FREEZE_TEMP_THRESHOLD_F, generates one alert per
    freeze-sensitive SKU (Batteries, Antifreeze, Cooling categories).

    Returns:
        One alert per freeze-sensitive SKU when a cold event is forecast.
        Empty list when the next week stays above the threshold.
    """
    lookahead_end = (today + timedelta(days=FREEZE_LOOKAHEAD_DAYS)).isoformat()
    weather_rows = _paginate(
        client, "weather_log",
        "log_date,temp_min_f",
        gte_filters={"log_date": today.isoformat()},
        lte_filters={"log_date": lookahead_end},
    )

    cold_days = [
        r for r in weather_rows
        if r.get("temp_min_f") is not None
        and float(r["temp_min_f"]) < FREEZE_TEMP_THRESHOLD_F
    ]
    if not cold_days:
        return []

    coldest   = min(cold_days, key=lambda r: float(r["temp_min_f"]))
    cold_temp = float(coldest["temp_min_f"])
    cold_date = coldest["log_date"]
    cold_count = len(cold_days)

    # Fetch all active SKUs and filter for freeze-sensitive categories
    sku_rows = _paginate(
        client, "sku_master",
        "sku_id,part_category,sub_category",
        eq_bool={"is_active": True},
    )

    freeze_skus = [
        r for r in sku_rows
        if (r.get("part_category") or "").lower() in FREEZE_SENSITIVE_PART_CATEGORIES
        or (r.get("sub_category") or "").lower() in FREEZE_SENSITIVE_SUBCATEGORIES
    ]
    if not freeze_skus:
        return []

    alerts: list[dict] = []
    for r in freeze_skus:
        sku_id = r["sku_id"]
        cat    = r.get("sub_category") or r.get("part_category") or "unknown"
        alerts.append(_make_alert(
            alert_date=today,
            alert_type="FREEZE_ALERT",
            severity="warning",
            sku_id=sku_id,
            message=(
                f"Freeze event forecast: {cold_count} day(s) below "
                f"{FREEZE_TEMP_THRESHOLD_F:.0f}°F in the next {FREEZE_LOOKAHEAD_DAYS} days "
                f"(coldest: {cold_temp:.1f}°F on {cold_date}).  "
                f"Review {sku_id} ({cat}) inventory ahead of expected demand spike."
            ),
            alert_key=f"FREEZE_ALERT|{sku_id}",
        ))
    return alerts


def _alert_supplier_risk(client: Any, today: date) -> list[dict]:
    """SUPPLIER_RISK — red-flag supplier has open purchase orders.

    Identifies suppliers whose most recent score has risk_flag = 'red',
    then finds all open PO lines routed through those suppliers.
    Generates one alert per (supplier_id, sku_id) open PO line.

    Returns:
        One alert per open PO line with a red-flag supplier.
    """
    score_cutoff = (today - timedelta(days=90)).isoformat()
    score_rows = _paginate(
        client, "supplier_scores",
        "supplier_id,score_date,risk_flag,composite_score",
        gte_filters={"score_date": score_cutoff},
    )

    # Most recent score per supplier
    latest_score: dict[str, dict] = {}
    for r in score_rows:
        sid = r.get("supplier_id")
        if not sid:
            continue
        if sid not in latest_score or r["score_date"] > latest_score[sid]["score_date"]:
            latest_score[sid] = r

    red_suppliers = {
        sid for sid, s in latest_score.items()
        if s.get("risk_flag") == "red"
    }
    if not red_suppliers:
        return []

    po_rows = _paginate(
        client, "purchase_orders",
        "po_number,sku_id,supplier_id,qty_ordered,po_date",
        in_filters={
            "status":      ["open", "partial"],
            "supplier_id": list(red_suppliers),
        },
    )
    if not po_rows:
        return []

    alerts: list[dict] = []
    for r in po_rows:
        sid    = r["supplier_id"]
        sku_id = r["sku_id"]
        score  = latest_score.get(sid, {}).get("composite_score")
        score_str = f"{score:.1f}/100" if score is not None else "n/a"
        alerts.append(_make_alert(
            alert_date=today,
            alert_type="SUPPLIER_RISK",
            severity="warning",
            sku_id=sku_id,
            supplier_id=sid,
            message=(
                f"Open PO {r['po_number']} for {sku_id} ({r['qty_ordered']} units) "
                f"is routed through {sid} — risk_flag=RED (score {score_str}).  "
                f"Consider alternate sourcing or expedite status check."
            ),
            alert_key=f"SUPPLIER_RISK|{sid}|{sku_id}|{r['po_number']}",
        ))
    return alerts


def _alert_dead_stock(client: Any, today: date) -> list[dict]:
    """DEAD_STOCK — SKUs with is_dead_stock = TRUE sitting more than DEAD_STOCK_DAYS.

    Reads is_dead_stock and last_sale_date from sku_master.  Alerts when:
        is_dead_stock = TRUE  AND
        (last_sale_date IS NULL OR last_sale_date < today − DEAD_STOCK_DAYS)

    Returns one alert per dead-stock SKU.
    """
    sku_rows = _paginate(
        client, "sku_master",
        "sku_id,part_category,sub_category,last_sale_date,is_dead_stock",
        eq_bool={"is_dead_stock": True},
    )

    dead_cutoff = (today - timedelta(days=DEAD_STOCK_DAYS)).isoformat()
    alerts: list[dict] = []
    for r in sku_rows:
        last_sale = r.get("last_sale_date")
        if last_sale and last_sale >= dead_cutoff:
            continue  # Sold recently enough — not yet stale

        sku_id   = r["sku_id"]
        days_ago = (
            (today - date.fromisoformat(last_sale)).days
            if last_sale else None
        )
        days_str = f"{days_ago} days ago" if days_ago is not None else "never sold"
        alerts.append(_make_alert(
            alert_date=today,
            alert_type="DEAD_STOCK",
            severity="info",
            sku_id=sku_id,
            message=(
                f"{sku_id} ({r.get('sub_category') or r.get('part_category', 'unknown')}) "
                f"is flagged dead stock.  Last sale: {days_str}.  "
                f"Consider liquidation, return, or de-listing."
            ),
            alert_key=f"DEAD_STOCK|{sku_id}",
        ))
    return alerts


def _alert_transfer_opportunity(client: Any, today: date) -> list[dict]:
    """TRANSFER_OPPORTUNITY — single summary alert for all pending transfers.

    Reads today's reorder_recommendations where recommendation_type = 'transfer'
    and is_approved = FALSE.  Returns ONE summary alert (not one-per-transfer)
    pointing the user at the reorder recommendations panel.

    Why a summary instead of per-transfer alerts:
      * The reorder engine routinely emits 15k+ unapproved transfers per day.
      * One alert per transfer flooded the alerts table (125k+ rows in a single
        nightly run on 2026-04-19) and pushed the alerts upsert past Supabase's
        statement timeout (57014).
      * The detail already lives in reorder_recommendations and is rendered on
        the dashboard's transfer panel — duplicating each row as an alert added
        no signal, only noise.

    The summary captures the headline numbers (count, distinct SKUs, distinct
    destination locations, total units, total covered days) so the alert text
    is informative on its own without forcing the user to open the panel.

    Returns a single-element list (or empty list when there are no pending
    transfers today).  Keeps the same return-shape as the other generators
    so run_alerts() needs no special-casing.
    """
    recs = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,transfer_from_location,qty_to_order,days_of_supply_remaining",
        filters={
            "recommendation_date":  today.isoformat(),
            "recommendation_type":  "transfer",
        },
        eq_bool={"is_approved": False},
    )

    if not recs:
        return []

    n_transfers   = len(recs)
    distinct_skus = len({r.get("sku_id") for r in recs if r.get("sku_id")})
    distinct_dest = len({r.get("location_id") for r in recs if r.get("location_id")})
    total_qty     = sum(float(r.get("qty_to_order") or 0) for r in recs)

    message = (
        f"{n_transfers:,} transfer opportunities identified today "
        f"across {distinct_skus:,} SKU(s) and {distinct_dest} destination "
        f"location(s) — {total_qty:,.0f} total units pending approval. "
        f"See the reorder recommendations panel for the per-SKU detail."
    )

    return [_make_alert(
        alert_date=today,
        alert_type="TRANSFER_OPPORTUNITY",
        severity="info",
        # No sku_id / location_id — this is a network-wide rollup, not a
        # per-SKU/location row.  Leaving them NULL keeps the dashboard from
        # double-displaying the same alert under specific SKU panels.
        message=message,
        # Stable per-day key so a same-day re-run of the alert engine
        # collapses to a single row (upsert ignore_duplicates).
        alert_key="TRANSFER_OPPORTUNITY|SUMMARY",
    )]


def _alert_forecast_accuracy_drop(client: Any, today: date) -> list[dict]:
    """FORECAST_ACCURACY_DROP — weekly MAPE > MAPE_THRESHOLD_PCT for an ABC class.

    Compares recent past forecasts (forecast_date in last MAPE_LOOKBACK_DAYS days)
    against actual sales in sales_transactions for the same dates.

    MAPE per (abc_class, model_type):
        MAPE = mean(|actual − predicted| / max(actual, 1)) × 100

    For lightgbm forecasts with location_id = 'ALL', actual demand is the
    network-wide sum across all locations for that SKU and date.

    Returns one alert per (abc_class, model_type) combination where MAPE
    exceeds the threshold.  Returns empty list when no past forecasts exist.
    """
    window_start = (today - timedelta(days=MAPE_LOOKBACK_DAYS)).isoformat()
    yesterday    = (today - timedelta(days=1)).isoformat()

    # Fetch past forecast rows
    fc_rows = _paginate(
        client, "forecast_results",
        "sku_id,location_id,forecast_date,model_type,predicted_qty,run_date",
        gte_filters={"forecast_date": window_start},
        lte_filters={"forecast_date": yesterday},
        in_filters={"model_type": ["lightgbm", "rolling_avg"]},
    )
    if not fc_rows:
        log.debug("FORECAST_ACCURACY_DROP: no past forecast rows in window — skipping.")
        return []

    # Keep only the latest run_date per (sku_id, location_id, model_type)
    latest_run: dict[tuple[str, str, str], str] = {}
    for r in fc_rows:
        key = (r["sku_id"], r["location_id"], r["model_type"])
        if r.get("run_date", "") > latest_run.get(key, ""):
            latest_run[key] = r["run_date"]

    forecast_map: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in fc_rows:
        key = (r["sku_id"], r["location_id"], r["model_type"])
        if r.get("run_date") == latest_run.get(key):
            d = str(r.get("forecast_date", ""))[:10]
            if d:
                forecast_map[key][d] = float(r.get("predicted_qty") or 0)

    # Fetch actual sales for the same window
    sales_rows = _paginate(
        client, "sales_transactions",
        "sku_id,location_id,transaction_date,qty_sold",
        gte_filters={"transaction_date": window_start},
        lte_filters={"transaction_date": yesterday},
    )

    # Build actuals lookup {(sku_id, location_id, date_str): qty}
    actuals: dict[tuple[str, str, str], float] = defaultdict(float)
    # And network-level {(sku_id, date_str): qty} for 'ALL' lightgbm forecasts
    actuals_all: dict[tuple[str, str], float] = defaultdict(float)
    for r in sales_rows:
        d = str(r.get("transaction_date", ""))[:10]
        actuals[(r["sku_id"], r["location_id"], d)] += float(r.get("qty_sold") or 0)
        actuals_all[(r["sku_id"], d)] += float(r.get("qty_sold") or 0)

    # Fetch abc_class lookup for all SKUs in forecast window
    sku_ids_in_fc = {k[0] for k in forecast_map}
    sku_rows = _paginate(
        client, "sku_master",
        "sku_id,abc_class",
        in_filters={"sku_id": list(sku_ids_in_fc)},
    )
    abc_map: dict[str, str] = {
        r["sku_id"]: (r.get("abc_class") or "?") for r in sku_rows
    }

    # Accumulate absolute percentage errors per (abc_class, model_type)
    ape_by_class: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (sku_id, loc_id, model), date_qty in forecast_map.items():
        abc = abc_map.get(sku_id, "?")
        for d, predicted in date_qty.items():
            if loc_id == "ALL":
                actual = actuals_all.get((sku_id, d), 0.0)
            else:
                actual = actuals.get((sku_id, loc_id, d), 0.0)
            ape = abs(actual - predicted) / max(actual, 1.0)
            ape_by_class[(abc, model)].append(ape)

    alerts: list[dict] = []
    for (abc, model), apes in ape_by_class.items():
        if not apes:
            continue
        mape = (sum(apes) / len(apes)) * 100
        if mape <= MAPE_THRESHOLD_PCT:
            continue
        alerts.append(_make_alert(
            alert_date=today,
            alert_type="FORECAST_ACCURACY_DROP",
            severity="warning",
            message=(
                f"{abc}-class {model} weekly MAPE is {mape:.1f}% "
                f"(threshold: {MAPE_THRESHOLD_PCT:.0f}%, "
                f"computed over {len(apes)} forecast observations in the last "
                f"{MAPE_LOOKBACK_DAYS} days).  "
                f"Review model inputs and retrain if data drift is suspected."
            ),
            alert_key=f"FORECAST_ACCURACY_DROP|{abc}|{model}",
        ))
    return alerts


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_alerts(dry_run: bool = False) -> int:
    """Execute all alert generators and write results to the alerts table.

    Args:
        dry_run: When True, compute all alerts but skip the DB write.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    t0 = time.monotonic()
    banner = "=" * 60
    log.info(banner)
    log.info("partswatch-ai — engine.alerts")
    log.info(
        "  freeze_threshold=%.0f°F  low_supply=%.0fd  dead_stock=%dd  "
        "mape_threshold=%.0f%%",
        FREEZE_TEMP_THRESHOLD_F, LOW_SUPPLY_DAYS,
        DEAD_STOCK_DAYS, MAPE_THRESHOLD_PCT,
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
    log.info("Alert date: %s", today.isoformat())
    log.info("-" * 60)

    generators = [
        ("CRITICAL_STOCKOUT",      _alert_critical_stockout),
        ("LOW_SUPPLY",             _alert_low_supply),
        ("FREEZE_ALERT",           _alert_freeze),
        ("SUPPLIER_RISK",          _alert_supplier_risk),
        ("DEAD_STOCK",             _alert_dead_stock),
        ("TRANSFER_OPPORTUNITY",   _alert_transfer_opportunity),
        ("FORECAST_ACCURACY_DROP", _alert_forecast_accuracy_drop),
    ]

    all_alerts: list[dict] = []
    counts: dict[str, int] = {}

    for name, fn in generators:
        try:
            result = fn(client, today)
            counts[name] = len(result)
            all_alerts.extend(result)
            log.info("  %-30s  %d alert(s)", name, len(result))
        except Exception:
            log.exception("Error in %s — skipping this alert type.", name)
            counts[name] = 0

    log.info("-" * 60)
    log.info("Total alerts generated: %d", len(all_alerts))

    # ------------------------------------------------------------------
    # Write to database
    # ------------------------------------------------------------------
    rows_written = 0

    if all_alerts and not dry_run:
        total = len(all_alerts)
        log.info(
            "Writing %d alert(s) to alerts table in batches of %d …",
            total, WRITE_BATCH_SIZE,
        )

        # One-shot location_name fallback: if the very first batch fails
        # because the alerts table predates migration 016, strip the column
        # from EVERY remaining row (not just the failing batch) so we don't
        # repeatedly hit the same error on each subsequent batch.
        location_name_stripped = False
        # Holder so retry paths can swap in a freshly-reconnected client.
        client_holder: list = [client]
        next_progress_at = WRITE_PROGRESS_INTERVAL

        for offset in range(0, total, WRITE_BATCH_SIZE):
            batch = all_alerts[offset:offset + WRITE_BATCH_SIZE]

            # Manual attempt counter so the location_name schema fallback
            # can retry the batch WITHOUT consuming a transient-retry slot.
            # `for attempt in range(...)` would auto-increment every iteration
            # — using a while loop with explicit ++ on transient failures only
            # keeps the schema-fallback retry "free" as documented.
            attempt = 1
            while True:
                try:
                    resp = (
                        client_holder[0].table("alerts")
                        .upsert(
                            batch,
                            on_conflict="alert_date,alert_key",
                            ignore_duplicates=True,
                        )
                        .execute()
                    )
                    rows_written += len(resp.data or [])
                    break

                except Exception as exc:
                    msg = str(exc)

                    # ----------------------------------------------------------
                    # Schema fallback — older alerts tables lack location_name.
                    # Strip it from THIS batch and all remaining ones, then
                    # retry the current batch immediately WITHOUT incrementing
                    # `attempt` (this is a known schema mismatch, not a
                    # transient failure, and shouldn't burn retry budget).
                    # ----------------------------------------------------------
                    if "location_name" in msg and not location_name_stripped:
                        log.warning(
                            "location_name column missing — stripping from "
                            "all remaining alerts and retrying batch."
                        )
                        for a in all_alerts:
                            a.pop("location_name", None)
                        location_name_stripped = True
                        continue  # retry without bumping `attempt`

                    # ----------------------------------------------------------
                    # Transient timeout / dropped-connection — reconnect with
                    # a fresh client and retry up to _WRITE_MAX_RETRIES times.
                    # ----------------------------------------------------------
                    if _is_write_retryable(exc) and attempt < _WRITE_MAX_RETRIES:
                        log.warning(
                            "  alerts write retry %d/%d "
                            "(offset=%d, batch=%d): %s — reconnecting in %.0fs …",
                            attempt, _WRITE_MAX_RETRIES, offset, len(batch),
                            type(exc).__name__, _WRITE_RETRY_DELAY,
                        )
                        time.sleep(_WRITE_RETRY_DELAY)
                        client_holder[0] = _get_fresh_client()
                        attempt += 1
                        continue

                    log.exception(
                        "Failed to write alerts batch at offset %d (size=%d).",
                        offset, len(batch),
                    )
                    return 1

            written_so_far = offset + len(batch)
            if written_so_far >= next_progress_at or written_so_far == total:
                log.info(
                    "  [ALERTS] Written %d / %d alerts (%.1f%%)",
                    written_so_far, total,
                    100.0 * written_so_far / total,
                )
                while next_progress_at <= written_so_far:
                    next_progress_at += WRITE_PROGRESS_INTERVAL

        log.info(
            "  Rows inserted (new): %d  (existing acknowledged alerts preserved)",
            rows_written,
        )
    elif dry_run:
        rows_written = len(all_alerts)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - t0
    log.info("=" * 60)
    log.info("Alert engine complete  (%.2fs)", elapsed)
    for name, cnt in counts.items():
        log.info("  %-30s  %d", name, cnt)
    log.info("  %-30s  %d", "TOTAL", len(all_alerts))
    log.info("  Rows written to DB:  %d%s",
             rows_written,
             "  (DRY RUN — no writes made)" if dry_run else "")
    log.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="partswatch-ai daily alert generator",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute alerts but do not write to the database.",
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
    return run_alerts(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
