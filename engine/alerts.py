"""engine/alerts.py — Daily alert generation for the partswatch-ai dashboard.

Runs nightly after engine/reorder.py.  Generates nine categories of alerts
and writes them to the ``alerts`` table.

ALERT TYPES
-----------
 1. CRITICAL_STOCKOUT       — SKU at zero stock with active demand
 2. LOW_SUPPLY              — days_of_supply_remaining < LOW_SUPPLY_DAYS
 3. FREEZE_ALERT            — extreme cold forecast; freeze-sensitive SKUs
 4. SUPPLIER_RISK           — red-flag supplier has open purchase orders
 5. DEAD_STOCK              — is_dead_stock = TRUE, no sale in DEAD_STOCK_DAYS
 6. TRANSFER_OPPORTUNITY    — unapproved transfer recommendation pending
 7. FORECAST_ACCURACY_DROP  — ABC-class weekly MAPE > MAPE_THRESHOLD_PCT
 8. CHURN_RISK              — customer flagged AT_RISK / CHURNED / LOST
 9. OPSL_GAP                — HIGH OPSL flag not yet in reorder queue

DESIGN RULES
------------
- Each alert type is its own isolated function — no shared state between types.
- Deduplication: if an alert_key already exists in a resolved=FALSE alert from
  the last DEDUP_LOOKBACK_DAYS days, the existing row's days_active counter is
  incremented instead of creating a new row.  first_seen_date is preserved.
- Auto-resolve: after writing, open alerts whose condition no longer holds are
  marked resolved=TRUE / resolved_date=today.  Resolvable types: CRITICAL_STOCKOUT,
  LOW_SUPPLY, CHURN_RISK, OPSL_GAP.
- Financial impact: every alert carries a financial_impact NUMERIC(14,2) field.
  Alerts are sorted descending by financial_impact within each severity tier.
- Dashboard queries must filter resolved=FALSE to exclude auto-resolved rows.

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
    """Return a brand-new Supabase client (bypasses lru_cache when available)."""
    try:
        from db.connection import get_new_client
        return get_new_client()
    except ImportError:
        return get_client()


# ---------------------------------------------------------------------------
# Batched-write tunables
# ---------------------------------------------------------------------------
WRITE_BATCH_SIZE: int = 200
"""How many alert rows to upsert per network round-trip."""

WRITE_PROGRESS_INTERVAL: int = 5_000
"""Log a progress line every N alerts written."""

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
# Tunable constants
# ---------------------------------------------------------------------------

# CRITICAL_STOCKOUT
ACTIVE_DEMAND_LOOKBACK_DAYS: int = 30

# LOW_SUPPLY
LOW_SUPPLY_DAYS: float = 3.0

# FREEZE_ALERT
FREEZE_TEMP_THRESHOLD_F: float = 20.0
FREEZE_LOOKAHEAD_DAYS: int = 7
FREEZE_SENSITIVE_PART_CATEGORIES: frozenset[str] = frozenset({
    "electrical",
    "cooling",
})
FREEZE_SENSITIVE_SUBCATEGORIES: frozenset[str] = frozenset({
    "batteries",
    "antifreeze",
    "coolant",
    "water pumps",
    "radiator hoses",
})

# DEAD_STOCK
DEAD_STOCK_DAYS: int = 180

# FORECAST_ACCURACY_DROP
MAPE_THRESHOLD_PCT: float = 25.0
MAPE_LOOKBACK_DAYS: int = 7

# Deduplication — look back this many days for existing open alerts with the
# same alert_key before deciding to increment vs insert.
DEDUP_LOOKBACK_DAYS: int = 30

# CHURN_RISK — financial_impact = baseline_monthly_spend × this multiplier
# (one quarter of annualized revenue at risk)
CHURN_QUARTER_MULTIPLIER: float = 3.0
CHURN_ACTIVE_FLAGS: tuple[str, ...] = ("AT_RISK", "CHURNED", "LOST")

_PAGE_SIZE: int = 1_000

# Severity sort order for financial-impact ranking within tier
_SEV_RANK: dict[str, int] = {"critical": 0, "warning": 1, "info": 2}

# Alert types whose condition can be auto-detected as resolved
_AUTO_RESOLVE_TYPES: frozenset[str] = frozenset({
    "CRITICAL_STOCKOUT", "LOW_SUPPLY", "CHURN_RISK", "OPSL_GAP",
})


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
    """Paginate through a Supabase table and return all matching rows."""
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
    sku_id:           str | None = None,
    location_id:      str | None = None,
    supplier_id:      str | None = None,
    customer_id:      str | None = None,
    financial_impact: float = 0.0,
) -> dict:
    """Build a single alert dict ready for DB insertion."""
    return {
        "alert_date":       alert_date.isoformat(),
        "alert_type":       alert_type,
        "severity":         severity,
        "sku_id":           sku_id,
        "location_id":      location_id,
        "location_name":    _LOCATION_NAMES.get(location_id or "") or None,
        "supplier_id":      supplier_id,
        "customer_id":      customer_id,
        "message":          message,
        "alert_key":        alert_key,
        "financial_impact": round(float(financial_impact or 0), 2),
        "is_acknowledged":  False,
        "acknowledged_by":  None,
        "acknowledged_at":  None,
    }


# ---------------------------------------------------------------------------
# Shared pre-fetch helpers
# ---------------------------------------------------------------------------

def _fetch_unit_cost_map(client: Any) -> dict[str, float]:
    """Return sku_id → unit_cost for all SKUs that have a non-null unit_cost."""
    rows = _paginate(client, "sku_master", "sku_id,unit_cost")
    return {
        r["sku_id"]: float(r["unit_cost"])
        for r in rows
        if r.get("sku_id") and r.get("unit_cost") is not None
    }


def _fetch_existing_open_alerts(
    client: Any,
    today: date,
    lookback_days: int = DEDUP_LOOKBACK_DAYS,
) -> dict[str, dict]:
    """Return the most-recent open alert row per alert_key from the last N days.

    Includes today's rows so that a same-day re-run can detect alerts already
    written in a previous run and skip both INSERT and UPDATE for them.

    Returns:
        dict mapping alert_key → {id, alert_date, alert_type, days_active}
        May return {} if the resolved column doesn't exist yet (pre-migration).
    """
    cutoff = (today - timedelta(days=lookback_days)).isoformat()
    try:
        rows = _paginate(
            client, "alerts",
            "id,alert_key,alert_date,alert_type,days_active,resolved",
            gte_filters={"alert_date": cutoff},
        )
    except Exception as exc:
        log.warning(
            "Could not fetch existing open alerts (%s) — dedup disabled. "
            "Run migration 047 if resolved column is missing.",
            exc.__class__.__name__,
        )
        return {}

    by_key: dict[str, dict] = {}
    for r in rows:
        k = r["alert_key"]
        if k not in by_key or r["alert_date"] > by_key[k]["alert_date"]:
            by_key[k] = r
    return by_key


# ---------------------------------------------------------------------------
# Alert generators — each is fully isolated (own DB reads, own return value)
# ---------------------------------------------------------------------------

def _alert_critical_stockout(
    client: Any,
    today: date,
    unit_cost_map: dict[str, float] | None = None,
) -> list[dict]:
    """CRITICAL_STOCKOUT — SKU at zero stock per latest inventory snapshot.

    financial_impact = qty_to_order × unit_cost from today's reorder recs.
    Items with no reorder recommendation get financial_impact=0 and naturally
    sort below items with active demand, so no separate demand query is needed.
    """
    cutoff_inv = (today - timedelta(days=7)).isoformat()
    inv_rows = _paginate(
        client, "inventory_snapshots",
        "sku_id,location_id,snapshot_date,qty_on_hand,is_stockout",
        gte_filters={"snapshot_date": cutoff_inv},
    )

    latest: dict[tuple[str, str], dict] = {}
    for r in inv_rows:
        key = (r["sku_id"], r["location_id"])
        if key not in latest or r["snapshot_date"] > latest[key]["snapshot_date"]:
            latest[key] = r

    stockouts = {k: v for k, v in latest.items() if v.get("is_stockout")}
    if not stockouts:
        return []

    reorder_qty: dict[tuple[str, str], float] = {}
    try:
        rec_rows = _paginate(
            client, "reorder_recommendations",
            "sku_id,location_id,qty_to_order",
            filters={"recommendation_date": today.isoformat()},
        )
        reorder_qty = {
            (r["sku_id"], r["location_id"]): float(r.get("qty_to_order") or 0)
            for r in rec_rows
        }
    except Exception:
        pass

    ucm = unit_cost_map or {}
    alerts: list[dict] = []
    for (sku_id, loc_id), inv in stockouts.items():
        qty  = reorder_qty.get((sku_id, loc_id), 0.0)
        cost = ucm.get(sku_id, 0.0)
        alerts.append(_make_alert(
            alert_date=today,
            alert_type="CRITICAL_STOCKOUT",
            severity="critical",
            sku_id=sku_id,
            location_id=loc_id,
            message=f"{sku_id} is completely out of stock at {loc_id}.",
            alert_key=f"CRITICAL_STOCKOUT|{sku_id}|{loc_id}",
            financial_impact=qty * cost,
        ))
    return alerts


def _alert_low_supply(
    client: Any,
    today: date,
    unit_cost_map: dict[str, float] | None = None,
) -> list[dict]:
    """LOW_SUPPLY — days_of_supply_remaining below LOW_SUPPLY_DAYS.

    financial_impact = qty_to_order × unit_cost.
    """
    recs = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,days_of_supply_remaining,urgency,"
        "forecast_model_used,qty_to_order",
        filters={"recommendation_date": today.isoformat()},
    )

    ucm = unit_cost_map or {}
    alerts: list[dict] = []
    for r in recs:
        days = float(r.get("days_of_supply_remaining") or 0)
        if days <= 0 or days >= LOW_SUPPLY_DAYS:
            continue
        sku_id = r["sku_id"]
        loc_id = r["location_id"]
        qty    = float(r.get("qty_to_order") or 0)
        cost   = ucm.get(sku_id, 0.0)
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
            financial_impact=qty * cost,
        ))
    return alerts


def _alert_freeze(client: Any, today: date) -> list[dict]:
    """FREEZE_ALERT — extreme cold forecast; battery and antifreeze SKUs flagged."""
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

    coldest    = min(cold_days, key=lambda r: float(r["temp_min_f"]))
    cold_temp  = float(coldest["temp_min_f"])
    cold_date  = coldest["log_date"]
    cold_count = len(cold_days)

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


def _alert_supplier_risk(
    client: Any,
    today: date,
    unit_cost_map: dict[str, float] | None = None,
) -> list[dict]:
    """SUPPLIER_RISK — red-flag supplier has open purchase orders.

    financial_impact = qty_ordered × unit_cost (total open PO value).
    """
    score_cutoff = (today - timedelta(days=90)).isoformat()
    score_rows = _paginate(
        client, "supplier_scores",
        "supplier_id,score_date,risk_flag,composite_score",
        gte_filters={"score_date": score_cutoff},
    )

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

    ucm = unit_cost_map or {}
    alerts: list[dict] = []
    for r in po_rows:
        sid    = r["supplier_id"]
        sku_id = r["sku_id"]
        score  = latest_score.get(sid, {}).get("composite_score")
        score_str = f"{score:.1f}/100" if score is not None else "n/a"
        qty  = float(r.get("qty_ordered") or 0)
        cost = ucm.get(sku_id, 0.0)
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
            financial_impact=qty * cost,
        ))
    return alerts


def _alert_dead_stock(client: Any, today: date) -> list[dict]:
    """DEAD_STOCK — SKUs with is_dead_stock = TRUE, no sale in DEAD_STOCK_DAYS."""
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
            continue

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

    Returns ONE summary alert (not one-per-transfer) to avoid flooding the
    alerts table.  Detail lives in the reorder_recommendations panel.
    """
    recs = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,transfer_from_location,qty_to_order,days_of_supply_remaining",
        filters={
            "recommendation_date": today.isoformat(),
            "recommendation_type": "transfer",
        },
        eq_bool={"is_approved": False},
    )
    if not recs:
        return []

    n_transfers   = len(recs)
    distinct_skus = len({r.get("sku_id") for r in recs if r.get("sku_id")})
    distinct_dest = len({r.get("location_id") for r in recs if r.get("location_id")})
    total_qty     = sum(float(r.get("qty_to_order") or 0) for r in recs)

    return [_make_alert(
        alert_date=today,
        alert_type="TRANSFER_OPPORTUNITY",
        severity="info",
        message=(
            f"{n_transfers:,} transfer opportunities identified today "
            f"across {distinct_skus:,} SKU(s) and {distinct_dest} destination "
            f"location(s) — {total_qty:,.0f} total units pending approval. "
            f"See the reorder recommendations panel for the per-SKU detail."
        ),
        alert_key="TRANSFER_OPPORTUNITY|SUMMARY",
    )]


def _alert_forecast_accuracy_drop(client: Any, today: date) -> list[dict]:
    """FORECAST_ACCURACY_DROP — weekly MAPE > MAPE_THRESHOLD_PCT for an ABC class."""
    window_start   = (today - timedelta(days=MAPE_LOOKBACK_DAYS)).isoformat()
    run_date_start = (today - timedelta(days=7)).isoformat()
    yesterday      = (today - timedelta(days=1)).isoformat()

    # Limit run_date to last 7 days so PostgREST only touches recent forecast
    # rows — the full table spans years and times out without this constraint.
    # If the table still times out (no composite index), skip gracefully.
    try:
        fc_rows = _paginate(
            client, "forecast_results",
            "sku_id,location_id,forecast_date,model_type,predicted_qty,run_date",
            gte_filters={"forecast_date": window_start, "run_date": run_date_start},
            lte_filters={"forecast_date": yesterday},
            in_filters={"model_type": ["lightgbm", "rolling_avg"]},
        )
    except Exception as exc:
        log.warning(
            "FORECAST_ACCURACY_DROP: forecast_results query timed out (%s) — "
            "skipping. Add an index on (run_date, model_type) to enable.",
            exc.__class__.__name__,
        )
        return []
    if not fc_rows:
        log.debug("FORECAST_ACCURACY_DROP: no past forecast rows in window — skipping.")
        return []

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

    # Batch sales lookup by the specific forecast SKUs to avoid a full table scan.
    sku_ids_in_fc = list({k[0] for k in forecast_map})
    _FC_BATCH = 50
    actuals:     dict[tuple[str, str, str], float] = defaultdict(float)
    actuals_all: dict[tuple[str, str], float]      = defaultdict(float)
    for _i in range(0, max(1, len(sku_ids_in_fc)), _FC_BATCH):
        _batch = sku_ids_in_fc[_i:_i + _FC_BATCH]
        _rows = _paginate(
            client, "sales_transactions",
            "sku_id,location_id,transaction_date,qty_sold",
            gte_filters={"transaction_date": window_start},
            lte_filters={"transaction_date": yesterday},
            in_filters={"sku_id": _batch},
        )
        for r in _rows:
            d = str(r.get("transaction_date", ""))[:10]
            actuals[(r["sku_id"], r["location_id"], d)] += float(r.get("qty_sold") or 0)
            actuals_all[(r["sku_id"], d)]               += float(r.get("qty_sold") or 0)

    sku_rows = _paginate(
        client, "sku_master",
        "sku_id,abc_class",
        in_filters={"sku_id": sku_ids_in_fc},
    )
    abc_map: dict[str, str] = {r["sku_id"]: (r.get("abc_class") or "?") for r in sku_rows}

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


def _alert_churn_risk(client: Any, today: date) -> list[dict]:
    """CHURN_RISK — customers flagged AT_RISK / CHURNED / LOST.

    Reads customer_churn_flags for any customer whose flag is in
    CHURN_ACTIVE_FLAGS.  Deduplication (days_active increment) is handled by
    the orchestrator using alert_key; this function always returns the full
    current list regardless of whether the alert already exists.

    financial_impact = baseline_monthly_spend × CHURN_QUARTER_MULTIPLIER
    (one quarter of annualised revenue at risk).
    """
    rows = _paginate(
        client, "customer_churn_flags",
        "customer_id,location_id,flag,baseline_monthly_spend,last_purchase_date",
        in_filters={"flag": list(CHURN_ACTIVE_FLAGS)},
    )

    alerts: list[dict] = []
    for r in rows:
        cid      = r.get("customer_id") or ""
        loc      = r.get("location_id") or ""
        flag     = r.get("flag") or ""
        baseline = float(r.get("baseline_monthly_spend") or 0)
        impact   = round(baseline * CHURN_QUARTER_MULTIPLIER, 2)
        last_dt  = r.get("last_purchase_date") or "unknown"

        if not cid or not loc:
            continue

        alerts.append(_make_alert(
            alert_date=today,
            alert_type="CHURN_RISK",
            severity="warning",
            customer_id=cid,
            location_id=loc,
            message=(
                f"Customer {cid} at {loc} is flagged {flag}.  "
                f"Baseline monthly spend: ${baseline:,.0f}.  "
                f"Last purchase: {last_dt}.  "
                f"Quarterly revenue at risk: ${impact:,.0f}.  "
                f"Assign rep follow-up immediately."
            ),
            alert_key=f"CHURN_RISK|{cid}|{loc}",
            financial_impact=impact,
        ))
    return alerts


def _alert_opsl_gap(client: Any, today: date) -> list[dict]:
    """OPSL_GAP — HIGH OPSL flag not yet added to the reorder queue.

    Reads opsl_flags for flag='HIGH' AND in_reorder_queue=FALSE.  Each row
    represents a SKU+location that is regularly sourced outside (costing margin)
    but has no reorder recommendation yet.

    financial_impact = estimated_margin_recovery (pre-computed by the OPSL engine).
    """
    rows = _paginate(
        client, "opsl_flags",
        "prod_line_pn,location_id,opsl_count,estimated_margin_recovery,last_opsl_date",
        filters={"flag": "HIGH"},
        eq_bool={"in_reorder_queue": False},
    )

    alerts: list[dict] = []
    for r in rows:
        pn       = r.get("prod_line_pn") or ""
        loc      = r.get("location_id") or ""
        count    = int(r.get("opsl_count") or 0)
        recovery = float(r.get("estimated_margin_recovery") or 0)
        last_dt  = r.get("last_opsl_date") or "unknown"

        if not pn or not loc:
            continue

        alerts.append(_make_alert(
            alert_date=today,
            alert_type="OPSL_GAP",
            severity="info",
            sku_id=pn,
            location_id=loc,
            message=(
                f"{pn} at {loc} has {count} outside-purchase (OPSL) events "
                f"(last: {last_dt}) and is not in the reorder queue.  "
                f"Estimated margin recovery if stocked locally: ${recovery:,.0f}.  "
                f"Add to reorder queue to recover margin."
            ),
            alert_key=f"OPSL_GAP|{pn}|{loc}",
            financial_impact=round(recovery, 2),
        ))
    return alerts


# ---------------------------------------------------------------------------
# Auto-resolve
# ---------------------------------------------------------------------------

def _auto_resolve(
    client: Any,
    today: date,
    current_keys_by_type: dict[str, set[str]],
    dry_run: bool = False,
) -> int:
    """Mark open alerts resolved when their underlying condition no longer holds.

    Resolution rule (same for all resolvable types): if an open alert's
    alert_key is NOT present in today's newly-generated alerts for the same
    alert_type, the condition is no longer detected → resolve it.

    Resolvable types: CRITICAL_STOCKOUT, LOW_SUPPLY, CHURN_RISK, OPSL_GAP.
    Types not in this set (FREEZE_ALERT, DEAD_STOCK, etc.) require manual
    acknowledgement and are never auto-resolved.

    Args:
        current_keys_by_type: {alert_type: set of alert_keys} from today's run.
        dry_run: When True, count candidates but do not write.

    Returns:
        Number of alerts resolved (or that would be resolved in dry-run).
    """
    cutoff = (today - timedelta(days=DEDUP_LOOKBACK_DAYS)).isoformat()
    try:
        open_rows = _paginate(
            client, "alerts",
            "id,alert_type,alert_key",
            gte_filters={"alert_date": cutoff},
            in_filters={"alert_type": list(_AUTO_RESOLVE_TYPES)},
            eq_bool={"resolved": False},
        )
    except Exception as exc:
        log.warning(
            "auto-resolve fetch failed (%s) — skipping. "
            "Run migration 047 if resolved column is missing.",
            exc.__class__.__name__,
        )
        return 0

    to_resolve: list[int] = [
        r["id"]
        for r in open_rows
        if r["alert_key"] not in current_keys_by_type.get(r.get("alert_type", ""), set())
    ]

    if to_resolve and not dry_run:
        payload = {"resolved": True, "resolved_date": today.isoformat()}
        for i in range(0, len(to_resolve), 200):
            batch = to_resolve[i:i + 200]
            try:
                client.table("alerts").update(payload).in_("id", batch).execute()
            except Exception as exc:
                log.warning(
                    "auto-resolve UPDATE failed for batch of %d ids: %s",
                    len(batch), exc.__class__.__name__,
                )

    return len(to_resolve)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_alerts(dry_run: bool = False) -> int:
    """Execute all alert generators and write results to the alerts table.

    Args:
        dry_run: When True, compute and log all alerts but skip DB writes.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    t0 = time.monotonic()
    banner = "=" * 60
    log.info(banner)
    log.info("partswatch-ai — engine.alerts")
    log.info(
        "  freeze_threshold=%.0f°F  low_supply=%.0fd  dead_stock=%dd  "
        "mape_threshold=%.0f%%  dedup_window=%dd",
        FREEZE_TEMP_THRESHOLD_F, LOW_SUPPLY_DAYS,
        DEAD_STOCK_DAYS, MAPE_THRESHOLD_PCT, DEDUP_LOOKBACK_DAYS,
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

    # ------------------------------------------------------------------
    # Pre-fetch shared data
    # ------------------------------------------------------------------
    log.info("Pre-fetching unit costs …")
    try:
        unit_cost_map = _fetch_unit_cost_map(client)
        log.info("  %d SKU unit costs loaded.", len(unit_cost_map))
    except Exception:
        log.warning("unit_cost fetch failed — financial_impact will be 0 for reorder/PO alerts.")
        unit_cost_map = {}

    log.info("Pre-fetching existing open alerts (last %d days) …", DEDUP_LOOKBACK_DAYS)
    existing_open = _fetch_existing_open_alerts(client, today)
    log.info("  %d open alert(s) eligible for dedup.", len(existing_open))

    # ------------------------------------------------------------------
    # Run generators
    # ------------------------------------------------------------------
    generators = [
        ("CRITICAL_STOCKOUT",
         lambda c, t: _alert_critical_stockout(c, t, unit_cost_map)),
        ("LOW_SUPPLY",
         lambda c, t: _alert_low_supply(c, t, unit_cost_map)),
        ("FREEZE_ALERT",        _alert_freeze),
        ("SUPPLIER_RISK",
         lambda c, t: _alert_supplier_risk(c, t, unit_cost_map)),
        ("DEAD_STOCK",          _alert_dead_stock),
        ("TRANSFER_OPPORTUNITY", _alert_transfer_opportunity),
        ("FORECAST_ACCURACY_DROP", _alert_forecast_accuracy_drop),
        ("CHURN_RISK",          _alert_churn_risk),
        ("OPSL_GAP",            _alert_opsl_gap),
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
    # Sort by severity tier then financial_impact descending
    # ------------------------------------------------------------------
    all_alerts.sort(key=lambda a: (
        _SEV_RANK.get(a.get("severity", "info"), 2),
        -(float(a.get("financial_impact") or 0)),
    ))

    # ------------------------------------------------------------------
    # Dedup split: existing alert_key → UPDATE; new → INSERT
    # ------------------------------------------------------------------
    to_update: list[dict] = []   # {id, days_active, financial_impact, message, ...}
    to_insert: list[dict] = []   # full alert dicts ready for upsert

    today_iso = today.isoformat()
    for alert in all_alerts:
        key = alert["alert_key"]
        if key in existing_open:
            existing_row = existing_open[key]
            if existing_row["alert_date"] == today_iso:
                # Already written today (previous run) — skip entirely.
                continue
            if existing_row.get("resolved"):
                # Most-recent prior row was resolved; treat as new occurrence.
                alert["first_seen_date"] = today_iso
                alert["days_active"]     = 1
                to_insert.append(alert)
                continue
            to_update.append({
                "id":               existing_row["id"],
                "alert_date":       today_iso,
                "days_active":      (existing_row.get("days_active") or 1) + 1,
                "financial_impact": alert.get("financial_impact", 0.0),
                "message":          alert["message"],
                "severity":         alert["severity"],
            })
        else:
            alert["first_seen_date"] = today_iso
            alert["days_active"]     = 1
            to_insert.append(alert)

    log.info(
        "  Dedup: %d will increment days_active on existing rows  |  "
        "%d new rows to insert",
        len(to_update), len(to_insert),
    )

    # ------------------------------------------------------------------
    # Build current-keys lookup for auto-resolve
    # ------------------------------------------------------------------
    current_keys_by_type: dict[str, set[str]] = defaultdict(set)
    for a in all_alerts:
        current_keys_by_type[a["alert_type"]].add(a["alert_key"])

    # ------------------------------------------------------------------
    # Dry-run summary
    # ------------------------------------------------------------------
    if dry_run:
        resolve_count = _auto_resolve(
            client, today, dict(current_keys_by_type), dry_run=True,
        )
        log.info("-" * 60)
        log.info("DRY RUN SUMMARY")
        log.info("  Alerts generated:         %d", len(all_alerts))
        log.info("  Would deduplicate:        %d  (days_active++)", len(to_update))
        log.info("  Would insert (new):       %d", len(to_insert))
        log.info("  Would auto-resolve:       %d", resolve_count)
        log.info("  New CHURN_RISK alerts:    %d", counts.get("CHURN_RISK", 0))
        log.info("  New OPSL_GAP alerts:      %d", counts.get("OPSL_GAP", 0))
        if to_insert:
            top_n = min(10, len(to_insert))
            log.info("  Top %d new alerts by financial impact:", top_n)
            for a in to_insert[:top_n]:
                fi = float(a.get("financial_impact") or 0)
                log.info(
                    f"    {a['alert_type']:<22s}  {a['severity']:<8s}"
                    f"  ${fi:>10,.0f}  {a['alert_key'][:60]}"
                )
        log.info("=" * 60)
        elapsed = time.monotonic() - t0
        log.info("Alert engine complete  (%.2fs)  — DRY RUN, no writes made.", elapsed)
        return 0

    # ------------------------------------------------------------------
    # Live writes
    # ------------------------------------------------------------------
    rows_written = 0

    # --- 1. INSERT new alerts ---
    if to_insert:
        total = len(to_insert)
        log.info(
            "Writing %d new alert(s) in batches of %d …",
            total, WRITE_BATCH_SIZE,
        )

        location_name_stripped = False
        new_cols_stripped       = False
        client_holder: list    = [client]
        next_progress_at        = WRITE_PROGRESS_INTERVAL

        for offset in range(0, total, WRITE_BATCH_SIZE):
            batch = to_insert[offset:offset + WRITE_BATCH_SIZE]

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

                    if "location_name" in msg and not location_name_stripped:
                        log.warning(
                            "location_name column missing — stripping from "
                            "all remaining alerts and retrying batch."
                        )
                        for a in to_insert:
                            a.pop("location_name", None)
                        location_name_stripped = True
                        continue

                    # Strip new migration-047 columns if migration not yet applied
                    new_col_names = (
                        "financial_impact", "days_active", "first_seen_date",
                        "resolved", "resolved_date", "customer_id",
                    )
                    if any(c in msg for c in new_col_names) and not new_cols_stripped:
                        log.warning(
                            "New column(s) from migration 047 missing — "
                            "stripping and retrying.  Apply 047 to enable "
                            "financial_impact, dedup, and auto-resolve."
                        )
                        for a in to_insert:
                            for c in new_col_names:
                                a.pop(c, None)
                        new_cols_stripped = True
                        continue

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

    # --- 2. UPDATE deduped alerts (days_active++) ---
    # Group by new days_active so we can issue one UPDATE…WHERE id IN (…) per
    # group instead of 96K individual row calls, which time out.
    if to_update:
        log.info("Updating %d deduped alert row(s) …", len(to_update))
        by_days: dict[int, list[int]] = {}
        for upd in to_update:
            by_days.setdefault(upd["days_active"], []).append(upd["id"])

        _UPD_BATCH = 200
        update_ok   = 0
        update_fail = 0
        for new_days, row_ids in by_days.items():
            for _i in range(0, len(row_ids), _UPD_BATCH):
                batch_ids = row_ids[_i:_i + _UPD_BATCH]
                try:
                    client.table("alerts").update({
                        "alert_date":  today.isoformat(),
                        "days_active": new_days,
                    }).in_("id", batch_ids).execute()
                    update_ok += len(batch_ids)
                except Exception as exc:
                    err_str = str(exc)
                    if "23505" in err_str:
                        # today row already exists from a prior run —
                        # alert data is correct, days_active not incremented.
                        update_ok += len(batch_ids)
                        log.debug(
                            "Dedup batch skipped (today row exists, "
                            "days_active=%d, batch=%d).",
                            new_days, len(batch_ids),
                        )
                    else:
                        update_fail += len(batch_ids)
                        log.warning(
                            "Dedup batch UPDATE failed "
                            "(days_active=%d, batch=%d): %s — %s",
                            new_days, len(batch_ids),
                            exc.__class__.__name__, err_str[:200],
                        )

        log.info("  Dedup updates: %d succeeded, %d failed.", update_ok, update_fail)

    # --- 3. Auto-resolve ---
    log.info("Scanning for auto-resolvable alerts …")
    resolved_count = _auto_resolve(client, today, dict(current_keys_by_type))
    log.info("  Auto-resolved: %d alert(s).", resolved_count)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - t0
    log.info("=" * 60)
    log.info("Alert engine complete  (%.2fs)", elapsed)
    for name, cnt in counts.items():
        log.info("  %-30s  %d", name, cnt)
    log.info("  %-30s  %d", "TOTAL", len(all_alerts))
    log.info("  %-30s  %d", "Deduped (days_active++)", len(to_update))
    log.info("  %-30s  %d", "Inserted (new rows)", rows_written)
    log.info("  %-30s  %d", "Auto-resolved", resolved_count)
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
