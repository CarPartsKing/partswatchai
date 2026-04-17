"""dashboard/server.py — Flask server for the PartsWatch AI morning dashboard.

Serves the single-page dashboard at / and provides all data via a single
/api/dashboard JSON endpoint.  All Supabase queries run server-side so no
credentials are exposed to the browser.

Usage
-----
    python dashboard/server.py          # production (host 0.0.0.0:5000)
    python dashboard/server.py --dev    # Flask debug mode
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.claude_api import PurchasingAssistant
from db.connection import get_client, get_new_client
from utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

app = Flask(__name__, static_folder=str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Chat session store — one PurchasingAssistant per browser session
# ---------------------------------------------------------------------------

_sessions: dict[str, PurchasingAssistant] = {}
_MAX_SESSIONS = 50          # prune oldest when limit reached

_PAGE_SIZE = 1_000
_LOW_SUPPLY_DAYS = 3.0
_FREEZE_THRESHOLD_F = 32.0
_WEATHER_LOOKAHEAD_DAYS = 7
_MAPE_LOOKBACK_DAYS = 7

LOCATION_NAMES: dict[str, str] = {
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

RETIRED_LOCATIONS = {"LOC-014", "LOC-019", "LOC-022", "LOC-023", "LOC-030", "LOC-031"}


def _loc_display(loc_id: str) -> str:
    name = LOCATION_NAMES.get(loc_id)
    return f"{name} ({loc_id})" if name else loc_id


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def _paginate(
    client: Any,
    table: str,
    select: str,
    filters:     dict | None = None,
    gte_filters: dict | None = None,
    lte_filters: dict | None = None,
    eq_bool:     dict | None = None,
    in_filters:  dict | None = None,
    order_col:   str | None = None,
    order_desc:  bool = False,
    limit:       int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    page_size = min(_PAGE_SIZE, limit) if limit else _PAGE_SIZE
    while True:
        q = client.table(table).select(select)
        for col, val in (filters or {}).items():
            q = q.eq(col, val)
        for col, val in (gte_filters or {}).items():
            q = q.gte(col, val)
        for col, val in (lte_filters or {}).items():
            q = q.lte(col, val)
        for col, val in (eq_bool or {}).items():
            q = q.eq(col, val)
        for col, vals in (in_filters or {}).items():
            q = q.in_(col, vals)
        if order_col:
            q = q.order(order_col, desc=order_desc)
        page = q.range(offset, offset + page_size - 1).execute().data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        if limit and len(rows) >= limit:
            rows = rows[:limit]
            break
        offset += page_size
    return rows


# ---------------------------------------------------------------------------
# Data builders — each returns a JSON-serialisable dict/list
# ---------------------------------------------------------------------------

def _build_weather(client: Any, today: date) -> dict:
    end_date = (today + timedelta(days=_WEATHER_LOOKAHEAD_DAYS)).isoformat()
    rows = _paginate(
        client, "weather_log",
        "log_date,temp_min_f,temp_max_f,precipitation_in,"
        "snowfall_in,consecutive_freeze_days,freeze_thaw_cycle",
        gte_filters={"log_date": today.isoformat()},
        lte_filters={"log_date": end_date},
        order_col="log_date",
    )
    today_row = next((r for r in rows if r["log_date"] == today.isoformat()), None)
    forecast = [r for r in rows if r["log_date"] > today.isoformat()]

    freeze_days = [
        r for r in forecast
        if r.get("temp_min_f") is not None and float(r["temp_min_f"]) < _FREEZE_THRESHOLD_F
    ]
    coldest = min(freeze_days, key=lambda r: float(r["temp_min_f"])) if freeze_days else None

    return {
        "today": today_row,
        "forecast": forecast,
        "freeze_warning": len(freeze_days) > 0,
        "freeze_days_count": len(freeze_days),
        "coldest_temp": float(coldest["temp_min_f"]) if coldest else None,
        "coldest_date": coldest["log_date"] if coldest else None,
    }


def _build_alerts(client: Any, today: date) -> dict:
    rows = _paginate(
        client, "alerts",
        "alert_type,severity,sku_id,location_id,supplier_id,message,alert_key",
        filters={"alert_date": today.isoformat()},
        eq_bool={"is_acknowledged": False},
    )
    sev_order = {"critical": 0, "warning": 1, "info": 2}
    rows.sort(key=lambda r: sev_order.get(r.get("severity", "info"), 9))

    summary = {"critical": 0, "warning": 0, "info": 0}
    for r in rows:
        sev = r.get("severity", "info")
        if sev in summary:
            summary[sev] += 1
        r["location_display"] = _loc_display(r["location_id"]) if r.get("location_id") else ""

    return {
        "summary": summary,
        "items": rows,
    }


def _build_reorder(client: Any, today: date) -> dict:
    """Reorder recommendations summary for the most recent run.

    Returns a dict with:
      - kpis:  network roll-ups (totals, urgency split, type split,
               top suppliers, top destination locations)
      - items: top 20 most-urgent unapproved recommendations for the
               table view
      - recommendation_date: the date the recs were written for
    """
    # Use the latest run we actually have rows for, not strictly today —
    # the nightly pipeline writes recs dated for the *next* business day,
    # and a partial day (no run yet) would silently show "no recs".
    try:
        latest = (
            client.table("reorder_recommendations")
            .select("recommendation_date")
            .order("recommendation_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
        rec_date = latest[0]["recommendation_date"] if latest else today.isoformat()
    except Exception:
        rec_date = today.isoformat()

    # Pull EVERY row for that date (full network) so the KPIs reflect the
    # real volume the buying team faces.  The reorder writer batches at
    # 200/row so we can have ~20k+ recs/day.
    rows = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,recommendation_type,urgency,qty_to_order,"
        "supplier_id,days_of_supply_remaining,forecast_model_used,"
        "transfer_from_location,is_approved",
        filters={"recommendation_date": rec_date},
    )

    urgency_order = {"critical": 0, "warning": 1, "high": 1,
                     "normal": 2, "medium": 2, "low": 3}

    by_urgency = {"critical": 0, "warning": 0, "normal": 0}
    by_type = {"po": 0, "transfer": 0}
    by_supplier: dict[str, int] = defaultdict(int)
    by_dest_loc: dict[str, int] = defaultdict(int)
    total_qty = 0.0
    approved = 0

    for r in rows:
        u = (r.get("urgency") or "normal").lower()
        if u in by_urgency:
            by_urgency[u] += 1
        t = (r.get("recommendation_type") or "po").lower()
        if t in by_type:
            by_type[t] += 1
        sup = r.get("supplier_id") or ""
        if sup and t == "po":
            by_supplier[sup] += 1
        loc = r.get("location_id") or ""
        if loc:
            by_dest_loc[loc] += 1
        total_qty += float(r.get("qty_to_order") or 0)
        if r.get("is_approved"):
            approved += 1

    top_suppliers = sorted(by_supplier.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_dest_locs = sorted(by_dest_loc.items(), key=lambda kv: kv[1], reverse=True)[:5]

    # Top-20 most-urgent unapproved for the on-screen action list:
    # urgency DESC (critical → warning → normal), then qty_to_order DESC
    # so the biggest buys float to the top.
    unapproved = [r for r in rows if not r.get("is_approved")]
    unapproved.sort(key=lambda r: (
        urgency_order.get((r.get("urgency") or "low").lower(), 9),
        -float(r.get("qty_to_order") or 0),
    ))
    items = unapproved[:20]
    for r in items:
        r["location_display"] = _loc_display(r.get("location_id", ""))
        if r.get("transfer_from_location"):
            r["transfer_from_display"] = _loc_display(r["transfer_from_location"])

    return {
        "recommendation_date": rec_date,
        "kpis": {
            "total_recommendations":  len(rows),
            "approved":               approved,
            "pending":                len(rows) - approved,
            "total_qty_recommended":  round(total_qty, 1),
            "by_urgency":             by_urgency,
            "by_type":                by_type,
            "top_suppliers": [
                {"supplier_id": sid, "count": n} for sid, n in top_suppliers
            ],
            "top_destinations": [
                {"location_id": lid,
                 "location_display": _loc_display(lid),
                 "count": n}
                for lid, n in top_dest_locs
            ],
        },
        "items": items,
    }


def _build_supplier_health(client: Any, today: date) -> dict:
    cutoff = (today - timedelta(days=90)).isoformat()
    rows = _paginate(
        client, "supplier_scores",
        "supplier_id,score_date,risk_flag,composite_score",
        gte_filters={"score_date": cutoff},
    )
    latest: dict[str, dict] = {}
    for r in rows:
        sid = r.get("supplier_id")
        if not sid:
            continue
        if sid not in latest or r["score_date"] > latest[sid]["score_date"]:
            latest[sid] = r

    counts: dict[str, int] = defaultdict(int)
    red_suppliers: list[dict] = []
    for sid, s in latest.items():
        flag = (s.get("risk_flag") or "unknown").lower()
        counts[flag] += 1
        if flag == "red":
            red_suppliers.append({
                "supplier_id": sid,
                "score": s.get("composite_score"),
            })

    return {
        "green": counts.get("green", 0),
        "amber": counts.get("amber", 0),
        "red": counts.get("red", 0),
        "total": sum(counts.values()),
        "red_suppliers": sorted(red_suppliers, key=lambda r: r.get("score") or 999),
    }


def _build_inventory_health(client: Any, today: date) -> dict:
    """Network-wide inventory health from the latest per-location snapshots.

    inventory_snapshots can have 200K+ rows for a single date, so we never
    paginate the whole thing.  Instead we issue a handful of cheap server-
    side `count='exact', head=True` queries plus one `.limit(N)` fetch for
    the headline stockout examples.  Total round-trips: ~5 + one per active
    location for the per-location stockout breakdown.
    """
    # 1. Find the freshest snapshot date.
    try:
        latest_row = (
            client.table("inventory_snapshots")
            .select("snapshot_date")
            .order("snapshot_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
        latest_date = latest_row[0]["snapshot_date"] if latest_row else today.isoformat()
    except Exception:
        latest_date = today.isoformat()

    def _count(query) -> int:
        try:
            return int(query.execute().count or 0)
        except Exception:
            return 0

    base = lambda: client.table("inventory_snapshots").select(
        "*", count="exact", head=True
    ).eq("snapshot_date", latest_date)

    snapshot_count = _count(base())
    stockout_count = _count(base().eq("is_stockout", True))

    # Pull just the stockout rows (small subset) for examples + per-location
    # aggregation.  We cap at 5000 to stay under Supabase's 8s timeout even
    # in worst-case (the table is indexed on snapshot_date+is_stockout).
    stockout_rows: list[dict] = []
    try:
        stockout_rows = (
            client.table("inventory_snapshots")
            .select("sku_id,location_id")
            .eq("snapshot_date", latest_date)
            .eq("is_stockout", True)
            .limit(5000)
            .execute()
            .data or []
        )
    except Exception:
        pass

    by_location: dict[str, dict[str, int]] = defaultdict(
        lambda: {"snapshots": 0, "stockouts": 0, "below_reorder": 0}
    )
    stockouts: list[dict] = []
    for r in stockout_rows:
        loc = r.get("location_id") or ""
        by_location[loc]["stockouts"] += 1
        if len(stockouts) < 10:
            stockouts.append(r)

    # Pull per-location snapshot counts (cheap aggregation via the index).
    # We can't easily GROUP BY through PostgREST, so issue one head count
    # per active location — at ~27 locations × ~80ms each this is well
    # under a second.
    try:
        loc_ids = [r["location_id"] for r in (
            client.table("locations").select("location_id").execute().data or []
        )]
    except Exception:
        loc_ids = list(by_location.keys())

    for loc in loc_ids:
        snaps = _count(base().eq("location_id", loc))
        if snaps:
            by_location[loc]["snapshots"] = snaps

    # Top 5 locations by stockout count.
    top_stockout_locs = sorted(
        ((loc, stats["stockouts"], stats["snapshots"])
         for loc, stats in by_location.items()
         if stats["stockouts"] > 0),
        key=lambda x: x[1], reverse=True,
    )[:5]

    # The expensive aggregates (below_reorder, overstock, total_on_hand,
    # total_on_order) require a full table scan with arithmetic which the
    # PostgREST `head=True` count can't satisfy.  Surface them as None so
    # the dashboard can render "—" rather than blocking on a multi-minute
    # paginate.  A future RPC view can backfill these.
    below_reorder = None
    overstock = None
    total_on_hand = None
    total_on_order = None

    # Low-supply count from today's reorder recommendations (kept for
    # parity with the old metric — 'days_of_supply < 3').
    try:
        recs = _paginate(
            client, "reorder_recommendations",
            "days_of_supply_remaining",
            filters={"recommendation_date": today.isoformat()},
        )
        low_supply = sum(
            1 for r in recs
            if r.get("days_of_supply_remaining") is not None
            and 0 < float(r["days_of_supply_remaining"]) < _LOW_SUPPLY_DAYS
        )
    except Exception:
        low_supply = 0

    return {
        "snapshot_date":     latest_date,
        "snapshot_count":    snapshot_count,
        "stockout_count":    stockout_count,
        "below_reorder":     below_reorder,
        "overstock_count":   overstock,
        "low_supply_count":  low_supply,
        "total_on_hand_qty": round(total_on_hand, 0) if total_on_hand is not None else None,
        "total_on_order_qty":round(total_on_order, 0) if total_on_order is not None else None,
        "locations_covered": len(by_location),
        "stockout_pairs": [
            {"sku_id": r["sku_id"], "location_id": r["location_id"],
             "location_display": _loc_display(r["location_id"])}
            for r in stockouts
        ],
        "top_stockout_locations": [
            {"location_id": loc,
             "location_display": _loc_display(loc),
             "stockouts": cnt,
             "snapshots": snaps,
             "stockout_pct": round(100.0 * cnt / snaps, 1) if snaps else 0.0}
            for loc, cnt, snaps in top_stockout_locs
        ],
    }


def _build_transfer_activity(client: Any, today: date) -> dict:
    """Recent inter-location transfer activity (last 30 days).

    Surfaces top transfer routes (from→to) and top moving SKUs so the
    buying team can see whether the network is rebalancing itself
    organically before issuing transfer recommendations.
    """
    cutoff = (today - timedelta(days=30)).isoformat()
    try:
        rows = _paginate(
            client, "location_transfers",
            "sku_id,from_location,to_location,transfer_date,"
            "qty_transferred,transfer_cost",
            gte_filters={"transfer_date": cutoff},
        )
    except Exception:
        return {"total_transfers": 0, "total_qty": 0.0,
                "top_routes": [], "top_skus": []}

    by_route: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"qty": 0.0, "count": 0, "cost": 0.0}
    )
    by_sku: dict[str, float] = defaultdict(float)
    total_qty = 0.0
    total_cost = 0.0

    for r in rows:
        f = r.get("from_location") or ""
        t = r.get("to_location") or ""
        sku = r.get("sku_id") or ""
        qty = float(r.get("qty_transferred") or 0)
        cost = float(r.get("transfer_cost") or 0) if r.get("transfer_cost") else 0.0
        by_route[(f, t)]["qty"] += qty
        by_route[(f, t)]["count"] += 1
        by_route[(f, t)]["cost"] += cost
        by_sku[sku] += qty
        total_qty += qty
        total_cost += cost

    top_routes = sorted(
        by_route.items(), key=lambda kv: kv[1]["qty"], reverse=True,
    )[:8]
    top_skus = sorted(by_sku.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "total_transfers": len(rows),
        "total_qty":       round(total_qty, 1),
        "total_cost":      round(total_cost, 2),
        "window_days":     30,
        "top_routes": [
            {
                "from_location":     f,
                "to_location":       t,
                "from_display":      _loc_display(f),
                "to_display":        _loc_display(t),
                "qty":               round(stats["qty"], 1),
                "transfer_count":    int(stats["count"]),
                "cost":              round(stats["cost"], 2),
            }
            for (f, t), stats in top_routes
        ],
        "top_skus": [
            {"sku_id": sku, "qty": round(qty, 1)}
            for sku, qty in top_skus
        ],
    }


def _build_forecast_accuracy(client: Any, today: date) -> list[dict]:
    """Forecast MAPE by ABC class × model.

    The forecast_results table is 4M+ rows and there is no PostgREST view
    that can compute MAPE server-side, so the full Python join requires
    pulling hundreds of thousands of rows — well past Supabase's 8s
    statement timeout.  Until a SQL view (e.g. v_forecast_accuracy_daily)
    is added, prefer the pre-computed ml/accuracy.py output table instead.
    """
    yesterday = (today - timedelta(days=1)).isoformat()
    cutoff    = (today - timedelta(days=_MAPE_LOOKBACK_DAYS)).isoformat()

    # Try the materialised accuracy table first (cheap, indexed by date).
    try:
        rows = (
            client.table("forecast_accuracy_daily")
            .select("abc_class,model_type,mape_pct,n_obs")
            .gte("accuracy_date", cutoff)
            .lte("accuracy_date", yesterday)
            .limit(2000)
            .execute()
            .data or []
        )
    except Exception:
        rows = []

    if not rows:
        return []

    # Aggregate to one row per (abc_class, model_type).
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"weighted_ape": 0.0, "n": 0}
    )
    for r in rows:
        key = (r.get("abc_class") or "?", r.get("model_type") or "?")
        n   = int(r.get("n_obs") or 0)
        if n <= 0:
            continue
        agg[key]["weighted_ape"] += float(r.get("mape_pct") or 0) * n
        agg[key]["n"]            += n

    return [
        {
            "abc_class":       abc,
            "model_type":      model,
            "mape_pct":        round(v["weighted_ape"] / v["n"], 1) if v["n"] else 0.0,
            "n_obs":           v["n"],
            "above_threshold": (v["weighted_ape"] / v["n"]) > 25.0 if v["n"] else False,
        }
        for (abc, model), v in sorted(agg.items())
    ]


def _build_forecast_accuracy_legacy(client: Any, today: date) -> list[dict]:
    """Original on-the-fly MAPE compute — kept for reference, not wired up."""
    window_start = (today - timedelta(days=_MAPE_LOOKBACK_DAYS)).isoformat()
    yesterday    = (today - timedelta(days=1)).isoformat()
    fc_rows = _paginate(
        client, "forecast_results",
        "sku_id,location_id,forecast_date,model_type,predicted_qty,run_date",
        gte_filters={"forecast_date": window_start},
        lte_filters={"forecast_date": yesterday},
        in_filters={"model_type": ["lightgbm", "rolling_avg"]},
    )
    if not fc_rows:
        return []

    latest_run: dict[tuple[str, str, str], str] = {}
    for r in fc_rows:
        key = (r["sku_id"], r["location_id"], r["model_type"])
        if (r.get("run_date") or "") > latest_run.get(key, ""):
            latest_run[key] = r["run_date"]

    forecast_map: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in fc_rows:
        key = (r["sku_id"], r["location_id"], r["model_type"])
        if r.get("run_date") == latest_run.get(key):
            d = str(r.get("forecast_date", ""))[:10]
            if d:
                forecast_map[key][d] = float(r.get("predicted_qty") or 0)

    # Restrict the sales scan to only the SKUs we have a forecast for.  The
    # raw sales_transactions table has millions of rows and an unfiltered
    # 7-day pull blows past Supabase's 8s statement timeout (57014).  Most
    # forecasts cover only a few thousand SKUs, so an `in_filters` batch is
    # dramatically cheaper.
    sku_ids = list({k[0] for k in forecast_map})
    actuals: dict[tuple[str, str, str], float] = defaultdict(float)
    actuals_all: dict[tuple[str, str], float] = defaultdict(float)
    _IN_BATCH = 300
    for i in range(0, len(sku_ids), _IN_BATCH):
        batch = sku_ids[i:i + _IN_BATCH]
        sales_rows = _paginate(
            client, "sales_transactions",
            "sku_id,location_id,transaction_date,qty_sold",
            gte_filters={"transaction_date": window_start},
            lte_filters={"transaction_date": yesterday},
            in_filters={"sku_id": batch},
        )
        for r in sales_rows:
            d = str(r.get("transaction_date", ""))[:10]
            actuals[(r["sku_id"], r["location_id"], d)] += float(r.get("qty_sold") or 0)
            actuals_all[(r["sku_id"], d)] += float(r.get("qty_sold") or 0)

    abc_map: dict[str, str] = {}
    for i in range(0, len(sku_ids), _IN_BATCH):
        batch = sku_ids[i:i + _IN_BATCH]
        sku_rows = _paginate(client, "sku_master", "sku_id,abc_class",
                             in_filters={"sku_id": batch})
        for r in sku_rows:
            abc_map[r["sku_id"]] = r.get("abc_class") or "?"

    ape_by_class: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (sku_id, loc_id, model), date_qty in forecast_map.items():
        abc = abc_map.get(sku_id, "?")
        for d, predicted in date_qty.items():
            actual = (
                actuals_all.get((sku_id, d), 0.0)
                if loc_id == "ALL"
                else actuals.get((sku_id, loc_id, d), 0.0)
            )
            ape = abs(actual - predicted) / max(actual, 1.0)
            ape_by_class[(abc, model)].append(ape)

    results = []
    for (abc, model), apes in sorted(ape_by_class.items()):
        mape = (sum(apes) / len(apes)) * 100
        results.append({
            "abc_class": abc,
            "model_type": model,
            "mape_pct": round(mape, 1),
            "n_obs": len(apes),
            "above_threshold": mape > 25.0,
        })
    return results


def _build_location_performance(client: Any, today: date) -> dict:
    try:
        loc_rows = _paginate(
            client, "locations",
            "location_id,location_name,location_tier,composite_tier_score,"
            "revenue_score,sku_breadth_score,fill_rate_score,return_rate_score,is_active",
        )
    except Exception:
        loc_rows = _paginate(
            client, "locations",
            "location_id,location_tier,composite_tier_score,"
            "revenue_score,sku_breadth_score,fill_rate_score,return_rate_score",
        )
    tiers: dict[int, list[dict]] = defaultdict(list)
    scores: dict[str, float] = {}
    for r in loc_rows:
        if r.get("is_active") is False:
            continue
        tier = int(r.get("location_tier") or 2)
        loc_id = r.get("location_id", "?")
        loc_name = r.get("location_name") or LOCATION_NAMES.get(loc_id) or loc_id
        tiers[tier].append({
            "location_id": loc_id,
            "location_name": loc_name,
            "location_display": f"{loc_name} ({loc_id})",
            "composite_tier_score": float(r.get("composite_tier_score") or 0),
            "revenue_score": float(r.get("revenue_score") or 0),
            "sku_breadth_score": float(r.get("sku_breadth_score") or 0),
            "fill_rate_score": float(r.get("fill_rate_score") or 0),
            "return_rate_score": float(r.get("return_rate_score") or 0),
        })
        if r.get("composite_tier_score") is not None:
            scores[loc_id] = float(r["composite_tier_score"])

    for tier in tiers:
        tiers[tier].sort(key=lambda x: x["composite_tier_score"], reverse=True)

    tier3_locs = [l["location_id"] for l in tiers.get(3, [])]
    tier3_critical: list[str] = []
    if tier3_locs:
        alert_rows = _paginate(
            client, "alerts",
            "location_id,severity",
            filters={"alert_date": today.isoformat()},
            eq_bool={"is_acknowledged": False},
            in_filters={"location_id": tier3_locs},
        )
        tier3_critical = sorted({
            r["location_id"] for r in alert_rows
            if r.get("severity") == "critical"
        })

    return {
        "tier1": tiers.get(1, []),
        "tier2": tiers.get(2, []),
        "tier3": tiers.get(3, []),
        "tier3_critical": tier3_critical,
        "scores": scores,
    }


def _build_network_kpis(client: Any, today: date) -> dict:
    try:
        sku_total = client.table("sku_master").select("sku_id", count="exact").limit(1).execute()
        total_skus = sku_total.count or 0
    except Exception:
        total_skus = 0

    abc_counts: dict[str, int] = {}
    for cls in ("A", "B", "C"):
        try:
            r = client.table("sku_master").select("sku_id", count="exact").eq("abc_class", cls).limit(1).execute()
            abc_counts[cls] = r.count or 0
        except Exception:
            abc_counts[cls] = 0

    try:
        anom_r = client.table("sales_transactions").select("transaction_id", count="exact").eq("is_anomaly", True).limit(1).execute()
        anomaly_count = anom_r.count or 0
    except Exception:
        anomaly_count = 0

    try:
        loc_rows = _paginate(client, "locations", "location_id,is_active")
        location_count = sum(1 for r in loc_rows if r.get("is_active") is not False)
    except Exception:
        loc_rows = _paginate(client, "locations", "location_id")
        location_count = len(loc_rows)

    try:
        sup_rows = _paginate(client, "supplier_scores", "supplier_id,score_date")
        latest: dict[str, str] = {}
        for r in sup_rows:
            sid = r.get("supplier_id", "")
            if sid not in latest or (r.get("score_date") or "") > latest[sid]:
                latest[sid] = r.get("score_date") or ""
        supplier_count = len(latest)
    except Exception:
        supplier_count = 0

    return {
        "total_skus": total_skus,
        "abc_counts": abc_counts,
        "anomaly_count": anomaly_count,
        "location_count": location_count,
        "supplier_count": supplier_count,
    }


def _build_top_skus(client: Any, today: date) -> list[dict]:
    rows = _paginate(
        client, "sku_master",
        "sku_id,abc_class,avg_weekly_units,part_category,brand,description",
        filters={"abc_class": "A"},
        order_col="avg_weekly_units",
        order_desc=True,
        limit=15,
    )
    result = []
    for r in rows:
        avg_wk = float(r.get("avg_weekly_units") or 0)
        result.append({
            "sku_id": r["sku_id"],
            "abc_class": r.get("abc_class", "?"),
            "avg_weekly_units": round(avg_wk, 1),
            "category": r.get("part_category") or "",
            "brand": r.get("brand") or "",
            "description": r.get("description") or "",
        })
    return result


def _build_anomaly_summary(client: Any, today: date) -> dict:
    try:
        anom_r = client.table("sales_transactions").select("transaction_id", count="exact").eq("is_anomaly", True).limit(1).execute()
        total_flagged = anom_r.count or 0
    except Exception:
        total_flagged = 0

    # Tight 30-day window (was 90) and order by transaction_date (indexed)
    # instead of qty_sold (not indexed) so Postgres can use the
    # is_anomaly+transaction_date composite index and stop after 1000 rows.
    cutoff = (today - timedelta(days=30)).isoformat()
    recent_anom: list[dict] = []
    try:
        recent_anom = (
            client.table("sales_transactions")
            .select("sku_id,location_id,transaction_date,qty_sold,unit_price,total_revenue")
            .eq("is_anomaly", True)
            .gte("transaction_date", cutoff)
            .order("transaction_date", desc=True)
            .limit(1000)
            .execute()
            .data or []
        )
    except Exception:
        recent_anom = []

    sorted_by_qty = sorted(
        recent_anom,
        key=lambda r: float(r.get("qty_sold") or 0),
        reverse=True,
    )
    top_high = sorted_by_qty[:8]
    top_low  = sorted_by_qty[-5:][::-1] if len(sorted_by_qty) >= 5 else []

    loc_dist: dict[str, dict] = {}
    sample_locs = ["LOC-008", "LOC-025", "LOC-004", "LOC-001", "LOC-005"]
    for loc in sample_locs:
        try:
            r = client.table("sales_transactions").select("transaction_id", count="exact").eq("is_anomaly", True).eq("location_id", loc).gte("transaction_date", cutoff).limit(1).execute()
            loc_dist[loc] = {"count": r.count or 0, "display": _loc_display(loc)}
        except Exception:
            loc_dist[loc] = {"count": 0, "display": _loc_display(loc)}

    return {
        "total_flagged": total_flagged,
        "top_high": [
            {
                "sku_id": r["sku_id"],
                "location_id": r["location_id"],
                "location_display": _loc_display(r["location_id"]),
                "date": r.get("transaction_date", ""),
                "qty": float(r.get("qty_sold") or 0),
                "price": float(r.get("unit_price") or 0),
                "revenue": float(r.get("total_revenue") or 0),
            }
            for r in top_high
        ],
        "top_low": [
            {
                "sku_id": r["sku_id"],
                "location_id": r["location_id"],
                "location_display": _loc_display(r["location_id"]),
                "date": r.get("transaction_date", ""),
                "qty": float(r.get("qty_sold") or 0),
                "revenue": float(r.get("total_revenue") or 0),
            }
            for r in top_low
        ],
        "location_distribution": loc_dist,
    }


def _build_supplier_detail(client: Any, today: date) -> list[dict]:
    score_rows = _paginate(
        client, "supplier_scores",
        "supplier_id,supplier_name,score_date,composite_score,risk_flag,"
        "fill_rate_pct,on_time_delivery_pct,avg_lead_time_days,lead_time_variance_avg",
    )
    latest: dict[str, dict] = {}
    for r in score_rows:
        sid = r.get("supplier_id", "")
        if sid not in latest or (r.get("score_date") or "") > latest[sid].get("score_date", ""):
            latest[sid] = r

    result = []
    for sid, s in sorted(latest.items(), key=lambda x: float(x[1].get("composite_score") or 999)):
        po_rows = _paginate(
            client, "purchase_orders",
            "po_number,line_number,sku_id,qty_ordered,qty_received,unit_cost,"
            "status,expected_delivery_date,lead_time_variance",
            filters={"supplier_id": sid},
        )
        open_pos = [p for p in po_rows if p.get("status") not in ("delivered", "cancelled", "closed")]
        delivered = [p for p in po_rows if p.get("status") == "delivered"]

        total_ordered = sum(float(p.get("qty_ordered") or 0) for p in delivered)
        total_received = sum(float(p.get("qty_received") or 0) for p in delivered)
        actual_fill = (total_received / total_ordered * 100) if total_ordered > 0 else None

        ltv_vals = [float(p.get("lead_time_variance") or 0) for p in delivered if p.get("lead_time_variance") is not None]
        avg_ltv = (sum(ltv_vals) / len(ltv_vals)) if ltv_vals else None

        open_value = sum(float(p.get("qty_ordered") or 0) * float(p.get("unit_cost") or 0) for p in open_pos)

        open_po_list = []
        for p in open_pos[:6]:
            open_po_list.append({
                "po_number": p.get("po_number", ""),
                "sku_id": p.get("sku_id", ""),
                "qty_ordered": float(p.get("qty_ordered") or 0),
                "status": p.get("status", ""),
                "expected_delivery": p.get("expected_delivery_date", ""),
                "value": round(float(p.get("qty_ordered") or 0) * float(p.get("unit_cost") or 0), 2),
            })

        result.append({
            "supplier_id": sid,
            "supplier_name": s.get("supplier_name") or "",
            "composite_score": float(s.get("composite_score") or 0),
            "risk_flag": (s.get("risk_flag") or "unknown").lower(),
            "fill_rate_pct": s.get("fill_rate_pct"),
            "on_time_delivery_pct": s.get("on_time_delivery_pct"),
            "avg_lead_time_days": float(s.get("avg_lead_time_days") or 0) if s.get("avg_lead_time_days") else None,
            "lead_time_variance_avg": float(s.get("lead_time_variance_avg") or 0) if s.get("lead_time_variance_avg") else None,
            "actual_fill_rate": round(actual_fill, 1) if actual_fill is not None else None,
            "avg_lead_time_variance": round(avg_ltv, 1) if avg_ltv is not None else None,
            "open_po_count": len(open_pos),
            "open_po_value": round(open_value, 2),
            "delivered_count": len(delivered),
            "open_pos": open_po_list,
        })
    return result


def _build_pipeline_status(client: Any, today: date) -> list[dict]:
    """Last-run summary for the four core nightly stages.

    There is no pipeline_runs ledger table, so instead of guessing at
    job state we surface what the buyer actually cares about: when did
    each stage last produce output, and how much did it produce.

    Returns a list of dicts shaped:
      {stage, status, last_run, row_count, label}
    """
    def _latest(table: str, ts_col: str, date_only: bool = False) -> tuple[str, int]:
        """Return (latest_value, total_row_count_for_that_value)."""
        try:
            rows = (
                client.table(table)
                .select(ts_col)
                .order(ts_col, desc=True)
                .limit(1)
                .execute()
                .data or []
            )
            if not rows:
                return ("", 0)
            latest_val = rows[0].get(ts_col)
            if not latest_val:
                return ("", 0)
            # If it's a timestamp, slice to the date for the count filter.
            if date_only:
                count_filter = latest_val
            else:
                count_filter = str(latest_val)[:10]
                # Use a half-open day range so we count the whole day even
                # though latest_val is just one row's timestamp.
                next_day = (date.fromisoformat(count_filter) + timedelta(days=1)).isoformat()
                count_resp = (
                    client.table(table)
                    .select("*", count="exact", head=True)
                    .gte(ts_col, count_filter)
                    .lt(ts_col, next_day)
                    .execute()
                )
                return (str(latest_val), int(count_resp.count or 0))
            count_resp = (
                client.table(table)
                .select("*", count="exact", head=True)
                .eq(ts_col, count_filter)
                .execute()
            )
            return (str(latest_val), int(count_resp.count or 0))
        except Exception:
            log.exception("Pipeline status probe failed for %s.%s", table, ts_col)
            return ("", 0)

    # Each stage maps to the table whose freshness it owns.
    extract_ts,   extract_n   = _latest("sales_transactions",       "transaction_date",   date_only=True)
    forecast_ts,  forecast_n  = _latest("forecast_results",         "forecast_date",      date_only=True)
    reorder_ts,   reorder_n   = _latest("reorder_recommendations",  "recommendation_date", date_only=True)
    alerts_ts,    alerts_n    = _latest("alerts",                   "alert_date",         date_only=True)

    def _row(stage: str, label: str, ts: str, n: int) -> dict:
        return {
            "stage":     stage,
            "label":     label,
            "last_run":  ts or None,
            "row_count": n,
            "status":    "ok" if ts else "never_run",
        }

    return [
        _row("extract",  "Last extract",  extract_ts,  extract_n),
        _row("forecast", "Last forecast", forecast_ts, forecast_n),
        _row("reorder",  "Last reorder",  reorder_ts,  reorder_n),
        _row("alerts",   "Last alerts",   alerts_ts,   alerts_n),
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return send_from_directory(str(Path(__file__).parent), "index.html")


@app.route("/api/dashboard")
def dashboard_data():
    """Return all dashboard sections as a single JSON payload."""
    t0 = time.perf_counter()
    today = date.today()
    try:
        client = get_client()
    except Exception as exc:
        log.exception("Supabase connection failed.")
        return jsonify({"error": str(exc)}), 503

    payload: dict[str, Any] = {
        "generated_at": today.isoformat(),
    }

    sections = [
        ("network_kpis",         _build_network_kpis),
        ("weather",              _build_weather),
        ("alerts",               _build_alerts),
        ("reorder",              _build_reorder),
        ("supplier_health",      _build_supplier_health),
        ("supplier_detail",      _build_supplier_detail),
        ("inventory_health",     _build_inventory_health),
        ("transfers",            _build_transfer_activity),
        ("forecast_accuracy",    _build_forecast_accuracy),
        ("location_performance", _build_location_performance),
        ("top_skus",             _build_top_skus),
        ("anomaly_summary",      _build_anomaly_summary),
        ("pipeline_status",      _build_pipeline_status),
    ]

    # Run all sections in parallel.  Each section gets its OWN Supabase
    # client because supabase-py's underlying httpx session is not safe
    # to share across threads under load.  Falls back to the shared
    # client only if get_new_client() fails.
    section_timings: dict[str, int] = {}

    def _run_section(name: str, fn: Any) -> tuple[str, Any, int]:
        s_t0 = time.perf_counter()
        log.info("[section] %s START", name)
        try:
            section_client = get_new_client()
        except Exception:
            section_client = client
        try:
            data = fn(section_client, today)
        except Exception:
            log.exception("Dashboard section '%s' failed.", name)
            data = None
        ms = int((time.perf_counter() - s_t0) * 1000)
        log.info("[section] %s DONE in %dms", name, ms)
        return name, data, ms

    with ThreadPoolExecutor(max_workers=min(8, len(sections))) as ex:
        futures = [ex.submit(_run_section, name, fn) for name, fn in sections]
        for fut in as_completed(futures):
            name, data, ms = fut.result()
            payload[name] = data
            section_timings[name] = ms

    elapsed_ms = (time.perf_counter() - t0) * 1000
    payload["query_ms"] = round(elapsed_ms)
    payload["section_ms"] = section_timings
    slowest = sorted(section_timings.items(), key=lambda kv: kv[1], reverse=True)[:3]
    log.info(
        "Dashboard data served in %.0fms (slowest: %s)",
        elapsed_ms,
        ", ".join(f"{n}={ms}ms" for n, ms in slowest),
    )
    return jsonify(payload)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle a chat message, maintaining per-session conversation history.

    Request body (JSON):
        session_id  str  — browser-generated UUID; created server-side if absent
        message     str  — the buyer's question

    Response (JSON):
        reply       str  — Claude's response
        session_id  str  — echo the session_id (so a new one can be captured)
    """
    body = request.get_json(silent=True) or {}
    session_id = (body.get("session_id") or "").strip()
    message    = (body.get("message")    or "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400

    if not session_id:
        session_id = str(uuid.uuid4())

    # Prune oldest session when limit reached
    if session_id not in _sessions and len(_sessions) >= _MAX_SESSIONS:
        oldest = next(iter(_sessions))
        del _sessions[oldest]
        log.debug("Session pruned: %s  (limit=%d)", oldest, _MAX_SESSIONS)

    if session_id not in _sessions:
        try:
            _sessions[session_id] = PurchasingAssistant(get_client())
            log.info("New chat session created: %s", session_id)
        except Exception as exc:
            log.exception("Failed to create PurchasingAssistant for session %s", session_id)
            return jsonify({"error": str(exc)}), 503

    assistant = _sessions[session_id]
    try:
        reply = assistant.chat(message)
        return jsonify({"reply": reply, "session_id": session_id})
    except Exception as exc:
        log.exception("Chat failed for session %s", session_id)
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = "--dev" in sys.argv
    log.info("Starting PartsWatch AI dashboard on port %d  debug=%s", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
