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

# Locations that exist in source data but should never appear on the
# operations dashboard.  Includes retired physical locations *and* virtual
# locations (LOC-021 INTERNET) — INTERNET is e-commerce throughput, not a
# brick-and-mortar branch, so it would distort location-performance and
# transfer-routing panels if shown.
EXCLUDED_DISPLAY_LOCATIONS = RETIRED_LOCATIONS | {"LOC-021"}


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
    not_null_cols: list[str] | None = None,
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
        for col in (not_null_cols or []):
            q = q.not_.is_(col, "null")
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


def _build_dead_stock(client: Any, today: date) -> dict:
    """Dead-stock summary for the most recent dead_stock pipeline run.

    Returns:
        kpis:   capital-at-risk totals (LIQUIDATE + MARKDOWN)
        top10:  top-10 LIQUIDATE candidates by inventory dollar value
        report_date: the date of the report being shown
    """
    # Latest report — same pattern as _build_reorder so a missed nightly
    # run still shows the most recent valid data instead of an empty card.
    try:
        latest = (
            client.table("dead_stock_recommendations")
            .select("report_date")
            .order("report_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
        report_date = latest[0]["report_date"] if latest else today.isoformat()
    except Exception:
        report_date = today.isoformat()

    rows = _paginate(
        client, "dead_stock_recommendations",
        "sku_id,location_id,classification,action,total_inv_value,"
        "qty_on_hand,days_since_sale,sale_frequency,abc_class,supplier_id",
        filters={"report_date": report_date},
    )

    liquidate = [r for r in rows if r.get("classification") == "LIQUIDATE"]
    markdown  = [r for r in rows if r.get("classification") == "MARKDOWN"]

    liquidate_value = sum(float(r.get("total_inv_value") or 0) for r in liquidate)
    markdown_value  = sum(float(r.get("total_inv_value") or 0) for r in markdown)

    top10 = sorted(
        liquidate,
        key=lambda r: float(r.get("total_inv_value") or 0),
        reverse=True,
    )[:10]
    for r in top10:
        r["location_display"] = _loc_display(r.get("location_id", ""))

    return {
        "report_date": report_date,
        "kpis": {
            "capital_at_risk":   round(liquidate_value + markdown_value, 2),
            "liquidate_count":   len(liquidate),
            "liquidate_value":   round(liquidate_value, 2),
            "markdown_count":    len(markdown),
            "markdown_value":    round(markdown_value, 2),
            "total_positions":   len(rows),
        },
        "top10": top10,
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
    # Headline KPI must match the per-location rollup below, which excludes
    # retired branches + virtual INTERNET (LOC-021).  Counting unfiltered
    # would inflate the dashboard total above what any branch sees.
    excluded_list = list(EXCLUDED_DISPLAY_LOCATIONS)
    try:
        stockout_count = int(
            base().eq("is_stockout", True)
                  .not_.in_("location_id", excluded_list)
                  .execute().count or 0
        )
    except Exception:
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
        # Skip excluded (retired + virtual INTERNET) locations from the
        # network-wide stockout rollup so the panel only shows branches
        # where stockouts are actionable.
        if loc in EXCLUDED_DISPLAY_LOCATIONS:
            continue
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
    """Transfer opportunities from the latest reorder run.

    Reads the engine's transfer-type recommendations
    (`reorder_recommendations.recommendation_type = 'transfer'`) for the
    most recent recommendation_date, aggregates them into top routes and
    top moving SKUs, and surfaces a network-rebalance summary.

    Excludes any route that touches an EXCLUDED_DISPLAY_LOCATIONS member
    (retired branches + virtual INTERNET) so logistics doesn't see ghost
    moves.
    """
    # Find the latest run that actually has TRANSFER recs (filter on
    # recommendation_type, not just date) — if the most recent reorder
    # cycle was PO-only we still want to surface the prior day's
    # transfer opportunities rather than show an empty panel.
    try:
        latest = (
            client.table("reorder_recommendations")
            .select("recommendation_date")
            .eq("recommendation_type", "transfer")
            .order("recommendation_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        latest = []
    if not latest:
        return {"total_transfers": 0, "total_qty": 0.0,
                "top_routes": [], "top_skus": [],
                "recommendation_date": today.isoformat()}
    rec_date = latest[0]["recommendation_date"]

    try:
        rows = _paginate(
            client, "reorder_recommendations",
            "sku_id,location_id,transfer_from_location,qty_to_order,urgency",
            filters={
                "recommendation_date": rec_date,
                "recommendation_type": "transfer",
            },
        )
    except Exception:
        return {"total_transfers": 0, "total_qty": 0.0,
                "top_routes": [], "top_skus": [],
                "recommendation_date": rec_date}

    by_route: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"qty": 0.0, "count": 0, "critical": 0}
    )
    by_sku: dict[str, float] = defaultdict(float)
    total_qty = 0.0
    valid_rows = 0

    for r in rows:
        f = r.get("transfer_from_location") or ""
        t = r.get("location_id") or ""
        if not f or not t:
            continue
        if f in EXCLUDED_DISPLAY_LOCATIONS or t in EXCLUDED_DISPLAY_LOCATIONS:
            continue
        sku = r.get("sku_id") or ""
        qty = float(r.get("qty_to_order") or 0)
        by_route[(f, t)]["qty"] += qty
        by_route[(f, t)]["count"] += 1
        if (r.get("urgency") or "").lower() == "critical":
            by_route[(f, t)]["critical"] += 1
        by_sku[sku] += qty
        total_qty += qty
        valid_rows += 1

    top_routes = sorted(
        by_route.items(), key=lambda kv: kv[1]["qty"], reverse=True,
    )[:8]
    top_skus = sorted(by_sku.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "recommendation_date": rec_date,
        "total_transfers": valid_rows,
        "total_qty":       round(total_qty, 1),
        "total_cost":      0.0,            # cost not modeled in recs yet
        "window_days":     1,               # single-day rec snapshot
        "top_routes": [
            {
                "from_location":     f,
                "to_location":       t,
                "from_display":      _loc_display(f),
                "to_display":        _loc_display(t),
                "qty":               round(stats["qty"], 1),
                "transfer_count":    int(stats["count"]),
                "critical_count":    int(stats["critical"]),
                "cost":              0.0,
            }
            for (f, t), stats in top_routes
        ],
        "top_skus": [
            {"sku_id": sku, "qty": round(qty, 1)}
            for sku, qty in top_skus
        ],
    }


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
    loc_index: dict[str, dict] = {}

    for r in loc_rows:
        if r.get("is_active") is False:
            continue
        loc_id = r.get("location_id", "?")
        if loc_id in EXCLUDED_DISPLAY_LOCATIONS:
            continue
        tier = int(r.get("location_tier") or 2)
        loc_name = r.get("location_name") or LOCATION_NAMES.get(loc_id) or loc_id
        entry: dict = {
            "location_id":          loc_id,
            "location_name":        loc_name,
            "location_display":     f"{loc_name} ({loc_id})",
            "composite_tier_score": float(r.get("composite_tier_score") or 0),
            "revenue_score":        float(r.get("revenue_score") or 0),
            "sku_breadth_score":    float(r.get("sku_breadth_score") or 0),
            "fill_rate_score":      float(r.get("fill_rate_score") or 0),
            "return_rate_score":    float(r.get("return_rate_score") or 0),
            "sales_90d":            None,
            "gp_90d":               None,
            "gp_pct":               None,
            "sales_trend_pct":      None,
            "churn_count":          0,
        }
        tiers[tier].append(entry)
        loc_index[loc_id] = entry
        if r.get("composite_tier_score") is not None:
            scores[loc_id] = float(r["composite_tier_score"])

    for tier in tiers:
        tiers[tier].sort(key=lambda x: x["composite_tier_score"], reverse=True)

    # GP & trend — one RPC call returns all locations
    current_start = (today - timedelta(days=90)).isoformat()
    prior_end     = (today - timedelta(days=91)).isoformat()
    prior_start   = (today - timedelta(days=180)).isoformat()
    try:
        gp_rows = client.rpc("get_all_locations_gp_summary", {
            "p_current_start": current_start,
            "p_prior_start":   prior_start,
            "p_prior_end":     prior_end,
        }).execute().data or []
        for r in gp_rows:
            loc_id = r.get("location_id") or ""
            if loc_id not in loc_index:
                continue
            sales_90d   = float(r.get("sales_90d")       or 0)
            gp_90d      = float(r.get("gp_90d")          or 0)
            prior_sales = float(r.get("prior_sales_90d") or 0)
            entry = loc_index[loc_id]
            entry["sales_90d"] = round(sales_90d, 2)
            entry["gp_90d"]    = round(gp_90d, 2)
            entry["gp_pct"]    = round(gp_90d / sales_90d * 100, 1) if sales_90d > 0 else None
            entry["sales_trend_pct"] = (
                round((sales_90d - prior_sales) / prior_sales * 100, 1)
                if prior_sales > 0 else None
            )
    except Exception:
        log.exception("get_all_locations_gp_summary RPC failed.")

    # Churn count per location (CHURNED + DECLINING)
    try:
        churn_rows = _paginate(
            client, "customer_churn_flags",
            "location_id,flag",
            in_filters={"flag": ["CHURNED", "DECLINING"]},
        )
        churn_by_loc: dict[str, int] = defaultdict(int)
        for r in churn_rows:
            churn_by_loc[r["location_id"]] += 1
        for loc_id, entry in loc_index.items():
            entry["churn_count"] = churn_by_loc.get(loc_id, 0)
    except Exception:
        pass

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
    start_date = (today - timedelta(days=90)).isoformat()
    try:
        rows = client.rpc("get_top_skus_by_gp", {
            "p_start_date": start_date,
        }).execute().data or []
    except Exception:
        log.exception("get_top_skus_by_gp RPC failed.")
        rows = []
    result = []
    for r in rows:
        total_gp    = float(r.get("total_gp")    or 0)
        total_sales = float(r.get("total_sales") or 0)
        result.append({
            "sku_id":         r.get("sku_id") or "",
            "total_gp":       round(total_gp, 2),
            "total_sales":    round(total_sales, 2),
            "gp_pct":         round(total_gp / total_sales * 100, 1) if total_sales > 0 else None,
            "location_count": int(r.get("location_count") or 0),
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


def _build_morning_brief(client: Any, today: date) -> dict:
    """Single-line headline counts for the morning-brief banner.

    Reuses the same data the individual panels query but pre-aggregated
    so the header can render in one render pass without waiting for
    every panel to hydrate.
    """
    today_iso = today.isoformat()
    try:
        crit = (
            client.table("alerts")
            .select("alert_key", count="exact", head=True)
            .eq("alert_date", today_iso)
            .eq("severity", "critical")
            .eq("is_acknowledged", False)
            .execute()
        )
        critical_alerts = int(crit.count or 0)
    except Exception:
        critical_alerts = 0

    # Latest reorder run, then count critical/warning unapproved.
    urgent_reorders = 0
    transfer_ops = 0
    try:
        latest = (
            client.table("reorder_recommendations")
            .select("recommendation_date")
            .order("recommendation_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
        if latest:
            rec_date = latest[0]["recommendation_date"]
            ur = (
                client.table("reorder_recommendations")
                .select("recommendation_id", count="exact", head=True)
                .eq("recommendation_date", rec_date)
                .eq("is_approved", False)
                .in_("urgency", ["critical", "warning", "high"])
                .execute()
            )
            urgent_reorders = int(ur.count or 0)
            tr = (
                client.table("reorder_recommendations")
                .select("recommendation_id", count="exact", head=True)
                .eq("recommendation_date", rec_date)
                .eq("recommendation_type", "transfer")
                .execute()
            )
            transfer_ops = int(tr.count or 0)
    except Exception:
        pass

    return {
        "critical_alerts": critical_alerts,
        "urgent_reorders": urgent_reorders,
        "transfer_ops":    transfer_ops,
    }


def _build_critical_actions(client: Any, today: date) -> list[dict]:
    """Aggregated 'do this now' list across alerts, POs, and supplier risk.

    Items:
      - Critical stockouts on active-demand SKUs (today's critical alerts
        with alert_type LIKE 'stockout%')
      - Overdue purchase orders (status='open', expected_delivery < today)
      - RED-flagged suppliers with at least one open PO

    Returns one combined list, severity-sorted, capped at 30 entries so
    the panel stays scannable.
    """
    today_iso = today.isoformat()
    actions: list[dict] = []

    # --- 1. Today's critical alerts (any type — buyer triages) ---
    try:
        alerts = _paginate(
            client, "alerts",
            "alert_type,severity,sku_id,location_id,supplier_id,message,alert_key",
            filters={"alert_date": today_iso, "severity": "critical"},
            eq_bool={"is_acknowledged": False},
        )
    except Exception:
        alerts = []
    for a in alerts[:20]:
        loc = a.get("location_id") or ""
        actions.append({
            "kind":        "alert",
            "severity":    "critical",
            "title":       (a.get("alert_type") or "alert").replace("_", " ").title(),
            "description": (a.get("message") or "")[:200],
            "sku_id":      a.get("sku_id"),
            "location_id": loc,
            "location_display": _loc_display(loc) if loc else "",
            "supplier_id": a.get("supplier_id"),
            "alert_key":   a.get("alert_key"),
            "action":      "Acknowledge",
        })

    # --- 2. Overdue open POs ---
    try:
        po_rows = _paginate(
            client, "purchase_orders",
            "po_number,sku_id,supplier_id,qty_ordered,unit_cost,"
            "expected_delivery_date,status",
            filters={"status": "open"},
        )
    except Exception:
        po_rows = []
    overdue = [
        p for p in po_rows
        if p.get("expected_delivery_date")
        and str(p["expected_delivery_date"])[:10] < today_iso
    ]
    overdue.sort(key=lambda p: str(p.get("expected_delivery_date") or ""))
    for p in overdue[:15]:
        try:
            value = float(p.get("qty_ordered") or 0) * float(p.get("unit_cost") or 0)
        except Exception:
            value = 0.0
        actions.append({
            "kind":        "overdue_po",
            "severity":    "critical",
            "title":       "Overdue PO",
            "description": (
                f"PO {p.get('po_number','')} for {p.get('sku_id','')} from "
                f"{p.get('supplier_id','')} — expected "
                f"{p.get('expected_delivery_date','')}, qty {p.get('qty_ordered',0)} "
                f"(${value:,.0f})"
            ),
            "sku_id":      p.get("sku_id"),
            "supplier_id": p.get("supplier_id"),
            "po_number":   p.get("po_number"),
            "value":       round(value, 2),
            "action":      "Chase supplier",
        })

    # --- 3. RED suppliers with any open PO ---
    open_po_by_sup: dict[str, int] = defaultdict(int)
    for p in po_rows:
        sid = p.get("supplier_id") or ""
        if sid:
            open_po_by_sup[sid] += 1
    try:
        score_rows = _paginate(
            client, "supplier_scores",
            "supplier_id,score_date,risk_flag,composite_score,supplier_name",
        )
    except Exception:
        score_rows = []
    latest_score: dict[str, dict] = {}
    for r in score_rows:
        sid = r.get("supplier_id") or ""
        if not sid:
            continue
        if sid not in latest_score or (r.get("score_date") or "") > latest_score[sid].get("score_date", ""):
            latest_score[sid] = r
    for sid, s in latest_score.items():
        if (s.get("risk_flag") or "").lower() == "red" and open_po_by_sup.get(sid, 0) > 0:
            actions.append({
                "kind":        "red_supplier",
                "severity":    "warning",
                "title":       "Red-flagged supplier with open POs",
                "description": (
                    f"{s.get('supplier_name') or sid} has "
                    f"{open_po_by_sup[sid]} open PO(s) — composite score "
                    f"{float(s.get('composite_score') or 0):.1f}"
                ),
                "supplier_id": sid,
                "action":      "Review supplier",
            })

    # Cap final list and pin critical first.
    sev_rank = {"critical": 0, "warning": 1, "info": 2}
    actions.sort(key=lambda a: sev_rank.get(a.get("severity", "info"), 9))
    return actions[:30]


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


def _build_stocking_gaps(client: Any, today: date) -> dict:
    """Stocking Gap Intelligence — chronic transfer-pattern gaps by location.

    Returns the most-recent stocking_gaps analysis grouped by location:
      {
        analysis_date: 'YYYY-MM-DD',
        total_chronic: int,
        total_annual_savings: float,
        locations: [{location_id, location_name, chronic_count,
                     total_annual_savings, rows: [...top20...]}],
        default_location_id: <location with most chronic gaps>
      }

    All locations returned in one payload so the dropdown switches client-side.
    Gracefully returns an empty payload if migration 030 hasn't been applied.
    """
    try:
        latest = (
            client.table("stocking_gaps")
            .select("analysis_date")
            .order("analysis_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        return {"analysis_date": None, "locations": [],
                "total_chronic": 0, "total_annual_savings": 0.0,
                "default_location_id": None}
    if not latest:
        return {"analysis_date": None, "locations": [],
                "total_chronic": 0, "total_annual_savings": 0.0,
                "default_location_id": None}
    analysis_date = latest[0]["analysis_date"]

    rows = _paginate(
        client, "stocking_gaps",
        "sku_id,location_id,location_name,transfer_from_location,"
        "transfer_frequency,transfer_streak,avg_qty_recommended,"
        "total_transfer_value,gap_score,gap_classification,"
        "suggested_stock_increase,current_reorder_point,"
        "annual_cost_savings,trend_direction",
        filters={"analysis_date": analysis_date},
    )

    by_loc: dict[str, dict] = {}
    total_chronic = 0
    total_savings = 0.0

    for r in rows:
        loc = r.get("location_id") or ""
        if not loc or loc in EXCLUDED_DISPLAY_LOCATIONS or loc == "LOC-025":
            continue
        cls = r.get("gap_classification") or "OCCASIONAL"
        savings = float(r.get("annual_cost_savings") or 0)

        if cls == "CHRONIC":
            total_chronic += 1
            total_savings += savings

        bucket = by_loc.setdefault(loc, {
            "location_id":         loc,
            "location_name":       r.get("location_name") or LOCATION_NAMES.get(loc, loc),
            "chronic_count":       0,
            "total_annual_savings":0.0,
            "rows":                [],
        })
        if cls == "CHRONIC":
            bucket["chronic_count"]        += 1
            bucket["total_annual_savings"] += savings

        bucket["rows"].append({
            "sku_id":                  r.get("sku_id"),
            "transfer_from_location":  r.get("transfer_from_location"),
            "transfer_from_display":   _loc_display(r["transfer_from_location"])
                                       if r.get("transfer_from_location") else "",
            "transfer_frequency":      int(r.get("transfer_frequency") or 0),
            "transfer_streak":         int(r.get("transfer_streak") or 0),
            "avg_qty_recommended":     float(r.get("avg_qty_recommended") or 0),
            "total_transfer_value":    float(r.get("total_transfer_value") or 0),
            "gap_score":               float(r.get("gap_score") or 0),
            "gap_classification":      cls,
            "suggested_stock_increase":float(r.get("suggested_stock_increase") or 0)
                                       if r.get("suggested_stock_increase") is not None else None,
            "current_reorder_point":   float(r.get("current_reorder_point") or 0)
                                       if r.get("current_reorder_point") is not None else None,
            "annual_cost_savings":     savings if cls == "CHRONIC" else None,
            "trend_direction":         r.get("trend_direction") or "STABLE",
        })

    # Keep top 20 chronic-first per location, round totals
    for b in by_loc.values():
        cls_rank = {"CHRONIC": 0, "RECURRING": 1, "OCCASIONAL": 2}
        b["rows"].sort(key=lambda r: (
            cls_rank.get(r["gap_classification"], 9),
            -(r.get("annual_cost_savings") or 0),
            -r["transfer_frequency"],
        ))
        b["rows"] = b["rows"][:20]
        b["total_annual_savings"] = round(b["total_annual_savings"], 2)

    locations = sorted(by_loc.values(), key=lambda b: b["location_name"])
    default_loc = (
        max(by_loc.values(), key=lambda b: b["chronic_count"])["location_id"]
        if by_loc else None
    )

    return {
        "analysis_date":        analysis_date,
        "total_chronic":        total_chronic,
        "total_annual_savings": round(total_savings, 2),
        "locations":            locations,
        "default_location_id":  default_loc,
    }


def _build_understocking(client: Any, today: date) -> dict:
    """Chronic Understocking Report — by location.

    Returns the most-recent understocking_report grouped by location:
      {
        report_date: 'YYYY-MM-DD',
        locations:  [{location_id, location_name, total_value_at_risk,
                      sku_count, rows: [...]}],
        default_location_id: <location with highest total_value_at_risk>
      }

    The frontend uses default_location_id to pick the dropdown's initial
    selection.  All locations are returned in one payload so the dropdown
    can switch between them client-side without refetching.
    """
    # Filter on run_completed_at IS NOT NULL so we never read a
    # mid-write partial report.  See engine/understocking.py for the
    # producer-side publication pattern.
    try:
        latest = (
            client.table("understocking_report")
            .select("report_date")
            .not_.is_("run_completed_at", "null")
            .order("report_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        latest = []
    if not latest:
        return {"report_date": None, "locations": [], "default_location_id": None}
    report_date = latest[0]["report_date"]

    rows = _paginate(
        client, "understocking_report",
        "location_id,location_name,sku_id,sku_description,"
        "stockout_days_pct,days_observed,days_below_reorder,"
        "avg_daily_demand,current_min_qty,suggested_min_qty,min_qty_gap,"
        "unit_cost,inventory_value_at_risk,transfer_recommended_count,"
        "priority_score",
        filters={"report_date": report_date},
        not_null_cols=["run_completed_at"],
    )

    by_loc: dict[str, dict] = {}
    for r in rows:
        loc = r.get("location_id") or ""
        if not loc or loc in EXCLUDED_DISPLAY_LOCATIONS or loc == "LOC-025":
            continue
        bucket = by_loc.setdefault(loc, {
            "location_id":         loc,
            "location_name":       r.get("location_name")
                                   or LOCATION_NAMES.get(loc) or loc,
            "total_value_at_risk": 0.0,
            "sku_count":           0,
            "rows":                [],
        })
        v = float(r.get("inventory_value_at_risk") or 0)
        bucket["total_value_at_risk"] += v
        bucket["sku_count"] += 1
        bucket["rows"].append({
            "sku_id":                     r.get("sku_id"),
            "sku_description":            r.get("sku_description") or "",
            "stockout_days_pct":          float(r.get("stockout_days_pct") or 0),
            "days_observed":              int(r.get("days_observed") or 0),
            "days_below_reorder":         int(r.get("days_below_reorder") or 0),
            "avg_daily_demand":           float(r.get("avg_daily_demand") or 0),
            "current_min_qty":            float(r.get("current_min_qty") or 0),
            "suggested_min_qty":          float(r.get("suggested_min_qty") or 0),
            "min_qty_gap":                float(r.get("min_qty_gap") or 0),
            "unit_cost":                  float(r.get("unit_cost") or 0),
            "inventory_value_at_risk":    v,
            "transfer_recommended_count": int(r.get("transfer_recommended_count") or 0),
            "priority_score":             float(r.get("priority_score") or 0),
        })

    # Sort each location's rows by priority_score desc; round totals.
    for b in by_loc.values():
        b["rows"].sort(key=lambda r: r["priority_score"], reverse=True)
        b["total_value_at_risk"] = round(b["total_value_at_risk"], 2)

    locations = sorted(by_loc.values(), key=lambda b: b["location_name"])
    default_loc = (
        max(by_loc.values(), key=lambda b: b["total_value_at_risk"])["location_id"]
        if by_loc else None
    )

    return {
        "report_date":         report_date,
        "locations":           locations,
        "default_location_id": default_loc,
    }


def _build_churn_summary(client: Any, today: date) -> dict:
    """Lightweight CHURNED/DECLINING/STABLE counts for the dashboard teaser card."""
    try:
        rows = _paginate(client, "customer_churn_flags", "flag,run_date")
    except Exception:
        return {"churned": 0, "declining": 0, "stable": 0, "total": 0, "run_date": None}
    counts: dict[str, int] = {"CHURNED": 0, "DECLINING": 0, "STABLE": 0}
    run_date = None
    for r in rows:
        flag = r.get("flag") or "STABLE"
        counts[flag] = counts.get(flag, 0) + 1
        rd = r.get("run_date")
        if rd and (run_date is None or rd > run_date):
            run_date = rd
    return {
        "churned":  counts.get("CHURNED",  0),
        "declining": counts.get("DECLINING", 0),
        "stable":   counts.get("STABLE",   0),
        "total":    sum(counts.values()),
        "run_date": run_date,
    }


def _build_opsl_intelligence(client: Any, today: date) -> dict:
    """HIGH/MEDIUM/LOW outside-purchase counts + HIGH detail rows for the dashboard."""
    try:
        rows = _paginate(
            client, "opsl_flags",
            "prod_line_pn,location_id,opsl_count,total_opsl_sales,"
            "avg_gp_pct,estimated_margin_recovery,flag,last_opsl_date,in_reorder_queue,run_date",
        )
    except Exception:
        return {"high": 0, "medium": 0, "low": 0, "total_recovery": 0.0, "run_date": None, "high_rows": []}
    counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    total_recovery = 0.0
    run_date = None
    high_rows: list[dict] = []
    for r in rows:
        flag = r.get("flag") or "LOW"
        counts[flag] = counts.get(flag, 0) + 1
        total_recovery += float(r.get("estimated_margin_recovery") or 0)
        rd = r.get("run_date")
        if rd and (run_date is None or rd > run_date):
            run_date = rd
        if flag == "HIGH":
            loc_id = r.get("location_id") or ""
            high_rows.append({
                "prod_line_pn":              r.get("prod_line_pn"),
                "location_id":               loc_id,
                "location_display":          _loc_display(loc_id),
                "opsl_count":                int(r.get("opsl_count") or 0),
                "total_opsl_sales":          round(float(r.get("total_opsl_sales") or 0), 2),
                "avg_gp_pct":                round(float(r.get("avg_gp_pct") or 0) * 100, 1),
                "estimated_margin_recovery": round(float(r.get("estimated_margin_recovery") or 0), 2),
                "in_reorder_queue":          bool(r.get("in_reorder_queue")),
            })
    high_rows.sort(key=lambda x: x["estimated_margin_recovery"], reverse=True)
    return {
        "high":           counts["HIGH"],
        "medium":         counts["MEDIUM"],
        "low":            counts["LOW"],
        "total_recovery": round(total_recovery, 2),
        "run_date":       run_date,
        "high_rows":      high_rows,
    }


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
        ("critical_actions",     _build_critical_actions),
        ("morning_brief",        _build_morning_brief),
        ("alerts",               _build_alerts),
        ("reorder",              _build_reorder),
        ("dead_stock",           _build_dead_stock),
        ("churn_summary",        _build_churn_summary),
        ("opsl_intelligence",    _build_opsl_intelligence),
        ("inventory_health",     _build_inventory_health),
        ("understocking",        _build_understocking),
        ("stocking_gaps",        _build_stocking_gaps),
        ("transfers",            _build_transfer_activity),
        ("location_performance", _build_location_performance),
        ("top_skus",             _build_top_skus),
        ("supplier_health",      _build_supplier_health),
        ("supplier_detail",      _build_supplier_detail),
        ("anomaly_summary",      _build_anomaly_summary),
        ("network_kpis",         _build_network_kpis),
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

    with ThreadPoolExecutor(max_workers=min(10, len(sections))) as ex:
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


@app.route("/api/dead-stock/export.csv")
def dead_stock_export():
    """Stream a CSV of every LIQUIDATE candidate from the latest report.

    Used by the dashboard's "Export Full Liquidation List" button.
    """
    import csv
    import io
    from flask import Response

    try:
        client = get_client()
    except Exception as exc:
        log.exception("Supabase connection failed for dead-stock export.")
        return jsonify({"error": str(exc)}), 503

    try:
        latest = (
            client.table("dead_stock_recommendations")
            .select("report_date")
            .order("report_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        latest = []
    if not latest:
        return jsonify({"error": "no dead-stock report available yet"}), 404
    report_date = latest[0]["report_date"]

    rows = _paginate(
        client, "dead_stock_recommendations",
        "sku_id,location_id,classification,action,dead_stock_score,"
        "total_inv_value,qty_on_hand,unit_cost,days_since_sale,"
        "sale_frequency,abc_class,supplier_id,part_category,sub_category",
        filters={"report_date": report_date, "classification": "LIQUIDATE"},
    )
    rows.sort(key=lambda r: float(r.get("total_inv_value") or 0), reverse=True)

    buf = io.StringIO()
    headers = [
        "sku_id", "location_id", "location_name", "classification", "action",
        "dead_stock_score", "total_inv_value", "qty_on_hand", "unit_cost",
        "days_since_sale", "sale_frequency", "abc_class", "supplier_id",
        "part_category", "sub_category",
    ]
    # CSV-injection guard (CWE-1236): cells whose first char is `=`, `+`, `-`,
    # `@`, TAB, or CR can trigger formula execution when the file is opened
    # in Excel/Sheets.  Prefix any such value with a single quote so the
    # spreadsheet treats it as text.  SKU/supplier IDs in this dataset can
    # legitimately start with `-`, so we have to neutralize, not reject.
    def _safe(v):
        if v is None:
            return ""
        s = str(v)
        if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
            return "'" + s
        return s

    w = csv.writer(buf)
    w.writerow(headers)
    for r in rows:
        w.writerow([
            _safe(r.get("sku_id")), _safe(r.get("location_id")),
            _safe(_loc_display(r.get("location_id", ""))),
            _safe(r.get("classification")), _safe(r.get("action")),
            _safe(r.get("dead_stock_score")), _safe(r.get("total_inv_value")),
            _safe(r.get("qty_on_hand")), _safe(r.get("unit_cost")),
            _safe(r.get("days_since_sale")), _safe(r.get("sale_frequency")),
            _safe(r.get("abc_class")), _safe(r.get("supplier_id")),
            _safe(r.get("part_category")), _safe(r.get("sub_category")),
        ])

    fname = f"liquidation_list_{report_date}.csv"
    log.info("Dead-stock CSV export: %d LIQUIDATE rows for %s", len(rows), report_date)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


# ---------------------------------------------------------------------------
# Generic CSV helpers — shared by every export endpoint.
# ---------------------------------------------------------------------------

def _csv_safe(v: Any) -> str:
    """Neutralize CSV-injection (CWE-1236) attempts.

    Cells whose first character is `=`, `+`, `-`, `@`, TAB, or CR can
    trigger formula execution when the file is opened in Excel/Sheets.
    Prefix any such value with a single quote so the spreadsheet treats
    it as literal text.  SKU/supplier IDs in this dataset can legitimately
    start with `-` so we neutralize, not reject.
    """
    if v is None:
        return ""
    s = str(v)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s


def _csv_response(headers: list[str], rows: list[list[Any]], filename: str):
    import csv as _csv
    import io as _io
    from flask import Response as _Response
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(headers)
    for r in rows:
        w.writerow([_csv_safe(c) for c in r])
    return _Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/api/reorder/export.csv")
def reorder_export():
    """CSV of every pending reorder rec for the latest run.

    Used by the Reorder panel's "Export to CSV" button so the buying
    team can drop the day's queue into spreadsheet workflows.
    """
    try:
        client = get_client()
    except Exception as exc:
        log.exception("Supabase connection failed for reorder export.")
        return jsonify({"error": str(exc)}), 503
    try:
        latest = (
            client.table("reorder_recommendations")
            .select("recommendation_date")
            .order("recommendation_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        latest = []
    if not latest:
        return jsonify({"error": "no reorder recommendations available yet"}), 404
    rec_date = latest[0]["recommendation_date"]

    rows = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,recommendation_type,urgency,qty_to_order,"
        "supplier_id,days_of_supply_remaining,forecast_model_used,"
        "transfer_from_location,is_approved,recommendation_date",
        filters={"recommendation_date": rec_date, "is_approved": False},
    )
    urgency_rank = {"critical": 0, "warning": 1, "high": 1, "normal": 2, "medium": 2, "low": 3}
    rows.sort(key=lambda r: (
        urgency_rank.get((r.get("urgency") or "low").lower(), 9),
        -float(r.get("qty_to_order") or 0),
    ))

    headers = [
        "recommendation_date", "urgency", "type", "sku_id", "location_id",
        "location_name", "transfer_from", "qty_to_order",
        "days_of_supply_remaining", "supplier_id", "forecast_model_used",
    ]
    out = []
    for r in rows:
        out.append([
            r.get("recommendation_date"),
            r.get("urgency"),
            r.get("recommendation_type"),
            r.get("sku_id"),
            r.get("location_id"),
            _loc_display(r.get("location_id", "")),
            _loc_display(r.get("transfer_from_location", "")) if r.get("transfer_from_location") else "",
            r.get("qty_to_order"),
            r.get("days_of_supply_remaining"),
            r.get("supplier_id"),
            r.get("forecast_model_used"),
        ])
    log.info("Reorder CSV export: %d pending rows for %s", len(out), rec_date)
    return _csv_response(headers, out, f"reorder_recommendations_{rec_date}.csv")


@app.route("/api/transfers/export.csv")
def transfers_export():
    """CSV of every transfer-type recommendation for the latest run.

    Used by the Transfer Opportunities panel's export button so logistics
    can dispatch inter-location moves from a single spreadsheet.
    """
    try:
        client = get_client()
    except Exception as exc:
        log.exception("Supabase connection failed for transfers export.")
        return jsonify({"error": str(exc)}), 503
    try:
        latest = (
            client.table("reorder_recommendations")
            .select("recommendation_date")
            .order("recommendation_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        latest = []
    if not latest:
        return jsonify({"error": "no reorder recommendations available yet"}), 404
    rec_date = latest[0]["recommendation_date"]

    rows = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,recommendation_type,urgency,qty_to_order,"
        "transfer_from_location,days_of_supply_remaining",
        filters={"recommendation_date": rec_date, "recommendation_type": "transfer"},
    )
    rows.sort(key=lambda r: -float(r.get("qty_to_order") or 0))

    headers = [
        "recommendation_date", "urgency", "sku_id",
        "from_location_id", "from_location_name",
        "to_location_id", "to_location_name",
        "qty_to_transfer", "days_of_supply_remaining",
    ]
    out = []
    for r in rows:
        out.append([
            rec_date,
            r.get("urgency"),
            r.get("sku_id"),
            r.get("transfer_from_location"),
            _loc_display(r.get("transfer_from_location", "")) if r.get("transfer_from_location") else "",
            r.get("location_id"),
            _loc_display(r.get("location_id", "")),
            r.get("qty_to_order"),
            r.get("days_of_supply_remaining"),
        ])
    log.info("Transfers CSV export: %d transfer rows for %s", len(out), rec_date)
    return _csv_response(headers, out, f"transfer_recommendations_{rec_date}.csv")


@app.route("/api/understocking/export.csv")
def understocking_export():
    """Stream a CSV of every chronic-understocking row from the latest report.

    Optional query string: `?location_id=LOC-005` filters to one location.
    Used by the dashboard's "Download Restocking Plan" button.
    """
    import csv
    import io
    from flask import Response

    try:
        client = get_client()
    except Exception as exc:
        log.exception("Supabase connection failed for understocking export.")
        return jsonify({"error": str(exc)}), 503

    location_filter = (request.args.get("location_id") or "").strip()

    # Filter on run_completed_at IS NOT NULL — see understocking
    # builder for rationale (avoid mid-write partial reads).
    try:
        latest = (
            client.table("understocking_report")
            .select("report_date")
            .not_.is_("run_completed_at", "null")
            .order("report_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        latest = []
    if not latest:
        return jsonify({"error": "no understocking report available yet"}), 404
    report_date = latest[0]["report_date"]

    filters: dict[str, Any] = {"report_date": report_date}
    if location_filter:
        filters["location_id"] = location_filter
    rows = _paginate(
        client, "understocking_report",
        "report_date,location_id,location_name,sku_id,sku_description,"
        "days_observed,days_below_reorder,stockout_days_pct,"
        "avg_daily_demand,current_min_qty,suggested_min_qty,min_qty_gap,"
        "unit_cost,inventory_value_at_risk,transfer_recommended_count,"
        "priority_score",
        filters=filters,
        not_null_cols=["run_completed_at"],
    )
    rows.sort(key=lambda r: float(r.get("priority_score") or 0), reverse=True)

    headers = [
        "report_date", "location_id", "location_name",
        "sku_id", "sku_description",
        "days_observed", "days_below_reorder", "stockout_days_pct",
        "avg_daily_demand", "current_min_qty", "suggested_min_qty",
        "min_qty_gap", "unit_cost", "inventory_value_at_risk",
        "transfer_recommended_count", "priority_score",
    ]

    def _safe(v):
        if v is None:
            return ""
        s = str(v)
        if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
            return "'" + s
        return s

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for r in rows:
        w.writerow([_safe(r.get(h)) for h in headers])

    suffix = f"_{location_filter}" if location_filter else ""
    fname = f"understocking_report_{report_date}{suffix}.csv"
    log.info("Understocking CSV export: %d rows for %s%s",
             len(rows), report_date,
             f" loc={location_filter}" if location_filter else "")
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.route("/api/stocking-gaps/export.csv")
def stocking_gaps_export():
    """CSV of stocking gaps for the latest analysis run.

    Optional query string: `?location_id=LOC-005` filters to one location.
    Used by the Stocking Gap Intelligence panel's Export CSV button.
    """
    try:
        client = get_client()
    except Exception as exc:
        log.exception("Supabase connection failed for stocking-gaps export.")
        return jsonify({"error": str(exc)}), 503

    location_filter = (request.args.get("location_id") or "").strip()

    try:
        latest = (
            client.table("stocking_gaps")
            .select("analysis_date")
            .order("analysis_date", desc=True)
            .limit(1)
            .execute()
            .data or []
        )
    except Exception:
        latest = []
    if not latest:
        return jsonify({"error": "no stocking gap analysis available yet"}), 404
    analysis_date = latest[0]["analysis_date"]

    filters: dict = {"analysis_date": analysis_date}
    if location_filter:
        filters["location_id"] = location_filter

    rows = _paginate(
        client, "stocking_gaps",
        "analysis_date,sku_id,location_id,location_name,transfer_from_location,"
        "transfer_frequency,transfer_streak,avg_qty_recommended,total_transfer_value,"
        "gap_score,gap_classification,suggested_stock_increase,current_reorder_point,"
        "annual_cost_savings,trend_direction",
        filters=filters,
    )

    cls_rank = {"CHRONIC": 0, "RECURRING": 1, "OCCASIONAL": 2}
    rows.sort(key=lambda r: (
        cls_rank.get(r.get("gap_classification") or "OCCASIONAL", 9),
        -(float(r.get("annual_cost_savings") or 0)),
    ))

    headers = [
        "analysis_date", "location_id", "location_name", "sku_id",
        "transfer_from_location", "gap_classification", "gap_score",
        "transfer_frequency", "transfer_streak", "avg_qty_recommended",
        "total_transfer_value", "suggested_stock_increase", "current_reorder_point",
        "annual_cost_savings", "trend_direction",
    ]
    out = []
    for r in rows:
        out.append([r.get(h) for h in headers])

    suffix = f"_{location_filter}" if location_filter else ""
    fname  = f"stocking_gaps_{analysis_date}{suffix}.csv"
    log.info("Stocking gaps CSV export: %d rows for %s%s",
             len(out), analysis_date,
             f" loc={location_filter}" if location_filter else "")
    return _csv_response(headers, out, fname)


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


# ---------------------------------------------------------------------------
# Churn Intelligence
# ---------------------------------------------------------------------------

def _build_churn(client: Any) -> dict:
    rows = _paginate(
        client, "customer_churn_flags",
        "customer_id,location_id,baseline_monthly_spend,last_90_days_spend,"
        "pct_change,flag,last_purchase_date,run_date",
        order_col="baseline_monthly_spend",
        order_desc=True,
    )

    summary: dict[str, int] = {"CHURNED": 0, "DECLINING": 0, "STABLE": 0}
    locations_seen: set[str] = set()
    for r in rows:
        f = r.get("flag", "STABLE")
        if f in summary:
            summary[f] += 1
        loc = r.get("location_id")
        if loc:
            locations_seen.add(loc)

    at_risk = []
    for r in rows:
        if r.get("flag") not in ("CHURNED", "DECLINING"):
            continue
        loc = r.get("location_id", "")
        r["location_name"] = LOCATION_NAMES.get(loc, "")
        at_risk.append(r)

    run_date = rows[0].get("run_date") if rows else None

    return {
        "run_date":      run_date,
        "summary": {
            "churned":   summary["CHURNED"],
            "declining": summary["DECLINING"],
            "stable":    summary["STABLE"],
            "total":     len(rows),
        },
        "locations":      sorted(locations_seen),
        "location_names": {loc: LOCATION_NAMES.get(loc, loc) for loc in locations_seen},
        "rows":           at_risk,
    }


@app.route("/churn")
def churn_page():
    return send_from_directory(str(Path(__file__).parent), "churn.html")


@app.route("/api/churn")
def api_churn():
    try:
        client = get_new_client()
        return jsonify(_build_churn(client))
    except Exception as exc:
        log.exception("Error building churn data")
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/rpc-debug")
def rpc_debug():
    """Diagnostic endpoint — tests both GP RPCs and returns raw results + date params."""
    today = date.today()
    current_start = (today - timedelta(days=90)).isoformat()
    prior_start   = (today - timedelta(days=180)).isoformat()
    prior_end     = (today - timedelta(days=91)).isoformat()
    start_date    = current_start

    result: dict[str, Any] = {
        "today": today.isoformat(),
        "params": {
            "current_start": current_start,
            "prior_start":   prior_start,
            "prior_end":     prior_end,
        },
    }

    try:
        client = get_client()
    except Exception as exc:
        return jsonify({"error": f"Supabase connect failed: {exc}"}), 503

    # Test get_all_locations_gp_summary
    try:
        gp_resp = client.rpc("get_all_locations_gp_summary", {
            "p_current_start": current_start,
            "p_prior_start":   prior_start,
            "p_prior_end":     prior_end,
        }).execute()
        result["gp_summary"] = {
            "row_count": len(gp_resp.data or []),
            "sample":    (gp_resp.data or [])[:3],
            "error":     None,
        }
    except Exception as exc:
        result["gp_summary"] = {"row_count": 0, "sample": [], "error": str(exc)}

    # Test get_top_skus_by_gp
    try:
        sku_resp = client.rpc("get_top_skus_by_gp", {
            "p_start_date": start_date,
        }).execute()
        result["top_skus"] = {
            "row_count": len(sku_resp.data or []),
            "sample":    (sku_resp.data or [])[:3],
            "error":     None,
        }
    except Exception as exc:
        result["top_skus"] = {"row_count": 0, "sample": [], "error": str(exc)}

    # Probe actual date range in the table (cheap — uses the tran_date index)
    try:
        min_row = client.table("sales_detail_transactions").select("tran_date").order("tran_date", desc=False).limit(1).execute().data or []
        max_row = client.table("sales_detail_transactions").select("tran_date").order("tran_date", desc=True).limit(1).execute().data or []
        result["table_date_range"] = {
            "min_tran_date": min_row[0]["tran_date"] if min_row else None,
            "max_tran_date": max_row[0]["tran_date"] if max_row else None,
        }
    except Exception as exc:
        result["table_date_range"] = {"error": str(exc)}

    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import socket as _socket

    debug = "--dev" in sys.argv
    requested = int(os.environ.get("PORT", 5000))
    # Try requested port, then two fallbacks.  Replit's preview proxy will
    # still find the app on 5001/5002, and we log the chosen port loudly so
    # the workflow log shows it.
    candidates = [requested, requested + 1, requested + 2]
    chosen: int | None = None
    for p in candidates:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
            except OSError:
                log.warning("Port %d already in use, trying next.", p)
                continue
        chosen = p
        break
    if chosen is None:
        log.error("All candidate ports busy: %s", candidates)
        sys.exit(1)
    log.info("Starting PartsWatch AI dashboard on port %d  debug=%s", chosen, debug)
    app.run(host="0.0.0.0", port=chosen, debug=debug, threaded=True)
