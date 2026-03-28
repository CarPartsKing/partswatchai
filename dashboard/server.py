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

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.claude_api import PurchasingAssistant
from db.connection import get_client
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

    return {
        "summary": summary,
        "items": rows,
    }


def _build_reorder(client: Any, today: date) -> list[dict]:
    rows = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,recommendation_type,urgency,qty_to_order,"
        "days_of_supply_remaining,forecast_model_used,transfer_from_location",
        filters={"recommendation_date": today.isoformat()},
        eq_bool={"is_approved": False},
        limit=100,
    )
    urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    rows.sort(key=lambda r: urgency_order.get(r.get("urgency", "low"), 9))
    return rows[:10]


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
    cutoff = (today - timedelta(days=7)).isoformat()
    inv_rows = _paginate(
        client, "inventory_snapshots",
        "sku_id,location_id,snapshot_date,is_stockout",
        gte_filters={"snapshot_date": cutoff},
    )
    latest: dict[tuple[str, str], dict] = {}
    for r in inv_rows:
        key = (r["sku_id"], r["location_id"])
        if key not in latest or r["snapshot_date"] > latest[key]["snapshot_date"]:
            latest[key] = r

    stockouts = [v for v in latest.values() if v.get("is_stockout")]

    recs = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,days_of_supply_remaining",
        filters={"recommendation_date": today.isoformat()},
    )
    low_supply = [
        r for r in recs
        if r.get("days_of_supply_remaining") is not None
        and 0 < float(r["days_of_supply_remaining"]) < _LOW_SUPPLY_DAYS
    ]

    return {
        "stockout_count": len(stockouts),
        "low_supply_count": len(low_supply),
        "snapshot_count": len(latest),
        "stockout_pairs": [
            {"sku_id": r["sku_id"], "location_id": r["location_id"]}
            for r in stockouts[:10]
        ],
    }


def _build_forecast_accuracy(client: Any, today: date) -> list[dict]:
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

    sales_rows = _paginate(
        client, "sales_transactions",
        "sku_id,location_id,transaction_date,qty_sold",
        gte_filters={"transaction_date": window_start},
        lte_filters={"transaction_date": yesterday},
    )
    actuals: dict[tuple[str, str, str], float] = defaultdict(float)
    actuals_all: dict[tuple[str, str], float] = defaultdict(float)
    for r in sales_rows:
        d = str(r.get("transaction_date", ""))[:10]
        actuals[(r["sku_id"], r["location_id"], d)] += float(r.get("qty_sold") or 0)
        actuals_all[(r["sku_id"], d)] += float(r.get("qty_sold") or 0)

    sku_ids = {k[0] for k in forecast_map}
    sku_rows = _paginate(client, "sku_master", "sku_id,abc_class",
                         in_filters={"sku_id": list(sku_ids)})
    abc_map = {r["sku_id"]: (r.get("abc_class") or "?") for r in sku_rows}

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
    loc_rows = _paginate(
        client, "locations",
        "location_id,location_tier,composite_tier_score",
    )
    tiers: dict[int, list[str]] = defaultdict(list)
    scores: dict[str, float] = {}
    for r in loc_rows:
        tier = int(r.get("location_tier") or 2)
        loc_id = r.get("location_id", "?")
        tiers[tier].append(loc_id)
        if r.get("composite_tier_score") is not None:
            scores[loc_id] = float(r["composite_tier_score"])

    tier3_locs = tiers.get(3, [])
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
        ("weather",              _build_weather),
        ("alerts",               _build_alerts),
        ("reorder",              _build_reorder),
        ("supplier_health",      _build_supplier_health),
        ("inventory_health",     _build_inventory_health),
        ("forecast_accuracy",    _build_forecast_accuracy),
        ("location_performance", _build_location_performance),
    ]

    for name, fn in sections:
        try:
            payload[name] = fn(client, today)
        except Exception:
            log.exception("Dashboard section '%s' failed.", name)
            payload[name] = None

    elapsed_ms = (time.perf_counter() - t0) * 1000
    payload["query_ms"] = round(elapsed_ms)
    log.info("Dashboard data served in %.0fms", elapsed_ms)
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
