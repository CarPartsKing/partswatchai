"""assistant/context_builder.py — Live Supabase context for the purchasing assistant.

Queried fresh on every chat message so Claude always sees current data.
The assembled string is injected into Claude's system prompt alongside
the static business context in claude_api.py.

SECTIONS (in order, each isolated function with its own DB reads)
-----------------------------------------------------------------
1. Alerts          — today's unacknowledged alerts, severity-ordered
2. Reorder         — top 10 critical reorder recommendations
3. Weather         — today's conditions + 7-day NE Ohio outlook
4. Supplier health — green / amber / red counts
5. Inventory health— stockout count, low-supply count
6. Forecast accuracy— latest MAPE by model type (last 7 days)
7. Location performance — tier 1/2/3 counts, tier-3 critical alerts

Target output: < 2 000 tokens (≈ 8 000 characters).  Each section is
kept compact; verbose alert messages are truncated to 120 chars.

Usage
-----
    from assistant.context_builder import build_context
    from db.connection import get_client

    ctx = build_context(get_client())   # returns a str
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from utils.logging_config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGE_SIZE: int = 1_000

# Sections that exceed this many characters are truncated with a note.
_MAX_SECTION_CHARS: int = 1_200

# Severity ordering for alert display
_SEVERITY_ORDER: dict[str, int] = {"critical": 0, "warning": 1, "info": 2}

# Days of supply below which a reorder line counts as "low supply"
_LOW_SUPPLY_DAYS: float = 3.0

# MAPE lookback window (days) — must match engine/alerts.py constant
_MAPE_LOOKBACK_DAYS: int = 7

# Weather forecast lookahead for the context window
_WEATHER_LOOKAHEAD_DAYS: int = 7

# Freeze threshold (°F) for weather highlights
_FREEZE_THRESHOLD_F: float = 20.0

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


def _loc_display(loc_id: str) -> str:
    name = LOCATION_NAMES.get(loc_id)
    return f"{name} ({loc_id})" if name else loc_id


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
    eq_bool:     dict | None = None,
    in_filters:  dict | None = None,
    order_col:   str | None = None,
    order_desc:  bool = False,
    limit:       int | None = None,
) -> list[dict]:
    """Paginate through a Supabase table and return matching rows.

    Args:
        client:      Active Supabase client.
        table:       Table name.
        select:      PostgREST column selector string.
        filters:     {col: value} exact equality.
        gte_filters: {col: value} for col >= value.
        lte_filters: {col: value} for col <= value.
        eq_bool:     {col: bool} boolean equality.
        in_filters:  {col: [values]} for col IN (values).
        order_col:   Column to order by (optional).
        order_desc:  If True, order descending.
        limit:       Stop fetching after this many rows total (None = all).

    Returns:
        All matching rows as a list of dicts.
    """
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
# Section 1 — Alerts
# ---------------------------------------------------------------------------

def _section_alerts(client: Any, today: date) -> str:
    """Unacknowledged alerts for today, ordered critical → warning → info.

    Shows up to 20 alerts; surplus is summarised as a count.  Alert
    messages are truncated to 120 characters to stay token-budget-friendly.
    """
    rows = _paginate(
        client, "alerts",
        "alert_type,severity,sku_id,location_id,supplier_id,message",
        filters={"alert_date": today.isoformat()},
        eq_bool={"is_acknowledged": False},
    )

    if not rows:
        return "[ALERTS]\nNone — no unacknowledged alerts today.\n"

    rows.sort(key=lambda r: _SEVERITY_ORDER.get(r.get("severity", "info"), 9))

    display_limit = 20
    shown = rows[:display_limit]
    overflow = len(rows) - display_limit

    lines = [f"[ALERTS — {len(rows)} unacknowledged]"]
    for r in shown:
        sev  = (r.get("severity") or "?").upper()[:8]
        atype = (r.get("alert_type") or "?")[:25]
        loc_id = r.get("location_id")
        ctx_parts = [p for p in [
            r.get("sku_id"), _loc_display(loc_id) if loc_id else None, r.get("supplier_id")
        ] if p]
        ctx = "  " + " | ".join(ctx_parts) if ctx_parts else ""
        msg = (r.get("message") or "")[:120]
        lines.append(f"  {sev:<8}  {atype:<25}{ctx}")
        lines.append(f"           {msg}")

    if overflow > 0:
        lines.append(f"  ... and {overflow} more alert(s) not shown.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section 2 — Reorder recommendations
# ---------------------------------------------------------------------------

def _section_reorder(client: Any, today: date) -> str:
    """Top 10 reorder recommendations for today, critical urgency first.

    Covers both POs and inter-location transfers.
    """
    rows = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,recommendation_type,urgency,"
        "qty_to_order,days_of_supply_remaining,forecast_model_used,"
        "transfer_from_location",
        filters={"recommendation_date": today.isoformat()},
        order_col="urgency",
        limit=50,
    )

    if not rows:
        return "[REORDER RECOMMENDATIONS]\nNone generated for today.\n"

    urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    rows.sort(key=lambda r: urgency_order.get(r.get("urgency", "low"), 9))
    top10 = rows[:10]
    total = len(rows)

    lines = [f"[REORDER — {total} total today, top 10 shown]"]
    for r in top10:
        sku   = r.get("sku_id", "?")
        loc   = _loc_display(r.get("location_id", "?"))
        rtype = (r.get("recommendation_type") or "po").upper()[:8]
        urg   = (r.get("urgency") or "?").upper()[:8]
        qty   = r.get("qty_to_order") or 0
        days  = r.get("days_of_supply_remaining")
        model = (r.get("forecast_model_used") or "?")[:12]
        days_str = f"{float(days):.1f}d" if days is not None else "N/A"
        from_loc = r.get("transfer_from_location")
        transfer_str = f"  ← from {_loc_display(from_loc)}" if from_loc and rtype == "TRANSFER" else ""
        lines.append(
            f"  {urg:<8}  {sku:<14}  {loc:<20}  "
            f"{rtype:<8}  qty={qty}  supply={days_str}  [{model}]{transfer_str}"
        )

    if total > 10:
        lines.append(f"  ... {total - 10} more recommendation(s) not shown.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section 3 — Weather
# ---------------------------------------------------------------------------

def _section_weather(client: Any, today: date) -> str:
    """Today's NE Ohio conditions plus a _WEATHER_LOOKAHEAD_DAYS-day outlook."""
    end_date = (today + timedelta(days=_WEATHER_LOOKAHEAD_DAYS)).isoformat()

    rows = _paginate(
        client, "weather_log",
        "log_date,temp_min_f,temp_max_f,precipitation_in,"
        "snowfall_in,consecutive_freeze_days,freeze_thaw_cycle",
        gte_filters={"log_date": today.isoformat()},
        lte_filters={"log_date": end_date},
        order_col="log_date",
    )

    if not rows:
        return "[WEATHER — NE Ohio]\nNo current weather data available.\n"

    def _fmt_row(r: dict, label: str) -> str:
        hi   = r.get("temp_max_f")
        lo   = r.get("temp_min_f")
        prec = r.get("precipitation_in") or 0
        snow = r.get("snowfall_in") or 0
        hi_s = f"{float(hi):.0f}°F" if hi is not None else "?°F"
        lo_s = f"{float(lo):.0f}°F" if lo is not None else "?°F"
        extras = []
        if snow and float(snow) > 0:
            extras.append(f"snow={float(snow):.1f}in")
        if prec and float(prec) > 0:
            extras.append(f"precip={float(prec):.2f}in")
        freeze_days = r.get("consecutive_freeze_days") or 0
        if freeze_days and int(freeze_days) > 0:
            extras.append(f"freeze_streak={int(freeze_days)}d")
        if r.get("freeze_thaw_cycle"):
            extras.append("freeze_thaw=YES")
        extra_str = "  " + "  ".join(extras) if extras else ""
        return f"  {label:<12}  hi={hi_s}  lo={lo_s}{extra_str}"

    lines = ["[WEATHER — NE Ohio]"]
    today_row = next((r for r in rows if r["log_date"] == today.isoformat()), None)
    if today_row:
        lines.append(_fmt_row(today_row, "Today"))
    else:
        lines.append("  Today         no data")

    forecast_rows = [r for r in rows if r["log_date"] > today.isoformat()]
    if forecast_rows:
        lines.append(f"  7-day outlook ({len(forecast_rows)} days):")
        for r in forecast_rows:
            d = r["log_date"][5:]  # MM-DD
            lines.append(_fmt_row(r, d))

    freeze_days_ahead = [
        r for r in forecast_rows
        if r.get("temp_min_f") is not None and float(r["temp_min_f"]) < _FREEZE_THRESHOLD_F
    ]
    if freeze_days_ahead:
        coldest = min(freeze_days_ahead, key=lambda r: float(r["temp_min_f"]))
        lines.append(
            f"  ** FREEZE WARNING: {len(freeze_days_ahead)} day(s) forecast below "
            f"{_FREEZE_THRESHOLD_F:.0f}°F  "
            f"(coldest: {float(coldest['temp_min_f']):.1f}°F on {coldest['log_date']}) **"
        )
    else:
        lines.append(f"  No freeze events forecast in next {_WEATHER_LOOKAHEAD_DAYS} days.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section 4 — Supplier health
# ---------------------------------------------------------------------------

def _section_supplier_health(client: Any, today: date) -> str:
    """Count of green / amber / red suppliers based on most recent scores."""
    cutoff = (today - timedelta(days=90)).isoformat()
    rows = _paginate(
        client, "supplier_scores",
        "supplier_id,score_date,risk_flag,composite_score",
        gte_filters={"score_date": cutoff},
    )

    if not rows:
        return "[SUPPLIER HEALTH]\nNo supplier scores available in last 90 days.\n"

    latest: dict[str, dict] = {}
    for r in rows:
        sid = r.get("supplier_id")
        if not sid:
            continue
        if sid not in latest or r["score_date"] > latest[sid]["score_date"]:
            latest[sid] = r

    counts: dict[str, int] = defaultdict(int)
    red_suppliers: list[str] = []
    for sid, s in latest.items():
        flag = (s.get("risk_flag") or "unknown").lower()
        counts[flag] += 1
        if flag == "red":
            red_suppliers.append(sid)

    total = sum(counts.values())
    lines = [
        "[SUPPLIER HEALTH]",
        f"  Green: {counts.get('green', 0):<4}  "
        f"Amber: {counts.get('amber', 0):<4}  "
        f"Red: {counts.get('red', 0):<4}  "
        f"Total: {total}",
    ]
    if red_suppliers:
        red_list = ", ".join(red_suppliers[:10])
        suffix = f" (+{len(red_suppliers) - 10} more)" if len(red_suppliers) > 10 else ""
        lines.append(f"  Red-flag suppliers: {red_list}{suffix}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section 5 — Inventory health
# ---------------------------------------------------------------------------

def _section_inventory_health(client: Any, today: date) -> str:
    """Active stockout count and low-supply SKU count as of today."""
    cutoff_inv = (today - timedelta(days=7)).isoformat()
    inv_rows = _paginate(
        client, "inventory_snapshots",
        "sku_id,location_id,snapshot_date,is_stockout",
        gte_filters={"snapshot_date": cutoff_inv},
    )

    latest: dict[tuple[str, str], dict] = {}
    for r in inv_rows:
        key = (r["sku_id"], r["location_id"])
        if key not in latest or r["snapshot_date"] > latest[key]["snapshot_date"]:
            latest[key] = r

    stockout_pairs = [(k, v) for k, v in latest.items() if v.get("is_stockout")]
    total_snapshots = len(latest)

    low_recs = _paginate(
        client, "reorder_recommendations",
        "sku_id,location_id,days_of_supply_remaining",
        filters={"recommendation_date": today.isoformat()},
    )
    low_supply_count = sum(
        1 for r in low_recs
        if r.get("days_of_supply_remaining") is not None
        and 0 < float(r["days_of_supply_remaining"]) < _LOW_SUPPLY_DAYS
    )

    lines = [
        "[INVENTORY HEALTH]",
        f"  Active stockouts (SKU×location):  {len(stockout_pairs)}",
        f"  Low supply (< {_LOW_SUPPLY_DAYS:.0f}d, not zero):    {low_supply_count}",
        f"  Snapshot pairs tracked:           {total_snapshots}",
    ]
    if stockout_pairs:
        sample = stockout_pairs[:5]
        sample_str = "  Stockout sample: " + ", ".join(
            f"{k[0]}@{_loc_display(k[1])}" for k, _ in sample
        )
        if len(stockout_pairs) > 5:
            sample_str += f" (+{len(stockout_pairs) - 5} more)"
        lines.append(sample_str)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section 6 — Forecast accuracy
# ---------------------------------------------------------------------------

def _section_forecast_accuracy(client: Any, today: date) -> str:
    """Weekly MAPE by (abc_class, model_type) over the last MAPE_LOOKBACK_DAYS days.

    Uses the same computation as engine/alerts.py so numbers are consistent.
    When no past forecast data exists (e.g. seed data), reports clearly.
    """
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
        return (
            "[FORECAST ACCURACY — last 7 days]\n"
            "  No past forecast rows in window — accuracy tracking starts once\n"
            "  forecast_date values enter the historical window.\n"
        )

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
    sku_rows = _paginate(
        client, "sku_master", "sku_id,abc_class",
        in_filters={"sku_id": list(sku_ids)},
    )
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

    lines = [f"[FORECAST ACCURACY — last {_MAPE_LOOKBACK_DAYS} days]"]
    for (abc, model), apes in sorted(ape_by_class.items()):
        mape = (sum(apes) / len(apes)) * 100
        flag = "  *** ABOVE THRESHOLD" if mape > 25.0 else ""
        lines.append(
            f"  {abc}-class  {model:<12}  MAPE={mape:.1f}%  "
            f"(n={len(apes)} obs){flag}"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section 7 — Location performance
# ---------------------------------------------------------------------------

def _section_location_performance(client: Any, today: date) -> str:
    """Tier 1/2/3 counts and any tier-3 locations with critical alerts today."""
    try:
        loc_rows = _paginate(
            client, "locations",
            "location_id,location_name,location_tier,composite_tier_score,is_active",
        )
    except Exception:
        loc_rows = _paginate(
            client, "locations",
            "location_id,location_tier,composite_tier_score",
        )

    if not loc_rows:
        return (
            "[LOCATION PERFORMANCE]\n"
            "  No location tiers available — run transform.location_classify first.\n"
        )

    tiers: dict[int, list[str]] = defaultdict(list)
    tier_scores: dict[str, float] = {}
    for r in loc_rows:
        if r.get("is_active") is False:
            continue
        tier = int(r.get("location_tier") or 2)
        loc_id = r.get("location_id", "?")
        loc_name = r.get("location_name") or LOCATION_NAMES.get(loc_id) or loc_id
        display = f"{loc_name} ({loc_id})"
        tiers[tier].append(display)
        score = r.get("composite_tier_score")
        if score is not None:
            tier_scores[loc_id] = float(score)

    lines = [
        "[LOCATION PERFORMANCE]",
        f"  Tier 1 (first-call):   {len(tiers.get(1, []))} locations  "
        f"{', '.join(tiers.get(1, []))}",
        f"  Tier 2 (second-call):  {len(tiers.get(2, []))} locations  "
        f"{', '.join(tiers.get(2, []))}",
        f"  Tier 3 (third-call):   {len(tiers.get(3, []))} locations  "
        f"{', '.join(tiers.get(3, []))}",
    ]

    tier3_loc_ids = [
        r.get("location_id", "?") for r in loc_rows
        if int(r.get("location_tier") or 2) == 3 and r.get("is_active") is not False
    ]
    if tier3_loc_ids:
        alert_rows = _paginate(
            client, "alerts",
            "location_id,severity,alert_type",
            filters={"alert_date": today.isoformat()},
            eq_bool={"is_acknowledged": False},
            in_filters={"location_id": tier3_loc_ids},
        )
        critical_t3 = [
            r for r in alert_rows
            if r.get("severity") == "critical"
        ]
        if critical_t3:
            locs_with_crit = sorted({_loc_display(r["location_id"]) for r in critical_t3})
            lines.append(
                f"  ** Tier 3 locations with critical alerts: "
                f"{', '.join(locs_with_crit)} **"
            )
        else:
            lines.append("  No critical alerts on Tier 3 locations today.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section 8 — Basket rules (co-purchase intelligence)
# ---------------------------------------------------------------------------

def _section_dead_stock(client: Any, today: date) -> str:
    """Dead-stock summary for the most recent dead_stock pipeline run.

    Surfaces the headline capital-at-risk numbers and the top liquidation
    candidate so the assistant can answer questions like "what's our worst
    dead stock?" or "how much capital is tied up?" without re-running ML.
    """
    # Latest report date (a missed nightly run shouldn't blank the section).
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
        return "[DEAD STOCK]\nNo dead-stock report available yet.\n"
    report_date = latest[0]["report_date"]

    rows = _paginate(
        client, "dead_stock_recommendations",
        "sku_id,location_id,classification,total_inv_value,days_since_sale,action",
        filters={"report_date": report_date},
    )
    if not rows:
        return f"[DEAD STOCK — {report_date}]\nNo LIQUIDATE/MARKDOWN positions.\n"

    liquidate = [r for r in rows if r.get("classification") == "LIQUIDATE"]
    markdown  = [r for r in rows if r.get("classification") == "MARKDOWN"]
    total_val = sum(float(r.get("total_inv_value") or 0) for r in rows)
    liq_val   = sum(float(r.get("total_inv_value") or 0) for r in liquidate)
    md_val    = sum(float(r.get("total_inv_value") or 0) for r in markdown)

    distinct_skus = len({r.get("sku_id") for r in rows if r.get("sku_id")})

    top_liq = max(
        liquidate,
        key=lambda r: float(r.get("total_inv_value") or 0),
        default=None,
    )

    lines = [f"[DEAD STOCK — {report_date}]"]
    lines.append(
        f"  ${total_val/1_000_000:.1f}M capital at risk across "
        f"{distinct_skus:,} SKU positions ({len(rows):,} SKU×location pairs)."
    )
    lines.append(
        f"  LIQUIDATE: {len(liquidate):,} positions  (${liq_val/1_000_000:.1f}M)"
    )
    lines.append(
        f"  MARKDOWN:  {len(markdown):,} positions  (${md_val/1_000_000:.1f}M)"
    )
    if top_liq:
        days = top_liq.get("days_since_sale")
        days_str = f"{int(days)}d idle" if days is not None else "idle"
        lines.append(
            f"  Top liquidation candidate: {top_liq.get('sku_id')} "
            f"@ {_loc_display(top_liq.get('location_id', ''))}  "
            f"${float(top_liq.get('total_inv_value') or 0):,.0f}  "
            f"{days_str}  → {top_liq.get('action', 'LIQUIDATE')}"
        )

    return "\n".join(lines) + "\n"


def _section_basket_rules(client: Any, today: date) -> str:
    try:
        rows = _paginate(
            client, "basket_rules",
            "antecedent_sku,consequent_sku,confidence,lift,transaction_count",
            limit=50,
        )
    except Exception:
        return "[BASKET RULES]\n  Table not available yet.\n"

    if not rows:
        return "[BASKET RULES]\n  No co-purchase rules generated yet.\n"

    sorted_rows = sorted(rows, key=lambda r: float(r.get("lift", 0)), reverse=True)
    top5 = sorted_rows[:5]

    lines = ["[BASKET RULES — co-purchase intelligence]"]
    for r in top5:
        ant = r.get("antecedent_sku", "?")
        con = r.get("consequent_sku", "?")
        conf = float(r.get("confidence", 0)) * 100
        lift = float(r.get("lift", 0))
        lines.append(
            f"  When customers buy {ant} they almost always also buy {con} "
            f"(confidence: {conf:.0f}%, lift: {lift:.1f}x)"
        )

    txn_count = top5[0].get("transaction_count", 0) if top5 else 0
    if txn_count:
        lines.append(f"  Based on {txn_count:,} baskets analyzed.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

def build_context(client: Any) -> str:
    """Build a structured live-data context string for Claude's system prompt.

    Queries all seven data sections from Supabase and assembles them into
    a single string targeted at < 2 000 tokens.  Each section is isolated —
    an error in one section logs a warning and inserts a placeholder rather
    than failing the whole context build.

    Args:
        client: Active Supabase client from db.connection.get_client().

    Returns:
        Multi-line string ready to be appended to Claude's system prompt.
    """
    t0 = time.perf_counter()
    today = date.today()

    sections = [
        ("Alerts",               _section_alerts),
        ("Reorder",              _section_reorder),
        ("Weather",              _section_weather),
        ("Supplier health",      _section_supplier_health),
        ("Inventory health",     _section_inventory_health),
        ("Forecast accuracy",    _section_forecast_accuracy),
        ("Location performance", _section_location_performance),
        ("Dead stock",           _section_dead_stock),
        ("Basket rules",         _section_basket_rules),
    ]

    parts: list[str] = [
        f"=== PARTSWATCH LIVE CONTEXT — {today.isoformat()} ===\n"
    ]

    for name, fn in sections:
        try:
            parts.append(fn(client, today))
        except Exception:
            log.warning("Context section '%s' failed — using placeholder.", name, exc_info=True)
            parts.append(f"[{name.upper()}]\n  Data temporarily unavailable.\n")

    parts.append(f"=== END CONTEXT  (built in {(time.perf_counter() - t0) * 1000:.0f}ms) ===")

    context = "\n".join(parts)
    log.debug(
        "Context built: %d chars (~%d tokens)",
        len(context), len(context) // 4,
    )
    return context
