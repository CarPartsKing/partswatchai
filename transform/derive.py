"""
transform/derive.py — Derived field calculations for partswatch-ai.

Runs after transform/clean.py on every nightly pull.  Reads raw data from the
database, computes derived metrics, and writes results back.  Source rows are
never deleted — only targeted fields are updated via upsert.

DERIVATIONS (run in this order)
    1. lost_sales_imputation — estimate unrecorded demand during stockout events
    2. abc_classification    — rank SKUs A/B/C by 90-day revenue
    3. xyz_classification    — X/Y/Z by CV of weekly demand (must follow ABC)
    4. supplier_scores       — fill rate, lead time, on-time, composite score
    5. weather_sensitivity   — Pearson r between daily qty sold and temp_min_f
    6. sku_metrics           — last_sale_date and avg_weekly_units per SKU

DESIGN PRINCIPLES
    - Each derivation is an isolated function. Adding a new one = write one
      function and append it to DERIVATIONS. Nothing else changes.
    - No source records are deleted or structurally modified.
    - Every function returns {"rows_updated": N} for the orchestrator to log.
    - Timing is measured and logged by the orchestrator for every derivation.

PREREQUISITE
    Run db/migrations/004_supplier_risk_flag.sql in the Supabase SQL Editor
    before first use.

USAGE
    python -m transform.derive
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from statistics import mean, pstdev
from typing import Any, Callable

from utils.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Window (days) around a stockout date to find peer-location sales for imputation
IMPUTATION_WINDOW_DAYS: int = 30

# Revenue lookback for ABC classification
ABC_LOOKBACK_DAYS: int = 90

# Lookback window for XYZ demand-variability classification (weeks of sales data)
XYZ_LOOKBACK_DAYS: int = 90   # same 90-day window as ABC for a consistent snapshot

# Minimum number of distinct weeks needed to compute a meaningful CV
MIN_XYZ_WEEKS: int = 4

# CV thresholds that define each class boundary
XYZ_CV_X: float = 0.5    # CV < this  → X (consistent)
XYZ_CV_Y: float = 1.0    # CV < this  → Y (variable); else → Z (erratic)

# Purchase-order lookback for supplier scoring
SUPPLIER_LOOKBACK_DAYS: int = 90

# Historical weather lookback for sensitivity correlation
WEATHER_LOOKBACK_DAYS: int = 730  # 2 years

# Minimum data points needed to compute a Pearson correlation
PEARSON_MIN_SAMPLES: int = 5

# Supabase page size
_PAGE_SIZE: int = 1000

# Batch size for upserts
BATCH_SIZE: int = 500


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

DeriveFn = Callable[[Any], dict]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fetch_all(
    client: Any,
    table: str,
    select: str = "*",
    gte: dict[str, str] | None = None,
    eq: dict[str, Any] | None = None,
) -> list[dict]:
    """Return rows from a Supabase table with server-side filtering.

    Args:
        client: Active Supabase client.
        table:  Table to query.
        select: PostgREST column selector string.
        gte:    Dict of {column: value} for >= filters (pushed server-side).
        eq:     Dict of {column: value} for == filters (pushed server-side).

    Returns:
        All matching rows as a list of dicts.
    """
    all_rows: list[dict] = []
    offset = 0
    while True:
        q = client.table(table).select(select)
        if gte:
            for col, val in gte.items():
                q = q.gte(col, val)
        if eq:
            for col, val in eq.items():
                q = q.eq(col, val)
        resp = q.range(offset, offset + _PAGE_SIZE - 1).execute()
        page: list[dict] = resp.data or []
        all_rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
        if offset % 50_000 == 0:
            log.info("    … fetched %d rows so far from %s", offset, table)
    return all_rows


def _fetch_chunked_by_date(
    client: Any,
    table: str,
    select: str,
    date_col: str,
    since: str,
    until: str | None = None,
    chunk_days: int = 7,
    extra_eq: dict[str, Any] | None = None,
    callback: Callable[[list[dict]], None] | None = None,
) -> int:
    """Fetch rows in weekly date chunks, calling *callback* for each page.

    Instead of loading millions of rows into one giant list, this streams
    them through *callback* so the caller can accumulate only the tiny
    per-SKU metric it needs.

    Returns total rows fetched.
    """
    from datetime import date as _date
    start = _date.fromisoformat(since)
    end = _date.fromisoformat(until) if until else _date.today()
    total = 0
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
            if callback and page:
                callback(page)
            total += len(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        chunk_start = chunk_end + timedelta(days=1)
        if total > 0 and total % 100_000 < (chunk_days * 7000):
            log.info("    … streamed %d rows so far from %s", total, table)
    log.info("    … streamed %d total rows from %s", total, table)
    return total


def _upsert(client: Any, table: str, rows: list[dict], on_conflict: str) -> None:
    """Upsert rows in BATCH_SIZE chunks.

    Args:
        client:      Active Supabase client.
        table:       Target table name.
        rows:        Full list of rows to upsert.
        on_conflict: Comma-separated conflict column string.
    """
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        client.table(table).upsert(batch, on_conflict=on_conflict).execute()


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Compute the Pearson correlation coefficient between two equal-length lists.

    Returns None if the inputs are too short, or if either variable has zero
    variance (constant values produce a degenerate denominator).

    Args:
        xs: Independent variable values (e.g. temp_min_f).
        ys: Dependent variable values (e.g. daily qty sold).

    Returns:
        Pearson r in [-1, 1], or None if not computable.
    """
    n = len(xs)
    if n < PEARSON_MIN_SAMPLES or len(ys) != n:
        return None

    sum_x  = sum(xs)
    sum_y  = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    sum_y2 = sum(y * y for y in ys)

    numerator = n * sum_xy - sum_x * sum_y
    denom_sq  = (n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)

    if denom_sq <= 0:
        return None

    return numerator / math.sqrt(denom_sq)


# ---------------------------------------------------------------------------
# Derivation 1 — Lost sales imputation
# ---------------------------------------------------------------------------

def derive_lost_sales_imputation(client: Any) -> dict:
    """Estimate unrecorded demand for every stockout transaction.

    For each sales_transactions row where is_stockout=True, we estimate how
    many additional units could have been sold by looking at what the same SKU
    sold at peer locations during a ±IMPUTATION_WINDOW_DAYS window around the
    stockout date.  The mean units-per-sale-event at peers becomes the
    lost_sales_imputation for that row.

    Writes to:
        sales_transactions.lost_sales_imputation

    Returns:
        {"rows_updated": N}
    """
    stockouts = _fetch_all(
        client,
        "sales_transactions",
        "transaction_id,sku_id,location_id,transaction_date,qty_sold,is_stockout",
        eq={"is_stockout": True},
    )

    if not stockouts:
        log.info("  No stockout transactions — skipping imputation.")
        return {"rows_updated": 0}

    log.info("  Found %d stockout transaction(s) to impute.", len(stockouts))

    stockout_skus = {r.get("sku_id", "") for r in stockouts} - {""}

    non_stockouts: list[dict] = []
    for sku in stockout_skus:
        rows = _fetch_all(
            client,
            "sales_transactions",
            "sku_id,location_id,transaction_date,qty_sold",
            eq={"sku_id": sku, "is_stockout": False},
        )
        non_stockouts.extend(rows)

    # Build lookup: sku_id → list of non-stockout sales at peer locations
    peer_sales: dict[str, list[dict]] = {}
    for r in non_stockouts:
        sku = r.get("sku_id", "")
        raw_date = str(r.get("transaction_date", ""))[:10]
        if not sku or not raw_date:
            continue
        try:
            tx_date = date.fromisoformat(raw_date)
        except ValueError:
            continue
        peer_sales.setdefault(sku, []).append({
            "location_id": r.get("location_id"),
            "date":        tx_date,
            "qty":         float(r.get("qty_sold") or 0),
        })

    updates: list[dict] = []
    window = timedelta(days=IMPUTATION_WINDOW_DAYS)

    for row in stockouts:
        sku = row.get("sku_id", "")
        loc = row.get("location_id", "")
        raw_date = str(row.get("transaction_date", ""))[:10]
        try:
            tx_date = date.fromisoformat(raw_date)
        except ValueError:
            continue

        # Peer sales: same SKU, different location, within ±window
        peers = [
            s["qty"]
            for s in peer_sales.get(sku, [])
            if s["location_id"] != loc
            and abs((s["date"] - tx_date).days) <= IMPUTATION_WINDOW_DAYS
        ]

        imputed = round(mean(peers), 2) if peers else 0.0

        updates.append({
            "transaction_id":       row["transaction_id"],
            "lost_sales_imputation": imputed,
        })

    # Use UPDATE (not upsert) — we are patching one field on an existing row.
    # Upsert would try to INSERT a new row with nulls for all other NOT NULL
    # columns when the conflict key is not found, which fails the constraint.
    for row in updates:
        client.table("sales_transactions").update(
            {"lost_sales_imputation": row["lost_sales_imputation"]}
        ).eq("transaction_id", row["transaction_id"]).execute()

    log.info(
        "  Updated lost_sales_imputation for %d stockout transaction(s).",
        len(updates),
    )
    return {"rows_updated": len(updates)}


# ---------------------------------------------------------------------------
# Derivation 2 — ABC classification
# ---------------------------------------------------------------------------

def derive_abc_classification(client: Any) -> dict:
    """Rank all SKUs A/B/C by total revenue over the last ABC_LOOKBACK_DAYS days.

    Ranking thresholds (by SKU count, not by revenue value):
        A — top 20 %
        B — next 30 %  (i.e. ranks 20–50 %)
        C — bottom 50 % + any SKU with no sales in the window

    Writes to:
        sku_master.abc_class

    Returns:
        {"rows_updated": N}
    """
    cutoff = (date.today() - timedelta(days=ABC_LOOKBACK_DAYS)).isoformat()

    log.info("  Streaming transactions since %s for ABC classification …", cutoff)
    revenue: dict[str, float] = {}

    def _acc_revenue(page: list[dict]) -> None:
        for r in page:
            sku = r.get("sku_id", "")
            if sku:
                revenue[sku] = revenue.get(sku, 0.0) + float(r.get("total_revenue") or 0)

    _fetch_chunked_by_date(
        client, "sales_transactions", "sku_id,total_revenue",
        date_col="transaction_date", since=cutoff,
        callback=_acc_revenue,
    )

    # Sort SKUs by revenue descending
    ranked = sorted(revenue.items(), key=lambda kv: kv[1], reverse=True)
    n      = len(ranked)

    # Percentile cut points (by index)
    a_end = max(1, round(n * 0.20))       # exclusive end of A tier
    b_end = max(a_end + 1, round(n * 0.50))  # exclusive end of B tier (A+B = 50%)

    updates: list[dict] = []
    class_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}

    for idx, (sku_id, rev) in enumerate(ranked):
        cls = "A" if idx < a_end else ("B" if idx < b_end else "C")
        class_counts[cls] += 1
        updates.append({"sku_id": sku_id, "abc_class": cls})

    # SKUs in sku_master with no sales in the window default to C
    all_skus = _fetch_all(client, "sku_master", "sku_id")
    seen_skus = {u["sku_id"] for u in updates}
    for r in all_skus:
        if r["sku_id"] not in seen_skus:
            updates.append({"sku_id": r["sku_id"], "abc_class": "C"})
            class_counts["C"] += 1

    _upsert(client, "sku_master", updates, on_conflict="sku_id")

    log.info(
        "  ABC classified %d SKU(s): A=%d  B=%d  C=%d",
        len(updates), class_counts["A"], class_counts["B"], class_counts["C"],
    )
    return {"rows_updated": len(updates)}


# ---------------------------------------------------------------------------
# Derivation 3 — XYZ demand-variability classification
# ---------------------------------------------------------------------------

def derive_xyz_classification(client: Any) -> dict:
    """Classify every SKU X / Y / Z by the variability of its weekly demand.

    The coefficient of variation (CV = population_std / mean) is computed from
    the SKU's weekly sales totals over the last XYZ_LOOKBACK_DAYS days.  Weeks
    inside the window with no recorded sales are treated as 0 — a product that
    only sells some weeks is inherently more erratic than one that sells every
    week.

    CV thresholds:
        X — CV < 0.5   : demand is stable and predictable week-to-week
        Y — 0.5 ≤ CV < 1.0 : demand varies but has a detectable pattern
        Z — CV ≥ 1.0   : demand is erratic and hard to forecast

    Special cases (assigned without computing CV):
        No sales in the window       → Z  (dead / never-moved)
        Fewer than MIN_XYZ_WEEKS     → Y  (not enough history for reliable CV)
        Mean of weekly series = 0    → Z  (technically impossible if no-sales→Z,
                                           but guards against float-zero errors)

    After classifying, reads the abc_class already written to sku_master by
    derive_abc_classification and combines the two into abc_xyz_class
    (e.g. "AX", "BZ", "CY").

    Writes to:
        sku_master.xyz_class       (CHAR 1: X / Y / Z)
        sku_master.abc_xyz_class   (CHAR 2: AX / AY / AZ / BX … CZ)

    Returns:
        {"rows_updated": N}
    """
    try:
        client.table("sku_master").select("xyz_class").limit(1).execute()
    except Exception:
        log.warning(
            "  xyz_class column not found in sku_master — "
            "apply migration 010_xyz_classification.sql in the Supabase SQL Editor first."
        )
        return {"rows_updated": 0}

    today  = date.today()
    cutoff = (today - timedelta(days=XYZ_LOOKBACK_DAYS)).isoformat()

    log.info("  Streaming transactions since %s for XYZ classification …", cutoff)

    all_iso_weeks: list[tuple[int, int]] = []
    cursor = today - timedelta(days=XYZ_LOOKBACK_DAYS)
    seen_weeks: set[tuple[int, int]] = set()
    while cursor <= today:
        iso_cal = cursor.isocalendar()
        wk = (iso_cal[0], iso_cal[1])
        if wk not in seen_weeks:
            all_iso_weeks.append(wk)
            seen_weeks.add(wk)
        cursor += timedelta(days=7)
    n_weeks = len(all_iso_weeks)

    weekly: dict[str, dict[tuple[int, int], float]] = defaultdict(
        lambda: defaultdict(float)
    )

    def _acc_weekly(page: list[dict]) -> None:
        for r in page:
            sku   = r.get("sku_id", "")
            qty   = float(r.get("qty_sold") or 0)
            d_str = str(r.get("transaction_date", ""))[:10]
            if not sku or not d_str:
                continue
            iso_cal = date.fromisoformat(d_str).isocalendar()
            weekly[sku][(iso_cal[0], iso_cal[1])] += qty

    _fetch_chunked_by_date(
        client, "sales_transactions", "sku_id,qty_sold,transaction_date",
        date_col="transaction_date", since=cutoff,
        callback=_acc_weekly,
    )

    # Fetch current abc_class for the combined field
    sku_rows = _fetch_all(client, "sku_master", "sku_id,abc_class")
    abc_map  = {r["sku_id"]: (r.get("abc_class") or "C") for r in sku_rows}
    all_skus = list(abc_map.keys())

    updates: list[dict] = []
    class_counts: dict[str, int] = {"X": 0, "Y": 0, "Z": 0}

    for sku_id in all_skus:
        week_data = weekly.get(sku_id, {})

        if not week_data:
            # No sales at all in the window
            xyz = "Z"
        else:
            # Build the full weekly series including zero-sales weeks
            series = [week_data.get(wk, 0.0) for wk in all_iso_weeks]
            total  = sum(series)

            if total == 0:
                xyz = "Z"
            elif n_weeks < MIN_XYZ_WEEKS:
                # Too few weeks — flag Y (not enough data to confirm stability)
                xyz = "Y"
            else:
                mean_w = total / n_weeks
                std_w  = pstdev(series)   # population std dev
                cv     = std_w / mean_w if mean_w > 0 else float("inf")

                if cv < XYZ_CV_X:
                    xyz = "X"
                elif cv < XYZ_CV_Y:
                    xyz = "Y"
                else:
                    xyz = "Z"

        class_counts[xyz] += 1
        abc = abc_map.get(sku_id, "C")
        updates.append({
            "sku_id":        sku_id,
            "xyz_class":     xyz,
            "abc_xyz_class": abc + xyz,
        })

    _upsert(client, "sku_master", updates, on_conflict="sku_id")

    log.info(
        "  XYZ classified %d SKU(s): X=%d  Y=%d  Z=%d  "
        "(lookback=%dd  weeks=%d  min_weeks=%d  cv_x=%.1f  cv_y=%.1f)",
        len(updates), class_counts["X"], class_counts["Y"], class_counts["Z"],
        XYZ_LOOKBACK_DAYS, n_weeks, MIN_XYZ_WEEKS, XYZ_CV_X, XYZ_CV_Y,
    )
    return {"rows_updated": len(updates)}


# ---------------------------------------------------------------------------
# Derivation 4 — Supplier scores
# ---------------------------------------------------------------------------

def derive_supplier_scores(client: Any) -> dict:
    """Calculate rolling performance metrics for every supplier.

    Metrics (from purchase_orders in last SUPPLIER_LOOKBACK_DAYS days):
        fill_rate_pct        — avg(qty_received / qty_ordered) for delivered lines
        avg_lead_time_days   — avg days from po_date to actual_delivery_date
        lead_time_variance_avg — avg |actual_delivery - expected_delivery| in days
        on_time_delivery_pct — fraction of lines where actual ≤ expected delivery
        composite_score      — fill_rate×40% + on_time×40% + var_component×20%
        risk_flag            — green (>80), amber (60–80), red (<60)

    Writes to:
        supplier_scores  (upsert on supplier_id, score_date)

    Returns:
        {"rows_updated": N}
    """
    cutoff = (date.today() - timedelta(days=SUPPLIER_LOOKBACK_DAYS)).isoformat()
    today  = date.today().isoformat()

    po_rows = _fetch_all(
        client,
        "purchase_orders",
        "supplier_id,po_date,expected_delivery_date,actual_delivery_date,"
        "qty_ordered,qty_received,status",
    )
    po_rows = [r for r in po_rows if str(r.get("po_date", ""))[:10] >= cutoff]

    if not po_rows:
        log.warning("  No purchase orders in the last %d days — skipping.", SUPPLIER_LOOKBACK_DAYS)
        return {"rows_updated": 0}

    # Group by supplier
    by_supplier: dict[str, list[dict]] = {}
    for r in po_rows:
        sup = r.get("supplier_id", "")
        if sup:
            by_supplier.setdefault(sup, []).append(r)

    updates: list[dict] = []

    for supplier_id, lines in by_supplier.items():

        # ── Fill rate: received and partial lines only ─────────────────────
        attempted = [l for l in lines if l.get("status") in ("received", "partial")]
        fill_rates: list[float] = []
        for l in attempted:
            qty_ord = float(l.get("qty_ordered") or 0)
            qty_rec = float(l.get("qty_received") or 0)
            if qty_ord > 0:
                fill_rates.append(min(qty_rec / qty_ord, 1.0))
        fill_rate = mean(fill_rates) if fill_rates else 0.0

        # ── Lead time and on-time: lines with actual_delivery_date ─────────
        delivered = [l for l in lines if l.get("actual_delivery_date")]
        lead_times: list[float] = []
        variances:  list[float] = []
        on_times:   list[float] = []

        for l in delivered:
            try:
                actual   = date.fromisoformat(str(l["actual_delivery_date"])[:10])
                expected = date.fromisoformat(str(l["expected_delivery_date"])[:10])
                po_dt    = date.fromisoformat(str(l["po_date"])[:10])
                lead_times.append(float((actual - po_dt).days))
                variances.append(float(abs((actual - expected).days)))
                on_times.append(1.0 if actual <= expected else 0.0)
            except (ValueError, TypeError, KeyError):
                continue

        avg_lead_time = mean(lead_times) if lead_times else None
        avg_variance  = mean(variances)  if variances  else None
        on_time_pct   = mean(on_times)   if on_times   else None

        # ── Composite score (0–100) ────────────────────────────────────────
        fill_score    = fill_rate * 100                                # 0–100
        on_time_score = (on_time_pct * 100) if on_time_pct is not None else 50.0
        # Variance component: 0 days = 100, each extra day costs 10 pts, floor 0
        var_score     = max(0.0, 100.0 - (avg_variance or 0) * 10.0)
        composite     = fill_score * 0.40 + on_time_score * 0.40 + var_score * 0.20

        risk_flag = "green" if composite > 80 else ("amber" if composite >= 60 else "red")

        log.info(
            "  %-12s  fill=%5.1f%%  on_time=%5.1f%%  var=%.1fd  "
            "composite=%5.1f  [%s]",
            supplier_id,
            fill_rate * 100,
            (on_time_pct or 0) * 100,
            avg_variance or 0,
            composite,
            risk_flag.upper(),
        )

        updates.append({
            "supplier_id":           supplier_id,
            "score_date":            today,
            "fill_rate_pct":         round(fill_rate, 4),
            "avg_lead_time_days":    round(avg_lead_time, 2)  if avg_lead_time is not None else None,
            "lead_time_variance_avg": round(avg_variance, 2)  if avg_variance  is not None else None,
            "on_time_delivery_pct":  round(on_time_pct, 4)    if on_time_pct   is not None else None,
            "composite_score":       round(composite, 2),
            "risk_flag":             risk_flag,
        })

    _upsert(client, "supplier_scores", updates, on_conflict="supplier_id,score_date")
    return {"rows_updated": len(updates)}


# ---------------------------------------------------------------------------
# Derivation 4 — Weather sensitivity score
# ---------------------------------------------------------------------------

def derive_weather_sensitivity(client: Any) -> dict:
    """Correlate daily sales volume with minimum temperature per SKU category.

    For each part_category, computes the Pearson r between:
        - temp_min_f (from weather_log, historical rows only)
        - sum of qty_sold per day (from sales_transactions)
    over the last WEATHER_LOOKBACK_DAYS days.

    Interpretation:
        r ≈ –1  Cold weather drives sales (batteries, block heaters)
        r ≈ +1  Warm weather drives sales (A/C parts, coolant hoses)
        r ≈  0  No weather relationship

    Writes the category-level r value to:
        sku_master.weather_sensitivity_score  (for every SKU in that category)

    Returns:
        {"rows_updated": N}
    """
    cutoff = (date.today() - timedelta(days=WEATHER_LOOKBACK_DAYS)).isoformat()

    sku_rows     = _fetch_all(client, "sku_master", "sku_id,part_category")
    sku_to_cat:  dict[str, str] = {r["sku_id"]: r.get("part_category", "") for r in sku_rows}

    cats_with_data = {c for c in sku_to_cat.values() if c}
    if not cats_with_data:
        log.info("  No part_category data in sku_master — skipping weather sensitivity.")
        return {"rows_updated": 0}

    weather_rows = _fetch_all(
        client, "weather_log", "log_date,temp_min_f,is_forecast",
        gte={"log_date": cutoff},
        eq={"is_forecast": False},
    )

    temp_by_date: dict[str, float] = {}
    for r in weather_rows:
        if r.get("temp_min_f") is not None:
            temp_by_date[str(r["log_date"])[:10]] = float(r["temp_min_f"])

    if not temp_by_date:
        log.info("  No historical weather data — skipping weather sensitivity.")
        return {"rows_updated": 0}

    log.info("  Streaming transactions since %s for weather sensitivity …", cutoff)
    daily_qty: dict[tuple[str, str], float] = {}

    def _acc_weather(page: list[dict]) -> None:
        for r in page:
            date_str = str(r.get("transaction_date", ""))[:10]
            sku      = r.get("sku_id", "")
            category = sku_to_cat.get(sku, "")
            if not date_str or not category or date_str not in temp_by_date:
                continue
            qty = float(r.get("qty_sold") or 0)
            key = (date_str, category)
            daily_qty[key] = daily_qty.get(key, 0.0) + qty

    _fetch_chunked_by_date(
        client, "sales_transactions", "sku_id,transaction_date,qty_sold",
        date_col="transaction_date", since=cutoff,
        callback=_acc_weather,
    )

    # Assemble (temp, qty) pairs per category
    cat_pairs: dict[str, list[tuple[float, float]]] = {}
    for (date_str, category), qty in daily_qty.items():
        temp = temp_by_date.get(date_str)
        if temp is not None:
            cat_pairs.setdefault(category, []).append((temp, qty))

    # Compute Pearson r per category
    cat_corr: dict[str, float] = {}
    for category, pairs in cat_pairs.items():
        temps = [p[0] for p in pairs]
        qtys  = [p[1] for p in pairs]
        r_val = _pearson_r(temps, qtys)
        if r_val is not None:
            cat_corr[category] = round(r_val, 4)
            log.info(
                "  Category %-22s  n=%3d  r = %+.4f",
                category, len(pairs), r_val,
            )
        else:
            log.info(
                "  Category %-22s  n=%3d  r = n/a (insufficient data)",
                category, len(pairs),
            )

    # Update every SKU with its category's correlation coefficient
    updates: list[dict] = []
    for r in sku_rows:
        cat  = r.get("part_category", "")
        corr = cat_corr.get(cat)
        if corr is not None:
            updates.append({"sku_id": r["sku_id"], "weather_sensitivity_score": corr})

    if updates:
        _upsert(client, "sku_master", updates, on_conflict="sku_id")

    return {"rows_updated": len(updates)}


# ---------------------------------------------------------------------------
# Derivation 5 — SKU metrics (last_sale_date, avg_weekly_units)
# ---------------------------------------------------------------------------

SKU_METRICS_LOOKBACK_DAYS: int = 365


def derive_sku_metrics(client: Any) -> dict:
    """Update last_sale_date and avg_weekly_units in sku_master for every SKU.

    last_sale_date   — date of most recent transaction for each SKU
    avg_weekly_units — total qty_sold ÷ weeks in lookback window

    Uses a 365-day lookback for avg_weekly_units to avoid fetching all 8M+ rows.

    Writes to:
        sku_master.last_sale_date
        sku_master.avg_weekly_units

    Returns:
        {"rows_updated": N}
    """
    cutoff = (date.today() - timedelta(days=SKU_METRICS_LOOKBACK_DAYS)).isoformat()
    total_weeks = SKU_METRICS_LOOKBACK_DAYS / 7.0

    log.info("  Streaming transactions since %s for SKU metrics (%.0f weeks) …",
             cutoff, total_weeks)

    sku_max_date: dict[str, str] = {}
    sku_qty:      dict[str, float] = {}

    def _acc_metrics(page: list[dict]) -> None:
        for r in page:
            sku      = r.get("sku_id", "")
            date_str = str(r.get("transaction_date", ""))[:10]
            qty      = float(r.get("qty_sold") or 0)
            if not sku:
                continue
            if sku not in sku_max_date or date_str > sku_max_date[sku]:
                sku_max_date[sku] = date_str
            sku_qty[sku] = sku_qty.get(sku, 0.0) + qty

    _fetch_chunked_by_date(
        client, "sales_transactions", "sku_id,transaction_date,qty_sold",
        date_col="transaction_date", since=cutoff,
        callback=_acc_metrics,
    )

    if not sku_qty:
        log.info("  No sales transactions found — skipping SKU metrics.")
        return {"rows_updated": 0}

    updates: list[dict] = []
    for sku, max_d in sku_max_date.items():
        updates.append({
            "sku_id":          sku,
            "last_sale_date":  max_d,
            "avg_weekly_units": round(sku_qty.get(sku, 0.0) / total_weeks, 2),
        })

    _upsert(client, "sku_master", updates, on_conflict="sku_id")
    return {"rows_updated": len(updates)}


# ---------------------------------------------------------------------------
# Derivation registry
# New derivation: write one function above + add one tuple here. Done.
# ---------------------------------------------------------------------------

DERIVATIONS: list[tuple[str, DeriveFn]] = [
    ("Lost sales imputation",       derive_lost_sales_imputation),
    ("ABC classification",          derive_abc_classification),
    ("XYZ demand variability",      derive_xyz_classification),   # after ABC (reads abc_class)
    ("Supplier scores",             derive_supplier_scores),
    ("Weather sensitivity",         derive_weather_sensitivity),
    ("SKU metrics",                 derive_sku_metrics),
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_derivations() -> int:
    """Run every registered derivation in order and log timing + row counts.

    Returns:
        Exit code: 0 on success, 1 if any derivation raised an exception.
    """
    from db.connection import get_client
    client = get_client()

    failed: list[str] = []
    total_rows = 0

    log.info("Running %d derivation(s) …", len(DERIVATIONS))
    log.info("=" * 60)

    for label, fn in DERIVATIONS:
        log.info("▶  %s", label)
        t0 = time.perf_counter()
        try:
            stats   = fn(client)
            elapsed = time.perf_counter() - t0
            rows    = stats.get("rows_updated", 0)
            total_rows += rows
            log.info("   ✓ %d row(s) updated  (%.2fs)", rows, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error("   ✗ FAILED after %.2fs: %s", elapsed, exc, exc_info=True)
            failed.append(label)
        log.info("-" * 60)

    log.info("Derivations complete — %d total row(s) updated.", total_rows)
    if failed:
        log.error("Failed derivations: %s", ", ".join(failed))
    log.info("=" * 60)

    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse arguments and run all derivations."""
    parser = argparse.ArgumentParser(
        description="partswatch-ai: compute and persist all derived fields.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prerequisite: run db/migrations/004_supplier_risk_flag.sql\n"
            "in the Supabase SQL Editor before first use.\n"
        ),
    )
    # Reserved for future flags (e.g. --only=abc_classification)
    parser.parse_args()

    log.info("=" * 60)
    log.info("partswatch-ai — derive")
    log.info("=" * 60)

    return run_derivations()


if __name__ == "__main__":
    sys.exit(main())
