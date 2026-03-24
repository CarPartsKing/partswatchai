"""
transform/derive.py — Derived field calculations for partswatch-ai.

Runs after transform/clean.py on every nightly pull.  Reads raw data from the
database, computes derived metrics, and writes results back.  Source rows are
never deleted — only targeted fields are updated via upsert.

DERIVATIONS (run in this order)
    1. lost_sales_imputation — estimate unrecorded demand during stockout events
    2. abc_classification    — rank SKUs A/B/C by 90-day revenue
    3. supplier_scores       — fill rate, lead time, on-time, composite score
    4. weather_sensitivity   — Pearson r between daily qty sold and temp_min_f
    5. sku_metrics           — last_sale_date and avg_weekly_units per SKU

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
from datetime import date, timedelta
from statistics import mean
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

def _fetch_all(client: Any, table: str, select: str = "*") -> list[dict]:
    """Return every row from a Supabase table, handling the 1000-row page cap.

    Args:
        client: Active Supabase client.
        table:  Table to query.
        select: PostgREST column selector string.

    Returns:
        All matching rows as a list of dicts.
    """
    all_rows: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table(table)
            .select(select)
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        page: list[dict] = resp.data or []
        all_rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return all_rows


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
    tx_rows = _fetch_all(
        client,
        "sales_transactions",
        "transaction_id,sku_id,location_id,transaction_date,qty_sold,is_stockout",
    )

    stockouts     = [r for r in tx_rows if r.get("is_stockout")]
    non_stockouts = [r for r in tx_rows if not r.get("is_stockout")]

    if not stockouts:
        log.info("  No stockout transactions — skipping imputation.")
        return {"rows_updated": 0}

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

    _upsert(client, "sales_transactions", updates, on_conflict="transaction_id")

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

    tx_rows = _fetch_all(
        client, "sales_transactions", "sku_id,total_revenue,transaction_date"
    )
    tx_rows = [r for r in tx_rows if str(r.get("transaction_date", ""))[:10] >= cutoff]

    # Sum revenue per SKU over the window
    revenue: dict[str, float] = {}
    for r in tx_rows:
        sku = r.get("sku_id", "")
        rev = float(r.get("total_revenue") or 0)
        revenue[sku] = revenue.get(sku, 0.0) + rev

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
# Derivation 3 — Supplier scores
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

    tx_rows      = _fetch_all(client, "sales_transactions",  "sku_id,transaction_date,qty_sold")
    sku_rows     = _fetch_all(client, "sku_master",          "sku_id,part_category")
    weather_rows = _fetch_all(client, "weather_log",         "log_date,temp_min_f,is_forecast")

    # Filter to window, historical weather only
    tx_rows      = [r for r in tx_rows      if str(r.get("transaction_date", ""))[:10] >= cutoff]
    weather_rows = [r for r in weather_rows
                    if str(r.get("log_date", ""))[:10] >= cutoff
                    and not r.get("is_forecast")]

    # Build lookups
    sku_to_cat:      dict[str, str]   = {r["sku_id"]: r.get("part_category", "") for r in sku_rows}
    temp_by_date:    dict[str, float] = {}
    for r in weather_rows:
        if r.get("temp_min_f") is not None:
            temp_by_date[str(r["log_date"])[:10]] = float(r["temp_min_f"])

    # Aggregate daily qty sold by (date, category)
    daily_qty: dict[tuple[str, str], float] = {}
    for r in tx_rows:
        date_str = str(r.get("transaction_date", ""))[:10]
        sku      = r.get("sku_id", "")
        category = sku_to_cat.get(sku, "")
        if not date_str or not category or date_str not in temp_by_date:
            continue
        qty = float(r.get("qty_sold") or 0)
        key = (date_str, category)
        daily_qty[key] = daily_qty.get(key, 0.0) + qty

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

def derive_sku_metrics(client: Any) -> dict:
    """Update last_sale_date and avg_weekly_units in sku_master for every SKU.

    last_sale_date   — date of most recent transaction for each SKU
    avg_weekly_units — total qty_sold ÷ total weeks spanned by the dataset

    The week denominator uses the full date range of all transactions in the
    database (not just a lookback window) so the metric reflects the true
    average selling rate over the observed history.

    Writes to:
        sku_master.last_sale_date
        sku_master.avg_weekly_units

    Returns:
        {"rows_updated": N}
    """
    tx_rows = _fetch_all(
        client, "sales_transactions", "sku_id,transaction_date,qty_sold"
    )

    if not tx_rows:
        log.info("  No sales transactions found — skipping SKU metrics.")
        return {"rows_updated": 0}

    # Global date range → week denominator
    all_dates: list[str] = [
        str(r["transaction_date"])[:10]
        for r in tx_rows
        if r.get("transaction_date")
    ]
    if not all_dates:
        return {"rows_updated": 0}

    global_min = date.fromisoformat(min(all_dates))
    global_max = date.fromisoformat(max(all_dates))
    total_weeks = max((global_max - global_min).days / 7.0, 1.0)

    log.info(
        "  Date range %s → %s  (%.1f weeks)",
        global_min, global_max, total_weeks,
    )

    # Aggregate per SKU
    sku_dates: dict[str, list[str]]  = {}
    sku_qty:   dict[str, float]      = {}

    for r in tx_rows:
        sku      = r.get("sku_id", "")
        date_str = str(r.get("transaction_date", ""))[:10]
        qty      = float(r.get("qty_sold") or 0)
        if not sku:
            continue
        sku_dates.setdefault(sku, []).append(date_str)
        sku_qty[sku] = sku_qty.get(sku, 0.0) + qty

    updates: list[dict] = []
    for sku, dates in sku_dates.items():
        updates.append({
            "sku_id":          sku,
            "last_sale_date":  max(dates),
            "avg_weekly_units": round(sku_qty[sku] / total_weeks, 2),
        })

    _upsert(client, "sku_master", updates, on_conflict="sku_id")
    return {"rows_updated": len(updates)}


# ---------------------------------------------------------------------------
# Derivation registry
# New derivation: write one function above + add one tuple here. Done.
# ---------------------------------------------------------------------------

DERIVATIONS: list[tuple[str, DeriveFn]] = [
    ("Lost sales imputation",   derive_lost_sales_imputation),
    ("ABC classification",      derive_abc_classification),
    ("Supplier scores",         derive_supplier_scores),
    ("Weather sensitivity",     derive_weather_sensitivity),
    ("SKU metrics",             derive_sku_metrics),
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
