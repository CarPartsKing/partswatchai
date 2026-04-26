"""ml/stocking_intelligence.py — Transfer-pattern stocking gap analysis.

Answers: "What should each store stock locally instead of waiting for transfers?"

Analyzes historical transfer recommendations to identify chronic stocking gaps
at each location.  When a SKU is repeatedly transferred TO a location, that
location should stock it locally — transfers cost time and $2.50 in handling
each time.

ALGORITHM
---------
1. Fetch reorder_recommendations where recommendation_type='transfer',
   recommendation_date >= 90 days ago.  Pages in batches of 1000 SKUs.

2. For each (sku_id, to_location) pair compute:
     transfer_frequency   — distinct recommendation_date values in 90-day window
     transfer_streak      — max consecutive calendar days with a recommendation
     avg_qty_recommended  — mean qty_to_order across all transfer rows
     trend_direction      — INCREASING / STABLE / DECREASING (last 30d vs prior 30d)
     total_transfer_value — avg_qty × unit_cost × transfer_frequency

3. Gap score and classification:
     CHRONIC   (score > 0.7) — freq >= 10 OR streak >= 5  → stock locally
     RECURRING (0.4–0.7)     — freq  5–9                  → raise Min Qty
     OCCASIONAL (< 0.4)      — freq < 5                   → normal

4. For CHRONIC cases:
     suggested_stock_increase = avg_qty × CHRONIC_GAP_MULTIPLIER × 1.2
     annual_cost_savings = (freq / LOOKBACK_DAYS * 365) × avg_qty × TRANSFER_HANDLING_COST

5. Upsert to stocking_gaps (UNIQUE analysis_date, sku_id, location_id).

USAGE
-----
    python -m ml.stocking_intelligence            # full run
    python -m ml.stocking_intelligence --dry-run  # compute only, no DB write
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHRONIC_THRESHOLD_DAYS: int   = 10
"""Transfers in 90 days at or above this → CHRONIC gap."""

STREAK_THRESHOLD: int         = 5
"""Consecutive days recommended at or above this → CHRONIC gap."""

TRANSFER_HANDLING_COST: float = 2.50
"""Dollar cost per transfer handled (picking, packing, inter-store logistics)."""

CHRONIC_GAP_MULTIPLIER: float = 1.5
"""Multiplier applied to avg_qty when computing suggested_stock_increase."""

LOOKBACK_DAYS: int            = 90
"""Days of transfer history to analyse."""

SAFETY_FACTOR: float          = 1.2
"""Conservative safety buffer applied on top of CHRONIC_GAP_MULTIPLIER."""

TREND_INCREASE_RATIO: float   = 1.20
"""last30 / prior30 >= this → INCREASING."""

TREND_DECREASE_RATIO: float   = 0.80
"""last30 / prior30 <= this → DECREASING."""

SKU_BATCH_SIZE: int           = 1_000
"""Max SKUs per .in_() query — keeps calls inside Supabase's 57014 timeout."""

WRITE_BATCH_SIZE: int         = 500
"""Rows per upsert round-trip."""

PROGRESS_LOG_EVERY: int       = 5_000
"""Log a progress line after processing this many SKUs."""

PAGE_SIZE: int                = 1_000

NORMAL_GP_PCT: float          = 0.35
"""Fallback GP% when a location has no SL/SL-I history in get_location_gp_baselines."""

# ---------------------------------------------------------------------------
# Location name lookup (matches all other modules)
# ---------------------------------------------------------------------------

LOCATION_NAMES: dict[str, str] = {
    "LOC-001": "BROOKPARK",        "LOC-002": "NOLMSTEAD",
    "LOC-003": "S.EUCLID",         "LOC-004": "CLARK AUTO",
    "LOC-005": "PARMA",            "LOC-006": "MEDINA",
    "LOC-007": "BOARDMAN",         "LOC-008": "ELYRIA",
    "LOC-009": "AKRON-GRANT",      "LOC-010": "MIDWAY CROSSINGS",
    "LOC-011": "ERIE ST",          "LOC-012": "MAYFIELD",
    "LOC-013": "CANTON",           "LOC-015": "JUNIATA",
    "LOC-016": "ARCHWOOD",         "LOC-017": "EUCLID",
    "LOC-018": "WARREN",           "LOC-020": "ROOTSTOWN",
    "LOC-021": "INTERNET",         "LOC-024": "MENTOR",
    "LOC-025": "MAIN DC",          "LOC-026": "COPLEY",
    "LOC-027": "CHARDON",          "LOC-028": "STRONGSVILLE",
    "LOC-029": "MIDDLEBURG",       "LOC-032": "PERRY",
    "LOC-033": "CRYSTAL",
}

EXCLUDED_LOCATIONS: frozenset[str] = frozenset({
    "LOC-021", "LOC-025",
    "LOC-014", "LOC-019", "LOC-022", "LOC-023", "LOC-030", "LOC-031",
})

# ---------------------------------------------------------------------------
# Retry helpers — mirrors engine/reorder.py pattern
# ---------------------------------------------------------------------------

_MAX_RETRIES: int   = 5
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
    "ReadError",
    "RemoteDisconnected",
    "Server disconnected",
)


def _is_retryable(exc: Exception) -> bool:
    blob = type(exc).__name__ + " " + str(exc)
    return any(tok in blob for tok in _RETRYABLE_TOKENS)


def _get_fresh_client() -> Any:
    try:
        from db.connection import get_new_client  # type: ignore[attr-defined]
        return get_new_client()
    except (ImportError, AttributeError):
        return get_client()


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def _paginate(
    client_holder: list,
    table:  str,
    select: str,
    filters:     dict | None = None,
    gte_filters: dict | None = None,
    lte_filters: dict | None = None,
    in_filters:  dict | None = None,
    order_by:    list[tuple[str, bool]] | None = None,
) -> list[dict]:
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
                for col, desc in (order_by or []):
                    q = q.order(col, desc=desc)
                page = q.range(offset, offset + PAGE_SIZE - 1).execute().data or []
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
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
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# Step 1 — fetch transfer recommendations
# ---------------------------------------------------------------------------

def _fetch_transfer_recs(
    client_holder: list,
    cutoff: date,
    trend_cutoff: date,
) -> dict[tuple[str, str], dict]:
    """Return per-(sku_id, location_id) transfer stats from the last LOOKBACK_DAYS.

    Paginates the full transfer history then aggregates in Python.  The result
    dict maps (sku_id, to_location) → stat block containing all derived metrics.
    """
    log.info("  Fetching transfer recommendations since %s …", cutoff.isoformat())
    rows = _paginate(
        client_holder,
        "reorder_recommendations",
        "sku_id,location_id,transfer_from_location,recommendation_date,qty_to_order",
        filters={"recommendation_type": "transfer"},
        gte_filters={"recommendation_date": cutoff.isoformat()},
        order_by=[("recommendation_date", False), ("sku_id", False)],
    )
    log.info("  Transfer rows fetched: %d", len(rows))

    # Accumulate per (sku, to_location):
    #   dates: set of recommendation_date strings
    #   qtys:  list of float qty_to_order
    #   from_locs: most common transfer_from_location
    #   last30_dates: dates in last 30 days
    #   prior30_dates: dates 30-60 days ago

    today = date.today()
    last30_cutoff  = (today - timedelta(days=30)).isoformat()
    prior30_start  = (today - timedelta(days=60)).isoformat()
    prior30_end    = (today - timedelta(days=31)).isoformat()

    raw: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "dates": set(),
        "qtys": [],
        "from_loc_counts": defaultdict(int),
        "last30_dates": set(),
        "prior30_dates": set(),
    })

    for r in rows:
        sku = r.get("sku_id")
        to_loc = r.get("location_id")
        rec_date = str(r.get("recommendation_date", ""))[:10]
        if not sku or not to_loc or not rec_date:
            continue
        if to_loc in EXCLUDED_LOCATIONS:
            continue

        key = (sku, to_loc)
        entry = raw[key]
        entry["dates"].add(rec_date)
        qty = float(r.get("qty_to_order") or 0)
        if qty > 0:
            entry["qtys"].append(qty)
        from_loc = r.get("transfer_from_location")
        if from_loc:
            entry["from_loc_counts"][from_loc] += 1

        if rec_date >= last30_cutoff:
            entry["last30_dates"].add(rec_date)
        if prior30_start <= rec_date <= prior30_end:
            entry["prior30_dates"].add(rec_date)

    log.info("  Unique (sku, location) pairs with transfers: %d", len(raw))
    return raw


# ---------------------------------------------------------------------------
# Step 2 — compute gap statistics per pair
# ---------------------------------------------------------------------------

def _compute_streak(dates: set[str]) -> int:
    """Return the longest run of consecutive calendar days in a set of date strings."""
    if not dates:
        return 0
    sorted_dates = sorted(date.fromisoformat(d) for d in dates)
    max_streak = 1
    current    = 1
    for i in range(1, len(sorted_dates)):
        if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 1
    return max_streak


def _compute_trend(last30: int, prior30: int) -> str:
    if prior30 == 0:
        return "INCREASING" if last30 > 0 else "STABLE"
    ratio = last30 / prior30
    if ratio >= TREND_INCREASE_RATIO:
        return "INCREASING"
    if ratio <= TREND_DECREASE_RATIO:
        return "DECREASING"
    return "STABLE"


def _gap_score_and_class(
    transfer_frequency: int,
    transfer_streak: int,
) -> tuple[float, str]:
    """Return (gap_score, gap_classification)."""
    is_chronic = (transfer_frequency >= CHRONIC_THRESHOLD_DAYS
                  or transfer_streak >= STREAK_THRESHOLD)
    if is_chronic:
        freq_ratio   = min(transfer_frequency / CHRONIC_THRESHOLD_DAYS, 1.0)
        streak_ratio = min(transfer_streak    / STREAK_THRESHOLD,       1.0)
        score = 0.70 + 0.30 * max(freq_ratio, streak_ratio)
        return round(min(score, 1.0), 4), "CHRONIC"

    if transfer_frequency >= 5:
        score = 0.40 + 0.30 * (transfer_frequency - 5) / 5.0
        return round(min(score, 0.699), 4), "RECURRING"

    score = (transfer_frequency / 5.0) * 0.40 if transfer_frequency > 0 else 0.0
    return round(score, 4), "OCCASIONAL"


def _build_gap_records(
    raw: dict[tuple[str, str], dict],
    unit_costs: dict[str, float],
    reorder_points: dict[tuple[str, str], float],
    analysis_date: date,
) -> list[dict]:
    """Convert raw accumulator into stocking_gaps rows ready for upsert."""
    records: list[dict] = []

    for (sku_id, location_id), entry in raw.items():
        dates          = entry["dates"]
        qtys           = entry["qtys"]
        last30_dates   = entry["last30_dates"]
        prior30_dates  = entry["prior30_dates"]
        from_loc_counts = entry["from_loc_counts"]

        transfer_frequency  = len(dates)
        transfer_streak     = _compute_streak(dates)
        avg_qty             = sum(qtys) / len(qtys) if qtys else 0.0
        trend               = _compute_trend(len(last30_dates), len(prior30_dates))
        gap_score, gap_cls  = _gap_score_and_class(transfer_frequency, transfer_streak)

        # Most common source location for this transfer pair
        transfer_from_location = (
            max(from_loc_counts, key=lambda k: from_loc_counts[k])
            if from_loc_counts else None
        )

        unit_cost = unit_costs.get(sku_id, 0.0)
        total_transfer_value = round(avg_qty * unit_cost * transfer_frequency, 2)

        # CHRONIC-only derived fields
        suggested_stock_increase: float | None = None
        annual_cost_savings:      float | None = None
        if gap_cls == "CHRONIC":
            suggested_stock_increase = round(
                avg_qty * CHRONIC_GAP_MULTIPLIER * SAFETY_FACTOR, 2
            )
            annual_freq = transfer_frequency / LOOKBACK_DAYS * 365
            annual_cost_savings = round(
                annual_freq * avg_qty * TRANSFER_HANDLING_COST, 2
            )

        current_reorder_point = reorder_points.get((sku_id, location_id))

        records.append({
            "analysis_date":           analysis_date.isoformat(),
            "sku_id":                  sku_id,
            "location_id":             location_id,
            "location_name":           LOCATION_NAMES.get(location_id, ""),
            "transfer_from_location":  transfer_from_location,
            "transfer_frequency":      transfer_frequency,
            "transfer_streak":         transfer_streak,
            "avg_qty_recommended":     round(avg_qty, 4),
            "total_transfer_value":    total_transfer_value,
            "gap_score":               gap_score,
            "gap_classification":      gap_cls,
            "suggested_stock_increase": suggested_stock_increase,
            "current_reorder_point":   (
                round(current_reorder_point, 2)
                if current_reorder_point is not None else None
            ),
            "annual_cost_savings":     annual_cost_savings,
            "trend_direction":         trend,
        })

    return records


# ---------------------------------------------------------------------------
# Fetch helpers — unit costs and reorder points
# ---------------------------------------------------------------------------

def _fetch_unit_costs(
    client_holder: list,
    sku_ids: list[str],
) -> dict[str, float]:
    """Return unit_cost per sku_id, batched via in_() to avoid 57014."""
    costs: dict[str, float] = {}
    n_batches = math.ceil(len(sku_ids) / SKU_BATCH_SIZE)
    processed = 0

    for i in range(n_batches):
        batch = sku_ids[i * SKU_BATCH_SIZE:(i + 1) * SKU_BATCH_SIZE]
        # Prefer inventory_snapshots.unit_cost, fall back to sku_master
        rows = _paginate(
            client_holder,
            "sku_master",
            "sku_id,unit_cost",
            in_filters={"sku_id": batch},
        )
        for r in rows:
            sid = r.get("sku_id")
            uc  = r.get("unit_cost")
            if sid and uc is not None and sid not in costs:
                try:
                    cf = float(uc)
                    if cf > 0:
                        costs[sid] = cf
                except (TypeError, ValueError):
                    pass
        processed += len(batch)
        if processed % PROGRESS_LOG_EVERY < SKU_BATCH_SIZE or processed == len(sku_ids):
            log.info("  [STOCKING] Unit cost fetch: %d / %d SKUs", processed, len(sku_ids))

    return costs


def _fetch_reorder_points(
    client_holder: list,
    sku_ids: list[str],
    lookback_days: int = 7,
) -> dict[tuple[str, str], float]:
    """Return latest reorder_point per (sku_id, location_id), batched."""
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    latest: dict[tuple[str, str], dict] = {}
    n_batches = math.ceil(len(sku_ids) / SKU_BATCH_SIZE)

    for i in range(n_batches):
        batch = sku_ids[i * SKU_BATCH_SIZE:(i + 1) * SKU_BATCH_SIZE]
        rows = _paginate(
            client_holder,
            "inventory_snapshots",
            "sku_id,location_id,snapshot_date,reorder_point",
            gte_filters={"snapshot_date": cutoff},
            in_filters={"sku_id": batch},
            order_by=[("snapshot_date", True)],
        )
        for r in rows:
            key  = (r["sku_id"], r["location_id"])
            snap = r.get("snapshot_date", "")
            rp   = r.get("reorder_point")
            if rp is None:
                continue
            existing = latest.get(key)
            if existing is None or snap > existing.get("snapshot_date", ""):
                latest[key] = {"snapshot_date": snap, "reorder_point": float(rp)}

    return {k: v["reorder_point"] for k, v in latest.items()}


# ---------------------------------------------------------------------------
# OPSL cross-reference helpers
# ---------------------------------------------------------------------------

def _fetch_gp_baselines(client_holder: list, cutoff: date) -> dict[str, float]:
    """Call get_location_gp_baselines RPC; returns {location_id: avg_gp_pct}.

    Falls back to {} with a warning if the RPC is unavailable — callers use
    NORMAL_GP_PCT as the default in that case.
    """
    try:
        resp = client_holder[0].rpc(
            "get_location_gp_baselines",
            {"p_start_date": cutoff.isoformat()},
        ).execute()
        baselines: dict[str, float] = {}
        for row in resp.data or []:
            loc = row.get("location_id")
            pct = row.get("avg_gp_pct")
            if loc and pct is not None:
                try:
                    baselines[loc] = float(pct)
                except (TypeError, ValueError):
                    pass
        log.info("  GP baselines fetched: %d locations", len(baselines))
        return baselines
    except Exception:
        log.warning(
            "  get_location_gp_baselines RPC failed — using %.2f fallback.",
            NORMAL_GP_PCT,
        )
        return {}


def _fetch_opsl_flags(client_holder: list) -> set[tuple[str, str]]:
    """Return set of (prod_line_pn, location_id) for HIGH/MEDIUM opsl_flags rows.

    These are used to mark stocking gaps as double_confirmed when the same
    SKU+location pair shows up in both the transfer-pattern analysis and the
    OPSL outside-purchase analysis.
    """
    try:
        rows = _paginate(
            client_holder,
            "opsl_flags",
            "prod_line_pn,location_id,flag",
            in_filters={"flag": ["HIGH", "MEDIUM"]},
        )
        result: set[tuple[str, str]] = set()
        for r in rows:
            pn  = r.get("prod_line_pn")
            loc = r.get("location_id")
            if pn and loc:
                result.add((pn, loc))
        log.info(
            "  OPSL flags (HIGH/MEDIUM): %d (sku, location) pairs",
            len(result),
        )
        return result
    except Exception:
        log.warning("  opsl_flags fetch failed — skipping OPSL cross-reference.")
        return set()


def _fetch_opsl_savings(
    client_holder: list,
    cutoff: date,
    gp_baselines: dict[str, float],
) -> dict[tuple[str, str], float]:
    """Compute annualised margin-recovery savings from OPSL events in sales_detail_transactions.

    OPSL event: stock_flag='N', tran_code='SL' — a sale fulfilled as an outside
    purchase because the item wasn't in stock locally.

    For each (prod_line_pn, location_id) pair the margin loss per 90-day window is:
        sum(baseline_gp_pct × sales − gross_profit)
    Annualised by × 4 (90d × 4 ≈ 365d).

    Returns {(prod_line_pn, location_id): annualised_savings_dollars}.
    Only pairs where margin_loss > 0 are included.
    """
    try:
        rows = _paginate(
            client_holder,
            "sales_detail_transactions",
            "prod_line_pn,location_id,sales,gross_profit",
            filters={
                "tran_code":  "SL",
                "stock_flag": "N",
            },
            gte_filters={"tran_date": cutoff.isoformat()},
        )

        # Accumulate total_sales and total_gross_profit per (pn, loc)
        acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for r in rows:
            pn  = r.get("prod_line_pn")
            loc = r.get("location_id")
            if not pn or not loc:
                continue
            try:
                acc[(pn, loc)][0] += float(r.get("sales") or 0)
                acc[(pn, loc)][1] += float(r.get("gross_profit") or 0)
            except (TypeError, ValueError):
                pass

        savings: dict[tuple[str, str], float] = {}
        for (pn, loc), (total_sales, total_gp) in acc.items():
            if total_sales <= 0:
                continue
            normal_gp_pct  = gp_baselines.get(loc, NORMAL_GP_PCT)
            margin_loss_90d = normal_gp_pct * total_sales - total_gp
            if margin_loss_90d > 0:
                savings[(pn, loc)] = round(margin_loss_90d * 4, 2)

        log.info(
            "  OPSL savings data: %d (sku, location) pairs with recoverable margin",
            len(savings),
        )
        return savings
    except Exception:
        log.warning(
            "  sales_detail_transactions OPSL savings fetch failed — "
            "falling back to transfer estimates for all gaps."
        )
        return {}


def _enrich_with_opsl(
    records: list[dict],
    opsl_flags: set[tuple[str, str]],
    opsl_savings: dict[tuple[str, str], float],
) -> None:
    """Mutate records in-place: add double_confirmed, confidence, savings_source.

    Applies to all classification tiers.  OPSL_ACTUAL savings only replace the
    transfer-cost estimate for CHRONIC gaps (the only tier that had savings before).
    """
    for rec in records:
        key = (rec["sku_id"], rec["location_id"])

        rec["double_confirmed"] = key in opsl_flags
        rec["confidence"]       = "HIGH" if rec["double_confirmed"] else "MEDIUM"

        if rec["gap_classification"] == "CHRONIC":
            if key in opsl_savings:
                rec["annual_cost_savings"] = opsl_savings[key]
                rec["savings_source"]      = "OPSL_ACTUAL"
            else:
                rec["savings_source"] = "TRANSFER_ESTIMATE"
        else:
            rec["savings_source"] = None


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def _upsert_gaps(
    client_holder: list,
    rows: list[dict],
    dry_run: bool,
) -> int:
    """Upsert stocking_gaps rows in WRITE_BATCH_SIZE chunks.  Returns written count."""
    if dry_run:
        log.info("DRY RUN — would upsert %d rows to stocking_gaps.", len(rows))
        return 0

    written = 0
    total   = len(rows)
    next_log_at = WRITE_BATCH_SIZE * 5

    for offset in range(0, total, WRITE_BATCH_SIZE):
        batch = rows[offset:offset + WRITE_BATCH_SIZE]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = (
                    client_holder[0]
                    .table("stocking_gaps")
                    .upsert(
                        batch,
                        on_conflict="analysis_date,sku_id,location_id",
                    )
                    .execute()
                )
                written += len(resp.data or [])
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    log.warning(
                        "  stocking_gaps upsert retry %d/%d (offset=%d): %s — "
                        "reconnecting in %.0fs …",
                        attempt, _MAX_RETRIES, offset,
                        type(exc).__name__, _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _get_fresh_client()
                    continue
                log.exception(
                    "Failed to upsert stocking_gaps batch at offset %d (size=%d).",
                    offset, len(batch),
                )
                raise

        done = offset + len(batch)
        if done >= next_log_at or done == total:
            log.info("  [STOCKING] Upserted %d / %d rows (%.1f%%)",
                     done, total, 100.0 * done / total)
            next_log_at += WRITE_BATCH_SIZE * 5

    return written


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_stocking_intelligence(dry_run: bool = False) -> int:
    """Analyse transfer-pattern stocking gaps and write to stocking_gaps.

    Returns 0 on success, 1 on fatal error.
    """
    t0     = time.monotonic()
    today  = date.today()
    cutoff = today - timedelta(days=LOOKBACK_DAYS)
    trend_cutoff = today - timedelta(days=60)

    banner = "=" * 64
    log.info(banner)
    log.info("partswatch-ai — ml.stocking_intelligence")
    log.info(
        "  lookback=%dd  chronic_threshold=%d  streak_threshold=%d  "
        "handling_cost=$%.2f%s",
        LOOKBACK_DAYS, CHRONIC_THRESHOLD_DAYS, STREAK_THRESHOLD,
        TRANSFER_HANDLING_COST,
        "  [DRY RUN]" if dry_run else "",
    )
    log.info(banner)
    log.info("Analysis date: %s  (lookback since %s)",
             today.isoformat(), cutoff.isoformat())

    try:
        client_holder: list = [_get_fresh_client()]
    except Exception:
        log.exception("Failed to initialise Supabase client.")
        return 1

    # ── Step 1: fetch transfer recommendations ────────────────────────
    log.info("Step 1 — fetching transfer recommendations …")
    try:
        raw = _fetch_transfer_recs(client_holder, cutoff, trend_cutoff)
    except Exception:
        log.exception("Transfer fetch failed.")
        return 1

    if not raw:
        log.warning("No transfer recommendations found in last %d days. "
                    "Nothing to analyse.", LOOKBACK_DAYS)
        return 0

    all_skus = sorted({sku for sku, _ in raw})
    log.info("  Distinct SKUs with transfers: %d", len(all_skus))

    # ── Step 2: fetch unit costs and reorder points ───────────────────
    log.info("Step 2 — fetching unit costs (%d SKUs in batches of %d) …",
             len(all_skus), SKU_BATCH_SIZE)
    try:
        unit_costs = _fetch_unit_costs(client_holder, all_skus)
        log.info("  SKUs with unit costs: %d", len(unit_costs))
    except Exception:
        log.exception("Unit cost fetch failed (non-fatal — using 0).")
        unit_costs = {}

    log.info("Step 3 — fetching current reorder points …")
    try:
        reorder_points = _fetch_reorder_points(client_holder, all_skus)
        log.info("  (sku, location) pairs with reorder points: %d", len(reorder_points))
    except Exception:
        log.exception("Reorder-point fetch failed (non-fatal — using NULL).")
        reorder_points = {}

    # ── Step 3: compute gap records ───────────────────────────────────
    log.info("Step 4 — computing gap scores for %d (sku, location) pairs …",
             len(raw))
    try:
        records = _build_gap_records(raw, unit_costs, reorder_points, today)
    except Exception:
        log.exception("Gap score computation failed.")
        return 1

    chronic    = [r for r in records if r["gap_classification"] == "CHRONIC"]
    recurring  = [r for r in records if r["gap_classification"] == "RECURRING"]
    occasional = [r for r in records if r["gap_classification"] == "OCCASIONAL"]

    log.info("  CHRONIC:    %d pairs  (score > 0.7)", len(chronic))
    log.info("  RECURRING:  %d pairs  (0.4–0.7)", len(recurring))
    log.info("  OCCASIONAL: %d pairs  (< 0.4)", len(occasional))

    # ── Step 4: OPSL cross-reference and improved savings ─────────────
    log.info("Step 5 — OPSL cross-reference and savings enrichment …")
    try:
        gp_baselines = _fetch_gp_baselines(client_holder, cutoff)
        opsl_flags   = _fetch_opsl_flags(client_holder)
        opsl_savings = _fetch_opsl_savings(client_holder, cutoff, gp_baselines)
        _enrich_with_opsl(records, opsl_flags, opsl_savings)
    except Exception:
        log.exception("OPSL enrichment failed (non-fatal — proceeding without it).")
        for rec in records:
            rec.setdefault("double_confirmed", False)
            rec.setdefault("confidence", "MEDIUM")
            if rec["gap_classification"] == "CHRONIC":
                rec.setdefault("savings_source", "TRANSFER_ESTIMATE")
            else:
                rec.setdefault("savings_source", None)

    # Re-compute chronic list after enrichment (savings may have changed)
    chronic = [r for r in records if r["gap_classification"] == "CHRONIC"]

    double_confirmed = [r for r in records if r.get("double_confirmed")]
    opsl_actual      = [r for r in chronic  if r.get("savings_source") == "OPSL_ACTUAL"]

    total_savings = sum(
        r["annual_cost_savings"] for r in chronic
        if r.get("annual_cost_savings") is not None
    )

    log.info("  Double-confirmed gaps (transfer + OPSL): %d", len(double_confirmed))
    log.info("  CHRONIC savings source — OPSL_ACTUAL: %d  TRANSFER_ESTIMATE: %d",
             len(opsl_actual), len(chronic) - len(opsl_actual))
    log.info("  Revised total annual savings (CHRONIC): $%.0f", total_savings)

    # Top 10 double-confirmed chronic gaps by savings
    top_dc = sorted(
        [r for r in double_confirmed if r["gap_classification"] == "CHRONIC"],
        key=lambda r: r.get("annual_cost_savings") or 0,
        reverse=True,
    )[:10]

    if top_dc:
        log.info("-" * 64)
        log.info("Top double-confirmed CHRONIC gaps (by annual savings):")
        for i, r in enumerate(top_dc, 1):
            src = r.get("savings_source", "?")[:4]
            log.info(
                "  %2d. %-14s @ %-12s  freq=%2d  streak=%2d  "
                "avg_qty=%.1f  savings=$%.0f/yr  src=%s  trend=%s",
                i, r["sku_id"], r["location_id"],
                r["transfer_frequency"], r["transfer_streak"],
                r["avg_qty_recommended"],
                r.get("annual_cost_savings") or 0,
                src, r["trend_direction"],
            )
        log.info("-" * 64)
    elif chronic:
        # Fall back to top 10 chronic by savings (no double-confirmed this run)
        top_chronic = sorted(chronic,
                             key=lambda r: r.get("annual_cost_savings") or 0,
                             reverse=True)[:10]
        log.info("-" * 64)
        log.info("Top CHRONIC stocking gaps (by annual savings) — no double-confirmed:")
        for i, r in enumerate(top_chronic, 1):
            log.info(
                "  %2d. %-14s @ %-12s  freq=%2d  streak=%2d  "
                "avg_qty=%.1f  savings=$%.0f/yr  trend=%s",
                i, r["sku_id"], r["location_id"],
                r["transfer_frequency"], r["transfer_streak"],
                r["avg_qty_recommended"],
                r.get("annual_cost_savings") or 0,
                r["trend_direction"],
            )
        log.info("-" * 64)

    # ── Step 5: write to stocking_gaps ────────────────────────────────
    log.info("Step 6 — upserting %d rows to stocking_gaps …", len(records))
    try:
        written = _upsert_gaps(client_holder, records, dry_run)
    except Exception:
        log.exception("Stocking gaps write failed.")
        return 1

    elapsed = time.monotonic() - t0
    log.info(banner)
    log.info(
        "Stocking intelligence complete  (%.1fs)  "
        "pairs=%d  chronic=%d  recurring=%d  occasional=%d  "
        "double_confirmed=%d  written=%d%s",
        elapsed, len(records),
        len(chronic), len(recurring), len(occasional),
        len(double_confirmed),
        written,
        "  (DRY RUN)" if dry_run else "",
    )
    log.info("  Revised annual savings (CHRONIC): $%.0f", total_savings)
    log.info(banner)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transfer-pattern stocking gap analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Compute scores but do not write to stocking_gaps.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_stocking_intelligence(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
