"""
ml/dead_stock.py — Capital-weighted dead stock detection and liquidation ranking.

Identifies inventory that is tying up capital with little prospect of near-term
sale.  Unlike a simple "days since last sale" check, every SKU×location pair
is scored by the *dollar value* of the capital tied up, adjusted for how long
it has sat and how infrequently it sells.  The result is a ranked liquidation
priority list: the most expensive, slowest-moving stock appears first.

SCORING
    For every (sku_id, location_id) pair with inventory on hand:

    unit_cost           — most recent received-PO unit cost for the SKU
                          (falls back to DEFAULT_UNIT_COST if no PO exists)
    total_inv_value     — qty_on_hand × unit_cost
    days_since_sale     — calendar days since the last transaction AT THIS
                          location (not the global sku_master value, which
                          would hide locations where the SKU never moves)
    sale_frequency      — distinct transaction dates in the last 365 days
                          at this location  (0 = never sold here in a year)
    dead_stock_score    — total_inv_value
                          × (days_since_sale / LOOKBACK_DAYS)
                          / (sale_frequency + 1)

    Higher score = more capital locked up in slower, older inventory.

CLASSIFICATION (per SKU×location)
    LIQUIDATE  — days_since_sale ≥ 180  AND  (sale_frequency ≤ LOW_FREQ OR
                 total_inv_value ≥ LIQUIDATE_MIN_VALUE)
    MARKDOWN   — 90 ≤ days_since_sale < 180
    MONITOR    — 60 ≤ days_since_sale < 90
    HEALTHY    — days_since_sale < 60  (or actively selling)

    Override to HEALTHY if sale_frequency ≥ HIGH_FREQ_OVERRIDE regardless
    of days — fast-movers that happen to have a temporary gap are not dead.

LIQUIDATION ACTIONS (LIQUIDATE candidates only)
    Return to vendor   — supplier identified in purchase_orders history
    Write off          — total_inv_value < WRITE_OFF_THRESHOLD  and no vendor
    Markdown           — abc_class A/B  or  avg_weekly_units > HIGH_VELOCITY
    Liquidate / delist — everything else

is_dead_stock FLAG
    sku_master.is_dead_stock is set TRUE for any SKU with at least one
    LIQUIDATE location.  It is cleared (FALSE) for SKUs that no longer
    qualify.  This feeds the nightly engine.alerts DEAD_STOCK generator.

WEEKLY REPORT
    A structured summary is printed to console and can optionally be written
    to a JSON file with --report-file <path>.

USAGE
    python -m ml.dead_stock               # full run
    python -m ml.dead_stock --dry-run     # score only, no DB writes
    python -m ml.dead_stock --report-file /tmp/dead_stock_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

setup_logging()
log     = get_logger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOKBACK_DAYS:          int   = 365    # window for sale_frequency count
LIQUIDATE_DAYS:         int   = 180    # kept for backward-compat; scoring uses urgency_score
MARKDOWN_DAYS:          int   = 90     # kept for backward-compat
MONITOR_DAYS:           int   = 60     # days_since_sale below this → HEALTHY regardless of score

LOW_FREQ_THRESHOLD:     int   = 2      # ≤ this many sales/year = low frequency
HIGH_FREQ_OVERRIDE:     int   = 12     # ≥ this many sales/year = always HEALTHY
LIQUIDATE_MIN_VALUE:    float = 50.0   # minimum inv value ($) to flag LIQUIDATE (legacy)
WRITE_OFF_THRESHOLD:    float = 25.0   # below this ($) with no vendor → write off
HIGH_VELOCITY_UNITS:    float = 1.0    # avg_weekly_units above which → markdown
DEFAULT_UNIT_COST:      float = 10.0   # fallback when no PO cost found

# Urgency-score thresholds (v2 scoring)
# urgency_score = (days_since_sale / 365) × log10(total_inv_value + 1)
URGENCY_SCORE_LIQUIDATE: float = 2.0   # score ≥ this → LIQUIDATE
URGENCY_SCORE_MARKDOWN:  float = 1.0   # score ≥ this → MARKDOWN; below → MONITOR

# Alert and conflict thresholds
MIN_ALERT_VALUE:         float = 500.0  # min total_inv_value ($) to emit DEAD_STOCK alert
SALES_DETAIL_LOOKBACK:   int   = 365    # days window for data_conflict check

PAGE_SIZE: int = 1_000

# A single sales_transactions query against the entire ~317 K SKU catalog
# blows past Supabase's statement timeout (error 57014) even with a 365-day
# cutoff (~2 M rows).  1000 SKUs per batch keeps each call well inside the
# timeout window — same value used by ml/forecast_rolling.py.
SKU_BATCH_SIZE: int = 1_000

# Cadence (in number of SKUs processed) for progress logging during the
# location-sales fetch.  Reported as: "Processed N / total SKUs".
PROGRESS_LOG_EVERY: int = 10_000

# ---------------------------------------------------------------------------
# Retry configuration for transient Supabase errors (57014 statement
# timeout, dropped HTTP/2 streams, read timeouts, …).  Mirrors the pattern
# used in engine/reorder.py and ml/forecast_rolling.py.
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = 5
_RETRY_DELAY: float = 5.0
_RETRYABLE_TOKENS: tuple[str, ...] = (
    "57014",
    "statement timeout",
    "canceling statement",
    "ConnectionTerminated",
    "RemoteProtocolError",
    "ReadTimeout",
    "ReadError",
    "ProtocolError",
    "RemoteDisconnected",
    "Server disconnected",
    "Connection aborted",
)


def _is_retryable_error(exc: Exception) -> bool:
    """True if exc looks like a Supabase timeout / dropped-connection error."""
    blob = type(exc).__name__ + " " + str(exc)
    return any(tok in blob for tok in _RETRYABLE_TOKENS)


def _get_fresh_client() -> Any:
    """Return a brand-new Supabase client (bypasses lru_cache when available).

    Only ImportError / AttributeError fall back to the cached client — any
    other failure (auth, network, config) must surface so we don't paper
    over a real outage by reusing a known-bad cached connection.
    """
    try:
        from db.connection import get_new_client  # type: ignore[attr-defined]
        return get_new_client()
    except (ImportError, AttributeError):
        return get_client()

# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------

CLASS_LIQUIDATE = "LIQUIDATE"
CLASS_MARKDOWN  = "MARKDOWN"
CLASS_MONITOR   = "MONITOR"
CLASS_HEALTHY   = "HEALTHY"

ACTION_RETURN    = "Return to vendor"
ACTION_MARKDOWN  = "Markdown — price reduction"
ACTION_WRITEOFF  = "Write off"
ACTION_LIQUIDATE = "Liquidate / delist"

# Normalized codes persisted to dead_stock_recommendations.action.  The human
# strings above stay the canonical labels for log/console output, but the DB
# stores the codes so the dashboard, CSV export, and assistant context all
# branch on a stable vocabulary instead of doing fragile string-matching on
# free-form labels (architect-flagged inconsistency).
_ACTION_CODE: dict[str, str] = {
    ACTION_WRITEOFF:  "WRITEOFF",
    ACTION_RETURN:    "RETURN",
    ACTION_MARKDOWN:  "MARKDOWN",
    ACTION_LIQUIDATE: "LIQUIDATE",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InventoryPosition:
    """All data needed to score one (sku_id, location_id) pair."""
    sku_id:            str
    location_id:       str
    qty_on_hand:       float
    unit_cost:         float
    days_since_sale:   int          # days since last tx at this location
    sale_frequency:    int          # distinct sale days in last 365 days
    abc_class:         str          # A / B / C
    avg_weekly_units:  float        # from sku_master
    supplier_id:       str | None   # from latest purchase order
    part_category:     str
    sub_category:      str

    @property
    def total_inv_value(self) -> float:
        return self.qty_on_hand * self.unit_cost

    @property
    def dead_stock_score(self) -> float:
        """Capital-weighted staleness score.  Higher = worse."""
        return (
            self.total_inv_value
            * (self.days_since_sale / LOOKBACK_DAYS)
            / (self.sale_frequency + 1)
        )


@dataclass
class ScoredPosition:
    """Classification output for one (sku_id, location_id) pair."""
    sku_id:           str
    location_id:      str
    classification:   str
    action:           str
    dead_stock_score: float
    urgency_score:    float
    total_inv_value:  float
    qty_on_hand:      float
    unit_cost:        float
    days_since_sale:  int
    sale_frequency:   int
    abc_class:        str
    avg_weekly_units: float
    supplier_id:      str | None
    part_category:    str
    sub_category:     str
    # v2 enrichment fields (populated after initial scoring)
    action_type:                 str       = ""
    transfer_candidate_location: str|None  = None
    data_conflict:               bool      = False


# ---------------------------------------------------------------------------
# Supabase pagination helper
# ---------------------------------------------------------------------------

def _paginate(
    client_holder: list,
    table:  str,
    select: str,
    filters:     dict | None = None,
    gte_filters: dict | None = None,
    eq_bool:     dict | None = None,
    not_null_cols: list[str] | None = None,
    order_col:   str | None  = None,
    order_desc:  bool        = False,
) -> list[dict]:
    """Paginated fetch with retry + reconnect on transient Supabase errors.

    ``client_holder`` is a single-element list; the reference is replaced
    in-place when a 57014 / dropped-connection error forces a reconnect.

    ``not_null_cols`` filters out rows where any of the listed columns is
    NULL (server-side, via ``.not_.is_(col, "null")``).
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page: list[dict] | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                q = client_holder[0].table(table).select(select)
                for col, val in (filters  or {}).items():
                    q = q.eq(col, val)
                for col, val in (gte_filters or {}).items():
                    q = q.gte(col, val)
                for col, val in (eq_bool or {}).items():
                    q = q.eq(col, val)
                for col in (not_null_cols or []):
                    q = q.not_.is_(col, "null")
                if order_col:
                    q = q.order(order_col, desc=order_desc)
                page = q.range(offset, offset + PAGE_SIZE - 1).execute().data or []
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
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _fetch_latest_inventory(client_holder: list) -> dict[tuple[str, str], dict]:
    """Latest inventory snapshot per (sku_id, location_id)."""
    rows = _paginate(client_holder, "inventory_snapshots",
                     "sku_id,location_id,snapshot_date,qty_on_hand")
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["sku_id"], r["location_id"])
        if key not in latest or r["snapshot_date"] > latest[key]["snapshot_date"]:
            latest[key] = r
    return {k: v for k, v in latest.items() if (v.get("qty_on_hand") or 0) > 0}


def _fetch_sku_master(client_holder: list) -> dict[str, dict]:
    """sku_id → sku_master row."""
    rows = _paginate(client_holder, "sku_master",
                     "sku_id,abc_class,last_sale_date,avg_weekly_units,"
                     "is_dead_stock,part_category,sub_category")
    return {r["sku_id"]: r for r in rows}


def _fetch_unit_costs(client_holder: list) -> dict[str, float]:
    """sku_id → per-unit cost.

    Resolution order (first non-null wins):
      1. ``inventory_snapshots.unit_cost`` — the source of truth, written
         by extract/autocube_product_pull.py from the Product cube's
         [Measures].[Unit Cost] (with [Ext Cost On Hand] / qty fallback).
         For each SKU we keep the MOST RECENT non-null cost across all
         locations and snapshot dates.  This is preferred over
         sku_master.unit_cost because (a) it is updated every extract
         even if the post-extract sku_master propagation aborts, and
         (b) it captures cost variance across locations naturally.
      2. ``sku_master.unit_cost`` — denormalized convenience copy written
         after a successful inventory extract.  Used only if (1) returns
         nothing for a SKU.
      3. Most recent ``purchase_orders.unit_cost`` for received POs —
         fallback for SKUs with no on-hand inventory anywhere.
    Anything still missing falls through to ``DEFAULT_UNIT_COST`` at the
    scoring call site.

    All three lookups degrade gracefully if their column/table is missing
    (e.g. migration 021 not yet applied).
    """
    costs: dict[str, float] = {}

    def _is_missing_unit_cost_col(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "unit_cost" in msg
            and ("does not exist" in msg or "could not find" in msg
                 or "schema cache" in msg)
        )

    # ---- 1. inventory_snapshots.unit_cost (source of truth) -----------
    # Pull only the most recent snapshot_date — that's the current
    # replacement cost.  Scanning all 2M+ historical snapshot rows just
    # for cost lookup would be wasteful.  If the most recent snapshot
    # is missing some SKUs (e.g. extract crashed mid-way), step (2) and
    # (3) fill the gaps.
    snap_rows: list[dict[str, Any]] = []
    try:
        client = client_holder[0]
        latest = (
            client.table("inventory_snapshots")
            .select("snapshot_date")
            .order("snapshot_date", desc=True)
            .limit(1)
            .execute()
        )
        latest_date = latest.data[0]["snapshot_date"] if latest.data else None
        if latest_date:
            log.info(
                "  Pulling unit_cost from inventory_snapshots for "
                "snapshot_date=%s …", latest_date,
            )
            snap_rows = _paginate(
                client_holder, "inventory_snapshots",
                "sku_id,unit_cost",
                filters={"snapshot_date": latest_date},
                not_null_cols=["unit_cost"],
                # Deterministic order is required for offset-based
                # pagination to avoid duplicate/skipped rows across pages.
                order_col="sku_id",
            )
    except Exception as exc:
        if _is_missing_unit_cost_col(exc):
            log.warning(
                "inventory_snapshots.unit_cost missing — apply migration "
                "021 to use Product-cube costs.  Continuing with "
                "sku_master + purchase_orders only."
            )
        else:
            raise

    for r in snap_rows:
        sid = r.get("sku_id")
        c   = r.get("unit_cost")
        if not sid or c is None or sid in costs:
            continue
        try:
            cf = float(c)
        except (TypeError, ValueError):
            continue
        if cf > 0:
            costs[sid] = cf

    snap_hits = len(costs)

    # ---- 2. sku_master.unit_cost (denormalized, fills gaps) ----------
    try:
        master_rows = _paginate(
            client_holder, "sku_master", "sku_id,unit_cost",
        )
    except Exception as exc:
        if _is_missing_unit_cost_col(exc):
            master_rows = []
        else:
            raise

    for r in master_rows:
        sid = r.get("sku_id")
        c   = r.get("unit_cost")
        if not sid or sid in costs or c is None:
            continue
        try:
            cf = float(c)
        except (TypeError, ValueError):
            continue
        if cf > 0:
            costs[sid] = cf

    master_hits = len(costs) - snap_hits

    # ---- 3. purchase_orders fallback ---------------------------------
    po_rows = _paginate(
        client_holder, "purchase_orders",
        "sku_id,unit_cost,actual_delivery_date",
        filters={"status": "received"},
        order_col="actual_delivery_date", order_desc=True,
    )
    for r in po_rows:
        sid = r.get("sku_id")
        if not sid or sid in costs or r.get("unit_cost") is None:
            continue
        try:
            costs[sid] = float(r["unit_cost"])
        except (TypeError, ValueError):
            continue

    log.info(
        "  Unit cost sources: inventory_snapshots=%d  +  sku_master=%d  "
        "+  purchase_orders=%d  →  total=%d SKUs",
        snap_hits, master_hits, len(costs) - snap_hits - master_hits,
        len(costs),
    )
    return costs


def _fetch_supplier_map(client_holder: list) -> dict[str, str]:
    """sku_id → supplier_id from the most recent received PO."""
    rows = _paginate(client_holder, "purchase_orders",
                     "sku_id,supplier_id,actual_delivery_date",
                     filters={"status": "received"},
                     order_col="actual_delivery_date", order_desc=True)
    suppliers: dict[str, str] = {}
    for r in rows:
        sid = r.get("sku_id")
        sup = r.get("supplier_id")
        if sid and sup and sid not in suppliers:
            suppliers[sid] = sup
    return suppliers


def _fetch_transactions_for_sku_batch(
    client_holder: list,
    cutoff:        str,
    sku_batch:     list[str],
) -> list[dict]:
    """Fetch sales_transactions for one SKU batch with retry + reconnect.

    Restricting each query to ``len(sku_batch) ≤ SKU_BATCH_SIZE`` keeps the
    call inside Supabase's statement timeout (error 57014) — the same
    pattern used by ml/forecast_rolling.py.  Each page is retried on
    transient errors, with a fresh client minted on repeat failures.
    """
    select = "sku_id,location_id,transaction_date"
    rows: list[dict] = []
    offset = 0
    while True:
        page: list[dict] | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # Exclude warranty replacements — they're not real demand and
                # would skew "last sold" toward a recent date for SKUs that
                # haven't actually sold (causing missed dead-stock detection).
                page = (
                    client_holder[0].table("sales_transactions")
                    .select(select)
                    .in_("sku_id", sku_batch)
                    .gte("transaction_date", cutoff)
                    .eq("is_warranty", False)
                    .range(offset, offset + PAGE_SIZE - 1)
                    .execute()
                    .data
                    or []
                )
                break
            except Exception as exc:
                if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                    log.warning(
                        "  tx fetch retry %d/%d (offset=%d, %d SKUs): %s — "
                        "reconnecting in %.0fs …",
                        attempt, _MAX_RETRIES, offset, len(sku_batch),
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


def _fetch_location_sales(
    client_holder: list,
    today:         date,
    sku_ids:       list[str],
) -> tuple[dict[tuple[str, str], date], dict[tuple[str, str], int]]:
    """Per-(sku_id, location_id): last sale date and 365-day sale frequency.

    Fetches sales_transactions in SKU-id batches of ``SKU_BATCH_SIZE`` rather
    than as a single all-catalog scan.  The whole-catalog scan — even with a
    365-day cutoff — produces ~2 M rows and exceeds Supabase's statement
    timeout (error 57014).  Per-batch fetching is the same pattern that
    fixed forecast_rolling, forecast_lgbm, and reorder.

    Each batch's rows are folded into the accumulator maps immediately and
    then released, so peak memory is bounded by one batch's transactions
    rather than the full ~2 M result set.

    Returns
    -------
    last_sale_map  : (sku_id, location_id) → most recent transaction_date
    frequency_map  : (sku_id, location_id) → count of distinct sale days
    """
    cutoff = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    total  = len(sku_ids)

    if total == 0:
        log.warning("  _fetch_location_sales received empty sku_ids list.")
        return {}, {}

    n_batches = (total + SKU_BATCH_SIZE - 1) // SKU_BATCH_SIZE
    log.info(
        "  Fetching sales_transactions since %s (last %d days) "
        "in %d SKU batch(es) of up to %d …",
        cutoff, LOOKBACK_DAYS, n_batches, SKU_BATCH_SIZE,
    )

    last_sale_map: dict[tuple[str, str], date]      = {}
    freq_dates:    dict[tuple[str, str], set[str]]  = defaultdict(set)

    processed       = 0
    next_log_at     = PROGRESS_LOG_EVERY
    total_tx_rows   = 0

    for start in range(0, total, SKU_BATCH_SIZE):
        sku_batch = sku_ids[start:start + SKU_BATCH_SIZE]
        rows = _fetch_transactions_for_sku_batch(
            client_holder, cutoff, sku_batch,
        )
        total_tx_rows += len(rows)

        # Fold this batch's rows into the accumulator maps immediately so
        # we don't carry the entire ~2 M-row dataset in memory at once.
        for r in rows:
            sid = r.get("sku_id")
            lid = r.get("location_id")
            if not sid or not lid:
                continue
            d = str(r.get("transaction_date", ""))[:10]
            if not d:
                continue
            key = (sid, lid)
            freq_dates[key].add(d)
            try:
                tx_date = date.fromisoformat(d)
            except ValueError:
                continue
            if key not in last_sale_map or tx_date > last_sale_map[key]:
                last_sale_map[key] = tx_date

        processed += len(sku_batch)
        if processed >= next_log_at or processed >= total:
            log.info(
                "[DEAD_STOCK] Processed %d / %d SKUs  (tx rows so far: %d)",
                processed, total, total_tx_rows,
            )
            # Bump the next checkpoint past `processed` in PROGRESS_LOG_EVERY
            # increments so we don't spam at end-of-run.
            while next_log_at <= processed:
                next_log_at += PROGRESS_LOG_EVERY

    log.info(
        "  Done: %d transaction row(s) across %d SKU batch(es); "
        "%d (sku, location) pair(s) had at least one sale.",
        total_tx_rows, n_batches, len(freq_dates),
    )

    frequency_map = {k: len(v) for k, v in freq_dates.items()}
    return last_sale_map, frequency_map


# ---------------------------------------------------------------------------
# Scoring and classification
# ---------------------------------------------------------------------------

def _urgency_score(pos: InventoryPosition) -> float:
    """Combined value-age score: (days_idle/365) × log10(total_inv_value + 1).

    High days_idle × high dollar value → highest urgency regardless of the
    arbitrary LIQUIDATE_DAYS threshold.  Items with low value or short idle
    time score near zero.
    """
    return (pos.days_since_sale / 365.0) * math.log10(pos.total_inv_value + 1)


def _classify(pos: InventoryPosition) -> tuple[str, str, float]:
    """Return (classification, action, urgency_score) for one inventory position.

    Classification is driven by the urgency_score formula rather than fixed
    day-count thresholds.  Fast movers and recently-sold items remain HEALTHY.

    Thresholds:
        urgency_score >= URGENCY_SCORE_LIQUIDATE  → LIQUIDATE
        urgency_score >= URGENCY_SCORE_MARKDOWN   → MARKDOWN
        urgency_score <  URGENCY_SCORE_MARKDOWN   → MONITOR
        days_since_sale < MONITOR_DAYS            → HEALTHY (overrides score)
        sale_frequency  >= HIGH_FREQ_OVERRIDE     → HEALTHY (overrides score)
    """
    # Fast movers are always healthy regardless of idle gap
    if pos.sale_frequency >= HIGH_FREQ_OVERRIDE:
        return CLASS_HEALTHY, "", 0.0

    # Items sold recently are always healthy
    if pos.days_since_sale < MONITOR_DAYS:
        return CLASS_HEALTHY, "", 0.0

    score = _urgency_score(pos)

    if score >= URGENCY_SCORE_LIQUIDATE:
        # Determine recommended action for LIQUIDATE
        if pos.total_inv_value < WRITE_OFF_THRESHOLD:
            return CLASS_LIQUIDATE, ACTION_WRITEOFF, score
        if pos.supplier_id:
            return CLASS_LIQUIDATE, ACTION_RETURN, score
        if pos.avg_weekly_units >= HIGH_VELOCITY_UNITS or pos.abc_class in ("A", "B"):
            return CLASS_LIQUIDATE, ACTION_MARKDOWN, score
        return CLASS_LIQUIDATE, ACTION_LIQUIDATE, score

    if score >= URGENCY_SCORE_MARKDOWN:
        return CLASS_MARKDOWN, ACTION_MARKDOWN, score

    return CLASS_MONITOR, "", score


def _fetch_transfer_candidates(
    client_holder: list,
    dead_sku_ids:  list[str],
    today:         date,
) -> dict[str, str]:
    """Return sku_id → first needing location for dead-stock SKUs.

    Checks reorder_recommendations (today's date, qty_to_order > 0) then
    understocking_report.  Fetched in bulk before the main scoring loop so
    the logic runs in O(batches) not O(SKUs).
    """
    if not dead_sku_ids:
        return {}

    result: dict[str, str] = {}
    today_iso = today.isoformat()
    _BATCH = 200

    # ── reorder_recommendations ───────────────────────────────────────
    for start in range(0, len(dead_sku_ids), _BATCH):
        batch = dead_sku_ids[start:start + _BATCH]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                rows = (
                    client_holder[0].table("reorder_recommendations")
                    .select("sku_id,location_id")
                    .in_("sku_id", batch)
                    .eq("recommendation_date", today_iso)
                    .gt("qty_to_order", 0)
                    .execute()
                    .data or []
                )
                for r in rows:
                    sid, lid = r.get("sku_id"), r.get("location_id")
                    if sid and lid and sid not in result:
                        result[sid] = lid
                break
            except Exception as exc:
                if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _get_fresh_client()
                    continue
                log.warning(
                    "  reorder_recommendations transfer check failed (%s) — "
                    "skipping batch.", exc.__class__.__name__,
                )
                break

    # ── understocking_report (fills gaps not covered by reorder recs) ─
    still_needed = [s for s in dead_sku_ids if s not in result]
    for start in range(0, len(still_needed), _BATCH):
        batch = still_needed[start:start + _BATCH]
        try:
            rows = (
                client_holder[0].table("understocking_report")
                .select("sku_id,location_id")
                .in_("sku_id", batch)
                .execute()
                .data or []
            )
            for r in rows:
                sid, lid = r.get("sku_id"), r.get("location_id")
                if sid and lid and sid not in result:
                    result[sid] = lid
        except Exception as exc:
            log.warning(
                "  understocking_report transfer check failed (%s) — "
                "skipping batch.", exc.__class__.__name__,
            )

    log.info(
        "  Transfer candidates: %d dead-stock SKU(s) needed at another location.",
        len(result),
    )
    return result


def _fetch_sales_detail_active(
    client_holder: list,
    dead_sku_ids:  list[str],
    today:         date,
) -> set[str]:
    """Return set of sku_ids with any sales_detail_transactions in last 365 days.

    Uses prod_line_pn = sku_id.  A hit means the is_dead_stock flag may be
    stale — these items are tagged data_conflict=TRUE and excluded from alerts
    and the liquidation total.  Fetched in bulk (not per-SKU).
    """
    if not dead_sku_ids:
        return set()

    cutoff = (today - timedelta(days=SALES_DETAIL_LOOKBACK)).isoformat()
    active: set[str] = set()
    _BATCH = 100  # smaller batch — prod_line_pn IN clause on a wide table

    for start in range(0, len(dead_sku_ids), _BATCH):
        batch = dead_sku_ids[start:start + _BATCH]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                rows = (
                    client_holder[0].table("sales_detail_transactions")
                    .select("prod_line_pn")
                    .in_("prod_line_pn", batch)
                    .gte("tran_date", cutoff)
                    .execute()
                    .data or []
                )
                active.update(r["prod_line_pn"] for r in rows if r.get("prod_line_pn"))
                break
            except Exception as exc:
                if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _get_fresh_client()
                    continue
                log.warning(
                    "  sales_detail_transactions conflict check failed (%s) — "
                    "skipping batch.", exc.__class__.__name__,
                )
                break

    log.info(
        "  Data conflicts: %d dead-stock SKU(s) have sales_detail activity "
        "in the last %d days.",
        len(active), SALES_DETAIL_LOOKBACK,
    )
    return active


def score_positions(
    inventory:    dict[tuple[str, str], dict],
    sku_master:   dict[str, dict],
    unit_costs:   dict[str, float],
    suppliers:    dict[str, str],
    last_sale_map: dict[tuple[str, str], date],
    freq_map:      dict[tuple[str, str], int],
    today:         date,
    transfer_map:  dict[str, str]   | None = None,
    conflict_skus: set[str]         | None = None,
) -> list[ScoredPosition]:
    """Build and score every inventory position.  Returns all positions."""
    _transfer = transfer_map  or {}
    _conflicts = conflict_skus or set()
    results: list[ScoredPosition] = []

    for (sku_id, loc_id), snap in inventory.items():
        sku = sku_master.get(sku_id, {})
        qty = float(snap.get("qty_on_hand") or 0)
        if qty <= 0:
            continue

        unit_cost = unit_costs.get(sku_id, DEFAULT_UNIT_COST)
        last_sale = last_sale_map.get((sku_id, loc_id))
        if last_sale is None:
            global_ls = sku.get("last_sale_date")
            last_sale = date.fromisoformat(global_ls) if global_ls else None

        days_since = (today - last_sale).days if last_sale else LOOKBACK_DAYS
        freq       = freq_map.get((sku_id, loc_id), 0)

        pos = InventoryPosition(
            sku_id           = sku_id,
            location_id      = loc_id,
            qty_on_hand      = qty,
            unit_cost        = unit_cost,
            days_since_sale  = days_since,
            sale_frequency   = freq,
            abc_class        = sku.get("abc_class") or "C",
            avg_weekly_units = float(sku.get("avg_weekly_units") or 0),
            supplier_id      = suppliers.get(sku_id),
            part_category    = sku.get("part_category") or "",
            sub_category     = sku.get("sub_category") or "",
        )
        cls, action, u_score = _classify(pos)

        # v2 enrichment: TRANSFER override and data_conflict flag
        xfer_loc   = _transfer.get(sku_id)
        is_conflict = sku_id in _conflicts
        if xfer_loc and cls in (CLASS_LIQUIDATE, CLASS_MARKDOWN):
            action_type = "TRANSFER"
        else:
            action_type = cls  # LIQUIDATE / MARKDOWN / MONITOR / HEALTHY

        results.append(ScoredPosition(
            sku_id           = sku_id,
            location_id      = loc_id,
            classification   = cls,
            action           = action,
            dead_stock_score = pos.dead_stock_score,
            urgency_score    = round(u_score, 4),
            total_inv_value  = pos.total_inv_value,
            qty_on_hand      = qty,
            unit_cost        = unit_cost,
            days_since_sale  = days_since,
            sale_frequency   = freq,
            abc_class        = pos.abc_class,
            avg_weekly_units = pos.avg_weekly_units,
            supplier_id      = pos.supplier_id,
            part_category    = pos.part_category,
            sub_category     = pos.sub_category,
            action_type                  = action_type,
            transfer_candidate_location  = xfer_loc,
            data_conflict                = is_conflict,
        ))

    # Ranked: highest urgency_score first (v2); ties broken by dead_stock_score
    results.sort(key=lambda r: (r.urgency_score, r.dead_stock_score), reverse=True)
    return results


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def _upsert_with_retry(
    client_holder: list,
    table: str,
    rows: list[dict],
    on_conflict: str,
) -> None:
    """Upsert rows with retry/reconnect on transient Supabase errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client_holder[0].table(table).upsert(
                rows, on_conflict=on_conflict,
            ).execute()
            return
        except Exception as exc:
            if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                log.warning(
                    "  upsert retry %d/%d (%d rows): %s — reconnecting in %.0fs …",
                    attempt, _MAX_RETRIES, len(rows),
                    type(exc).__name__, _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _get_fresh_client()
                continue
            raise


def _update_is_dead_stock(
    client_holder: list,
    scored:        list[ScoredPosition],
    dry_run:       bool,
) -> tuple[int, int]:
    """Set is_dead_stock = TRUE for LIQUIDATE SKUs; clear for HEALTHY SKUs.

    Returns (set_true_count, set_false_count).
    """
    liquidate_skus: set[str] = {
        s.sku_id for s in scored if s.classification == CLASS_LIQUIDATE
    }
    healthy_skus: set[str] = {
        s.sku_id for s in scored
        if s.classification == CLASS_HEALTHY
        and s.sku_id not in liquidate_skus
    }

    set_true  = [{"sku_id": s, "is_dead_stock": True}  for s in liquidate_skus]
    set_false = [{"sku_id": s, "is_dead_stock": False} for s in healthy_skus]

    if dry_run:
        log.info("DRY RUN — would set is_dead_stock=TRUE  for %d SKU(s)", len(set_true))
        log.info("DRY RUN — would set is_dead_stock=FALSE for %d SKU(s)", len(set_false))
        return len(set_true), len(set_false)

    PAGE = 100
    for batch in [set_true[i:i+PAGE] for i in range(0, len(set_true), PAGE)]:
        _upsert_with_retry(client_holder, "sku_master", batch, "sku_id")
    for batch in [set_false[i:i+PAGE] for i in range(0, len(set_false), PAGE)]:
        _upsert_with_retry(client_holder, "sku_master", batch, "sku_id")

    log.info("is_dead_stock flag updated: TRUE=%d  FALSE=%d", len(set_true), len(set_false))
    return len(set_true), len(set_false)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _cls_badge(cls: str) -> str:
    return {
        CLASS_LIQUIDATE: "[red]LIQUIDATE[/red]",
        CLASS_MARKDOWN:  "[yellow]MARKDOWN[/yellow]",
        CLASS_MONITOR:   "[cyan]MONITOR[/cyan]",
        CLASS_HEALTHY:   "[green]HEALTHY[/green]",
    }.get(cls, cls)


def _persist_recommendations(
    client_holder: list,
    scored: list[ScoredPosition],
    report_date: date,
    dry_run: bool,
) -> None:
    """Upsert all LIQUIDATE+MARKDOWN positions to dead_stock_recommendations.

    Idempotent: primary key (report_date, sku_id, location_id) means re-runs
    overwrite the same day's rows in place.  MONITOR/HEALTHY are intentionally
    skipped — only actionable rows are persisted.
    """
    actionable = [p for p in scored
                  if p.classification in (CLASS_LIQUIDATE, CLASS_MARKDOWN)]

    if dry_run:
        log.info("DRY RUN — would upsert %d row(s) to dead_stock_recommendations",
                 len(actionable))
        return

    # Idempotency: a same-day rerun where a SKU's classification CHANGES
    # (e.g. moved from LIQUIDATE → MONITOR after fresh sales) would otherwise
    # leave the stale row in place — upsert overwrites matching keys but
    # never deletes vanished ones.  Wipe the day's rows up front so the
    # write reflects exactly the current scoring run.  Same crash-safety
    # tradeoff as elsewhere: process death between delete and upsert leaves
    # the day empty until the next run; acceptable here because dead_stock
    # is read-only by the dashboard/assistant (no downstream pipeline
    # consumes this table) and the next nightly run self-heals.
    try:
        client_holder[0].table("dead_stock_recommendations") \
            .delete() \
            .eq("report_date", report_date.isoformat()) \
            .execute()
    except Exception:
        log.exception("Pre-upsert wipe of dead_stock_recommendations[%s] "
                      "failed — proceeding; stale rows may persist.",
                      report_date.isoformat())

    if not actionable:
        log.info("No LIQUIDATE/MARKDOWN positions to persist.")
        return

    rows = [{
        "report_date":                report_date.isoformat(),
        "sku_id":                     p.sku_id,
        "location_id":                p.location_id,
        "classification":             p.classification,
        "action":                     _ACTION_CODE.get(p.action, p.action),
        "dead_stock_score":           round(p.dead_stock_score, 2),
        "total_inv_value":            round(p.total_inv_value, 2),
        "qty_on_hand":                p.qty_on_hand,
        "unit_cost":                  round(p.unit_cost, 4) if p.unit_cost else None,
        "days_since_sale":            p.days_since_sale,
        "sale_frequency":             round(p.sale_frequency, 2),
        "abc_class":                  p.abc_class,
        "supplier_id":                p.supplier_id,
        "part_category":              p.part_category or None,
        "sub_category":               p.sub_category or None,
        # v2 columns (migration 048) — stripped gracefully if not yet applied
        "action_type":                p.action_type or None,
        "transfer_candidate_location": p.transfer_candidate_location,
        "data_conflict":              p.data_conflict,
    } for p in actionable]

    BATCH = 500
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        try:
            _upsert_with_retry(
                client_holder,
                "dead_stock_recommendations",
                chunk,
                on_conflict="report_date,sku_id,location_id",
            )
        except Exception as exc:
            # Schema fallback: strip v2 columns if migration 048 not applied.
            if not _v2_stripped and any(c in str(exc) for c in _v2_cols):
                log.warning(
                    "migration 048 columns missing — stripping action_type, "
                    "transfer_candidate_location, data_conflict and retrying. "
                    "Apply 048_dead_stock_v2.sql to enable v2 features."
                )
                _v2_stripped = True
                rows = [{k: v for k, v in r.items() if k not in _v2_cols}
                        for r in rows]
                chunk = rows[i : i + BATCH]
                _upsert_with_retry(
                    client_holder, "dead_stock_recommendations",
                    chunk, on_conflict="report_date,sku_id,location_id",
                )
            else:
                raise

    log.info("Persisted %d row(s) to dead_stock_recommendations for %s.",
             len(rows), report_date.isoformat())


def _render_liquidation_table(positions: list[ScoredPosition], top_n: int = 20) -> None:
    """Print the ranked liquidation priority table to console."""
    candidates = [p for p in positions if p.classification in (CLASS_LIQUIDATE, CLASS_MARKDOWN)]
    if not candidates:
        console.print(Panel("[green]No liquidation candidates found.[/green]",
                            title="Liquidation Priority", border_style="green"))
        return

    tbl = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    tbl.add_column("#",            width=3,  style="dim")
    tbl.add_column("SKU",          width=14)
    tbl.add_column("Location",     width=8)
    tbl.add_column("Class",        width=11)
    tbl.add_column("Inv Value",    width=10, justify="right")
    tbl.add_column("Qty",          width=6,  justify="right")
    tbl.add_column("Unit Cost",    width=9,  justify="right")
    tbl.add_column("Days Idle",    width=9,  justify="right")
    tbl.add_column("Sales/yr",     width=8,  justify="right")
    tbl.add_column("Score",        width=8,  justify="right")
    tbl.add_column("Action",       width=22)

    for i, p in enumerate(candidates[:top_n], 1):
        tbl.add_row(
            str(i),
            p.sku_id,
            p.location_id,
            _cls_badge(p.classification),
            f"${p.total_inv_value:,.2f}",
            f"{int(p.qty_on_hand)}",
            f"${p.unit_cost:.2f}",
            str(p.days_since_sale),
            str(p.sale_frequency),
            f"{p.dead_stock_score:.1f}",
            p.action or "—",
        )

    title = f"Liquidation Priority  ({len(candidates)} candidate(s)"
    if len(candidates) > top_n:
        title += f", showing top {top_n}"
    title += ")"
    console.print(Panel(tbl, title=title, border_style="red"))


def _build_report(
    scored:     list[ScoredPosition],
    today:      date,
    elapsed_s:  float,
    set_true:   int,
    set_false:  int,
    dry_run:    bool,
) -> dict:
    """Build the structured weekly performance report dict."""
    by_class: dict[str, list[ScoredPosition]] = defaultdict(list)
    for p in scored:
        by_class[p.classification].append(p)

    liquidate = by_class[CLASS_LIQUIDATE]
    markdown  = by_class[CLASS_MARKDOWN]
    monitor   = by_class[CLASS_MONITOR]

    # v2 stats: conflict and transfer enrichment
    transfer_count  = sum(1 for p in liquidate + markdown if p.action_type == "TRANSFER")
    conflict_count  = sum(1 for p in liquidate + markdown if p.data_conflict)
    clean_positions = [p for p in liquidate + markdown if not p.data_conflict]
    alert_eligible  = [p for p in clean_positions if p.total_inv_value >= MIN_ALERT_VALUE]

    total_at_risk       = sum(p.total_inv_value for p in liquidate + markdown)
    clean_at_risk       = sum(p.total_inv_value for p in clean_positions)
    liquidate_val       = sum(p.total_inv_value for p in liquidate)
    markdown_val        = sum(p.total_inv_value for p in markdown)

    action_counts: dict[str, int] = defaultdict(int)
    for p in liquidate:
        action_counts[p.action] += 1

    return {
        "report_date":  today.isoformat(),
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "dry_run":      dry_run,
        "elapsed_s":    round(elapsed_s, 2),
        "summary": {
            "total_positions_scored":  len(scored),
            "liquidate_count":         len(liquidate),
            "markdown_count":          len(markdown),
            "monitor_count":           len(monitor),
            "healthy_count":           len(by_class[CLASS_HEALTHY]),
            "capital_at_risk":         round(total_at_risk, 2),
            "clean_capital_at_risk":   round(clean_at_risk, 2),
            "liquidate_value":         round(liquidate_val, 2),
            "markdown_value":          round(markdown_val, 2),
            "transfer_count":          transfer_count,
            "data_conflict_count":     conflict_count,
            "alert_eligible_count":    len(alert_eligible),
            "min_alert_value":         MIN_ALERT_VALUE,
            "is_dead_stock_set":       set_true,
            "is_dead_stock_cleared":   set_false,
        },
        "action_breakdown": dict(action_counts),
        "top_liquidate": [
            {
                "sku_id":                    p.sku_id,
                "location_id":               p.location_id,
                "total_inv_value":           round(p.total_inv_value, 2),
                "urgency_score":             round(p.urgency_score, 3),
                "days_since_sale":           p.days_since_sale,
                "sale_frequency":            p.sale_frequency,
                "action":                    p.action,
                "action_type":               p.action_type,
                "transfer_candidate_location": p.transfer_candidate_location,
                "data_conflict":             p.data_conflict,
            }
            for p in sorted(liquidate, key=lambda x: x.urgency_score, reverse=True)[:20]
        ],
    }


def _render_summary_panel(report: dict, dry_run: bool) -> None:
    s = report["summary"]
    dry_tag = "  [yellow]DRY RUN[/yellow]" if dry_run else ""

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("Metric",  style="bold", width=32)
    tbl.add_column("Value",   justify="right", width=16)

    rows = [
        ("Positions scored",           str(s["total_positions_scored"])),
        ("LIQUIDATE",                  f"[red]{s['liquidate_count']}[/red]"),
        ("MARKDOWN",                   f"[yellow]{s['markdown_count']}[/yellow]"),
        ("MONITOR",                    f"[cyan]{s['monitor_count']}[/cyan]"),
        ("HEALTHY",                    f"[green]{s['healthy_count']}[/green]"),
        ("",                           ""),
        ("Capital at risk (total)",    f"[red]${s['capital_at_risk']:,.2f}[/red]"),
        ("  — Liquidate value",        f"${s['liquidate_value']:,.2f}"),
        ("  — Markdown value",         f"${s['markdown_value']:,.2f}"),
        ("Clean capital at risk",      f"[yellow]${s['clean_capital_at_risk']:,.2f}[/yellow]"),
        ("",                           ""),
        ("TRANSFER candidates",        f"[cyan]{s['transfer_count']}[/cyan]"),
        ("Data conflicts (excluded)",  f"[dim]{s['data_conflict_count']}[/dim]"),
        (f"Alert-eligible (>=${s['min_alert_value']:,.0f})",
                                       f"[bold]{s['alert_eligible_count']}[/bold]"),
        ("",                           ""),
        ("is_dead_stock set TRUE",     str(s["is_dead_stock_set"])),
        ("is_dead_stock cleared",      str(s["is_dead_stock_cleared"])),
    ]
    for label, val in rows:
        tbl.add_row(label, val)

    if report["action_breakdown"]:
        tbl.add_row("", "")
        for action, count in report["action_breakdown"].items():
            tbl.add_row(f"  {action}", str(count))

    console.print(Panel(tbl,
        title=f"Dead Stock Summary — {report['report_date']}{dry_tag}",
        border_style="red" if s["liquidate_count"] > 0 else "green"))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_dead_stock(
    dry_run:     bool = False,
    report_file: str | None = None,
) -> int:
    """Execute the full capital-weighted dead stock detection pipeline.

    Args:
        dry_run:     When True, score and classify everything but skip all
                     DB writes and is_dead_stock flag updates.
        report_file: Optional file path to write the JSON performance report.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    t0  = time.monotonic()
    banner = "=" * 64
    log.info(banner)
    log.info("partswatch-ai — ml.dead_stock  (v2: urgency-score + transfer + conflict)")
    log.info(
        "  monitor_days=%d  urgency_liquidate=%.1f  urgency_markdown=%.1f  "
        "min_alert=$%.0f  write_off=$%.0f",
        MONITOR_DAYS, URGENCY_SCORE_LIQUIDATE, URGENCY_SCORE_MARKDOWN,
        MIN_ALERT_VALUE, WRITE_OFF_THRESHOLD,
    )
    log.info(banner)

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")

    today = date.today()
    log.info("Analysis date: %s", today.isoformat())

    # ── Connect ───────────────────────────────────────────────────────
    # Wrap the client in a single-element list so retry helpers can swap
    # in a freshly minted client when Supabase drops the connection.
    try:
        client_holder: list = [_get_fresh_client()]
    except Exception:
        log.exception("Failed to initialise Supabase client.")
        return 1

    # ── Fetch all data ────────────────────────────────────────────────
    log.info("Fetching inventory positions …")
    try:
        inventory    = _fetch_latest_inventory(client_holder)
        sku_master   = _fetch_sku_master(client_holder)
        unit_costs   = _fetch_unit_costs(client_holder)
        suppliers    = _fetch_supplier_map(client_holder)
        last_sale_map, freq_map = _fetch_location_sales(
            client_holder, today, sorted(sku_master.keys()),
        )
    except Exception:
        log.exception("Data fetch failed.")
        return 1

    log.info("  Inventory positions (on-hand > 0):  %d", len(inventory))
    log.info("  SKUs in master:                     %d", len(sku_master))
    log.info("  SKUs with known unit cost:          %d", len(unit_costs))
    log.info("  SKUs with known supplier:           %d", len(suppliers))
    log.info("  SKU x location pairs with sales data: %d", len(freq_map))

    if not inventory:
        log.warning("No inventory positions found — nothing to score.")
        return 0

    # ── Score and classify (v2: urgency score) ────────────────────────
    log.info("Scoring %d inventory position(s) …", len(inventory))
    # First pass: score without enrichment to identify dead-stock SKUs.
    scored = score_positions(
        inventory, sku_master, unit_costs, suppliers,
        last_sale_map, freq_map, today,
    )

    # ── v2: Bulk-fetch transfer candidates + data conflicts ───────────
    actionable_skus = list({
        p.sku_id for p in scored
        if p.classification in (CLASS_LIQUIDATE, CLASS_MARKDOWN)
    })
    log.info("Fetching v2 enrichment for %d actionable SKU(s) …", len(actionable_skus))
    try:
        transfer_map  = _fetch_transfer_candidates(client_holder, actionable_skus, today)
        conflict_skus = _fetch_sales_detail_active(client_holder, actionable_skus, today)
    except Exception:
        log.exception("v2 enrichment fetch failed (non-fatal) — continuing without enrichment.")
        transfer_map  = {}
        conflict_skus = set()

    # Second pass: re-score with enrichment (same data, adds action_type + flags).
    scored = score_positions(
        inventory, sku_master, unit_costs, suppliers,
        last_sale_map, freq_map, today,
        transfer_map=transfer_map,
        conflict_skus=conflict_skus,
    )

    by_class: dict[str, list[ScoredPosition]] = defaultdict(list)
    for p in scored:
        by_class[p.classification].append(p)

    log.info("Classification results (v2 urgency score):")
    for cls in (CLASS_LIQUIDATE, CLASS_MARKDOWN, CLASS_MONITOR, CLASS_HEALTHY):
        log.info("  %-12s  %d", cls, len(by_class[cls]))

    liquidate      = by_class[CLASS_LIQUIDATE]
    markdown       = by_class[CLASS_MARKDOWN]
    liquidate_value = sum(p.total_inv_value for p in liquidate)
    markdown_value  = sum(p.total_inv_value for p in markdown)
    transfer_count  = sum(1 for p in liquidate + markdown if p.action_type == "TRANSFER")
    conflict_count  = sum(1 for p in liquidate + markdown if p.data_conflict)
    clean_value     = sum(p.total_inv_value for p in liquidate + markdown if not p.data_conflict)
    alert_count     = sum(
        1 for p in {p.sku_id for p in liquidate + markdown
                    if not p.data_conflict and p.total_inv_value >= MIN_ALERT_VALUE}
    )

    log.info("Capital at risk:  LIQUIDATE $%.2f  MARKDOWN $%.2f", liquidate_value, markdown_value)
    log.info(
        "v2 enrichment:  TRANSFER=%d  data_conflict=%d  "
        "clean_at_risk=$%.2f  alert_eligible_skus=%d (>=$%.0f)",
        transfer_count, conflict_count, clean_value, alert_count, MIN_ALERT_VALUE,
    )

    # ── Log top candidates (by urgency score) ─────────────────────────
    liquidate_sorted = sorted(liquidate, key=lambda p: p.urgency_score, reverse=True)
    if liquidate_sorted:
        log.info("-" * 64)
        log.info("Top LIQUIDATE candidates (by urgency score):")
        for i, p in enumerate(liquidate_sorted[:10], 1):
            xfer = f" -> TRANSFER:{p.transfer_candidate_location}" if p.action_type == "TRANSFER" else ""
            conflict = " [CONFLICT]" if p.data_conflict else ""
            log.info(
                f"  {i:2d}. {p.sku_id:<14s} @{p.location_id:<8s}"
                f"  ${p.total_inv_value:>9,.2f}  {p.days_since_sale:4d}d"
                f"  score={p.urgency_score:.3f}{xfer}{conflict}"
            )
        log.info("-" * 64)

    # ── Rich console table ────────────────────────────────────────────
    _render_liquidation_table(scored)

    # ── Update is_dead_stock flag ─────────────────────────────────────
    try:
        set_true, set_false = _update_is_dead_stock(client_holder, scored, dry_run)
    except Exception:
        log.exception("is_dead_stock update failed (non-fatal).")
        set_true = set_false = 0

    # ── Persist LIQUIDATE+MARKDOWN positions to DB ────────────────────
    # Powers the dashboard "Dead Stock" panel and the assistant's
    # dead-stock context section.  MONITOR/HEALTHY are skipped to keep
    # the table size proportional to the actionable workload (~175k rows
    # vs ~3M).  Per-day grain (report_date) so historical trending is
    # possible without overwriting prior reports.
    try:
        _persist_recommendations(client_holder, scored, today, dry_run)
    except Exception:
        log.exception("dead_stock_recommendations persistence failed (non-fatal).")

    # ── Build and render summary ──────────────────────────────────────
    elapsed_s = time.monotonic() - t0
    report    = _build_report(scored, today, elapsed_s, set_true, set_false, dry_run)

    _render_summary_panel(report, dry_run)

    # ── Optional JSON report file ─────────────────────────────────────
    if report_file:
        try:
            with open(report_file, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=str)
            log.info("Weekly performance report written → %s", report_file)
        except Exception:
            log.exception("Failed to write report file (non-fatal).")

    log.info(banner)
    log.info(
        "Dead stock analysis complete  (%.1fs)  "
        "LIQUIDATE=%d  MARKDOWN=%d  MONITOR=%d  HEALTHY=%d",
        elapsed_s,
        len(by_class[CLASS_LIQUIDATE]),
        len(by_class[CLASS_MARKDOWN]),
        len(by_class[CLASS_MONITOR]),
        len(by_class[CLASS_HEALTHY]),
    )
    log.info(banner)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capital-weighted dead stock detection and liquidation ranking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Score and classify without writing to the database.")
    p.add_argument("--report-file", metavar="PATH",
                   help="Write the JSON performance report to this file path.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_dead_stock(dry_run=args.dry_run, report_file=args.report_file)


if __name__ == "__main__":
    sys.exit(main())
