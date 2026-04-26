"""engine/understocking.py — Chronic Understocking Report.

Answers the owner's question:
    "What should each store be stocking more of instead of waiting
     for transfers?"

A SKU is "chronically understocked" at a location when its on-hand
quantity sat below the reorder point for >=30% of observed snapshot
days in the lookback window (target 90 days; the engine adapts to
whatever snapshot history actually exists).

For every active retail location (excluding retired branches, the
LOC-021 INTERNET virtual location, and the LOC-025 MAIN DC) the top
20 chronic SKUs are persisted, ranked by:

    priority_score = stockout_days_pct × avg_daily_demand × unit_cost

Companion display metric (what's at stake if we run out for the next
lead-time + buffer window):

    inventory_value_at_risk = avg_daily_demand × unit_cost × 21

ALGORITHM (paginated Python — matches ml/dead_stock.py pattern)
    1. Page through inventory_snapshots with server-side filters:
         snapshot_date >= cutoff, location_id NOT IN excluded,
         reorder_point > 0.  Accumulate per (sku, location) the
         total_days, days_below_reorder, latest reorder_point and
         latest unit_cost.  (~2M rows × 27 days history today; reads
         in batches of 1000.)
    2. Filter to (days_below_reorder / total_days) >= 30%.
    3. Bulk-fetch sku_master rows for those SKUs (in_() chunks of 200)
       to pull avg_weekly_units, description, and a unit_cost fallback
       when the snapshot's was NULL.
    4. Bulk-fetch reorder_recommendations of type 'transfer' from the
       last 30 days (small set, <few thousand rows) and count by
       destination (location, sku) as a confirmation signal.
    5. Compute:
         avg_daily_demand        = avg_weekly_units / 7
         suggested_min_qty       = avg_daily × (LEAD_TIME + BUFFER)   (= 21d)
         inventory_value_at_risk = avg_daily × unit_cost × 21
         priority_score          = stockout_pct × avg_daily × unit_cost
    6. Keep the top 20 per location by priority_score.
    7. DELETE today's existing rows then bulk INSERT — same-day
       reruns are fully idempotent.
    8. Post-write integrity check: query back today's row count,
       assert it equals what we built.

Pipeline position: WEEKLY_STAGES (after dead_stock).

CLI:
    python -m engine.understocking [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Policy parameters
# ---------------------------------------------------------------------------

LOOKBACK_DAYS: int = 90
"""Target lookback window for snapshot aggregation."""

STOCKOUT_PCT_THRESHOLD: float = 0.30
"""SKU is chronically understocked if below reorder_point this fraction of days."""

LEAD_TIME_DAYS: float = 7.0
"""Default lead time.  Mirrors engine/reorder.py DEFAULT_LEAD_TIME_DAYS."""

BUFFER_DAYS: float = 14.0
"""Safety buffer added on top of lead time when computing suggested_min_qty."""

LEAD_BUFFER_DAYS: float = LEAD_TIME_DAYS + BUFFER_DAYS  # 21.0
"""Sum used for suggested_min_qty and inventory_value_at_risk."""

TOP_N_PER_LOCATION: int = 20
"""How many chronic SKUs to persist per location."""

TRANSFER_REC_LOOKBACK_DAYS: int = 30
"""Window for counting transfer-IN recommendations as confirmation signal."""

# Network-level buying trigger
NETWORK_THRESHOLD: int = 5
"""Minimum number of locations simultaneously understocked to fire a DC bulk buy."""

NETWORK_SHORT_PCT_MIN: float = 0.50
"""A location counts toward the network threshold when short_pct exceeds this."""

NETWORK_SAFETY_BUFFER: float = 1.50
"""qty_to_order for the DC rec = sum(location shortfalls) × this multiplier."""

DC_LOCATION_ID: str = "LOC-025"
"""Reorder recommendation destination for network-triggered bulk purchases."""

# GP fetch for financial_severity
GP_TRAN_CODES: list[str] = ["SL", "SL-I"]
GP_LOOKBACK_DAYS: int = 90   # same as LOOKBACK_DAYS, kept separate for clarity

# v2 schema columns — stripped from inserts when migration 050 is not applied yet.
_UNDERSTOCK_V2_COLS: frozenset[str] = frozenset(
    {"financial_severity", "avg_gp_pct", "network_flag", "double_confirmed"}
)

# Same set as dashboard/server.py.EXCLUDED_DISPLAY_LOCATIONS plus
# LOC-025 MAIN DC (warehouse, not a buyer-facing branch).
EXCLUDED_LOCATIONS: list[str] = [
    "LOC-014", "LOC-019", "LOC-022", "LOC-023", "LOC-030", "LOC-031",  # retired
    "LOC-021",  # INTERNET (virtual)
    "LOC-025",  # MAIN DC (not retail)
]

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

# ---------------------------------------------------------------------------
# Pagination + retry helpers — pattern lifted from ml/dead_stock.py.
# ---------------------------------------------------------------------------

PAGE_SIZE: int = 1_000
SKU_BATCH_SIZE: int = 1_000       # SKUs per inventory_snapshots batch query
SKU_LOOKUP_CHUNK: int = 200       # batch size for sku_master.in_(sku_ids)
WRITE_BATCH_SIZE: int = 500
PROGRESS_LOG_EVERY: int = 10_000  # log cadence (SKU count) during snapshot fetch
_MAX_RETRIES: int = 5
_RETRY_DELAY: float = 5.0

_RETRYABLE_TOKENS = (
    "57014",                      # statement_timeout
    "JSON could not be generated",
    "502", "504", "Bad Gateway", "Gateway Time-out",
    "RemoteDisconnected", "Server disconnected", "Connection aborted",
    "EOF occurred",
    "ReadError",                  # httpx/httpcore read failure
    "WinError 10054",             # Windows: remote host forcibly closed connection
)

_RETRYABLE_TYPES = (ConnectionError, OSError)  # OSError covers WinError 10054


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    blob = type(exc).__name__ + " " + str(exc)
    return any(tok in blob for tok in _RETRYABLE_TOKENS)


def _fresh_client() -> Any:
    """Brand-new Supabase client (bypasses lru_cache when available)."""
    try:
        from db.connection import get_new_client  # type: ignore[attr-defined]
        return get_new_client()
    except (ImportError, AttributeError):
        return get_client()


# ---------------------------------------------------------------------------
# Step 1 — SKU-batched fetch of inventory_snapshots, accumulate per (sku, loc).
# ---------------------------------------------------------------------------

@dataclass
class SnapStat:
    total_days:           int = 0
    days_below_reorder:   int = 0
    latest_snapshot_date: str = ""
    latest_reorder_point: float = 0.0
    latest_unit_cost:     float | None = None  # None if every snapshot was NULL


def _fetch_all_sku_ids(client_holder: list) -> list[str]:
    """Return all sku_id values from sku_master via offset pagination."""
    sku_ids: list[str] = []
    offset = 0
    while True:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                rows = (
                    client_holder[0]
                    .table("sku_master")
                    .select("sku_id")
                    .range(offset, offset + PAGE_SIZE - 1)
                    .execute()
                    .data or []
                )
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    log.warning(
                        "  sku_master fetch retry %d/%d (offset=%d): %s — "
                        "reconnecting in %.0fs …",
                        attempt, _MAX_RETRIES, offset,
                        type(exc).__name__, _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _fresh_client()
                    continue
                raise
        sku_ids.extend(r["sku_id"] for r in rows)
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return sku_ids


def _accumulate_snapshot_stats(
    client_holder: list, cutoff: date,
) -> dict[tuple[str, str], SnapStat]:
    """SKU-batched pass over inventory_snapshots — no OFFSET pagination.

    Fetches all SKU IDs from sku_master first, then queries
    inventory_snapshots in batches of SKU_BATCH_SIZE using
    .in_("sku_id", batch) + .gte("snapshot_date", cutoff).
    This avoids large-offset timeouts on the 1.4M-row table.

    Server-side filters applied per batch:
      * sku_id IN <batch>
      * snapshot_date >= cutoff
      * reorder_point > 0       (drops SKUs we don't actively manage)
      * location_id NOT IN excluded
    """
    log.info("  Fetching all SKU IDs from sku_master …")
    all_sku_ids = _fetch_all_sku_ids(client_holder)
    total_skus = len(all_sku_ids)
    log.info("  %d SKUs to process in batches of %d", total_skus, SKU_BATCH_SIZE)

    stats: dict[tuple[str, str], SnapStat] = defaultdict(SnapStat)
    processed = 0
    next_log_at = PROGRESS_LOG_EVERY
    t0 = time.monotonic()

    for batch_num, batch_start in enumerate(range(0, total_skus, SKU_BATCH_SIZE)):
        sku_batch = all_sku_ids[batch_start:batch_start + SKU_BATCH_SIZE]

        # Proactively reconnect every 50 batches to prevent Supabase from
        # dropping long-lived connections (WinError 10054).
        if batch_num > 0 and batch_num % 50 == 0:
            client_holder[0] = _fresh_client()

        # Brief pause between batches to reduce connection pressure.
        if batch_num > 0:
            time.sleep(0.1)

        # Inner pagination for this SKU batch (bounded by rows-per-1000-SKUs,
        # not the full table size — large offsets never occur).
        batch_offset = 0
        while True:
            page: list[dict] | None = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    page = (
                        client_holder[0]
                        .table("inventory_snapshots")
                        .select("sku_id,location_id,snapshot_date,"
                                "qty_on_hand,reorder_point,unit_cost")
                        .in_("sku_id", sku_batch)
                        .gte("snapshot_date", cutoff.isoformat())
                        .gt("reorder_point", 0)
                        .not_.in_("location_id", EXCLUDED_LOCATIONS)
                        .range(batch_offset, batch_offset + PAGE_SIZE - 1)
                        .execute()
                        .data or []
                    )
                    break
                except Exception as exc:
                    if _is_retryable(exc) and attempt < _MAX_RETRIES:
                        log.warning(
                            "  snapshots fetch retry %d/%d "
                            "(batch_start=%d, batch_offset=%d): %s — "
                            "reconnecting in %.0fs …",
                            attempt, _MAX_RETRIES, batch_start, batch_offset,
                            type(exc).__name__, _RETRY_DELAY,
                        )
                        time.sleep(_RETRY_DELAY)
                        client_holder[0] = _fresh_client()
                        continue
                    raise

            assert page is not None
            for r in page:
                key = (r["sku_id"], r["location_id"])
                s = stats[key]
                s.total_days += 1
                qoh = float(r.get("qty_on_hand") or 0)
                rop = float(r.get("reorder_point") or 0)
                if qoh < rop:
                    s.days_below_reorder += 1
                snap_date = r["snapshot_date"]
                if snap_date >= s.latest_snapshot_date:
                    s.latest_snapshot_date = snap_date
                    s.latest_reorder_point = rop
                    uc = r.get("unit_cost")
                    if uc is not None:
                        s.latest_unit_cost = float(uc)

            if len(page) < PAGE_SIZE:
                break
            batch_offset += PAGE_SIZE

        processed += len(sku_batch)
        if processed >= next_log_at or processed >= total_skus:
            log.info(
                "[UNDERSTOCKING] Processed %d / %d SKUs",
                processed, total_skus,
            )
            while next_log_at <= processed:
                next_log_at += PROGRESS_LOG_EVERY

    log.info(
        "  snapshots fetch done: %d (sku,loc) pairs, %.1fs",
        len(stats), time.monotonic() - t0,
    )
    return stats


# ---------------------------------------------------------------------------
# Step 3 — bulk-fetch sku_master for chronic SKUs.
# ---------------------------------------------------------------------------

def _fetch_sku_master(
    client_holder: list, sku_ids: list[str],
) -> dict[str, dict]:
    """Return {sku_id: {description, avg_weekly_units, unit_cost}}."""
    out: dict[str, dict] = {}
    for i in range(0, len(sku_ids), SKU_LOOKUP_CHUNK):
        chunk = sku_ids[i : i + SKU_LOOKUP_CHUNK]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                rows = (
                    client_holder[0]
                    .table("sku_master")
                    .select("sku_id,description,avg_weekly_units,unit_cost")
                    .in_("sku_id", chunk)
                    .execute()
                    .data or []
                )
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _fresh_client()
                    continue
                raise
        for r in rows:
            out[r["sku_id"]] = r
    return out


# ---------------------------------------------------------------------------
# Step 4 — bulk-fetch transfer recommendations (small).
# ---------------------------------------------------------------------------

def _fetch_transfer_counts(
    client_holder: list, since: date,
) -> dict[tuple[str, str], int]:
    """Count of transfer recs per (location_id, sku_id) since `since`."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    offset = 0
    while True:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                rows = (
                    client_holder[0]
                    .table("reorder_recommendations")
                    .select("location_id,sku_id")
                    .eq("recommendation_type", "transfer")
                    .gte("recommendation_date", since.isoformat())
                    # Deterministic ordering for offset pagination —
                    # without it `.range(offset, ...)` can skip or
                    # duplicate rows between pages, miscounting
                    # transfer recs per (location, sku).
                    .order("recommendation_date", desc=False)
                    .order("location_id",        desc=False)
                    .order("sku_id",             desc=False)
                    .range(offset, offset + PAGE_SIZE - 1)
                    .execute()
                    .data or []
                )
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _fresh_client()
                    continue
                raise
        for r in rows:
            counts[(r["location_id"], r["sku_id"])] += 1
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return counts


# ---------------------------------------------------------------------------
# Step 3a — avg GP per unit from sales_detail_transactions.
# ---------------------------------------------------------------------------

def _fetch_gp_per_unit(
    client_holder: list, sku_ids: list[str], cutoff: date,
) -> dict[str, float]:
    """Return {sku_id: avg_gp_per_unit} using prod_line_pn as the sku_id proxy.

    avg_gp_per_unit = sum(gross_profit) / sum(qty_ship) over the last
    GP_LOOKBACK_DAYS days, tran_code IN GP_TRAN_CODES, per prod_line_pn.
    Returns 0.0 for any sku_id with no matching sales detail rows.
    """
    gp_sum:  dict[str, float] = defaultdict(float)
    qty_sum: dict[str, float] = defaultdict(float)

    for i in range(0, max(len(sku_ids), 1), SKU_LOOKUP_CHUNK):
        chunk = sku_ids[i : i + SKU_LOOKUP_CHUNK]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                rows = (
                    client_holder[0]
                    .table("sales_detail_transactions")
                    .select("prod_line_pn,gross_profit,qty_ship")
                    .in_("prod_line_pn", chunk)
                    .in_("tran_code", GP_TRAN_CODES)
                    .gte("tran_date", cutoff.isoformat())
                    .execute().data or []
                )
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _fresh_client()
                    continue
                raise
        for r in rows:
            pn = r.get("prod_line_pn") or ""
            if not pn:
                continue
            gp_sum[pn]  += float(r.get("gross_profit") or 0)
            qty_sum[pn] += float(r.get("qty_ship") or 0)

    return {
        pn: (gp_sum[pn] / qty_sum[pn]) if qty_sum.get(pn, 0) > 0 else 0.0
        for pn in gp_sum
    }


# ---------------------------------------------------------------------------
# Step 4a — OPSL flags cross-reference.
# ---------------------------------------------------------------------------

def _fetch_opsl_flags(client_holder: list) -> set[tuple[str, str]]:
    """Return {(prod_line_pn, location_id)} for HIGH and MEDIUM OPSL flags."""
    out: set[tuple[str, str]] = set()
    offset = 0
    while True:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                rows = (
                    client_holder[0]
                    .table("opsl_flags")
                    .select("prod_line_pn,location_id")
                    .in_("flag", ["HIGH", "MEDIUM"])
                    .range(offset, offset + PAGE_SIZE - 1)
                    .execute().data or []
                )
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _fresh_client()
                    continue
                raise
        for r in rows:
            pn  = r.get("prod_line_pn") or ""
            loc = r.get("location_id") or ""
            if pn and loc:
                out.add((pn, loc))
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return out


# ---------------------------------------------------------------------------
# Network trigger — write DC bulk-purchase recs to reorder_recommendations.
# ---------------------------------------------------------------------------

def _write_network_recs(
    client_holder: list, today: date, triggers: list[dict],
) -> int:
    """Upsert one DC bulk-purchase rec per triggered SKU.  Returns count written."""
    if not triggers:
        return 0

    recs = [
        {
            "sku_id":              t["sku_id"],
            "location_id":         DC_LOCATION_ID,
            "recommendation_date": today.isoformat(),
            "qty_to_order":        round(t["qty_to_order"], 2),
            "recommendation_type": "po",
            "urgency":             "critical",
            "forecast_model_used": "NETWORK_UNDERSTOCK",   # VARCHAR(20) — 18 chars, fits
            "source":              "NETWORK_UNDERSTOCK",   # VARCHAR(20) until 050 widened to 50
            "notes":               t["notes"][:2000],
            "is_approved":         False,
        }
        for t in triggers
    ]

    _v2_stripped = False
    written = 0
    for i in range(0, len(recs), WRITE_BATCH_SIZE):
        batch = recs[i : i + WRITE_BATCH_SIZE]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                client_holder[0].table("reorder_recommendations").upsert(
                    batch,
                    on_conflict="sku_id,location_id,recommendation_date",
                ).execute()
                written += len(batch)
                break
            except Exception as exc:
                err = str(exc)
                if "23505" in err:
                    # Today's rec already exists — approval state preserved, count as ok.
                    written += len(batch)
                    break
                if not _v2_stripped and (
                    "source" in err or "notes" in err or "column" in err.lower()
                ):
                    _v2_stripped = True
                    log.warning(
                        "  network_recs: source/notes columns absent "
                        "(migration 050 not applied) — retrying without them."
                    )
                    for row in batch:
                        row.pop("source", None)
                        row.pop("notes", None)
                    # retry this batch without v2 cols (attempt loop continues)
                    continue
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _fresh_client()
                    continue
                log.error(
                    "  network_recs write failed (batch %d): %s — %s",
                    i, exc.__class__.__name__, err[:200],
                )
                break
    return written


# ---------------------------------------------------------------------------
# Step 7 — write today's rows (idempotent: delete-then-insert).
# ---------------------------------------------------------------------------

def _write_today(
    client_holder: list, today: date, rows: list[dict],
) -> int:
    """Delete today's existing report, bulk-insert new rows.  Returns count.

    Inserted rows leave ``run_completed_at`` NULL.  The caller MUST
    invoke ``_mark_run_complete`` after a successful insert pass —
    dashboard reads filter on ``run_completed_at IS NOT NULL`` so
    consumers never observe a partial / mid-write state.
    """
    # 1. Delete today's existing rows (covers stale completed rows AND
    #    any half-written NULL-marker rows from an earlier crashed run).
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            (
                client_holder[0]
                .table("understocking_report")
                .delete()
                .eq("report_date", today.isoformat())
                .execute()
            )
            break
        except Exception as exc:
            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _fresh_client()
                continue
            raise

    if not rows:
        return 0

    # 2. Insert in batches with run_completed_at left NULL.
    written = 0
    for i in range(0, len(rows), WRITE_BATCH_SIZE):
        batch = rows[i : i + WRITE_BATCH_SIZE]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                client_holder[0].table("understocking_report").insert(batch).execute()
                written += len(batch)
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _fresh_client()
                    continue
                raise
    return written


def _mark_run_complete(client_holder: list, today: date) -> None:
    """Stamp run_completed_at = NOW() on every row for `today`.

    This is the publication step — until it runs, the dashboard
    treats today's rows as not-yet-visible.  Wrapped with the same
    retry/reconnect helpers as every other write.
    """
    now_iso = datetime.utcnow().isoformat() + "Z"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            (
                client_holder[0]
                .table("understocking_report")
                .update({"run_completed_at": now_iso})
                .eq("report_date", today.isoformat())
                .execute()
            )
            return
        except Exception as exc:
            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _fresh_client()
                continue
            raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_understocking(dry_run: bool = False) -> int:
    """Compute and persist the Chronic Understocking Report.

    Returns 0 on success, 1 on failure.
    """
    t0 = time.monotonic()
    today = date.today()
    cutoff = today - timedelta(days=LOOKBACK_DAYS)
    gp_cutoff = today - timedelta(days=GP_LOOKBACK_DAYS)
    transfer_since = today - timedelta(days=TRANSFER_REC_LOOKBACK_DAYS)

    log.info("=" * 64)
    log.info("CHRONIC UNDERSTOCKING REPORT  (lookback %d days, "
             "threshold %.0f%%, top %d/loc, network >=%d locs)%s",
             LOOKBACK_DAYS, STOCKOUT_PCT_THRESHOLD * 100,
             TOP_N_PER_LOCATION, NETWORK_THRESHOLD,
             "  [DRY RUN]" if dry_run else "")
    log.info("=" * 64)

    client_holder: list = [_fresh_client()]

    # ---------- Step 1: snapshot accumulation ----------
    log.info("Step 1 — SKU-batched inventory_snapshots fetch since %s ...", cutoff)
    try:
        stats = _accumulate_snapshot_stats(client_holder, cutoff)
    except Exception:
        log.exception("Snapshot accumulation failed.")
        return 1

    if not stats:
        log.warning("No qualifying snapshots found.  Nothing to report.")
        return 0

    actual_window_days = max(
        (s.total_days for s in stats.values()), default=0,
    )
    log.info("  Max observed window across locations: %d days", actual_window_days)

    # ---------- Step 2: filter to chronic ----------
    chronic: list[tuple[tuple[str, str], SnapStat, float]] = []
    for key, s in stats.items():
        if s.total_days <= 0:
            continue
        pct = s.days_below_reorder / s.total_days
        if pct >= STOCKOUT_PCT_THRESHOLD:
            chronic.append((key, s, pct))

    log.info("Step 2 — chronic candidates (>=%.0f%% below reorder): %d "
             "(out of %d (sku,loc) pairs)",
             STOCKOUT_PCT_THRESHOLD * 100, len(chronic), len(stats))

    if not chronic:
        log.warning("No chronic-understocking SKUs found.")
        if not dry_run:
            _write_today(client_holder, today, [])
        return 0

    # ---------- Step 3: sku_master enrichment ----------
    sku_ids = sorted({k[0] for k, _, _ in chronic})
    log.info("Step 3 — fetching sku_master for %d SKUs (chunks of %d) ...",
             len(sku_ids), SKU_LOOKUP_CHUNK)
    sku_master = _fetch_sku_master(client_holder, sku_ids)
    log.info("  resolved %d/%d SKUs from sku_master", len(sku_master), len(sku_ids))

    # ---------- Step 3a: avg GP per unit from sales_detail_transactions ----------
    log.info("Step 3a — fetching avg GP/unit from sales_detail_transactions "
             "since %s (%d SKUs) ...", gp_cutoff, len(sku_ids))
    try:
        gp_map = _fetch_gp_per_unit(client_holder, sku_ids, gp_cutoff)
    except Exception:
        log.warning("  avg GP fetch failed — financial_severity will be 0 for all rows.",
                    exc_info=True)
        gp_map = {}
    gp_hits = sum(1 for v in gp_map.values() if v > 0)
    log.info("  avg GP/unit: %d SKUs with positive GP (of %d fetched)",
             gp_hits, len(gp_map))

    # ---------- Step 4: transfer-rec confirmation counts ----------
    log.info("Step 4 — fetching transfer recommendations since %s ...", transfer_since)
    transfer_counts = _fetch_transfer_counts(client_holder, transfer_since)
    log.info("  %d (location, sku) pairs have transfer recs in last %dd",
             len(transfer_counts), TRANSFER_REC_LOOKBACK_DAYS)

    # ---------- Step 4a: OPSL cross-reference ----------
    log.info("Step 4a — fetching OPSL flags (HIGH + MEDIUM) ...")
    try:
        opsl_set = _fetch_opsl_flags(client_holder)
    except Exception:
        log.warning("  OPSL fetch failed — double_confirmed will be FALSE for all rows.",
                    exc_info=True)
        opsl_set = set()
    log.info("  OPSL HIGH/MEDIUM flags: %d (sku, location) pairs", len(opsl_set))

    # ---------- Step 5: compute metrics ----------
    enriched: list[dict] = []
    for (sku_id, location_id), s, pct in chronic:
        sm = sku_master.get(sku_id) or {}
        avg_weekly = float(sm.get("avg_weekly_units") or 0.0)
        if avg_weekly <= 0:
            continue
        avg_daily = avg_weekly / 7.0

        unit_cost = s.latest_unit_cost
        if unit_cost is None:
            uc = sm.get("unit_cost")
            unit_cost = float(uc) if uc is not None else 0.0
        if unit_cost <= 0:
            continue

        suggested_min      = avg_daily * LEAD_BUFFER_DAYS
        value_at_risk      = avg_daily * unit_cost * LEAD_BUFFER_DAYS
        priority           = pct * avg_daily * unit_cost
        if priority <= 0:
            continue

        # GP-weighted severity (improvement 1)
        gp_per_unit        = gp_map.get(sku_id, 0.0)
        fin_severity       = round(pct * gp_per_unit * avg_daily, 2)

        # OPSL cross-reference (improvement 3) — use sku_id as prod_line_pn proxy
        is_double          = (sku_id, location_id) in opsl_set

        enriched.append({
            "report_date":                today.isoformat(),
            "location_id":                location_id,
            "location_name":              LOCATION_NAMES.get(location_id, ""),
            "sku_id":                     sku_id,
            "sku_description":            (sm.get("description") or "")[:500],
            "days_observed":              s.total_days,
            "days_below_reorder":         s.days_below_reorder,
            "stockout_days_pct":          round(pct, 4),
            "avg_daily_demand":           round(avg_daily, 4),
            "current_min_qty":            round(s.latest_reorder_point, 2),
            "suggested_min_qty":          round(suggested_min, 2),
            "min_qty_gap":                round(suggested_min - s.latest_reorder_point, 2),
            "unit_cost":                  round(unit_cost, 2),
            "inventory_value_at_risk":    round(value_at_risk, 2),
            "transfer_recommended_count": transfer_counts.get((location_id, sku_id), 0),
            "priority_score":             round(priority, 4),
            # v2 columns
            "financial_severity":         fin_severity,
            "avg_gp_pct":                 round(min(gp_per_unit, 99.9999), 4),
            "network_flag":               False,   # set below after trigger detection
            "double_confirmed":           is_double,
        })

    log.info("Step 5 — enriched rows with positive priority: %d", len(enriched))
    log.info("  double_confirmed (understocked + OPSL HIGH/MED): %d",
             sum(1 for r in enriched if r["double_confirmed"]))

    # ---------- Step 6a: network-level trigger detection (improvement 2) ----------
    log.info("Step 6a — network trigger detection "
             "(>=%d locs, short_pct > %.0f%%) ...",
             NETWORK_THRESHOLD, NETWORK_SHORT_PCT_MIN * 100)

    # Group all chronic rows (not just top-N) by sku_id where short_pct is severe
    sku_severe: dict[str, list[dict]] = defaultdict(list)
    for r in enriched:
        if r["stockout_days_pct"] > NETWORK_SHORT_PCT_MIN:
            sku_severe[r["sku_id"]].append(r)

    network_triggers: list[dict] = []
    network_sku_ids:  set[str]   = set()

    for sku_id, rows_for_sku in sku_severe.items():
        if len(rows_for_sku) < NETWORK_THRESHOLD:
            continue
        total_shortfall = sum(r["min_qty_gap"] for r in rows_for_sku)
        qty_to_order    = max(total_shortfall * NETWORK_SAFETY_BUFFER, 1.0)
        affected_parts  = sorted(rows_for_sku,
                                 key=lambda x: x["stockout_days_pct"], reverse=True)
        affected_str    = ", ".join(
            f"{r['location_id']} ({r['stockout_days_pct'] * 100:.0f}%)"
            for r in affected_parts
        )
        notes = f"Network understocking trigger: {affected_str}"
        network_triggers.append({
            "sku_id":        sku_id,
            "qty_to_order":  round(qty_to_order, 2),
            "notes":         notes,
        })
        network_sku_ids.add(sku_id)
        log.info(
            "  NETWORK TRIGGER: %-20s  %d locs  qty=%.0f  [%s]",
            sku_id, len(rows_for_sku), qty_to_order, affected_str[:100],
        )

    # Mark network_flag on all enriched rows for triggered SKUs
    for r in enriched:
        if r["sku_id"] in network_sku_ids:
            r["network_flag"] = True

    log.info("  Network triggers: %d SKUs", len(network_triggers))

    # ---------- Step 6: top N per location — sort double_confirmed first, ----------
    #            then financial_severity (improvements 1 & 3 combined sort)
    by_loc: dict[str, list[dict]] = defaultdict(list)
    for r in enriched:
        by_loc[r["location_id"]].append(r)

    final: list[dict] = []
    for loc, items in by_loc.items():
        items.sort(
            key=lambda r: (r["double_confirmed"], r["financial_severity"]),
            reverse=True,
        )
        final.extend(items[:TOP_N_PER_LOCATION])

    # ---------- Dry-run summary ----------
    if dry_run:
        log.info("-" * 64)
        log.info("DRY RUN SUMMARY")
        log.info("  Rows that would be persisted:   %d", len(final))
        log.info("  Network triggers:               %d SKUs", len(network_triggers))
        if network_triggers:
            for t in network_triggers:
                log.info("    %s — qty=%.0f", t["sku_id"], t["qty_to_order"])
        log.info("  Double-confirmed (understock+OPSL): %d",
                 sum(1 for r in final if r["double_confirmed"]))
        log.info("  Top 10 by financial_severity:")
        top10 = sorted(final, key=lambda r: r["financial_severity"], reverse=True)[:10]
        for r in top10:
            log.info(
                "    %-20s  %-10s  sev=$%8.2f  gp/unit=$%6.2f  "
                "short=%.0f%%  dc=%s  net=%s",
                r["sku_id"], r["location_id"],
                r["financial_severity"], r["avg_gp_pct"],
                r["stockout_days_pct"] * 100,
                "Y" if r["double_confirmed"] else "N",
                "Y" if r["network_flag"] else "N",
            )
        log.info("-" * 64)
        log.info("=" * 64)
        log.info("Understocking report complete  (%.1fs)", time.monotonic() - t0)
        log.info("=" * 64)
        return 0

    # Per-location summary (live run)
    log.info("-" * 64)
    log.info("Per-location summary (sorted by financial severity):")
    loc_sev:   dict[str, float] = defaultdict(float)
    loc_count: dict[str, int]   = defaultdict(int)
    loc_top:   dict[str, dict]  = {}
    for r in final:
        loc_sev[r["location_id"]] += r["financial_severity"]
        loc_count[r["location_id"]] += 1
        if r["location_id"] not in loc_top \
                or r["financial_severity"] > loc_top[r["location_id"]]["financial_severity"]:
            loc_top[r["location_id"]] = r
    for loc, sev in sorted(loc_sev.items(), key=lambda kv: kv[1], reverse=True):
        top = loc_top[loc]
        log.info(
            "  %-22s %s — %2d SKUs  sev=$%.0f  "
            "top: %s (%.0f%% short, gp/unit=$%.2f)",
            LOCATION_NAMES.get(loc, "?"), loc, loc_count[loc], sev,
            top["sku_id"], top["stockout_days_pct"] * 100, top["avg_gp_pct"],
        )
    log.info("-" * 64)

    # ---------- Step 7: persist (delete-then-insert, v2 schema fallback) ----------
    log.info("Step 7 — clearing today's existing rows then writing %d "
             "new rows (batches of %d) ...", len(final), WRITE_BATCH_SIZE)
    try:
        written = _write_today(client_holder, today, final)
    except Exception as exc:
        err = str(exc)
        v2_keywords = list(_UNDERSTOCK_V2_COLS) + ["column"]
        if any(kw in err.lower() for kw in [k.lower() for k in v2_keywords]):
            log.warning(
                "  v2 columns absent (migration 050 not applied) — "
                "retrying without them."
            )
            stripped = [
                {k: v for k, v in r.items() if k not in _UNDERSTOCK_V2_COLS}
                for r in final
            ]
            try:
                written = _write_today(client_holder, today, stripped)
            except Exception:
                log.exception("Persist failed even after stripping v2 columns.")
                return 1
        else:
            log.exception("Persist failed.")
            return 1
    log.info("  wrote %d rows.", written)

    # ---------- Step 7b: write network-trigger DC recs ----------
    if network_triggers:
        log.info("Step 7b — writing %d network DC bulk-purchase recs ...",
                 len(network_triggers))
        try:
            net_written = _write_network_recs(client_holder, today, network_triggers)
            log.info("  wrote %d network recs to reorder_recommendations.", net_written)
        except Exception:
            log.warning("  network rec write failed — understocking report still published.",
                        exc_info=True)

    # ---------- Step 8: post-write integrity check ----------
    try:
        check = (
            client_holder[0]
            .table("understocking_report")
            .select("id", count="exact")
            .eq("report_date", today.isoformat())
            .limit(1)
            .execute()
        )
        persisted = int(check.count or 0)
    except Exception:
        log.exception("Post-write integrity read failed.")
        return 1

    if persisted != len(final):
        log.error(
            "Post-write integrity check FAILED: built=%d persisted=%d "
            "(report_date=%s).  Run NOT marked complete.",
            len(final), persisted, today,
        )
        return 1

    log.info("Post-write integrity OK: %d rows in table for %s.", persisted, today)

    try:
        _mark_run_complete(client_holder, today)
        log.info("Run marked complete — dashboard will now show this report.")
    except Exception:
        log.exception(
            "Failed to mark run complete.  Today's rows are persisted "
            "but invisible to the dashboard until next successful run."
        )
        return 1

    log.info("=" * 64)
    log.info("Understocking report complete  (%.1fs)", time.monotonic() - t0)
    log.info("  Rows persisted:          %d", written)
    log.info("  Network DC recs:         %d", len(network_triggers))
    log.info("  Double-confirmed:        %d",
             sum(1 for r in final if r["double_confirmed"]))
    log.info("=" * 64)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chronic Understocking Report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and report without writing to understocking_report.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_understocking(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
