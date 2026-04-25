"""engine/transfer_intelligence.py — Transfer Intelligence Report.

Combines two independent signals to answer definitively:
    "What should each store stock locally instead of waiting for transfers?"

SIGNALS
    1. Transfer pattern (stocking_gaps table, ml/stocking_intelligence.py):
         SKUs repeatedly transferred TO a location — CHRONIC or RECURRING.
    2. Chronic understocking (understocking_report table, engine/understocking.py):
         SKUs sitting below reorder point ≥30% of observed days.

TIERS
    Tier 1 — CONFIRMED GAP (both signals agree)
         Highest confidence.  Act on these first.
         "This part keeps getting transferred in AND we're always below reorder.
          Stock it locally now."

    Tier 2 — TRANSFER PATTERN ONLY
         Repeated transfers but reorder point looks OK on paper.
         "We keep pulling from other branches — raise the Min Qty."

    Tier 3 — UNDERSTOCKED ONLY
         Below reorder point frequently; no transfer history yet.
         "Review demand signal and Min Qty — we may under-order."

PRIORITY SCORE (ranking within each tier)
    Tier 1:  inventory_value_at_risk × (1 + gap_score)
    Tier 2:  annual_transfer_savings  (or gap_score × total_transfer_value)
    Tier 3:  priority_score from understocking_report

OUTPUT
    transfer_intelligence_report table — top TOP_N_PER_TIER rows per
    (tier, location_id), ranked by priority_score DESC.

    Idempotent: DELETE today's rows then INSERT, with run_completed_at
    publication marker (identical pattern to engine/understocking.py).

PIPELINE POSITION
    Weekly, after ml/stocking_intelligence and engine/understocking.

CLI
    python -m engine.transfer_intelligence [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAD_BUFFER_DAYS: float = 21.0   # matches engine/understocking.py
TOP_N_PER_TIER:   int   = 20     # rows persisted per (tier, location_id)
WRITE_BATCH_SIZE: int   = 500
PAGE_SIZE:        int   = 1_000
SKU_BATCH_SIZE:   int   = 200    # for sku_master enrichment

TIER_CONFIRMED:    int = 1       # both signals agree
TIER_TRANSFER:     int = 2       # transfer pattern only
TIER_UNDERSTOCKED: int = 3       # understocking only

TIER_LABELS: dict[int, str] = {
    TIER_CONFIRMED:    "CONFIRMED_GAP",
    TIER_TRANSFER:     "TRANSFER_PATTERN_ONLY",
    TIER_UNDERSTOCKED: "UNDERSTOCKED_ONLY",
}

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
# Retry helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES: int   = 5
_RETRY_DELAY: float = 5.0

_RETRYABLE_TOKENS: tuple[str, ...] = (
    "57014", "statement timeout", "canceling statement",
    "ConnectionTerminated", "RemoteProtocolError",
    "ReadTimeout", "ReadError", "RemoteDisconnected",
    "Server disconnected", "Connection aborted",
    "502", "504", "Bad Gateway", "Gateway Time-out",
    "WinError 10054",
)
_RETRYABLE_TYPES = (ConnectionError, OSError)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
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
    table:       str,
    select:      str,
    filters:     dict | None = None,
    in_filters:  dict | None = None,
    not_null_cols: list[str] | None = None,
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
                for col, vals in (in_filters or {}).items():
                    q = q.in_(col, vals)
                for col in (not_null_cols or []):
                    q = q.not_.is_(col, "null")
                for col, desc in (order_by or []):
                    q = q.order(col, desc=desc)
                page = q.range(offset, offset + PAGE_SIZE - 1).execute().data or []
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    log.warning(
                        "  %s fetch retry %d/%d (offset=%d): %s — "
                        "reconnecting in %.0fs …",
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

def _fetch_latest_stocking_gaps(
    client_holder: list,
) -> dict[tuple[str, str], dict]:
    """Fetch CHRONIC + RECURRING gaps from stocking_gaps (latest analysis_date).

    Returns {(sku_id, location_id): row_dict}.
    """
    resp = (
        client_holder[0]
        .table("stocking_gaps")
        .select("analysis_date")
        .order("analysis_date", desc=True)
        .limit(1)
        .execute()
    )
    if not (resp.data or []):
        log.warning("  stocking_gaps table is empty.")
        return {}
    analysis_date = resp.data[0]["analysis_date"]
    log.info("  Latest stocking_gaps analysis_date: %s", analysis_date)

    rows = _paginate(
        client_holder,
        "stocking_gaps",
        "sku_id,location_id,transfer_frequency,transfer_streak,"
        "avg_qty_recommended,total_transfer_value,gap_score,"
        "gap_classification,suggested_stock_increase,"
        "current_reorder_point,annual_cost_savings,trend_direction",
        filters={"analysis_date": analysis_date},
        in_filters={"gap_classification": ["CHRONIC", "RECURRING"]},
        order_by=[("gap_score", True)],
    )
    log.info("  stocking_gaps rows (CHRONIC+RECURRING): %d", len(rows))
    return {(r["sku_id"], r["location_id"]): r for r in rows}


def _fetch_latest_understocking(
    client_holder: list,
) -> dict[tuple[str, str], dict]:
    """Fetch all rows from the latest completed understocking_report.

    Returns {(sku_id, location_id): row_dict}.
    """
    resp = (
        client_holder[0]
        .table("understocking_report")
        .select("report_date")
        .not_.is_("run_completed_at", "null")
        .order("report_date", desc=True)
        .limit(1)
        .execute()
    )
    if not (resp.data or []):
        log.warning("  No completed understocking_report found.")
        return {}
    report_date = resp.data[0]["report_date"]
    log.info("  Latest understocking_report report_date: %s", report_date)

    rows = _paginate(
        client_holder,
        "understocking_report",
        "sku_id,location_id,sku_description,days_observed,"
        "days_below_reorder,stockout_days_pct,avg_daily_demand,"
        "current_min_qty,suggested_min_qty,min_qty_gap,"
        "unit_cost,inventory_value_at_risk,priority_score",
        filters={"report_date": report_date},
        not_null_cols=["run_completed_at"],
        order_by=[("priority_score", True)],
    )
    log.info("  understocking_report rows: %d", len(rows))
    return {(r["sku_id"], r["location_id"]): r for r in rows}


def _fetch_sku_master(
    client_holder: list,
    sku_ids: list[str],
) -> dict[str, dict]:
    """Return {sku_id: {description, avg_weekly_units, unit_cost}} in batches."""
    out: dict[str, dict] = {}
    for i in range(0, len(sku_ids), SKU_BATCH_SIZE):
        chunk = sku_ids[i : i + SKU_BATCH_SIZE]
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
                    client_holder[0] = _get_fresh_client()
                    continue
                raise
        for r in rows:
            out[r["sku_id"]] = r
    return out


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _compute_priority(
    tier:   int,
    under:  dict,
    gap:    dict,
) -> float:
    if tier == TIER_CONFIRMED:
        var    = float(under.get("inventory_value_at_risk") or 0.0)
        gap_s  = float(gap.get("gap_score") or 0.0)
        return round(var * (1.0 + gap_s), 4)
    if tier == TIER_TRANSFER:
        savings = gap.get("annual_cost_savings")
        if savings is not None:
            return round(float(savings), 4)
        return round(
            float(gap.get("gap_score") or 0.0)
            * float(gap.get("total_transfer_value") or 0.0),
            4,
        )
    # TIER_UNDERSTOCKED
    return round(float(under.get("priority_score") or 0.0), 4)


def _build_rows(
    gaps:       dict[tuple[str, str], dict],
    understocks: dict[tuple[str, str], dict],
    sku_master:  dict[str, dict],
    today:       date,
) -> list[dict]:
    rows: list[dict] = []

    for (sku_id, loc_id) in set(gaps) | set(understocks):
        in_gap   = (sku_id, loc_id) in gaps
        in_under = (sku_id, loc_id) in understocks
        gap   = gaps.get((sku_id, loc_id)) or {}
        under = understocks.get((sku_id, loc_id)) or {}
        sm    = sku_master.get(sku_id) or {}

        tier = (TIER_CONFIRMED if (in_gap and in_under)
                else TIER_TRANSFER if in_gap
                else TIER_UNDERSTOCKED)

        # Description: understocking_report already carries it; sku_master is fallback
        description = (under.get("sku_description") or sm.get("description") or "")[:500]

        # Unit cost: prefer understocking (snapshot-derived), fall back to sku_master
        uc_raw = under.get("unit_cost") if under.get("unit_cost") is not None else sm.get("unit_cost")
        unit_cost = float(uc_raw) if uc_raw is not None else None

        # Avg daily demand
        avg_daily_raw = under.get("avg_daily_demand")
        if avg_daily_raw is not None:
            avg_daily: float | None = float(avg_daily_raw)
        else:
            avg_weekly = float(sm.get("avg_weekly_units") or 0.0)
            avg_daily = avg_weekly / 7.0 if avg_weekly > 0 else None

        # Current / suggested min qty
        current_min_raw = (under.get("current_min_qty")
                           if under.get("current_min_qty") is not None
                           else gap.get("current_reorder_point"))
        current_min = float(current_min_raw) if current_min_raw is not None else None

        if in_under:
            smin_raw = under.get("suggested_min_qty")
            suggested_min = float(smin_raw) if smin_raw is not None else None
            min_gap_v = float(under.get("min_qty_gap") or 0.0)
        else:
            # Tier 2: current ROP + suggested_stock_increase
            crp_raw = gap.get("current_reorder_point")
            inc_raw = gap.get("suggested_stock_increase")
            if crp_raw is not None and inc_raw is not None:
                crp = float(crp_raw)
                inc = float(inc_raw)
                suggested_min = round(crp + inc, 2)
                min_gap_v     = round(inc, 2)
            else:
                suggested_min = None
                min_gap_v     = 0.0

        # Value at risk: prefer stored value; compute for Tier 2 if possible
        var_raw = under.get("inventory_value_at_risk")
        if var_raw is not None:
            value_at_risk: float | None = round(float(var_raw), 2)
        elif avg_daily is not None and unit_cost:
            value_at_risk = round(avg_daily * unit_cost * LEAD_BUFFER_DAYS, 2)
        else:
            value_at_risk = None

        priority = _compute_priority(tier, under, gap)

        rows.append({
            "report_date":             today.isoformat(),
            "location_id":             loc_id,
            "location_name":           LOCATION_NAMES.get(loc_id, ""),
            "sku_id":                  sku_id,
            "sku_description":         description,
            "tier":                    tier,
            "tier_label":              TIER_LABELS[tier],
            # Transfer signal (NULL for Tier 3)
            "transfer_frequency":      int(gap["transfer_frequency"]) if gap.get("transfer_frequency") is not None else None,
            "transfer_streak":         int(gap["transfer_streak"])    if gap.get("transfer_streak")    is not None else None,
            "gap_score":               round(float(gap["gap_score"]), 4) if gap.get("gap_score") is not None else None,
            "gap_classification":      gap.get("gap_classification"),
            "avg_qty_recommended":     round(float(gap["avg_qty_recommended"]), 4) if gap.get("avg_qty_recommended") is not None else None,
            "annual_transfer_savings": round(float(gap["annual_cost_savings"]),  2) if gap.get("annual_cost_savings")  is not None else None,
            # Understocking signal (NULL for Tier 2)
            "stockout_days_pct":       round(float(under["stockout_days_pct"]),  4) if under.get("stockout_days_pct")  is not None else None,
            "days_below_reorder":      int(under["days_below_reorder"])             if under.get("days_below_reorder") is not None else None,
            "days_observed":           int(under["days_observed"])                  if under.get("days_observed")      is not None else None,
            "avg_daily_demand":        round(avg_daily, 4) if avg_daily is not None else None,
            "inventory_value_at_risk": value_at_risk,
            # Min qty / cost
            "current_min_qty":         round(current_min, 2)   if current_min   is not None else None,
            "suggested_min_qty":       round(suggested_min, 2) if suggested_min is not None else None,
            "min_qty_gap":             round(min_gap_v, 2),
            "unit_cost":               round(unit_cost, 2)     if unit_cost     is not None else None,
            "priority_score":          priority,
        })

    return rows


def _top_n_per_tier_per_location(rows: list[dict], n: int) -> list[dict]:
    """Top-n rows per (tier, location_id), ranked by priority_score DESC."""
    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["tier"], r["location_id"])].append(r)
    out: list[dict] = []
    for items in buckets.values():
        items.sort(key=lambda r: r["priority_score"], reverse=True)
        out.extend(items[:n])
    return out


# ---------------------------------------------------------------------------
# Write helpers  (identical pattern to engine/understocking.py)
# ---------------------------------------------------------------------------

def _write_today(
    client_holder: list, today: date, rows: list[dict],
) -> int:
    """Delete today's rows, then insert new ones (run_completed_at=NULL).

    Returns the count of rows inserted.  Caller must invoke
    _mark_run_complete() after a successful integrity check.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            (
                client_holder[0]
                .table("transfer_intelligence_report")
                .delete()
                .eq("report_date", today.isoformat())
                .execute()
            )
            break
        except Exception as exc:
            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _get_fresh_client()
                continue
            raise

    if not rows:
        return 0

    written = 0
    for i in range(0, len(rows), WRITE_BATCH_SIZE):
        batch = rows[i : i + WRITE_BATCH_SIZE]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                client_holder[0].table("transfer_intelligence_report").insert(batch).execute()
                written += len(batch)
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _get_fresh_client()
                    continue
                raise
    return written


def _mark_run_complete(client_holder: list, today: date) -> None:
    """Stamp run_completed_at = NOW() — this is the dashboard publication step."""
    now_iso = datetime.utcnow().isoformat() + "Z"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            (
                client_holder[0]
                .table("transfer_intelligence_report")
                .update({"run_completed_at": now_iso})
                .eq("report_date", today.isoformat())
                .execute()
            )
            return
        except Exception as exc:
            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _get_fresh_client()
                continue
            raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_transfer_intelligence(dry_run: bool = False) -> int:
    """Build and persist the Transfer Intelligence Report.

    Returns 0 on success, 1 on failure.
    """
    t0    = time.monotonic()
    today = date.today()

    log.info("=" * 60)
    log.info("TRANSFER INTELLIGENCE REPORT%s",
             "  [DRY RUN]" if dry_run else "")
    log.info("  Combines stocking_gaps + understocking_report → 3-tier output")
    log.info("  Top %d per (tier, location)  |  LEAD_BUFFER=%dd",
             TOP_N_PER_TIER, int(LEAD_BUFFER_DAYS))
    log.info("=" * 60)

    client_holder: list = [_get_fresh_client()]

    # ── Step 1: fetch transfer-pattern gaps ──────────────────────────
    log.info("Step 1 — fetching stocking_gaps (CHRONIC + RECURRING) …")
    try:
        gaps = _fetch_latest_stocking_gaps(client_holder)
    except Exception:
        log.exception("stocking_gaps fetch failed.")
        return 1

    # ── Step 2: fetch chronic understocking rows ──────────────────────
    log.info("Step 2 — fetching latest completed understocking_report …")
    try:
        understocks = _fetch_latest_understocking(client_holder)
    except Exception:
        log.exception("understocking_report fetch failed.")
        return 1

    if not gaps and not understocks:
        log.warning("Both source tables are empty — nothing to build.")
        return 0

    log.info("  stocking_gaps pairs:   %d", len(gaps))
    log.info("  understocking pairs:   %d", len(understocks))

    confirmed_count = sum(
        1 for key in gaps if key in understocks
    )
    log.info("  Overlap (Tier 1):      %d", confirmed_count)

    # ── Step 3: enrich with sku_master ───────────────────────────────
    all_sku_ids = sorted({sku_id for (sku_id, _) in (set(gaps) | set(understocks))})
    log.info("Step 3 — fetching sku_master for %d unique SKUs …", len(all_sku_ids))
    sku_master: dict[str, dict] = {}
    if all_sku_ids:
        try:
            sku_master = _fetch_sku_master(client_holder, all_sku_ids)
            log.info("  resolved %d / %d SKUs", len(sku_master), len(all_sku_ids))
        except Exception:
            log.exception("sku_master fetch failed (non-fatal — descriptions/costs may be missing).")

    # ── Step 4: build and rank rows ───────────────────────────────────
    log.info("Step 4 — building transfer intelligence rows …")
    try:
        all_rows = _build_rows(gaps, understocks, sku_master, today)
    except Exception:
        log.exception("Row build failed.")
        return 1

    final = _top_n_per_tier_per_location(all_rows, TOP_N_PER_TIER)

    t1_n = sum(1 for r in final if r["tier"] == TIER_CONFIRMED)
    t2_n = sum(1 for r in final if r["tier"] == TIER_TRANSFER)
    t3_n = sum(1 for r in final if r["tier"] == TIER_UNDERSTOCKED)
    log.info(
        "  Built %d rows total  "
        "(Tier1-CONFIRMED=%d  Tier2-TRANSFER=%d  Tier3-UNDERSTOCKED=%d)",
        len(final), t1_n, t2_n, t3_n,
    )

    # Per-location summary
    log.info("-" * 60)
    loc_tiers: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    loc_var:   dict[str, float]          = defaultdict(float)
    for r in final:
        loc_tiers[r["location_id"]][r["tier"]] += 1
        loc_var[r["location_id"]] += float(r.get("inventory_value_at_risk") or 0)
    for loc, var in sorted(loc_var.items(), key=lambda kv: kv[1], reverse=True):
        tiers = loc_tiers[loc]
        log.info(
            "  %-22s %s — T1=%2d  T2=%2d  T3=%2d  $%9.0f at risk",
            LOCATION_NAMES.get(loc, "?"), loc,
            tiers.get(1, 0), tiers.get(2, 0), tiers.get(3, 0), var,
        )
    log.info("-" * 60)

    # ── Step 5: persist ───────────────────────────────────────────────
    if dry_run:
        log.info("DRY RUN — skipping write.  %d rows would be persisted.", len(final))
        log.info("=" * 60)
        log.info("Transfer intelligence complete  (%.1fs)  rows=%d",
                 time.monotonic() - t0, len(final))
        log.info("=" * 60)
        return 0

    log.info("Step 5 — writing %d rows (delete-then-insert, "
             "run_completed_at=NULL) …", len(final))
    try:
        written = _write_today(client_holder, today, final)
    except Exception:
        log.exception("Persist failed.")
        return 1
    log.info("  wrote %d rows.", written)

    # Integrity check
    try:
        check = (
            client_holder[0]
            .table("transfer_intelligence_report")
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
            "Integrity check FAILED: built=%d persisted=%d — "
            "run NOT marked complete.  Dashboard will show previous run.",
            len(final), persisted,
        )
        return 1

    log.info("Integrity OK: %d rows for %s.", persisted, today)

    try:
        _mark_run_complete(client_holder, today)
        log.info("Run marked complete — dashboard will now show this report.")
    except Exception:
        log.exception("Failed to mark run complete.")
        return 1

    log.info("=" * 60)
    log.info("Transfer intelligence complete  (%.1fs)  rows=%d  "
             "T1=%d  T2=%d  T3=%d",
             time.monotonic() - t0, len(final), t1_n, t2_n, t3_n)
    log.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transfer Intelligence Report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and report without writing to transfer_intelligence_report.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_transfer_intelligence(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
