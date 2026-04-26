"""ml/classify.py — GP-based ABC classification with recency weighting and location ranking.

Three improvements over the legacy sales-volume ABC classification in sku_master:

    1. GP-based ranking  — gross_profit from sales_detail_transactions
       (tran_code IN ('SL', 'SL-I')) replaces raw sales volume as the
       ABC ranking metric.  Falls back to qty_ship volume when fewer than
       MIN_SKU_COUNT_FOR_GP SKUs have positive GP data.

    2. Recency weighting — decay of GP_DECAY (0.9) per calendar month back.
       The current calendar month carries weight 1.0; 12 months ago carries
       GP_DECAY ** 11 ≈ 0.314.  Weights are applied before GP is summed and
       ranked.

    3. Location-level ABC — independent GP ranking per location written to
       sku_location_class.  The global sku_master.abc_class is also updated.

Outputs
-------
    sku_master          — abc_class (VARCHAR 1), gp_12m (NUMERIC 14,4) updated.
    sku_location_class  — one row per (sku_id, location_id) with per-location
                          abc_class, xyz_class (propagated from sku_master),
                          gp_12m, and sales_12m.

Prerequisite migrations
-----------------------
    049_sku_master_gp.sql — adds gp_12m to sku_master, creates sku_location_class.

Usage
-----
    python -m ml.classify            # live run
    python -m ml.classify --dry-run  # compute + compare, no DB writes
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

GP_WINDOW_MONTHS: int = 12     # calendar months of GP history to consume
GP_DECAY: float = 0.9          # weight multiplier per month back (0 = current month)
ABC_A_CUTOFF: float = 0.10     # top 10 % of SKUs by count -> A
ABC_B_CUTOFF: float = 0.30     # top 30 % cumulative (next 20 % = B)
MIN_SKU_COUNT_FOR_GP: int = 1_000  # fallback to volume when fewer SKUs have positive GP

TRAN_CODES: tuple[str, ...] = ("SL", "SL-I")

# Non-part catch-all codes that must not influence ABC classification.
# Any prod_line_pn / sku_id whose upper-cased value contains one of these
# strings is silently excluded from all GP aggregation and ranking.
EXCLUDE_SKU_PATTERNS: tuple[str, ...] = (
    "COUPON", "COU", "DELIVERY", "FEE", "MISC", "LABOR",
)

_PAGE_SIZE: int = 1_000
_UPD_BATCH: int = 500
_CHUNK_DAYS: int = 14   # date-chunked fetch to avoid statement timeouts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paginate(
    client: Any,
    table: str,
    select: str,
    filters: dict | None = None,
    gte_filters: dict | None = None,
    lte_filters: dict | None = None,
    in_filters: dict | None = None,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        q = client.table(table).select(select)
        for col, val in (filters or {}).items():
            q = q.eq(col, val)
        for col, val in (gte_filters or {}).items():
            q = q.gte(col, val)
        for col, val in (lte_filters or {}).items():
            q = q.lte(col, val)
        for col, vals in (in_filters or {}).items():
            q = q.in_(col, vals)
        page = q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
        if offset % 50_000 == 0:
            log.info("    ... fetched %d rows from %s", offset, table)
    return rows


def _months_back(today: date, tran_date: date) -> int:
    """Full calendar months between tran_date's month and today's month."""
    return (today.year - tran_date.year) * 12 + (today.month - tran_date.month)


def _is_excluded(sku_id: str) -> bool:
    """Return True when sku_id matches any EXCLUDE_SKU_PATTERNS (case-insensitive contains)."""
    upper = sku_id.upper()
    return any(pat in upper for pat in EXCLUDE_SKU_PATTERNS)


def _rank_to_abc(ranked_skus: list[str]) -> dict[str, str]:
    """Assign A/B/C to a list already sorted descending by metric."""
    n = len(ranked_skus)
    if n == 0:
        return {}
    a_count = max(1, round(n * ABC_A_CUTOFF))
    b_count = max(1, round(n * (ABC_B_CUTOFF - ABC_A_CUTOFF)))
    result: dict[str, str] = {}
    for i, sku in enumerate(ranked_skus):
        if i < a_count:
            result[sku] = "A"
        elif i < a_count + b_count:
            result[sku] = "B"
        else:
            result[sku] = "C"
    return result


# ---------------------------------------------------------------------------
# Stage 1: Fetch GP from sales_detail_transactions with decay weighting
# ---------------------------------------------------------------------------

def _fetch_gp_data(client: Any, today: date) -> tuple[
    dict[str, float],            # gp_weighted_global  {pn: decay-weighted GP}
    dict[str, float],            # gp_raw_global       {pn: unweighted 12m GP}
    dict[tuple[str, str], float], # gp_weighted_loc    {(pn, loc): weighted GP}
    dict[tuple[str, str], float], # gp_raw_loc         {(pn, loc): unweighted GP}
    dict[tuple[str, str], float], # sales_loc           {(pn, loc): raw sales $}
    dict[str, float],            # vol_weighted_global {pn: weighted qty_ship}
]:
    """Fetch 12 months of SL/SL-I GP, apply recency decay, return aggregated dicts."""
    cutoff = today - timedelta(days=GP_WINDOW_MONTHS * 31)
    log.info(
        "Stage 1: Fetching sales_detail_transactions since %s (tran_code IN %s) ...",
        cutoff.isoformat(), TRAN_CODES,
    )

    # Date-chunked fetch to stay under Supabase statement timeout
    rows: list[dict] = []
    chunk_start = cutoff
    while chunk_start <= today:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS - 1), today)
        offset = 0
        while True:
            q = (
                client.table("sales_detail_transactions")
                .select("prod_line_pn,location_id,tran_date,gross_profit,sales,qty_ship")
                .in_("tran_code", list(TRAN_CODES))
                .gte("tran_date", chunk_start.isoformat())
                .lte("tran_date", chunk_end.isoformat())
            )
            page = q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
            rows.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        chunk_start = chunk_end + timedelta(days=1)
        if len(rows) % 100_000 < (_CHUNK_DAYS * 2_000):
            if len(rows) > 0:
                log.info("    ... streamed %d rows so far ...", len(rows))

    log.info("  Loaded %d sales detail rows.", len(rows))

    gp_weighted_global: dict[str, float] = defaultdict(float)
    gp_raw_global: dict[str, float] = defaultdict(float)
    gp_weighted_loc: dict[tuple[str, str], float] = defaultdict(float)
    gp_raw_loc: dict[tuple[str, str], float] = defaultdict(float)
    sales_loc: dict[tuple[str, str], float] = defaultdict(float)
    vol_weighted_global: dict[str, float] = defaultdict(float)

    excluded_count = 0
    for r in rows:
        pn = r.get("prod_line_pn") or ""
        loc = r.get("location_id") or ""
        if not pn:
            continue
        if _is_excluded(pn):
            excluded_count += 1
            continue

        tran_date_str = str(r.get("tran_date", ""))[:10]
        if len(tran_date_str) < 10:
            continue
        try:
            tran_date = date.fromisoformat(tran_date_str)
        except ValueError:
            continue

        mb = max(0, min(_months_back(today, tran_date), GP_WINDOW_MONTHS - 1))
        weight = GP_DECAY ** mb

        gp = float(r.get("gross_profit") or 0)
        rev = float(r.get("sales") or 0)
        qty = float(r.get("qty_ship") or 0)

        gp_weighted_global[pn] += gp * weight
        gp_raw_global[pn] += gp
        vol_weighted_global[pn] += qty * weight

        if loc:
            gp_weighted_loc[(pn, loc)] += gp * weight
            gp_raw_loc[(pn, loc)] += gp
            sales_loc[(pn, loc)] += rev

    log.info(
        "  Distinct prod_line_pn values: %d  (across %d locations)  "
        "excluded catch-all rows: %d",
        len(gp_weighted_global),
        len({loc for (_, loc) in gp_weighted_loc}),
        excluded_count,
    )
    return (
        dict(gp_weighted_global),
        dict(gp_raw_global),
        dict(gp_weighted_loc),
        dict(gp_raw_loc),
        dict(sales_loc),
        dict(vol_weighted_global),
    )


# ---------------------------------------------------------------------------
# Stage 2: Global ABC classification -> sku_master
# ---------------------------------------------------------------------------

def _classify_global(
    client: Any,
    today: date,
    gp_weighted: dict[str, float],
    gp_raw: dict[str, float],
    vol_weighted: dict[str, float],
    dry_run: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Rank sku_master SKUs globally by weighted GP, assign ABC, update sku_master.

    Returns:
        (new_class_map, old_class_map) both keyed by sku_id.
    """
    log.info("Stage 2: Global GP-based ABC classification ...")

    sku_count_with_gp = sum(1 for v in gp_weighted.values() if v > 0)
    use_gp = sku_count_with_gp >= MIN_SKU_COUNT_FOR_GP
    if not use_gp:
        log.warning(
            "  Only %d SKUs have positive weighted GP (need >=%d) -- "
            "falling back to volume-based ranking.",
            sku_count_with_gp, MIN_SKU_COUNT_FOR_GP,
        )
        metric = vol_weighted
    else:
        log.info("  Using GP-based ranking (%d SKUs with positive GP).", sku_count_with_gp)
        metric = gp_weighted

    # Fetch current sku_master for comparison and update
    log.info("  Fetching sku_master ...")
    sm_rows = _paginate(client, "sku_master", "sku_id,abc_class,description")
    old_class: dict[str, str] = {
        r["sku_id"]: (r.get("abc_class") or "C") for r in sm_rows
    }
    desc_map: dict[str, str] = {r["sku_id"]: (r.get("description") or "") for r in sm_rows}

    # Rank sku_master.sku_ids by their GP metric (0.0 for SKUs with no sales detail data).
    # Excluded catch-all codes are removed from the ranking pool and will not receive
    # an abc_class update (their existing value in sku_master is left unchanged).
    sku_ids = [s for s in old_class if not _is_excluded(s)]
    excluded_sm = len(old_class) - len(sku_ids)
    if excluded_sm:
        log.info("  Excluded %d catch-all SKUs from ranking pool.", excluded_sm)
    ranked_skus = sorted(sku_ids, key=lambda s: metric.get(s, 0.0), reverse=True)
    new_class = _rank_to_abc(ranked_skus)

    # Class-change matrix (sku_master perspective only)
    matrix: dict[tuple[str, str], int] = defaultdict(int)
    for sku_id, old_c in old_class.items():
        new_c = new_class.get(sku_id, "C")
        matrix[(old_c, new_c)] += 1

    log.info("  Class-change matrix (old -> new):")
    for (old_c, new_c), cnt in sorted(matrix.items()):
        marker = "" if old_c == new_c else "  <-- changed"
        log.info("    %s -> %s : %6d%s", old_c, new_c, cnt, marker)

    total_changed = sum(cnt for (o, n), cnt in matrix.items() if o != n)
    log.info("  Total SKUs reclassified: %d / %d", total_changed, len(old_class))

    # Top 10 moving up to A
    up_to_a = sorted(
        (s for s in sku_ids if new_class.get(s) == "A" and old_class.get(s) != "A"),
        key=lambda s: metric.get(s, 0.0),
        reverse=True,
    )
    log.info("  Top 10 SKUs moving UP to A class:")
    if up_to_a:
        for sku in up_to_a[:10]:
            log.info(
                "    %-20s  %-38s  old=%-1s  weighted_gp=%10.2f",
                sku, desc_map.get(sku, "")[:38],
                old_class.get(sku, "?"), metric.get(sku, 0.0),
            )
    else:
        log.info("    (none)")

    # Top 10 dropping from A
    drop_from_a = sorted(
        (s for s in sku_ids if old_class.get(s) == "A" and new_class.get(s) != "A"),
        key=lambda s: metric.get(s, 0.0),
        reverse=True,
    )
    log.info("  Top 10 SKUs DROPPING from A class:")
    if drop_from_a:
        for sku in drop_from_a[:10]:
            log.info(
                "    %-20s  %-38s  new=%-1s  weighted_gp=%10.2f",
                sku, desc_map.get(sku, "")[:38],
                new_class.get(sku, "?"), metric.get(sku, 0.0),
            )
    else:
        log.info("    (none)")

    log.info(
        "  New global distribution:  A=%d  B=%d  C=%d  (of %d total)",
        sum(1 for c in new_class.values() if c == "A"),
        sum(1 for c in new_class.values() if c == "B"),
        sum(1 for c in new_class.values() if c == "C"),
        len(new_class),
    )

    if dry_run:
        log.info("  DRY RUN -- sku_master not updated.")
        return new_class, old_class

    # Write abc_class + gp_12m back to sku_master (skip excluded catch-all codes)
    updates = [
        {
            "sku_id":    r["sku_id"],
            "abc_class": new_class.get(r["sku_id"], "C"),
            "gp_12m":    round(gp_raw.get(r["sku_id"], 0.0), 4),
        }
        for r in sm_rows
        if not _is_excluded(r["sku_id"])
    ]
    _gp12m_available = True
    _written = 0
    for i in range(0, len(updates), _UPD_BATCH):
        batch = updates[i : i + _UPD_BATCH]
        try:
            client.table("sku_master").upsert(batch, on_conflict="sku_id").execute()
            _written += len(batch)
        except Exception as exc:
            err = str(exc)
            if _gp12m_available and ("gp_12m" in err or "column" in err.lower()):
                _gp12m_available = False
                log.warning(
                    "  gp_12m column absent -- migration 049 not applied. "
                    "Updating abc_class only."
                )
                for row in batch:
                    row.pop("gp_12m", None)
                client.table("sku_master").upsert(batch, on_conflict="sku_id").execute()
                _written += len(batch)
            else:
                log.error("  sku_master upsert failed (batch %d): %s", i, exc)

    log.info(
        "  Updated %d SKUs in sku_master (abc_class%s).",
        _written, " + gp_12m" if _gp12m_available else " only",
    )
    return new_class, old_class


# ---------------------------------------------------------------------------
# Stage 3: Location-level ABC -> sku_location_class
# ---------------------------------------------------------------------------

def _classify_by_location(
    client: Any,
    today: date,
    gp_weighted_loc: dict[tuple[str, str], float],
    gp_raw_loc: dict[tuple[str, str], float],
    sales_loc: dict[tuple[str, str], float],
    dry_run: bool = False,
) -> int:
    """Rank SKUs independently per location by weighted GP; write sku_location_class.

    Returns:
        Count of (sku_id, location_id) rows written (or that would be written).
    """
    log.info("Stage 3: Location-level ABC classification ...")

    # Propagate xyz_class from sku_master (global demand-variability metric)
    xyz_rows = _paginate(client, "sku_master", "sku_id,xyz_class")
    xyz_map: dict[str, str | None] = {r["sku_id"]: r.get("xyz_class") for r in xyz_rows}

    # Group (pn, loc) -> weighted_gp by location (excluded catch-all codes dropped)
    by_loc: dict[str, dict[str, float]] = defaultdict(dict)
    for (pn, loc), wgp in gp_weighted_loc.items():
        if not _is_excluded(pn):
            by_loc[loc][pn] = wgp

    rows_to_write: list[dict] = []
    for loc in sorted(by_loc):
        sku_gp = by_loc[loc]
        ranked = sorted(sku_gp, key=lambda k: sku_gp[k], reverse=True)
        loc_class = _rank_to_abc(ranked)
        for pn, cls in loc_class.items():
            rows_to_write.append({
                "sku_id":      pn,
                "location_id": loc,
                "abc_class":   cls,
                "xyz_class":   xyz_map.get(pn),
                "gp_12m":      round(gp_raw_loc.get((pn, loc), 0.0), 4),
                "sales_12m":   round(sales_loc.get((pn, loc), 0.0), 4),
                "run_date":    today.isoformat(),
            })

    log.info(
        "  Location-level rows: %d  (across %d locations, %d unique SKUs)",
        len(rows_to_write),
        len(by_loc),
        len({r["sku_id"] for r in rows_to_write}),
    )

    if dry_run:
        log.info("  DRY RUN -- sku_location_class not updated.")
        return len(rows_to_write)

    written = 0
    for i in range(0, len(rows_to_write), _UPD_BATCH):
        chunk = rows_to_write[i : i + _UPD_BATCH]
        try:
            client.table("sku_location_class").upsert(
                chunk, on_conflict="sku_id,location_id"
            ).execute()
            written += len(chunk)
        except Exception as exc:
            log.error(
                "  sku_location_class upsert failed (batch %d): %s -- %s",
                i, exc.__class__.__name__, str(exc)[:200],
            )

    log.info("  Written %d rows to sku_location_class.", written)
    return written


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_classify(dry_run: bool = False) -> int:
    """Run all three classification stages.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    t0 = time.monotonic()
    banner = "=" * 66
    log.info(banner)
    log.info("partswatch-ai -- ml.classify (GP-based ABC + location ranking)")
    log.info(
        "  window=%dm  decay=%.1f/month  A<=top%d%%  B<=top%d%%  "
        "min_gp_skus=%d",
        GP_WINDOW_MONTHS, GP_DECAY,
        int(ABC_A_CUTOFF * 100), int(ABC_B_CUTOFF * 100),
        MIN_SKU_COUNT_FOR_GP,
    )
    log.info(banner)

    try:
        client = get_client()
    except Exception:
        log.exception("Failed to initialise Supabase client.")
        return 1

    if dry_run:
        log.info("DRY RUN -- no database writes will be made.")

    today = date.today()
    log.info("Run date: %s", today.isoformat())
    log.info("-" * 66)

    try:
        gp_weighted, gp_raw, gp_weighted_loc, gp_raw_loc, sales_loc, vol_weighted = (
            _fetch_gp_data(client, today)
        )
    except Exception:
        log.exception("Stage 1 (GP data fetch) failed.")
        return 1

    log.info("-" * 66)

    try:
        new_class, old_class = _classify_global(
            client, today, gp_weighted, gp_raw, vol_weighted, dry_run=dry_run,
        )
    except Exception:
        log.exception("Stage 2 (global ABC classification) failed.")
        return 1

    log.info("-" * 66)

    try:
        loc_rows = _classify_by_location(
            client, today, gp_weighted_loc, gp_raw_loc, sales_loc, dry_run=dry_run,
        )
    except Exception:
        log.exception("Stage 3 (location ABC classification) failed.")
        return 1

    elapsed = time.monotonic() - t0
    changed = sum(1 for s in old_class if old_class[s] != new_class.get(s, "C"))
    log.info(banner)
    log.info("ml.classify complete  (%.2fs)", elapsed)
    log.info("  Global SKUs reclassified:   %d / %d", changed, len(old_class))
    log.info("  New distribution:           A=%d  B=%d  C=%d",
             sum(1 for c in new_class.values() if c == "A"),
             sum(1 for c in new_class.values() if c == "B"),
             sum(1 for c in new_class.values() if c == "C"))
    log.info("  Location-level rows:        %d", loc_rows)
    if dry_run:
        log.info("  (DRY RUN -- no writes made)")
    log.info(banner)
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="partswatch-ai GP-based ABC classification with recency weighting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute classifications without writing to the database.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from config import LOG_LEVEL
        setup_logging(LOG_LEVEL)
    except (ImportError, EnvironmentError):
        setup_logging("INFO")

    args = _parse_args()
    return run_classify(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
