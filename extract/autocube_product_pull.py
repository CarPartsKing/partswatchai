"""extract/autocube_product_pull.py — Product cube extraction from Autocube OLAP.

Extracts inventory snapshots, transfer history, product master enrichment,
and pricing tiers from the Autologue Product cube via XMLA/SOAP.

Runs as a companion to autocube_pull.py (which handles Sales Detail).

SECURITY
--------
- Read-only: only MDX SELECT queries are executed
- query_validator() checks every query before execution
- All MDX queries are hardcoded constants
- All writes go to Supabase only

MODES
-----
    python -m extract.autocube_product_pull --test
        Discover all dimension members and measures in the Product cube.

    python -m extract.autocube_product_pull --mode inventory
        Pull current inventory snapshots (On Hand, Min, Max, On Order, Cost).

    python -m extract.autocube_product_pull --mode full
        Alias for --mode inventory — loads every (SKU, location) on-hand
        position from the Product cube's [Prod PN Loc] hierarchy.

    python -m extract.autocube_product_pull --mode transfers
        Pull inter-location transfer history.

    python -m extract.autocube_product_pull --mode enrich
        Pull product master attributes (vendor, aisle/bin, core flag, etc).

    python -m extract.autocube_product_pull --mode pricing
        Pull pricing tier matrix (Price 1-10, A-C, S, ST with GP margins).

    python -m extract.autocube_product_pull --mode all
        Run all four extraction modes in sequence.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from extract.autocube_pull import (
    AutocubeClient,
    get_client,
    clean_numeric,
    _extract_location_code,
    _BATCH_SIZE,
)
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

_PRODUCT_CUBE = "Product"

# Active store + DC locations (numeric Autocube codes).  Retired locations
# 14, 19, 22, 23, 30, 31 are excluded.  Inventory extraction iterates this
# list and runs ONE MDX per location with a WHERE slicer + NON EMPTY on
# rows — Autologue's server cannot handle a full Prod Code × Loc crossjoin
# (out-of-memory after ~10 minutes), so location-by-location batching is
# the only viable approach.
ACTIVE_LOCATIONS: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 15, 16, 17, 18,
    20, 21, 24, 25, 26, 27, 28, 29,
    32, 33,
]

def _dedupe_by_keys(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Collapse a list of dict rows so each composite key appears once.

    Postgres' ON CONFLICT DO UPDATE raises 21000 ("cannot affect a row a
    second time") when a single statement tries to upsert two rows that
    map to the same conflict target.  Autocube can legitimately return
    duplicate (sku_id, location_id) pairs (e.g. the same SKU under two
    Prod Code aliases), so we deduplicate every batch on the same keys
    that appear in the upsert's on_conflict clause, keeping the LAST
    occurrence (most recent measure values win).
    """
    seen: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        seen[tuple(row.get(k) for k in keys)] = row
    return list(seen.values())


MDX_INVENTORY_BY_LOCATION = """\
SELECT
  {{
    [Measures].[On Hand Qty],
    [Measures].[Min Qty],
    [Measures].[Max Qty],
    [Measures].[Qty On Order],
    [Measures].[Ext Cost On Hand],
    [Measures].[Unit Cost on Hand],
    [Measures].[Cost]
  }} ON COLUMNS,
  NON EMPTY
    [Product].[Prod Code].[Prod Code].MEMBERS
  ON ROWS
FROM [Product]
WHERE ([Location].[Loc].&[{loc_num}])
"""
# ---------------------------------------------------------------------------
# Cost-measure selection — empirical findings 2026-04-18:
#   [Unit Cost]            — almost always NULL across the catalog (deprecated
#                            or not populated in this Autocube deployment).
#   [Unit Cost on Hand]    — true per-unit replacement cost of inventory
#                            CURRENTLY on hand.  NULL when qty_on_hand = 0.
#   [Cost]                 — catalog / master per-unit cost.  Populated for
#                            essentially every active SKU regardless of stock
#                            level.  Used as fallback when on-hand cost is
#                            absent (e.g. fully sold-down SKUs we still want
#                            to value historically).
#   [Ext Cost On Hand]     — extended (qty × cost) on hand value.  Used as
#                            secondary derivation when both unit-cost
#                            measures are absent but on-hand qty > 0.
# Resolution order in extraction: UCoH → Cost → ExtCoH/qty.
# ---------------------------------------------------------------------------

MDX_TRANSFERS = """\
SELECT
  NON EMPTY {{
    [Measures].[Qty Ship],
    [Measures].[Ext Cost]
  }} ON COLUMNS,
  NON EMPTY CROSSJOIN(
    [Product].[Prod Code].[Prod Code].MEMBERS,
    [Transfer Locs].[Loc From].[Loc From].MEMBERS,
    [Transfer Locs].[Loc To].[Loc To].MEMBERS
  ) ON ROWS
FROM [Product]
"""

MDX_PRODUCT_ENRICH = """\
SELECT
  NON EMPTY {{
    [Measures].[Product Count]
  }} ON COLUMNS,
  NON EMPTY CROSSJOIN(
    [Product].[Prod Code].[Prod Code].MEMBERS,
    [Product].[Vendor 1].[Vendor 1].MEMBERS
  ) ON ROWS
FROM [Product]
"""

MDX_PRODUCT_ATTRIBUTES = """\
SELECT
  NON EMPTY {{
    [Measures].[Product Count]
  }} ON COLUMNS,
  NON EMPTY CROSSJOIN(
    [Product].[Prod Code].[Prod Code].MEMBERS,
    [Product].[Prod Line].[Prod Line].MEMBERS
  ) ON ROWS
FROM [Product]
"""

# Prod Desc / Ext Desc / UOM live as MEMBER PROPERTIES of [Prod Code],
# not as their own hierarchies — so we read them via DIMENSION PROPERTIES.
MDX_PRODUCT_DESC = """\
SELECT
  NON EMPTY { [Measures].[Product Count] } ON COLUMNS,
  NON EMPTY [Product].[Prod Code].[Prod Code].MEMBERS
  DIMENSION PROPERTIES
    MEMBER_NAME,
    [Product].[Prod Code].[Prod Desc],
    [Product].[Prod Code].[Ext Desc],
    [Product].[Prod Code].[UOM]
  ON ROWS
FROM [Product]
"""

_PRICE_TIERS = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "A", "B", "C", "S", "ST",
]

MDX_PRICING = """\
SELECT
  NON EMPTY {{
    [Measures].[Price 1],
    [Measures].[Price 2],
    [Measures].[Price 3],
    [Measures].[Price 4],
    [Measures].[Price 5],
    [Measures].[Price 6],
    [Measures].[Price 7],
    [Measures].[Price 8],
    [Measures].[Price 9],
    [Measures].[Price 10],
    [Measures].[Price A],
    [Measures].[Price B],
    [Measures].[Price C],
    [Measures].[Price S],
    [Measures].[Price ST],
    [Measures].[Cost],
    [Measures].[Price 1 GP],
    [Measures].[Price 2 GP],
    [Measures].[Price 3 GP],
    [Measures].[Price 4 GP],
    [Measures].[Price 5 GP],
    [Measures].[Price 6 GP],
    [Measures].[Price 7 GP],
    [Measures].[Price 8 GP],
    [Measures].[Price 9 GP],
    [Measures].[Price 10 GP],
    [Measures].[Price A GP],
    [Measures].[Price B GP],
    [Measures].[Price C GP],
    [Measures].[Price S GP],
    [Measures].[Price ST GP]
  }} ON COLUMNS,
  NON EMPTY
    [Product].[Prod Code].[Prod Code].MEMBERS
  ON ROWS
FROM [Product]
"""


def _get_product_client() -> AutocubeClient:
    client = get_client()
    client._cube = _PRODUCT_CUBE
    return client


def run_test() -> int:
    log.info("=" * 70)
    log.info("  PRODUCT CUBE DISCOVERY")
    log.info("=" * 70)

    try:
        client = _get_product_client()
        endpoint = client.connect()
        log.info("Connected: %s", endpoint)
    except Exception:
        log.exception("Connection failed.")
        return 1

    log.info("")
    log.info("--- DIMENSIONS ---")
    try:
        dims = client.discover("MDSCHEMA_DIMENSIONS", {
            "CATALOG_NAME": config.AUTOCUBE_CATALOG,
            "CUBE_NAME": _PRODUCT_CUBE,
        })
        for d in dims:
            log.info("  Dim: %-30s  Type: %s",
                     d.get("DIMENSION_UNIQUE_NAME", "?"),
                     d.get("DIMENSION_TYPE", "?"))
    except Exception:
        log.exception("Failed to list dimensions.")

    log.info("")
    log.info("--- HIERARCHIES ---")
    try:
        hierarchies = client.discover("MDSCHEMA_HIERARCHIES", {
            "CATALOG_NAME": config.AUTOCUBE_CATALOG,
            "CUBE_NAME": _PRODUCT_CUBE,
        })
        for h in hierarchies:
            log.info("  Hierarchy: %-40s  Dimension: %s",
                     h.get("HIERARCHY_UNIQUE_NAME", "?"),
                     h.get("DIMENSION_UNIQUE_NAME", "?"))
    except Exception:
        log.exception("Failed to list hierarchies.")

    log.info("")
    log.info("--- MEASURES ---")
    try:
        measures = client.discover("MDSCHEMA_MEASURES", {
            "CATALOG_NAME": config.AUTOCUBE_CATALOG,
            "CUBE_NAME": _PRODUCT_CUBE,
        })
        for m in measures:
            log.info("  Measure: %-40s  Type: %s  Agg: %s",
                     m.get("MEASURE_UNIQUE_NAME", "?"),
                     m.get("DATA_TYPE", "?"),
                     m.get("MEASURE_AGGREGATOR", "?"))
        log.info("  Total: %d measures", len(measures))
    except Exception:
        log.exception("Failed to list measures.")

    log.info("")
    log.info("--- SAMPLE: Inventory snapshot (first 5 rows) ---")
    try:
        sample_mdx = (
            "SELECT NON EMPTY { [Measures].[On Hand Qty], [Measures].[Min Qty], "
            "[Measures].[Max Qty], [Measures].[Qty On Order] } ON COLUMNS, "
            "NON EMPTY HEAD(CROSSJOIN([Product].[Prod Code].[Prod Code].MEMBERS, "
            "[Location].[Loc].[Loc].MEMBERS), 5) ON ROWS FROM [Product]"
        )
        rows = client.execute_mdx(sample_mdx)
        for i, r in enumerate(rows):
            log.info("  Sample %d: %s", i + 1, r)
    except Exception:
        log.exception("Sample inventory query failed.")

    log.info("")
    log.info("--- SAMPLE: Transfer Locs (first 5 rows) ---")
    try:
        sample_xfer = (
            "SELECT NON EMPTY { [Measures].[Qty Ship], [Measures].[Ext Cost] } ON COLUMNS, "
            "NON EMPTY HEAD(CROSSJOIN([Product].[Prod Code].[Prod Code].MEMBERS, "
            "[Transfer Locs].[Loc From].[Loc From].MEMBERS, "
            "[Transfer Locs].[Loc To].[Loc To].MEMBERS), 5) ON ROWS FROM [Product]"
        )
        rows = client.execute_mdx(sample_xfer)
        for i, r in enumerate(rows):
            log.info("  Sample %d: %s", i + 1, r)
        if not rows:
            log.info("  (no transfer data returned)")
    except Exception:
        log.exception("Sample transfer query failed.")

    log.info("")
    log.info("--- SAMPLE: Product attributes (first 5 rows) ---")
    try:
        sample_attr = (
            "SELECT NON EMPTY { [Measures].[Product Count] } ON COLUMNS, "
            "NON EMPTY HEAD(CROSSJOIN([Product].[Prod Code].[Prod Code].MEMBERS, "
            "[Product].[Vendor 1].[Vendor 1].MEMBERS), 5) ON ROWS FROM [Product]"
        )
        rows = client.execute_mdx(sample_attr)
        for i, r in enumerate(rows):
            log.info("  Sample %d: %s", i + 1, r)
    except Exception:
        log.exception("Sample product attribute query failed.")

    log.info("")
    log.info("--- SAMPLE: Pricing (first 3 rows) ---")
    try:
        sample_price = (
            "SELECT NON EMPTY { [Measures].[Price 1], [Measures].[Cost], "
            "[Measures].[Price 1 GP] } ON COLUMNS, "
            "NON EMPTY HEAD([Product].[Prod Code].[Prod Code].MEMBERS, 3) "
            "ON ROWS FROM [Product]"
        )
        rows = client.execute_mdx(sample_price)
        for i, r in enumerate(rows):
            log.info("  Sample %d: %s", i + 1, r)
    except Exception:
        log.exception("Sample pricing query failed.")

    log.info("")
    log.info("Product cube discovery complete.")
    return 0


def run_inventory_extract(dry_run: bool = False) -> int:
    """Pull current inventory snapshots from the Product cube.

    Iterates ACTIVE_LOCATIONS one at a time, running an MDX query per
    location with a WHERE slicer + NON EMPTY on rows.  This is the only
    pattern that scales: a full Prod Code × Loc crossjoin OOMs the
    Autologue server after ~10 minutes, and the [Prod PN Loc] compound
    hierarchy is too wide for the server to materialize either.

    For each location, only SKUs with non-zero on-hand / min / max /
    on-order / cost are returned (NON EMPTY suppresses the rest).
    """
    t0 = time.perf_counter()
    today = date.today()
    log.info("=" * 60)
    log.info("  PRODUCT CUBE — INVENTORY SNAPSHOT%s", " [DRY RUN]" if dry_run else "")
    log.info("  Locations to query: %d  %s",
             len(ACTIVE_LOCATIONS), ACTIVE_LOCATIONS)
    log.info("=" * 60)

    try:
        client = _get_product_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    cleaned: list[dict[str, Any]] = []
    skipped_unparseable = 0
    failed_locations: list[int] = []
    per_location_counts: dict[int, int] = {}

    for idx, loc_num in enumerate(ACTIVE_LOCATIONS, start=1):
        location_id = f"LOC-{loc_num:03d}"
        mdx = MDX_INVENTORY_BY_LOCATION.format(loc_num=loc_num)
        loc_t0 = time.perf_counter()

        try:
            rows = client.execute_mdx(mdx)
        except Exception:
            log.exception(
                "  [%2d/%d] %s  MDX failed — skipping this location.",
                idx, len(ACTIVE_LOCATIONS), location_id,
            )
            failed_locations.append(loc_num)
            continue

        loc_cleaned = 0
        for r in rows:
            sku = r.get("[Product].[Prod Code].[Prod Code]", "").strip()
            if not sku:
                skipped_unparseable += 1
                continue

            qty_on_hand   = clean_numeric(r.get("[Measures].[On Hand Qty]")) or 0
            qty_on_order  = clean_numeric(r.get("[Measures].[Qty On Order]")) or 0
            unit_cost_oh  = clean_numeric(r.get("[Measures].[Unit Cost on Hand]"))
            catalog_cost  = clean_numeric(r.get("[Measures].[Cost]"))
            ext_cost      = clean_numeric(r.get("[Measures].[Ext Cost On Hand]"))

            # Cost resolution order:
            #   1. Unit Cost on Hand  — true per-unit cost of CURRENT stock
            #   2. Cost               — catalog/master cost (works for SKUs
            #                           with qty=0 too)
            #   3. Ext Cost On Hand / qty — derived (last-resort fallback)
            unit_cost = unit_cost_oh
            if (not unit_cost or unit_cost <= 0) and catalog_cost and catalog_cost > 0:
                unit_cost = catalog_cost
            if (not unit_cost or unit_cost <= 0) and ext_cost and qty_on_hand > 0:
                unit_cost = round(ext_cost / qty_on_hand, 4)

            cleaned.append({
                "sku_id":        sku,
                "location_id":   location_id,
                "snapshot_date": today.isoformat(),
                "qty_on_hand":   qty_on_hand,
                "qty_on_order":  qty_on_order,
                "reorder_point": clean_numeric(r.get("[Measures].[Min Qty]")),
                "reorder_qty":   clean_numeric(r.get("[Measures].[Max Qty]")),
                "unit_cost":     round(unit_cost, 4) if unit_cost is not None else None,
            })
            loc_cleaned += 1

        per_location_counts[loc_num] = loc_cleaned
        log.info(
            "  [%2d/%d] %s  rows=%d  cleaned=%d  (%.1fs)",
            idx, len(ACTIVE_LOCATIONS), location_id,
            len(rows), loc_cleaned, time.perf_counter() - loc_t0,
        )

    log.info(
        "Inventory extract: %d rows cleaned across %d/%d locations  "
        "(skipped %d unparseable, failed locations: %s)",
        len(cleaned),
        len(ACTIVE_LOCATIONS) - len(failed_locations),
        len(ACTIVE_LOCATIONS),
        skipped_unparseable,
        failed_locations or "none",
    )

    if dry_run:
        for i, r in enumerate(cleaned[:5]):
            log.info("[DRY RUN] Row %d: %s", i + 1, r)
        log.info("[DRY RUN] %d rows ready (not loaded).", len(cleaned))
        return 0

    from db.connection import get_client as get_db_client
    db = get_db_client()

    unique_skus = sorted({r["sku_id"] for r in cleaned})
    log.info("Ensuring %d unique SKUs exist in sku_master …", len(unique_skus))
    for i in range(0, len(unique_skus), _BATCH_SIZE):
        batch = [{"sku_id": s} for s in unique_skus[i:i + _BATCH_SIZE]]
        db.table("sku_master").upsert(batch, on_conflict="sku_id", ignore_duplicates=True).execute()
    log.info("sku_master populated.")

    pre_dedup = len(cleaned)
    cleaned = _dedupe_by_keys(cleaned, ("sku_id", "location_id", "snapshot_date"))
    if pre_dedup != len(cleaned):
        log.info(
            "Deduplicated inventory rows: %d → %d (removed %d duplicate keys).",
            pre_dedup, len(cleaned), pre_dedup - len(cleaned),
        )

    loaded = 0
    for i in range(0, len(cleaned), _BATCH_SIZE):
        batch = cleaned[i:i + _BATCH_SIZE]
        try:
            db.table("inventory_snapshots").upsert(
                batch, on_conflict="sku_id,location_id,snapshot_date"
            ).execute()
            loaded += len(batch)
        except Exception:
            log.exception("Upsert failed at batch offset %d.", i)
            raise

    # ----------------------------------------------------------------------
    # Propagate the latest non-null per-unit cost up to sku_master.unit_cost
    # as a denormalized convenience.  This is best-effort: dead_stock now
    # reads inventory_snapshots.unit_cost directly as the source of truth,
    # so a failure here doesn't break costing.  We keep the LAST observed
    # non-null cost per SKU across all locations processed this run
    # (locations are iterated in a stable order, so it is deterministic).
    # NOTE: this block runs at the very end of the extract — if extract
    # crashes mid-loop, sku_master.unit_cost may be stale, but
    # inventory_snapshots was already written per-batch above and remains
    # the authoritative cost source.
    # ----------------------------------------------------------------------
    sku_cost: dict[str, float] = {}
    for r in cleaned:
        c = r.get("unit_cost")
        if c is None or c <= 0:
            continue
        sku_cost[r["sku_id"]] = float(c)

    if sku_cost:
        cost_updates = [
            {"sku_id": sid, "unit_cost": round(c, 4)}
            for sid, c in sku_cost.items()
        ]
        log.info(
            "Propagating unit_cost into sku_master for %d SKU(s) "
            "(%d snapshot row(s) had no cost) …",
            len(cost_updates), len(cleaned) - sum(
                1 for r in cleaned
                if r.get("unit_cost") and r["unit_cost"] > 0
            ),
        )
        cost_loaded = 0
        for i in range(0, len(cost_updates), _BATCH_SIZE):
            batch = cost_updates[i:i + _BATCH_SIZE]
            try:
                db.table("sku_master").upsert(
                    batch, on_conflict="sku_id"
                ).execute()
                cost_loaded += len(batch)
            except Exception:
                log.exception(
                    "sku_master.unit_cost upsert failed at batch offset %d.", i,
                )
                raise
        log.info("sku_master.unit_cost updated for %d SKU(s).", cost_loaded)
    else:
        log.warning(
            "No non-null unit_cost values present in this extract — "
            "sku_master.unit_cost was not updated.",
        )

    elapsed = time.perf_counter() - t0
    log.info("Loaded %d inventory snapshots in %.1fs.", loaded, elapsed)
    return 0


def run_transfer_extract(dry_run: bool = False) -> int:
    t0 = time.perf_counter()
    log.info("=" * 60)
    log.info("  PRODUCT CUBE — TRANSFERS%s", " [DRY RUN]" if dry_run else "")
    log.info("=" * 60)

    try:
        client = _get_product_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    try:
        rows = client.execute_mdx(MDX_TRANSFERS)
        log.info("Transfer rows returned: %d", len(rows))
    except Exception:
        log.exception("Transfer extract failed.")
        return 1

    if not rows:
        log.info("No transfer data returned.")
        return 0

    today = date.today()
    cleaned: list[dict[str, Any]] = []
    for r in rows:
        sku = r.get("[Product].[Prod Code].[Prod Code]", "").strip()
        from_raw = r.get("[Transfer Locs].[Loc From].[Loc From]", "").strip()
        to_raw = r.get("[Transfer Locs].[Loc To].[Loc To]", "").strip()
        if not sku or not from_raw or not to_raw:
            continue
        from_loc = _extract_location_code(from_raw)
        to_loc = _extract_location_code(to_raw)
        qty = clean_numeric(r.get("[Measures].[Qty Ship]")) or 0
        cost = clean_numeric(r.get("[Measures].[Ext Cost]"))
        tid = f"XF-{today.isoformat()}-{sku}-{from_loc}-{to_loc}"
        cleaned.append({
            "transfer_id": tid,
            "sku_id": sku,
            "from_location": from_loc,
            "to_location": to_loc,
            "transfer_date": today.isoformat(),
            "qty_transferred": qty,
            "transfer_cost": cost,
        })

    log.info("Cleaned %d transfer rows.", len(cleaned))

    if dry_run:
        for i, r in enumerate(cleaned[:5]):
            log.info("[DRY RUN] Row %d: %s", i + 1, r)
        log.info("[DRY RUN] %d rows ready (not loaded).", len(cleaned))
        return 0

    from db.connection import get_client as get_db_client
    db = get_db_client()

    unique_skus = sorted({r["sku_id"] for r in cleaned})
    log.info("Ensuring %d unique SKUs exist in sku_master …", len(unique_skus))
    for i in range(0, len(unique_skus), _BATCH_SIZE):
        batch = [{"sku_id": s} for s in unique_skus[i:i + _BATCH_SIZE]]
        db.table("sku_master").upsert(batch, on_conflict="sku_id", ignore_duplicates=True).execute()

    pre_dedup = len(cleaned)
    cleaned = _dedupe_by_keys(cleaned, ("transfer_id",))
    if pre_dedup != len(cleaned):
        log.info(
            "Deduplicated transfer rows: %d → %d (removed %d duplicate keys).",
            pre_dedup, len(cleaned), pre_dedup - len(cleaned),
        )

    loaded = 0
    for i in range(0, len(cleaned), _BATCH_SIZE):
        batch = cleaned[i:i + _BATCH_SIZE]
        try:
            db.table("location_transfers").upsert(
                batch, on_conflict="transfer_id"
            ).execute()
            loaded += len(batch)
        except Exception:
            log.exception("Upsert failed at batch offset %d.", i)
            raise

    elapsed = time.perf_counter() - t0
    log.info("Loaded %d transfer records in %.1fs.", loaded, elapsed)
    return 0


def run_enrich_extract(dry_run: bool = False) -> int:
    t0 = time.perf_counter()
    log.info("=" * 60)
    log.info("  PRODUCT CUBE — SKU MASTER ENRICHMENT%s", " [DRY RUN]" if dry_run else "")
    log.info("=" * 60)

    try:
        client = _get_product_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    log.info("Pulling vendor mapping …")
    try:
        vendor_rows = client.execute_mdx(MDX_PRODUCT_ENRICH)
        log.info("Vendor rows: %d", len(vendor_rows))
    except Exception:
        log.exception("Vendor extract failed.")
        vendor_rows = []

    log.info("Pulling product line mapping …")
    try:
        line_rows = client.execute_mdx(MDX_PRODUCT_ATTRIBUTES)
        log.info("Product line rows: %d", len(line_rows))
    except Exception:
        log.exception("Product line extract failed.")
        line_rows = []

    log.info("Pulling product descriptions …")
    try:
        desc_rows = client.execute_mdx(MDX_PRODUCT_DESC)
        log.info("Description rows: %d", len(desc_rows))
    except Exception:
        log.exception("Description extract failed.")
        desc_rows = []

    enrichment: dict[str, dict[str, Any]] = {}

    for r in vendor_rows:
        sku = r.get("[Product].[Prod Code].[Prod Code]", "").strip()
        vendor = r.get("[Product].[Vendor 1].[Vendor 1]", "").strip()
        if sku and vendor:
            enrichment.setdefault(sku, {})["primary_supplier_id"] = vendor

    for r in line_rows:
        sku = r.get("[Product].[Prod Code].[Prod Code]", "").strip()
        prod_line = r.get("[Product].[Prod Line].[Prod Line]", "").strip()
        if sku and prod_line:
            enrichment.setdefault(sku, {})["product_line"] = prod_line

    # Description rows come back via DIMENSION PROPERTIES, so the SKU is in
    # the MEMBER_NAME slot rather than the usual caption slot.  Skip Autocube
    # filler rows ("---", placeholder "UNKNOWN" / "~").
    _NULLISH = {"", "~", "---", "UNKNOWN"}
    for r in desc_rows:
        sku = (r.get("[Product].[Prod Code].[Prod Code].[MEMBER_NAME]")
               or r.get("[Product].[Prod Code].[Prod Code]")
               or "").strip()
        if not sku or sku in _NULLISH:
            continue
        prod_desc = (r.get("[Product].[Prod Code].[Prod Code].[Prod Desc]") or "").strip()
        ext_desc  = (r.get("[Product].[Prod Code].[Prod Code].[Ext Desc]") or "").strip()
        uom       = (r.get("[Product].[Prod Code].[Prod Code].[UOM]") or "").strip()

        # Prefer the longer Ext Desc when it carries real content; fall
        # back to the short Prod Desc otherwise.
        description = ext_desc if ext_desc and ext_desc not in _NULLISH else prod_desc
        if description and description not in _NULLISH:
            enrichment.setdefault(sku, {})["description"] = description[:500]
        if uom and uom not in _NULLISH:
            enrichment.setdefault(sku, {})["unit_of_measure"] = uom[:16]

    if not enrichment:
        log.info("No enrichment data to apply.")
        return 0

    log.info("Enrichment data for %d SKUs.", len(enrichment))

    if dry_run:
        samples = list(enrichment.items())[:5]
        for sku, attrs in samples:
            log.info("[DRY RUN] %s → %s", sku, attrs)
        log.info("[DRY RUN] %d SKUs ready (not loaded).", len(enrichment))
        return 0

    from db.connection import get_client as get_db_client
    db = get_db_client()

    updates = [{"sku_id": sku, **attrs} for sku, attrs in enrichment.items()]

    loaded = 0
    for i in range(0, len(updates), _BATCH_SIZE):
        batch = updates[i:i + _BATCH_SIZE]
        try:
            db.table("sku_master").upsert(
                batch, on_conflict="sku_id"
            ).execute()
            loaded += len(batch)
        except Exception:
            log.exception("sku_master enrichment upsert failed at offset %d.", i)
            raise

    elapsed = time.perf_counter() - t0
    log.info("Enriched %d SKUs in %.1fs.", loaded, elapsed)
    return 0


def run_pricing_extract(dry_run: bool = False) -> int:
    t0 = time.perf_counter()
    today = date.today()
    log.info("=" * 60)
    log.info("  PRODUCT CUBE — PRICING TIERS%s", " [DRY RUN]" if dry_run else "")
    log.info("=" * 60)

    try:
        client = _get_product_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    try:
        rows = client.execute_mdx(MDX_PRICING)
        log.info("Pricing rows returned: %d", len(rows))
    except Exception:
        log.exception("Pricing extract failed.")
        return 1

    if not rows:
        log.info("No pricing data returned.")
        return 0

    cleaned: list[dict[str, Any]] = []
    for r in rows:
        sku = r.get("[Product].[Prod Code].[Prod Code]", "").strip()
        if not sku:
            continue
        unit_cost = clean_numeric(r.get("[Measures].[Cost]"))
        for tier in _PRICE_TIERS:
            price_val = clean_numeric(r.get(f"[Measures].[Price {tier}]"))
            gp_val = clean_numeric(r.get(f"[Measures].[Price {tier} GP]"))
            if price_val is not None and price_val != 0:
                cleaned.append({
                    "sku_id": sku,
                    "price_tier": tier,
                    "price_value": round(price_val, 4),
                    "gp_margin_pct": round(gp_val, 4) if gp_val is not None else None,
                    "unit_cost": round(unit_cost, 4) if unit_cost is not None else None,
                    "snapshot_date": today.isoformat(),
                })

    log.info("Cleaned %d pricing tier rows.", len(cleaned))

    if dry_run:
        for i, r in enumerate(cleaned[:10]):
            log.info("[DRY RUN] Row %d: %s", i + 1, r)
        log.info("[DRY RUN] %d rows ready (not loaded).", len(cleaned))
        return 0

    from db.connection import get_client as get_db_client
    db = get_db_client()

    unique_skus = sorted({r["sku_id"] for r in cleaned})
    log.info("Ensuring %d unique SKUs exist in sku_master …", len(unique_skus))
    for i in range(0, len(unique_skus), _BATCH_SIZE):
        batch = [{"sku_id": s} for s in unique_skus[i:i + _BATCH_SIZE]]
        db.table("sku_master").upsert(batch, on_conflict="sku_id", ignore_duplicates=True).execute()

    pre_dedup = len(cleaned)
    cleaned = _dedupe_by_keys(cleaned, ("sku_id", "price_tier", "snapshot_date"))
    if pre_dedup != len(cleaned):
        log.info(
            "Deduplicated pricing rows: %d → %d (removed %d duplicate keys).",
            pre_dedup, len(cleaned), pre_dedup - len(cleaned),
        )

    loaded = 0
    for i in range(0, len(cleaned), _BATCH_SIZE):
        batch = cleaned[i:i + _BATCH_SIZE]
        try:
            db.table("sku_pricing_tiers").upsert(
                batch, on_conflict="sku_id,price_tier,snapshot_date"
            ).execute()
            loaded += len(batch)
        except Exception:
            log.exception("Pricing upsert failed at offset %d.", i)
            raise

    elapsed = time.perf_counter() - t0
    log.info("Loaded %d pricing tier rows in %.1fs.", loaded, elapsed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Product cube extraction from Autocube OLAP via XMLA/SOAP",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Discover dimensions, hierarchies, measures + sample data",
    )
    parser.add_argument(
        "--mode",
        choices=["inventory", "full", "transfers", "enrich", "pricing", "all"],
        help="Extraction mode.  'full' is an alias for 'inventory'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and clean but do not write to Supabase",
    )
    args = parser.parse_args()

    if not args.test and not args.mode:
        parser.print_help()
        return 1

    if args.test:
        return run_test()

    if args.mode == "all":
        rc = 0
        for mode_fn, label in [
            (run_inventory_extract, "inventory"),
            (run_transfer_extract, "transfers"),
            (run_enrich_extract, "enrich"),
            (run_pricing_extract, "pricing"),
        ]:
            log.info("Running mode: %s …", label)
            result = mode_fn(dry_run=args.dry_run)
            if result != 0:
                log.error("Mode %s failed with rc=%d — continuing.", label, result)
                rc = 1
        return rc

    mode_map = {
        "inventory": run_inventory_extract,
        "full":      run_inventory_extract,  # alias
        "transfers": run_transfer_extract,
        "enrich":    run_enrich_extract,
        "pricing":   run_pricing_extract,
    }
    return mode_map[args.mode](dry_run=args.dry_run)


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
