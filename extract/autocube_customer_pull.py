"""extract/autocube_customer_pull.py — Customer master from Sales Summary cube.

The Sales Detail cube does NOT carry [Salesman], [Cust Type], or [Status]
attributes on its [Customer] dimension, but the Sales Summary cube does.
This module pulls those slowly-changing attributes per customer and upserts
them into the customer_master table (migration 028) so churn.py and other
engines can route per-customer recommendations to the right salesman.

Usage:
    python -m extract.autocube_customer_pull               # full pull
    python -m extract.autocube_customer_pull --dry-run     # show counts only

Security:
    Read-only MDX SELECTs against a non-PII dimension table.  No PII in flight.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from typing import Any

import config  # noqa: F401  (ensures env loaded the same way as siblings)
from extract.autocube_pull import get_client
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

_SALES_SUMMARY_CUBE = "Sales Summary"
_BATCH_SIZE = 500

# Each MDX query keeps Cust No on rows and crosses one attribute hierarchy
# at a time.  Crossing all four attributes in a single query risks the same
# memory blow-up we see on Product cube extracts; one-attribute-at-a-time is
# both safer and easier to debug.
_ATTR_QUERIES: list[tuple[str, str, str]] = [
    # (output_field, hierarchy, level_path)
    ("salesman_id",      "[Customer].[Salesman]",  "[Customer].[Salesman].[Salesman]"),
    ("customer_type",    "[Customer].[Cust Type]", "[Customer].[Cust Type].[Cust Type]"),
    ("customer_status",  "[Customer].[Status]",    "[Customer].[Status].[Status]"),
]


def _build_mdx(level_path: str) -> str:
    return (
        "SELECT NON EMPTY { [Measures].[Cust Count] } ON COLUMNS, "
        "NON EMPTY CROSSJOIN("
        "[Customer].[Cust No].[Cust No].MEMBERS, "
        f"{level_path}.MEMBERS"
        ") ON ROWS "
        "FROM [Sales Summary]"
    )


def _strip(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip()
    if s in {"~", "---", "UNKNOWN", "N/A"}:
        return ""
    return s


def run_customer_master(dry_run: bool = False) -> int:
    t0 = time.perf_counter()
    log.info("=" * 60)
    log.info("  SALES SUMMARY — CUSTOMER MASTER%s", " [DRY RUN]" if dry_run else "")
    log.info("=" * 60)

    try:
        client = get_client()
        client._cube = _SALES_SUMMARY_CUBE
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    enrichment: dict[str, dict[str, Any]] = {}

    for output_field, hierarchy, level_path in _ATTR_QUERIES:
        log.info("Pulling %s via %s …", output_field, hierarchy)
        mdx = _build_mdx(level_path)
        try:
            rows = client.execute_mdx(mdx)
        except Exception:
            log.exception("Failed to pull %s — continuing.", output_field)
            continue
        kept = 0
        cust_key = "[Customer].[Cust No].[Cust No]"
        attr_key = f"{level_path}"
        for r in rows:
            cust_id = _strip(r.get(cust_key))
            attr_val = _strip(r.get(attr_key))
            if not cust_id:
                continue
            entry = enrichment.setdefault(cust_id, {"customer_id": cust_id})
            if attr_val:
                entry[output_field] = attr_val[:50]
                kept += 1
        log.info("  → %d rows; %d non-empty %s values", len(rows), kept, output_field)

    if not enrichment:
        log.info("No customer attributes returned — nothing to load.")
        return 0

    today = date.today().isoformat()
    records: list[dict[str, Any]] = []
    for cust_id, attrs in enrichment.items():
        records.append({
            "customer_id":     cust_id[:50],
            "salesman_id":     attrs.get("salesman_id"),
            "customer_type":   attrs.get("customer_type"),
            "customer_status": (attrs.get("customer_status") or "")[:20] or None,
            "last_updated":    today,
        })

    # Coverage report — these counts answer "how complete is each attribute?"
    total = len(records)
    cov_sales = sum(1 for r in records if r["salesman_id"])
    cov_type  = sum(1 for r in records if r["customer_type"])
    cov_stat  = sum(1 for r in records if r["customer_status"])
    log.info("Customers discovered: %d", total)
    log.info("  with salesman_id    : %d (%.1f%%)", cov_sales, 100 * cov_sales / total)
    log.info("  with customer_type  : %d (%.1f%%)", cov_type,  100 * cov_type  / total)
    log.info("  with customer_status: %d (%.1f%%)", cov_stat,  100 * cov_stat  / total)

    if dry_run:
        for r in records[:5]:
            log.info("[DRY RUN] %s", r)
        log.info("[DRY RUN] %d customer rows ready (not loaded).", total)
        log.info("Dry run complete in %.1fs.", time.perf_counter() - t0)
        return 0

    from db.connection import get_client as get_db_client
    db = get_db_client()

    loaded = 0
    for i in range(0, total, _BATCH_SIZE):
        batch = records[i:i + _BATCH_SIZE]
        try:
            db.table("customer_master").upsert(batch, on_conflict="customer_id").execute()
            loaded += len(batch)
        except Exception:
            log.exception("customer_master upsert failed at offset %d.", i)
            raise

    log.info("Loaded %d customer_master rows in %.1fs.", loaded, time.perf_counter() - t0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull customer dimension (salesman/type/status) from Sales Summary cube",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract and report coverage but do not write to Supabase")
    args = parser.parse_args()
    return run_customer_master(dry_run=args.dry_run)


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
