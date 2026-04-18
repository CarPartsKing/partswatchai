"""ml/basket.py — Co-purchase basket analysis using mlxtend Apriori.

Finds which SKUs are almost always bought together so we can pre-stage
related inventory and detect demand for SKU B when SKU A moves.

PIPELINE POSITION
-----------------
Weekly stage.  Consumes:
  - sales_transactions (grouped by transaction_id → baskets)

Writes to:
  - basket_rules (one row per antecedent → consequent pair)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from db.connection import get_client, get_new_client
from utils.logging_config import get_logger, setup_logging

log = logging.getLogger(__name__)

MIN_SUPPORT    = 0.001
MIN_CONFIDENCE = 0.3
MIN_LIFT       = 1.5
HIGH_CONF      = 0.70
# Restored to 90 days after confirming the SKU-batched fetch (1000-SKU
# batches with `.in_("sku_id", ...)` + per-batch offset paging) reliably
# stays inside Supabase's statement_timeout on the ~8.4M-row table.
# 90 days gives Apriori a much fuller signal — seasonal, weekly, and
# slower-cadence pairs that were missed at 30.
LOOKBACK_DAYS  = 90
MAX_BASKET_SIZE = 50
_PAGE_SIZE     = 1000
_SKU_BATCH_SIZE = 1000   # mirrors forecast_rolling / dead_stock — keeps each
                         # `.in_("sku_id", batch)` query under timeout.
_PROGRESS_EVERY = 50_000
_MAX_RETRIES    = 5
_RETRY_DELAY    = 5.0


def _is_statement_timeout(exc: Exception) -> bool:
    """Detect Postgres 57014 (statement_timeout) inside Supabase errors."""
    s = str(exc)
    return "57014" in s or "statement timeout" in s.lower() or "canceling statement" in s.lower()


def _fetch_all_sku_ids(client) -> list[str]:
    """Return every sku_id from sku_master, fully paged.

    Drives the SKU-batched transaction fetch below — we need the universe
    of SKU IDs to slice `.in_("sku_id", batch)` over.
    """
    skus: list[str] = []
    offset = 0
    while True:
        # Deterministic order is REQUIRED for stable offset paging — without
        # it, Postgres is free to return rows in any order and concurrent
        # writes / plan changes can cause pages to skip or duplicate rows.
        resp = (
            client.table("sku_master")
            .select("sku_id")
            .order("sku_id", desc=False)
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        page = resp.data or []
        if not page:
            break
        skus.extend(r["sku_id"] for r in page if r.get("sku_id"))
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return skus


def _fetch_transactions_for_skus(
    client_holder: list,
    date_lo: str,
    date_hi: str,
    sku_batch: list[str],
) -> list[dict]:
    """Fetch all transactions in [date_lo, date_hi] for a batch of SKUs.

    Mirrors the SKU-batch pattern proven in `forecast_rolling._fetch_transactions_for_skus`
    and `dead_stock` — restricting each query to ≤ _SKU_BATCH_SIZE SKUs keeps
    the planner inside Supabase's statement_timeout (57014) on the 8.4M-row
    sales_transactions table where the previous keyset-by-date approach
    timed out at 90 days and still struggled at 30.

    We page within each SKU batch with `.range(offset, offset+PAGE-1)` because
    Supabase caps any single response at 1000 rows; a 1000-SKU × 30-day slice
    can still return many thousands of rows for high-velocity SKUs.

    On 57014 we reconnect with a fresh client and retry with backoff.
    """
    select = "invoice_number,transaction_date,location_id,sku_id"
    rows: list[dict] = []
    offset = 0
    while True:
        page: list[dict] | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # `.order("id")` is REQUIRED — offset paging without an
                # explicit deterministic order can skip or duplicate rows
                # under concurrent writes or planner changes.
                page = (
                    client_holder[0].table("sales_transactions")
                    .select(select)
                    .in_("sku_id", sku_batch)
                    .gte("transaction_date", date_lo)
                    .lte("transaction_date", date_hi)
                    .order("id", desc=False)
                    .range(offset, offset + _PAGE_SIZE - 1)
                    .execute()
                    .data
                    or []
                )
                break
            except Exception as exc:
                if _is_statement_timeout(exc) and attempt < _MAX_RETRIES:
                    backoff = min(2 ** attempt, 30)
                    log.warning(
                        "  tx fetch 57014 retry %d/%d (offset=%d, %d SKUs) — "
                        "reconnecting in %ds …",
                        attempt, _MAX_RETRIES, offset, len(sku_batch), backoff,
                    )
                    client_holder[0] = get_new_client()
                    time.sleep(backoff)
                    continue
                if attempt < _MAX_RETRIES:
                    log.warning(
                        "  tx fetch transient '%s' (offset=%d, %d SKUs) — "
                        "retrying in %.0fs …",
                        type(exc).__name__, offset, len(sku_batch), _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                    continue
                raise
        assert page is not None
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def _build_baskets(client, lookback_start: str, lookback_end: str) -> list[list[str]]:
    """Group sales lines into baskets keyed by REAL invoice number.

    `transaction_id` is synthesized from (date, sku, location) so it is NOT
    a valid basket key — every synthetic transaction contains exactly one
    SKU by construction.  The real grouping key is `invoice_number`, which
    is populated by migration 022 from the Sales Detail cube's
    [Invoice Nbr] dimension.  Rows with `invoice_number IS NULL` are legacy
    rows from before migration 022 and cannot participate in basket analysis.
    To scope the basket key to a single store-day (invoice numbers can repeat
    across locations and recycle over time) we key on
    (transaction_date, location_id, invoice_number).

    Fetch strategy (architect/user-mandated):
      1. Pull every sku_id from sku_master.
      2. For each batch of _SKU_BATCH_SIZE SKUs, fetch transactions in the
         date window.  Accumulate raw rows across all batches.
      3. Group all rows by (date, location, invoice_number) at the end.

    NOTE: We deliberately do NOT finalize baskets per SKU-batch — invoices
    routinely contain SKUs that fall into different batches, and prematurely
    grouping would split co-occurrences and destroy the very signal Apriori
    is looking for.  Grouping happens once, after all rows are in memory.
    """
    log.info("Fetching transactions in [%s, %s] (last %d days) "
             "via SKU-batched approach …",
             lookback_start, lookback_end, LOOKBACK_DAYS)

    client_holder = [client]

    # ── Step 1: enumerate SKU universe ────────────────────────────────────
    all_skus = _fetch_all_sku_ids(client)
    log.info("  sku_master universe: %d SKUs", len(all_skus))
    if not all_skus:
        return []

    # ── Step 2: batched transaction fetch ─────────────────────────────────
    n_batches = (len(all_skus) + _SKU_BATCH_SIZE - 1) // _SKU_BATCH_SIZE
    rows: list[dict] = []
    t_start = time.monotonic()
    next_log_threshold = _PROGRESS_EVERY
    for b_idx in range(n_batches):
        b_start = b_idx * _SKU_BATCH_SIZE
        sku_batch = all_skus[b_start: b_start + _SKU_BATCH_SIZE]
        batch_rows = _fetch_transactions_for_skus(
            client_holder, lookback_start, lookback_end, sku_batch,
        )
        rows.extend(batch_rows)
        if len(rows) >= next_log_threshold or b_idx == n_batches - 1:
            elapsed = time.monotonic() - t_start
            rate = len(rows) / elapsed if elapsed > 0 else 0
            log.info("  batch %d/%d  skus=%d  rows_in_batch=%d  total=%d  "
                     "(%.0f rows/s, %.1fs)",
                     b_idx + 1, n_batches, len(sku_batch), len(batch_rows),
                     len(rows), rate, elapsed)
            while next_log_threshold <= len(rows):
                next_log_threshold += _PROGRESS_EVERY

    log.info("  Fetched %d transaction rows total across %d SKU batches.",
             len(rows), n_batches)
    if not rows:
        return []

    groups: dict[tuple, set[str]] = defaultdict(set)
    no_invoice = 0
    for r in rows:
        inv = r.get("invoice_number")
        sku = r.get("sku_id")
        if not sku:
            continue
        if not inv:
            no_invoice += 1
            continue
        key = (r.get("transaction_date"), r.get("location_id"), str(inv))
        groups[key].add(str(sku))

    if no_invoice:
        pct = 100.0 * no_invoice / len(rows)
        log.warning("  %d rows (%.1f%%) had no invoice_number and were "
                    "excluded — these are legacy rows from before "
                    "migration 022 / the historical re-extract.",
                    no_invoice, pct)

    baskets = [
        list(skus) for skus in groups.values()
        if 2 <= len(skus) <= MAX_BASKET_SIZE
    ]
    log.info("  Built %d baskets (from %d groups, filtered 1-item and >%d-item).",
             len(baskets), len(groups), MAX_BASKET_SIZE)
    return baskets


def _run_apriori(baskets: list[list[str]]) -> pd.DataFrame:
    log.info("Running Apriori (min_support=%.4f) …", MIN_SUPPORT)

    te = TransactionEncoder()
    te_array = te.fit(baskets).transform(baskets)
    df = pd.DataFrame(te_array, columns=te.columns_)

    frequent = apriori(df, min_support=MIN_SUPPORT, use_colnames=True, max_len=2)
    log.info("  Frequent itemsets found: %d", len(frequent))
    if frequent.empty:
        return pd.DataFrame()

    rules = association_rules(frequent, metric="confidence",
                              min_threshold=MIN_CONFIDENCE, num_itemsets=len(frequent))
    rules = rules[rules["lift"] >= MIN_LIFT].copy()
    rules.sort_values("lift", ascending=False, inplace=True)

    log.info("  Association rules after filtering (confidence≥%.1f%%, lift≥%.1f): %d",
             MIN_CONFIDENCE * 100, MIN_LIFT, len(rules))
    return rules


def _write_rules(client, rules_df: pd.DataFrame, basket_count: int,
                 today: date, dry_run: bool) -> int:
    if rules_df.empty:
        log.info("No rules to write.")
        return 0

    records = []
    for _, row in rules_df.iterrows():
        ant = list(row["antecedents"])
        con = list(row["consequents"])
        if len(ant) != 1 or len(con) != 1:
            continue
        records.append({
            "antecedent_sku":    ant[0],
            "consequent_sku":    con[0],
            "support":           round(float(row["support"]), 6),
            "confidence":        round(float(row["confidence"]), 4),
            "lift":              round(float(row["lift"]), 4),
            "rule_date":         today.isoformat(),
            "transaction_count": basket_count,
        })

    log.info("Prepared %d pairwise rules for upsert.", len(records))
    if dry_run:
        log.info("  DRY RUN — skipping database write.")
        return len(records)

    batch_size = 500
    written = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            resp = (
                client.table("basket_rules")
                .upsert(batch, on_conflict="antecedent_sku,consequent_sku,rule_date")
                .execute()
            )
            written += len(resp.data or [])
        except Exception:
            log.exception("Failed to write basket_rules batch %d–%d.",
                          i, i + len(batch))
            return -1
    log.info("  Wrote %d rules to basket_rules.", written)
    return written


def run_basket(dry_run: bool = False) -> int:
    t0 = time.monotonic()
    today = date.today()
    lookback_start = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    lookback_end = today.isoformat()

    log.info("=" * 60)
    log.info("BASKET ANALYSIS — co-purchase association rules")
    log.info("  Lookback: %d days  [%s .. %s]",
             LOOKBACK_DAYS, lookback_start, lookback_end)
    log.info("  Thresholds: support≥%.4f  confidence≥%.2f  lift≥%.1f",
             MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT)
    log.info("=" * 60)

    client = get_client()

    baskets = _build_baskets(client, lookback_start, lookback_end)
    if not baskets:
        log.warning("No multi-item baskets found — skipping Apriori.")
        return 0

    rules_df = _run_apriori(baskets)

    rows_written = _write_rules(client, rules_df, len(baskets), today, dry_run)
    if rows_written < 0:
        return 1

    high_conf = len(rules_df[rules_df["confidence"] >= HIGH_CONF]) if not rules_df.empty else 0

    elapsed = time.monotonic() - t0
    log.info("=" * 60)
    log.info("Basket analysis complete  (%.2fs)", elapsed)
    log.info("  Total baskets analyzed:         %d", len(baskets))
    log.info("  Association rules found:        %d", len(rules_df))
    log.info("  High-confidence rules (>%.0f%%):  %d", HIGH_CONF * 100, high_conf)
    if not rules_df.empty:
        top10 = rules_df.head(10)
        log.info("  Top 10 rules by lift:")
        for _, r in top10.iterrows():
            ant = list(r["antecedents"])[0]
            con = list(r["consequents"])[0]
            log.info("    %s → %s  (conf=%.0f%% lift=%.1fx supp=%.4f)",
                     ant, con, r["confidence"] * 100, r["lift"], r["support"])
    log.info("  Rules written to DB:            %d%s",
             rows_written, "  (DRY RUN)" if dry_run else "")
    log.info("=" * 60)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="partswatch-ai co-purchase basket analysis",
    )
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Compute rules but do not write to the database.")
    return parser.parse_args()


def main() -> int:
    try:
        from config import LOG_LEVEL
        setup_logging(LOG_LEVEL)
    except (ImportError, EnvironmentError):
        setup_logging("INFO")

    args = _parse_args()
    return run_basket(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
