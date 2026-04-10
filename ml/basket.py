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

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

log = logging.getLogger(__name__)

MIN_SUPPORT    = 0.001
MIN_CONFIDENCE = 0.3
MIN_LIFT       = 1.5
HIGH_CONF      = 0.70
LOOKBACK_DAYS  = 180
MAX_BASKET_SIZE = 50
_PAGE_SIZE     = 1000


def _paginate(client, table, select, filters=None, gte_filters=None,
              lte_filters=None, order_col=None, order_desc=True, limit=None):
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
        if order_col:
            q = q.order(order_col, desc=order_desc)
        end = offset + _PAGE_SIZE - 1
        if limit and offset + _PAGE_SIZE > limit:
            end = offset + (limit - offset) - 1
        q = q.range(offset, end)
        resp = q.execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
        if limit and len(rows) >= limit:
            break
    return rows


def _build_baskets(client, lookback_start: str) -> list[list[str]]:
    log.info("Fetching transactions since %s …", lookback_start)

    rows = _paginate(
        client, "sales_transactions",
        "transaction_id,transaction_date,location_id,sku_id",
        gte_filters={"transaction_date": lookback_start},
    )
    log.info("  Fetched %d transaction rows.", len(rows))
    if not rows:
        return []

    groups: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        tid = r.get("transaction_id")
        if tid:
            key = str(tid)
        else:
            tdate = r.get("transaction_date") or "unknown"
            loc = r.get("location_id") or "unknown"
            key = f"{tdate}|{loc}"
        sku = r.get("sku_id")
        if sku:
            groups[key].add(str(sku))

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

    log.info("=" * 60)
    log.info("BASKET ANALYSIS — co-purchase association rules")
    log.info("  Lookback: %d days (since %s)", LOOKBACK_DAYS, lookback_start)
    log.info("  Thresholds: support≥%.4f  confidence≥%.2f  lift≥%.1f",
             MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT)
    log.info("=" * 60)

    client = get_client()

    baskets = _build_baskets(client, lookback_start)
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
