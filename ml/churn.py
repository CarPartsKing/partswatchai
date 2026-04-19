"""
ml/churn.py — Customer churn risk scoring (RFM-style).

Pulls the last LOOKBACK_DAYS days of sales_transactions, aggregates per
customer (one invoice = one "order"), computes recency / frequency-trend /
monetary-trend sub-scores, combines them into a single 0-100 churn risk
score, classifies into HIGH / MEDIUM / LOW tiers, and writes one row per
scored customer to customer_churn_scores.

WHY RFM
    Recency, Frequency, and Monetary are the canonical signals for "is
    this customer about to leave?" — we don't have a labeled churn
    dataset to train a model against, so a transparent rule-based score
    is both more defensible to the buyer team and more debuggable than a
    black-box classifier.  Each sub-score lives in its own column so the
    dashboard can show exactly WHY a customer was flagged.

SCORING (per customer, all in 0-100, higher = more at risk)
    recency_score   = days_since_last_order capped at RECENCY_FULL_DAYS
                      (e.g. 50 days idle → 100; 0 days idle → 0)

    frequency_score = 100 × (1 - order_count_30d / max(1, expected_30d))
                      where expected_30d = order_count_prior_60d / 2
                      (last 30 days vs the same-length stretch before that).
                      Clamped to [0, 100].  A customer ordering at half
                      the prior cadence scores ~50; complete halt → 100.

    monetary_score  = same shape as frequency_score but on revenue dollars.

    churn_score     = 0.4 * recency_score
                    + 0.3 * frequency_score
                    + 0.3 * monetary_score

    Recency is weighted highest because it is the most reliable single
    signal — a customer who has not purchased in 60 days is almost
    certainly disengaged regardless of what their prior trend looked like.

CLASSIFICATION
    HIGH    churn_score ≥ 70   →  at_risk_flag = TRUE
    MEDIUM  40 ≤ score < 70
    LOW     score < 40

ELIGIBILITY FILTERS (applied during the sales_transactions sweep)
    customer_id IS NOT NULL                — drop legacy / unscorable rows
    customer_status NOT IN ('Inactive')    — already-gone accounts aren't
                                              "churning", they're done.
                                              NULL status is INCLUDED (we
                                              don't yet know the cube's
                                              status enum exhaustively).
    is_residual_demand = FALSE             — third-call store traffic
    is_anomaly         = FALSE             — mean-shifted noise
    is_warranty        = FALSE             — replacement, not a real sale
    is_core_return     = FALSE             — negative-qty returns

    Customers with fewer than MIN_ORDERS_TO_SCORE distinct invoices in
    the lookback window are dropped — there's no signal in 1-2 orders.

USAGE
    python -m ml.churn               # full run
    python -m ml.churn --dry-run     # score only, no DB writes
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tunable constants — adjust without touching logic
# ---------------------------------------------------------------------------

LOOKBACK_DAYS:           int   = 90
"""Days of sales history considered for scoring.  90 days gives the
recency / frequency-trend / monetary-trend signals enough room to
diverge while staying inside what the extract pipeline currently has
populated with customer_id (migration 025 was applied 2026-04-19, so
older rows have NULL customer_id and are filtered out anyway)."""

TREND_WINDOW_DAYS:       int   = 30
"""Length of the "recent" window for frequency / monetary trends.
order_count_30d / revenue_30d are compared against the corresponding
prior 60-day window (LOOKBACK_DAYS - TREND_WINDOW_DAYS), normalized
so the comparison is per-30-days on both sides."""

RECENCY_FULL_DAYS:       int   = 50
"""Days idle at which recency_score reaches 100.  Set just below
TREND_WINDOW_DAYS + a small buffer — a customer who has not purchased
in 50+ days is materially disengaged for our weekly cadence."""

MIN_ORDERS_TO_SCORE:     int   = 3
"""Minimum distinct invoices in the lookback window required for a
customer to be scored.  Below this threshold the trend math is too
noisy to be meaningful (one "skipped" order would flip the score)."""

WEIGHT_RECENCY:          float = 0.40
WEIGHT_FREQUENCY:        float = 0.30
WEIGHT_MONETARY:         float = 0.30
"""Composite weights for churn_score.  MUST sum to 1.0."""

HIGH_RISK_THRESHOLD:     float = 70.0
MEDIUM_RISK_THRESHOLD:   float = 40.0
"""Tier cutoffs.  HIGH triggers at_risk_flag = TRUE."""

# customer_status values that mean "this account is already gone, don't
# pretend it's at risk".  Compared case-insensitively.  NULL status is
# treated as Active because we have not yet enumerated every status code
# the cube emits — exclude only what we are SURE is not active.
INACTIVE_STATUSES: frozenset[str] = frozenset({"INACTIVE", "CLOSED", "DELETED"})

PAGE_SIZE:               int   = 1_000
WRITE_BATCH_SIZE:        int   = 200
"""How many score rows to upsert per round-trip.  Same pattern as
engine/reorder.py — keeps each write inside Supabase's statement
timeout (error 57014)."""


# ---------------------------------------------------------------------------
# Per-customer aggregator
# ---------------------------------------------------------------------------

@dataclass
class _CustomerAgg:
    """Running aggregate for one customer over the lookback window.

    Only stores the minimum needed for the final score:
      * customer_type / customer_status — last-seen value (cube emits the
        same value on every row for a given customer; "last wins" is
        equivalent to "any wins" here).
      * last_date / first_date — for recency and tenure (tenure unused
        today, kept for future feature work).
      * orders_30d / orders_prior_60d — sets of invoice numbers; we use
        sets so duplicate sales_transactions rows for the same invoice
        (multi-line invoices) collapse into one "order".
      * revenue_30d / revenue_prior_60d — summed Ext Price.
    """
    customer_type:     str | None = None
    customer_status:   str | None = None
    last_date:         date | None = None
    first_date:        date | None = None
    orders_30d:        set         = None
    orders_prior_60d:  set         = None
    orders_all:        set         = None
    revenue_30d:       float       = 0.0
    revenue_prior_60d: float       = 0.0
    revenue_all:       float       = 0.0

    def __post_init__(self) -> None:
        if self.orders_30d is None:
            self.orders_30d = set()
        if self.orders_prior_60d is None:
            self.orders_prior_60d = set()
        if self.orders_all is None:
            self.orders_all = set()


# ---------------------------------------------------------------------------
# Data fetch — paginated date-windowed sweep of sales_transactions
# ---------------------------------------------------------------------------

def _fetch_and_aggregate(
    client: Any,
    today: date,
) -> dict[str, _CustomerAgg]:
    """Stream sales_transactions for the lookback window and aggregate per customer.

    Pages through sales_transactions filtered to the lookback window using
    PostgREST keyset paging on (transaction_date, transaction_id) so the
    result is deterministic across page boundaries.  Each row is fed into
    the customer's _CustomerAgg incrementally, so peak memory is bounded
    by `distinct customers × ~few KB` rather than `total transaction rows
    × ~few hundred bytes`.

    Filters applied server-side keep the network volume small:
      * transaction_date in [today - LOOKBACK_DAYS, today]
      * customer_id IS NOT NULL  (drops legacy pre-migration-025 rows)
      * is_anomaly = FALSE, is_residual_demand = FALSE,
        is_warranty = FALSE, is_core_return = FALSE

    Returns dict mapping customer_id → _CustomerAgg.
    """
    window_start = today - timedelta(days=LOOKBACK_DAYS)
    trend_start  = today - timedelta(days=TREND_WINDOW_DAYS)

    log.info(
        "Sweeping sales_transactions  window=[%s .. %s]  "
        "trend_split=%s  page=%d",
        window_start.isoformat(), today.isoformat(),
        trend_start.isoformat(), PAGE_SIZE,
    )

    select_cols = (
        "transaction_id,transaction_date,customer_id,customer_type,"
        "customer_status,invoice_number,total_revenue"
    )

    aggs: dict[str, _CustomerAgg] = defaultdict(_CustomerAgg)
    rows_seen = 0
    rows_kept = 0
    page = 0

    # Keyset cursor — (transaction_date, transaction_id) gives a stable
    # ordering even across rows sharing a date.  We read PAGE_SIZE rows,
    # then bump the cursor to the last row's (date, id) pair.
    cursor_date: str | None = None
    cursor_tid:  str | None = None

    while True:
        page += 1
        q = (
            client.table("sales_transactions")
            .select(select_cols)
            .gte("transaction_date", window_start.isoformat())
            .lte("transaction_date", today.isoformat())
            .not_.is_("customer_id", "null")
            .eq("is_anomaly",         False)
            .eq("is_residual_demand", False)
            .eq("is_warranty",        False)
            .eq("is_core_return",     False)
            .order("transaction_date", desc=False)
            .order("transaction_id",   desc=False)
            .limit(PAGE_SIZE)
        )
        if cursor_date is not None:
            # Strict-after: rows with the same date AND a greater tid, OR a
            # later date entirely.  PostgREST doesn't expose tuple > so we
            # emulate with two clauses combined via .or_.
            q = q.or_(
                f"transaction_date.gt.{cursor_date},"
                f"and(transaction_date.eq.{cursor_date},transaction_id.gt.{cursor_tid})"
            )

        try:
            rows = q.execute().data or []
        except Exception:
            log.exception("Page %d fetch failed at cursor (%s, %s).",
                          page, cursor_date, cursor_tid)
            raise

        if not rows:
            break

        rows_seen += len(rows)
        for r in rows:
            cust = r.get("customer_id")
            if not cust:
                continue  # Belt-and-suspenders; the .not_.is_ filter should catch this.

            status = r.get("customer_status")
            if status and status.strip().upper() in INACTIVE_STATUSES:
                continue  # Skip already-inactive accounts.

            tx_date_str = r.get("transaction_date")
            if not tx_date_str:
                continue
            try:
                tx_date = date.fromisoformat(tx_date_str[:10])
            except (ValueError, TypeError):
                continue

            invoice = r.get("invoice_number")
            # No invoice → can't dedupe order count, fall back to transaction_id
            # so each row counts as its own "order".  This is a graceful
            # degradation for legacy pre-migration-022 rows that may still
            # be in the table.
            order_key = invoice or r.get("transaction_id") or f"{cust}-{tx_date_str}"

            revenue = float(r.get("total_revenue") or 0.0)

            agg = aggs[cust]
            if agg.customer_type is None and r.get("customer_type"):
                agg.customer_type = str(r["customer_type"])
            if agg.customer_status is None and status:
                agg.customer_status = str(status)
            if agg.first_date is None or tx_date < agg.first_date:
                agg.first_date = tx_date
            if agg.last_date is None or tx_date > agg.last_date:
                agg.last_date = tx_date

            agg.orders_all.add(order_key)
            agg.revenue_all += revenue

            if tx_date >= trend_start:
                agg.orders_30d.add(order_key)
                agg.revenue_30d += revenue
            else:
                agg.orders_prior_60d.add(order_key)
                agg.revenue_prior_60d += revenue

            rows_kept += 1

        # Advance cursor; if we got a short page we're done.
        last = rows[-1]
        cursor_date = last.get("transaction_date")
        cursor_tid  = last.get("transaction_id")

        if page % 10 == 0 or len(rows) < PAGE_SIZE:
            log.info(
                "  page=%d rows=%d kept=%d distinct_customers=%d  cursor=%s",
                page, rows_seen, rows_kept, len(aggs), cursor_date,
            )

        if len(rows) < PAGE_SIZE:
            break

    log.info(
        "Sweep complete: %d rows seen, %d kept, %d distinct customers.",
        rows_seen, rows_kept, len(aggs),
    )
    return aggs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_customer(
    customer_id: str,
    agg:         _CustomerAgg,
    today:       date,
) -> dict[str, Any] | None:
    """Score one customer.  Returns None when the customer is unscorable.

    Unscorable means: zero orders, no last_date, or fewer than
    MIN_ORDERS_TO_SCORE distinct invoices in the lookback window.
    """
    order_count_90d       = len(agg.orders_all)
    if order_count_90d < MIN_ORDERS_TO_SCORE:
        return None
    if agg.last_date is None:
        return None

    order_count_30d       = len(agg.orders_30d)
    order_count_prior_60d = len(agg.orders_prior_60d)
    days_since_last       = (today - agg.last_date).days

    # ---------- recency_score ----------
    # Linear ramp 0 → 100 over [0, RECENCY_FULL_DAYS].  Clamped at 100 so
    # a customer who has not purchased in 200 days does not exceed the
    # cap (which would skew the composite weighting).
    recency_score = min(100.0, max(0.0, 100.0 * days_since_last / RECENCY_FULL_DAYS))

    # ---------- frequency_score ----------
    # Compare the count in the last TREND_WINDOW_DAYS days against the
    # PER-30-DAY rate from the prior LOOKBACK - TREND window.  We divide
    # the prior count by (prior_window_days / TREND_WINDOW_DAYS) so the
    # two sides compare like-for-like.
    prior_window_days = LOOKBACK_DAYS - TREND_WINDOW_DAYS
    prior_per_trend   = order_count_prior_60d / max(1.0, prior_window_days / TREND_WINDOW_DAYS)
    if prior_per_trend <= 0:
        # No prior baseline → can't measure trend.  Use a neutral 50 so
        # the recency component drives the composite score.
        frequency_score = 50.0
    else:
        ratio = order_count_30d / prior_per_trend
        # ratio = 1.0 → no change → 0
        # ratio = 0.5 → 50% drop → 50
        # ratio = 0.0 → complete halt → 100
        # ratio > 1.0 → growing → 0
        frequency_score = 100.0 * max(0.0, 1.0 - ratio)
        frequency_score = min(100.0, frequency_score)

    # ---------- monetary_score ----------
    # Same shape as frequency, but on revenue dollars.
    prior_revenue_per_trend = agg.revenue_prior_60d / max(
        1.0, prior_window_days / TREND_WINDOW_DAYS
    )
    if prior_revenue_per_trend <= 0:
        monetary_score = 50.0
    else:
        ratio = agg.revenue_30d / prior_revenue_per_trend
        monetary_score = 100.0 * max(0.0, 1.0 - ratio)
        monetary_score = min(100.0, monetary_score)

    # ---------- composite ----------
    churn_score = (
        WEIGHT_RECENCY   * recency_score
        + WEIGHT_FREQUENCY * frequency_score
        + WEIGHT_MONETARY  * monetary_score
    )

    if churn_score >= HIGH_RISK_THRESHOLD:
        risk_tier = "HIGH"
        at_risk   = True
    elif churn_score >= MEDIUM_RISK_THRESHOLD:
        risk_tier = "MEDIUM"
        at_risk   = False
    else:
        risk_tier = "LOW"
        at_risk   = False

    return {
        "customer_id":           customer_id,
        "run_date":              today.isoformat(),
        "customer_type":         agg.customer_type,
        "customer_status":       agg.customer_status,
        "order_count_90d":       order_count_90d,
        "order_count_30d":       order_count_30d,
        "order_count_prior_60d": order_count_prior_60d,
        "revenue_90d":           round(agg.revenue_all,       2),
        "revenue_30d":           round(agg.revenue_30d,       2),
        "revenue_prior_60d":     round(agg.revenue_prior_60d, 2),
        "days_since_last_order": days_since_last,
        "recency_score":         round(recency_score,   2),
        "frequency_score":       round(frequency_score, 2),
        "monetary_score":        round(monetary_score,  2),
        "churn_score":           round(churn_score,     2),
        "risk_tier":             risk_tier,
        "at_risk_flag":          at_risk,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _write_scores(
    client: Any,
    scores: list[dict[str, Any]],
    dry_run: bool,
) -> tuple[int, int]:
    """Upsert score rows in batches.

    Returns:
        (rows_written, batches_failed) — `batches_failed` lets the caller
        decide whether to fail the pipeline stage.  We do NOT raise on the
        first failure because we still want the remaining batches attempted
        (a transient 57014 on one batch shouldn't drop the rest), but the
        caller MUST check batches_failed before reporting success.
    """
    if dry_run:
        log.info("[DRY RUN] would write %d score rows.", len(scores))
        return 0, 0
    if not scores:
        return 0, 0

    log.info(
        "Writing %d churn score row(s) to customer_churn_scores in batches of %d …",
        len(scores), WRITE_BATCH_SIZE,
    )
    written        = 0
    batches_failed = 0
    for offset in range(0, len(scores), WRITE_BATCH_SIZE):
        batch = scores[offset:offset + WRITE_BATCH_SIZE]
        try:
            resp = (
                client.table("customer_churn_scores")
                .upsert(batch, on_conflict="customer_id,run_date")
                .execute()
            )
            written += len(resp.data or [])
        except Exception:
            batches_failed += 1
            log.exception(
                "Failed to write churn scores batch at offset %d (size=%d).",
                offset, len(batch),
            )
    log.info(
        "  Rows written: %d  batches_failed: %d", written, batches_failed,
    )
    return written, batches_failed


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_churn(dry_run: bool = False) -> int:
    """Run the customer churn predictor.

    Args:
        dry_run: When True, score everything but skip the DB write.

    Returns:
        Process exit code (0 = success, 1 = failure).
    """
    setup_logging()
    t0 = time.monotonic()
    today = date.today()

    log.info("=" * 60)
    log.info("partswatch-ai — ml.churn  (run_date=%s)", today.isoformat())
    log.info(
        "  Lookback=%d days  trend_window=%d days  recency_full=%d days",
        LOOKBACK_DAYS, TREND_WINDOW_DAYS, RECENCY_FULL_DAYS,
    )
    log.info(
        "  Thresholds: HIGH≥%.0f  MEDIUM≥%.0f  min_orders=%d",
        HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, MIN_ORDERS_TO_SCORE,
    )
    log.info(
        "  Weights: recency=%.2f  frequency=%.2f  monetary=%.2f",
        WEIGHT_RECENCY, WEIGHT_FREQUENCY, WEIGHT_MONETARY,
    )
    log.info("=" * 60)

    # Defensive: weights must sum to 1.0 or the score range collapses.
    weight_sum = WEIGHT_RECENCY + WEIGHT_FREQUENCY + WEIGHT_MONETARY
    if not math.isclose(weight_sum, 1.0, abs_tol=0.001):
        log.error("Weight sum %.3f != 1.0 — refusing to score.", weight_sum)
        return 1

    client = get_client()

    try:
        aggs = _fetch_and_aggregate(client, today)
    except Exception:
        log.exception("sales_transactions sweep failed.")
        return 1

    if not aggs:
        # No customer rows in the window means either (a) the extract has
        # not yet populated customer_id (migration 025 just applied), or
        # (b) all customers were filtered out.  Either way it's a no-op,
        # not a failure — return 0 so the nightly pipeline doesn't trip.
        log.warning(
            "No customers found in the lookback window — nothing to score. "
            "Likely cause: customer_id not yet populated in sales_transactions "
            "(migration 025 / 90-day re-extract may still be pending)."
        )
        return 0

    log.info("Scoring %d customers …", len(aggs))
    scores: list[dict[str, Any]] = []
    skipped_low_orders = 0
    for customer_id, agg in aggs.items():
        row = _score_customer(customer_id, agg, today)
        if row is None:
            skipped_low_orders += 1
            continue
        scores.append(row)

    if skipped_low_orders:
        log.info(
            "  Skipped %d customers with < %d orders in the lookback window.",
            skipped_low_orders, MIN_ORDERS_TO_SCORE,
        )

    # ---------- Tier breakdown ----------
    by_tier: dict[str, int] = defaultdict(int)
    for s in scores:
        by_tier[s["risk_tier"]] += 1

    log.info("Risk tier breakdown:")
    log.info("  HIGH    %6d  (at_risk_flag = TRUE)", by_tier["HIGH"])
    log.info("  MEDIUM  %6d", by_tier["MEDIUM"])
    log.info("  LOW     %6d", by_tier["LOW"])
    log.info("  TOTAL   %6d", len(scores))

    # ---------- Top 10 HIGH-risk accounts ----------
    high = [s for s in scores if s["risk_tier"] == "HIGH"]
    high.sort(key=lambda s: s["churn_score"], reverse=True)
    if high:
        log.info("-" * 60)
        log.info("Top 10 HIGH-risk customers (by churn_score):")
        for i, s in enumerate(high[:10], 1):
            log.info(
                "  %2d. %-12s  score=%5.1f  type=%-4s  "
                "idle=%3dd  orders 30/prior=%2d/%2d  rev 30/prior=$%.0f/$%.0f",
                i, s["customer_id"], s["churn_score"], s["customer_type"] or "—",
                s["days_since_last_order"],
                s["order_count_30d"], s["order_count_prior_60d"],
                s["revenue_30d"], s["revenue_prior_60d"],
            )
        log.info("-" * 60)

    # ---------- Persist ----------
    written, batches_failed = _write_scores(client, scores, dry_run=dry_run)

    # Integrity check: count what actually landed in the DB for today's
    # run_date.  If it doesn't match what we tried to write, something
    # went wrong silently (e.g. CHECK constraint rejection mid-batch) and
    # we MUST fail the stage so the nightly summary turns red.
    persistence_ok = True
    if not dry_run and scores:
        try:
            persisted = (
                client.table("customer_churn_scores")
                .select("customer_id", count="exact")
                .eq("run_date", today.isoformat())
                .limit(1)
                .execute()
            )
            persisted_count = getattr(persisted, "count", None)
            if persisted_count is not None and persisted_count < len(scores):
                log.error(
                    "Persistence integrity check FAILED: scored=%d  persisted=%d  "
                    "(missing=%d).",
                    len(scores), persisted_count, len(scores) - persisted_count,
                )
                persistence_ok = False
            else:
                log.info(
                    "Persistence integrity check OK: scored=%d  persisted=%s",
                    len(scores), persisted_count,
                )
        except Exception:
            log.warning(
                "Persistence integrity check failed to run (non-fatal).",
                exc_info=True,
            )

    elapsed = time.monotonic() - t0
    log.info("=" * 60)
    log.info("Churn analysis complete  (%.1fs)", elapsed)
    log.info(
        "  Customers scored: %d  HIGH=%d  MEDIUM=%d  LOW=%d  rows_written=%d%s",
        len(scores), by_tier["HIGH"], by_tier["MEDIUM"], by_tier["LOW"],
        written, "  (DRY RUN — no writes)" if dry_run else "",
    )
    log.info("=" * 60)

    # Fail the stage if any batch failed OR the integrity check found a
    # gap.  Caller (main.py) propagates this exit code into the nightly
    # summary so a partial write is loudly visible rather than silently
    # under-populating customer_churn_scores.
    if batches_failed > 0 or not persistence_ok:
        log.error(
            "Churn stage FAILED: batches_failed=%d  persistence_ok=%s",
            batches_failed, persistence_ok,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Customer churn risk scoring (RFM-style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Score and report without writing to customer_churn_scores.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_churn(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
