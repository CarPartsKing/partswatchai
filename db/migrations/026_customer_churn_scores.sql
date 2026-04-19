-- =============================================================================
-- 026_customer_churn_scores.sql — RFM-style churn risk scoring per customer
--
-- BACKGROUND
-- ----------
-- Migration 025 added customer_id / customer_type / customer_status to
-- sales_transactions.  This migration adds the destination table for
-- ml/churn.py — a per-customer, per-run-date snapshot of behavioral signals
-- (recency, frequency, monetary trend) and a 0-100 churn risk score.
--
-- One row per (customer_id, run_date) so historical scores are retained
-- and the dashboard / assistant can chart churn-risk trajectories without
-- the engine needing to backfill anything on subsequent runs.
--
-- TABLE PURPOSE
-- -------------
-- ml/churn.py reads the last 90 days of sales_transactions, aggregates per
-- customer, scores each customer 0-100, classifies into HIGH / MEDIUM / LOW
-- risk tiers, and writes one row per scored customer.  The dashboard's
-- "Customer churn" panel and the assistant's churn context section both
-- read the latest run_date from this table.
--
-- WHY NOT JUST USE customer_id AS PK
-- ----------------------------------
-- We want trend visibility ("this customer was MEDIUM last week, HIGH today")
-- and a single per-customer row would silently overwrite that history.  The
-- composite (customer_id, run_date) PK keeps every snapshot.
--
-- IDEMPOTENCY: every statement uses IF NOT EXISTS / IF EXISTS — safe to re-run.
-- =============================================================================

CREATE TABLE IF NOT EXISTS customer_churn_scores (
    customer_id           VARCHAR(50)    NOT NULL,
    run_date              DATE           NOT NULL,

    -- Denormalized customer attributes captured at scoring time.  Stored on
    -- the snapshot row so the dashboard / assistant can segment by type
    -- without joining back to sales_transactions for every render.
    -- NOTE: customer_type is currently a numeric code ('5', '6', etc.)
    -- emitted by the Sales Detail cube.  The mapping (wholesale vs retail
    -- vs fleet vs etc.) is unknown to us today — investigate with the
    -- buyer team and document here once known.
    customer_type         VARCHAR(50),
    customer_status       VARCHAR(20),

    -- ---------- Raw aggregates over the 90-day lookback ----------
    -- Distinct invoice count (an "order" = one invoice = one purchase event).
    order_count_90d       INTEGER        NOT NULL DEFAULT 0,
    order_count_30d       INTEGER        NOT NULL DEFAULT 0,
    order_count_prior_60d INTEGER        NOT NULL DEFAULT 0,

    revenue_90d           NUMERIC(14, 2) NOT NULL DEFAULT 0,
    revenue_30d           NUMERIC(14, 2) NOT NULL DEFAULT 0,
    revenue_prior_60d     NUMERIC(14, 2) NOT NULL DEFAULT 0,

    -- Days between run_date and the customer's most recent transaction.
    -- NULL when the customer has zero orders in the lookback window — the
    -- scorer skips those rather than writing a row, so in practice this is
    -- always populated, but kept nullable for safety.
    days_since_last_order INTEGER,

    -- ---------- Component sub-scores (0-100, higher = more at-risk) ----------
    -- recency_score:   scaled from days_since_last_order
    -- frequency_score: derived from order_count_30d vs (order_count_prior_60d / 2)
    -- monetary_score:  derived from revenue_30d vs (revenue_prior_60d / 2)
    recency_score         NUMERIC(5, 2)  NOT NULL DEFAULT 0
                                         CHECK (recency_score   BETWEEN 0 AND 100),
    frequency_score       NUMERIC(5, 2)  NOT NULL DEFAULT 0
                                         CHECK (frequency_score BETWEEN 0 AND 100),
    monetary_score        NUMERIC(5, 2)  NOT NULL DEFAULT 0
                                         CHECK (monetary_score  BETWEEN 0 AND 100),

    -- Composite weighted score: 0.4*recency + 0.3*frequency + 0.3*monetary.
    -- HIGH ≥ 70, MEDIUM 40-69, LOW < 40.  Tunable in ml/churn.py constants.
    churn_score           NUMERIC(5, 2)  NOT NULL
                                         CHECK (churn_score BETWEEN 0 AND 100),
    risk_tier             VARCHAR(10)    NOT NULL
                                         CHECK (risk_tier IN ('HIGH', 'MEDIUM', 'LOW')),

    -- "Just flag the account" — TRUE for HIGH-tier customers so the
    -- dashboard and engine.alerts can filter to actionable rows without
    -- re-thresholding.  Salesman routing is intentionally NOT here yet —
    -- the cube's [Customer].[Salesman] dimension comes through as NULL on
    -- our extract today; revisit when that data is populated.
    at_risk_flag          BOOLEAN        NOT NULL DEFAULT FALSE,

    created_at            TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (customer_id, run_date)
);

-- Dashboard / assistant always read the most-recent run_date and then
-- order by churn_score DESC (or filter at_risk_flag = TRUE).  These two
-- partial / composite indexes cover both access patterns.
CREATE INDEX IF NOT EXISTS idx_customer_churn_scores_run_score
    ON customer_churn_scores (run_date DESC, churn_score DESC);

CREATE INDEX IF NOT EXISTS idx_customer_churn_scores_at_risk
    ON customer_churn_scores (run_date DESC, churn_score DESC)
    WHERE at_risk_flag = TRUE;

COMMENT ON TABLE customer_churn_scores IS
    'Per-customer churn risk snapshot written by ml/churn.py.  One row per '
    '(customer_id, run_date).  Composite churn_score combines recency, '
    'frequency-trend, and monetary-trend sub-scores; at_risk_flag is TRUE '
    'when churn_score >= HIGH threshold (70 by default).';

-- Project rule (replit.md): every CREATE TABLE migration disables RLS so the
-- service-role client and ad-hoc SQL writes are never blocked by an empty
-- policy set.
ALTER TABLE customer_churn_scores DISABLE ROW LEVEL SECURITY;
