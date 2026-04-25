-- =============================================================================
-- 034_customer_churn_flags.sql
--
-- Customer churn detection flags — per customer per location, derived from
-- sales_detail_transactions by engine/churn_detection.py.
--
-- LOGIC (see churn_detection.py for constants)
-- ---------------------------------------------
-- Baseline  = avg monthly spend over months 1-18 of the 2-year lookback window
--             (window_start  ..  window_start + 18 months)
-- Comparison = total spend in the last 90 days
--
-- FLAG values
--   CHURNED   — baseline purchases >= threshold, zero spend last 90 days
--   DECLINING — last-90-day spend >= 30% below expected (baseline_monthly * 3)
--   STABLE    — within normal range
--
-- Each run overwrites the single row per (customer_id, location_id) so the
-- table always reflects the current snapshot.  Historical trajectory is NOT
-- retained here — use customer_churn_scores for that.
--
-- All statements are idempotent — safe to re-run.
-- =============================================================================

CREATE TABLE IF NOT EXISTS customer_churn_flags (
    customer_id              VARCHAR(200)   NOT NULL,
    location_id              VARCHAR(20)    NOT NULL,
    baseline_monthly_spend   NUMERIC(14, 4),
    last_90_days_spend       NUMERIC(14, 4),
    pct_change               NUMERIC(8, 2),
    flag                     VARCHAR(20)    NOT NULL
                                            CHECK (flag IN ('CHURNED', 'DECLINING', 'STABLE')),
    last_purchase_date       DATE,
    run_date                 DATE           NOT NULL,
    created_at               TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (customer_id, location_id)
);

CREATE INDEX IF NOT EXISTS idx_ccf_flag
    ON customer_churn_flags (flag);

CREATE INDEX IF NOT EXISTS idx_ccf_location_flag
    ON customer_churn_flags (location_id, flag);

CREATE INDEX IF NOT EXISTS idx_ccf_run_date
    ON customer_churn_flags (run_date DESC);

ALTER TABLE customer_churn_flags DISABLE ROW LEVEL SECURITY;
