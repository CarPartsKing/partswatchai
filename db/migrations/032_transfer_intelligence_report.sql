-- =============================================================================
-- 032_transfer_intelligence_report.sql
--
-- Transfer Intelligence Report — unified 3-tier output combining:
--   1. stocking_gaps (ml/stocking_intelligence.py) — transfer-pattern signal
--   2. understocking_report (engine/understocking.py) — chronic-below-reorder signal
--
-- Produced by engine/transfer_intelligence.py (weekly pipeline stage).
--
-- TIERS
--   1 = CONFIRMED_GAP        both signals agree — highest confidence, act first
--   2 = TRANSFER_PATTERN_ONLY repeated transfers but reorder point looks ok
--   3 = UNDERSTOCKED_ONLY    below reorder point frequently, no transfer history
--
-- Idempotent write pattern (same as understocking_report):
--   rows are inserted with run_completed_at = NULL; engine sets it to NOW()
--   only after a successful full write + integrity check.  Dashboard filters
--   run_completed_at IS NOT NULL so consumers never read a partial report.
-- =============================================================================

CREATE TABLE IF NOT EXISTS transfer_intelligence_report (
    id                       BIGSERIAL      PRIMARY KEY,
    report_date              DATE           NOT NULL,
    location_id              VARCHAR(20)    NOT NULL,
    location_name            VARCHAR(60),
    sku_id                   VARCHAR(50)    NOT NULL,
    sku_description          TEXT,

    -- Confidence tier
    tier                     SMALLINT       NOT NULL,        -- 1 / 2 / 3
    tier_label               VARCHAR(30)    NOT NULL,        -- CONFIRMED_GAP / TRANSFER_PATTERN_ONLY / UNDERSTOCKED_ONLY

    -- Transfer signal (from stocking_gaps; NULL for Tier 3)
    transfer_frequency       INT,            -- distinct recommendation days in 90d window
    transfer_streak          INT,            -- max consecutive days recommended
    gap_score                NUMERIC(6,4),   -- 0.0000–1.0000
    gap_classification       VARCHAR(12),    -- CHRONIC / RECURRING
    avg_qty_recommended      NUMERIC(12,4),
    annual_transfer_savings  NUMERIC(12,2),  -- (freq/90*365) × avg_qty × $2.50

    -- Understocking signal (from understocking_report; NULL for Tier 2)
    stockout_days_pct        NUMERIC(5,4),   -- fraction of days below reorder_point
    days_below_reorder       INT,
    days_observed            INT,
    avg_daily_demand         NUMERIC(12,4),
    inventory_value_at_risk  NUMERIC(14,2),  -- avg_daily × unit_cost × 21d

    -- Min qty / cost (populated for all tiers where data is available)
    current_min_qty          NUMERIC(12,2),
    suggested_min_qty        NUMERIC(12,2),
    min_qty_gap              NUMERIC(12,2),  -- suggested − current  (= "Increase By" in CSV)
    unit_cost                NUMERIC(12,2),

    -- Ranking (higher = act sooner)
    priority_score           NUMERIC(16,4)   NOT NULL DEFAULT 0,

    -- Publication marker — see engine/transfer_intelligence.py
    run_completed_at         TIMESTAMPTZ,
    created_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (report_date, location_id, sku_id)
);

-- Backfill the column if re-applying over an older schema.
ALTER TABLE transfer_intelligence_report
    ADD COLUMN IF NOT EXISTS run_completed_at TIMESTAMPTZ;

-- Backfill legacy rows so dashboard is not blank between migration and next run.
UPDATE transfer_intelligence_report
   SET run_completed_at = created_at
 WHERE run_completed_at IS NULL;

-- Indexes for dashboard query patterns
CREATE INDEX IF NOT EXISTS idx_ti_report_date
    ON transfer_intelligence_report (report_date DESC);

CREATE INDEX IF NOT EXISTS idx_ti_loc_date
    ON transfer_intelligence_report (location_id, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_ti_tier_priority
    ON transfer_intelligence_report (report_date DESC, tier, priority_score DESC);

CREATE INDEX IF NOT EXISTS idx_ti_confirmed
    ON transfer_intelligence_report (report_date DESC, priority_score DESC)
    WHERE tier = 1;

ALTER TABLE transfer_intelligence_report DISABLE ROW LEVEL SECURITY;
