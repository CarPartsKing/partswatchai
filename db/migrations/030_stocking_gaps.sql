-- =============================================================================
-- 030_stocking_gaps.sql
--
-- Stocking Gap Intelligence — answers:
--   "What should each store stock locally instead of waiting for transfers?"
--
-- Produced by ml/stocking_intelligence.py (weekly pipeline stage).
-- Each row is one (analysis_date, sku_id, location_id) gap assessment
-- derived from 90 days of transfer-type reorder recommendations.
--
-- A SKU is CHRONIC when it has been transferred to a location 10+ times
-- in 90 days OR on 5+ consecutive days — the store should stock it locally
-- rather than perpetually pulling from other branches.
-- =============================================================================

CREATE TABLE IF NOT EXISTS stocking_gaps (
    id                       BIGSERIAL     PRIMARY KEY,
    analysis_date            DATE          NOT NULL,
    sku_id                   VARCHAR(50)   NOT NULL,
    location_id              VARCHAR(20)   NOT NULL,
    location_name            VARCHAR(60),

    -- Transfer source (most common branch this SKU was sent FROM)
    transfer_from_location   VARCHAR(20),

    -- Transfer pattern metrics
    transfer_frequency       INT           NOT NULL DEFAULT 0,   -- unique days transferred in 90d
    transfer_streak          INT           NOT NULL DEFAULT 0,   -- max consecutive days
    avg_qty_recommended      NUMERIC(12,4) NOT NULL DEFAULT 0,
    total_transfer_value     NUMERIC(14,2),                      -- avg_qty × unit_cost × frequency
    trend_direction          VARCHAR(12)   NOT NULL DEFAULT 'STABLE',  -- INCREASING / STABLE / DECREASING

    -- Scoring
    gap_score                NUMERIC(6,4)  NOT NULL DEFAULT 0,   -- 0.0000–1.0000
    gap_classification       VARCHAR(12)   NOT NULL DEFAULT 'OCCASIONAL', -- CHRONIC / RECURRING / OCCASIONAL

    -- Recommendations (populated for CHRONIC only)
    suggested_stock_increase NUMERIC(12,2),   -- avg_qty × 1.5 × 1.2
    current_reorder_point    NUMERIC(12,2),   -- latest reorder_point from inventory_snapshots
    annual_cost_savings      NUMERIC(12,2),   -- (freq/90*365) × avg_qty × $2.50

    created_at               TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    UNIQUE (analysis_date, sku_id, location_id)
);

-- Indexes for common dashboard query patterns
CREATE INDEX IF NOT EXISTS idx_stocking_gaps_date
    ON stocking_gaps (analysis_date DESC);

CREATE INDEX IF NOT EXISTS idx_stocking_gaps_loc_date
    ON stocking_gaps (location_id, analysis_date DESC);

CREATE INDEX IF NOT EXISTS idx_stocking_gaps_classification
    ON stocking_gaps (analysis_date DESC, gap_classification, gap_score DESC);

CREATE INDEX IF NOT EXISTS idx_stocking_gaps_savings
    ON stocking_gaps (analysis_date DESC, annual_cost_savings DESC NULLS LAST)
    WHERE gap_classification = 'CHRONIC';

-- No RLS — internal analytics table, service-role access only
ALTER TABLE stocking_gaps DISABLE ROW LEVEL SECURITY;

-- Auto-stamp updated_at on every upsert
CREATE OR REPLACE FUNCTION _set_stocking_gaps_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$;

DROP TRIGGER IF EXISTS trg_stocking_gaps_updated_at ON stocking_gaps;
CREATE TRIGGER trg_stocking_gaps_updated_at
    BEFORE UPDATE ON stocking_gaps
    FOR EACH ROW EXECUTE FUNCTION _set_stocking_gaps_updated_at();
