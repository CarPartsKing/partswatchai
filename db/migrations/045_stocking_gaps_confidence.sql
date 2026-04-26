-- =============================================================================
-- 045_stocking_gaps_confidence.sql
--
-- Two additions supporting OPSL cross-reference and improved savings estimates:
--
--   1. stocking_gaps.double_confirmed — TRUE when the same SKU+location appears
--      in both stocking_gaps (chronic transfer pattern) AND opsl_flags (HIGH or
--      MEDIUM outside purchase flag).  Converging evidence from two independent
--      signals = higher confidence recommendation.
--
--   2. stocking_gaps.confidence — 'HIGH' for double-confirmed gaps, 'MEDIUM' for
--      single-source gaps.  Dashboard sorts HIGH above MEDIUM regardless of score.
--
--   3. stocking_gaps.savings_source — which method produced annual_cost_savings:
--      'OPSL_ACTUAL'      — derived from real margin-loss data in
--                           sales_detail_transactions (stock_flag='N', tran_code='SL');
--                           = sum(baseline_gp% × sales − gross_profit) × 4 (90d→annual)
--      'TRANSFER_ESTIMATE'— legacy: (freq/90×365) × avg_qty × $2.50 handling cost
--
-- All statements are idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Add double_confirmed
-- ---------------------------------------------------------------------------
ALTER TABLE stocking_gaps
    ADD COLUMN IF NOT EXISTS double_confirmed BOOLEAN NOT NULL DEFAULT FALSE;

-- ---------------------------------------------------------------------------
-- 2. Add confidence
-- ---------------------------------------------------------------------------
ALTER TABLE stocking_gaps
    ADD COLUMN IF NOT EXISTS confidence VARCHAR(20)
        CHECK (confidence IN ('HIGH', 'MEDIUM'));

-- ---------------------------------------------------------------------------
-- 3. Add savings_source
-- ---------------------------------------------------------------------------
ALTER TABLE stocking_gaps
    ADD COLUMN IF NOT EXISTS savings_source VARCHAR(20)
        CHECK (savings_source IN ('OPSL_ACTUAL', 'TRANSFER_ESTIMATE'));

-- ---------------------------------------------------------------------------
-- 4. Indexes supporting dashboard sort: double_confirmed DESC, then savings DESC
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_stocking_gaps_confirmed
    ON stocking_gaps (analysis_date DESC, double_confirmed DESC, annual_cost_savings DESC NULLS LAST)
    WHERE gap_classification = 'CHRONIC';

CREATE INDEX IF NOT EXISTS idx_stocking_gaps_confidence
    ON stocking_gaps (analysis_date DESC, confidence, gap_score DESC)
    WHERE confidence IS NOT NULL;
