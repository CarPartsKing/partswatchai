-- =============================================================================
-- 048_dead_stock_v2.sql
--
-- Extends dead_stock_recommendations with three new columns that support the
-- upgraded engine/dead_stock v2 pipeline:
--
--   action_type VARCHAR(20)
--       Recommended disposition — superset of classification.  Values:
--         LIQUIDATE | MARKDOWN | MONITOR   (carry through from classification)
--         TRANSFER                          (override when another location
--                                           needs the SKU — sourced from
--                                           reorder_recommendations or
--                                           understocking_report)
--
--   transfer_candidate_location VARCHAR(50)
--       Location ID of the first needing location when action_type = 'TRANSFER'.
--       NULL for all non-TRANSFER rows.
--
--   data_conflict BOOLEAN NOT NULL DEFAULT FALSE
--       Set TRUE when sales_detail_transactions has ≥ 1 transaction for this
--       SKU's prod_line_pn within the last 365 days.  Indicates that the
--       is_dead_stock flag may be stale or incorrect — these items are excluded
--       from the liquidation total and from DEAD_STOCK alerts.
--
-- Note: migration 044 was referenced in design docs but never created as a file.
-- This migration (048) serves as the authoritative DDL for all three columns.
--
-- Idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

ALTER TABLE dead_stock_recommendations
    ADD COLUMN IF NOT EXISTS action_type                 VARCHAR(20),
    ADD COLUMN IF NOT EXISTS transfer_candidate_location VARCHAR(50),
    ADD COLUMN IF NOT EXISTS data_conflict               BOOLEAN NOT NULL DEFAULT FALSE;

-- Dashboard: TRANSFER candidates by date
CREATE INDEX IF NOT EXISTS idx_dead_stock_recs_transfer
    ON dead_stock_recommendations (report_date, action_type)
    WHERE action_type = 'TRANSFER';

-- Dashboard: high-value LIQUIDATE items excluding data conflicts
-- (replaces/supplements the existing idx_dead_stock_recs_liquidate_value)
CREATE INDEX IF NOT EXISTS idx_dead_stock_recs_liquidate_clean
    ON dead_stock_recommendations (report_date, total_inv_value DESC)
    WHERE classification = 'LIQUIDATE' AND data_conflict = FALSE;

-- Alert engine: value-filtered open dead stock
CREATE INDEX IF NOT EXISTS idx_dead_stock_recs_alert_eligible
    ON dead_stock_recommendations (report_date, sku_id)
    WHERE data_conflict = FALSE;
