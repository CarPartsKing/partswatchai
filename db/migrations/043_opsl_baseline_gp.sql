-- =============================================================================
-- 043_opsl_baseline_gp.sql
--
-- Two changes supporting dynamic GP% baselines and the OPSL→reorder
-- feedback loop:
--
--   1. opsl_flags.baseline_gp_pct  — per-location GP% used for the
--      estimated_margin_recovery calculation.  Previously the engine used a
--      hardcoded NORMAL_GP_PCT = 0.35 for all locations; now it queries the
--      actual 90-day average from SL/SL-I rows and stores it alongside each
--      flag so dashboards can show which baseline was applied.
--
--   2. get_location_gp_baselines(p_start_date) RPC — GROUP BY query over
--      sales_detail_transactions; returns one row per location with the
--      actual average GP%.  Python falls back to 0.35 if a location returns
--      no rows (new location, no sales data yet, etc.).
--
--   3. reorder_recommendations.source / .notes — new columns so the OPSL
--      feedback loop can tag auto-generated recs with source='OPSL' and a
--      human-readable notes string.
--
--   4. Drop reorder_recommendations.sku_id FK — the OPSL engine writes
--      prod_line_pn values (e.g. "AC 12345") as sku_id.  These do not exist
--      in sku_master because prod_line_pn is a compound format.  The FK is
--      dropped so OPSL-sourced recs can be inserted without a constraint
--      violation.  The normal reorder engine continues to work unchanged
--      since it only writes sku_ids that DO exist in sku_master.
--
-- All statements are idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. opsl_flags — add baseline_gp_pct column
-- ---------------------------------------------------------------------------
ALTER TABLE opsl_flags
    ADD COLUMN IF NOT EXISTS baseline_gp_pct NUMERIC(6,4);

-- ---------------------------------------------------------------------------
-- 2. get_location_gp_baselines() RPC
--    Returns one row per location: the actual avg GP% from SL/SL-I sales in
--    the requested window.  Used by the OPSL engine instead of hardcoded 0.35.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_location_gp_baselines(
    p_start_date DATE
)
RETURNS TABLE (
    location_id TEXT,
    avg_gp_pct  NUMERIC
)
LANGUAGE sql STABLE SECURITY DEFINER
SET statement_timeout = '30s'
AS $$
    SELECT
        location_id::TEXT,
        COALESCE(SUM(gross_profit), 0) / NULLIF(SUM(sales), 0) AS avg_gp_pct
    FROM sales_detail_transactions
    WHERE tran_date     >= p_start_date
      AND tran_code     IN ('SL', 'SL-I')
      AND sales          > 0
      AND gross_profit   IS NOT NULL
    GROUP BY location_id
$$;

GRANT EXECUTE ON FUNCTION get_location_gp_baselines(DATE) TO anon, authenticated;

-- ---------------------------------------------------------------------------
-- 3. reorder_recommendations — add source and notes columns
-- ---------------------------------------------------------------------------
ALTER TABLE reorder_recommendations
    ADD COLUMN IF NOT EXISTS source VARCHAR(20);

ALTER TABLE reorder_recommendations
    ADD COLUMN IF NOT EXISTS notes TEXT;

CREATE INDEX IF NOT EXISTS idx_reorder_source
    ON reorder_recommendations (source)
    WHERE source IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 4. Drop sku_id FK so OPSL-sourced recs can use prod_line_pn as sku_id.
--    The normal reorder engine (engine/reorder.py) only writes sku_ids that
--    exist in sku_master, so this change does not affect it.
-- ---------------------------------------------------------------------------
ALTER TABLE reorder_recommendations
    DROP CONSTRAINT IF EXISTS reorder_recommendations_sku_id_fkey;
