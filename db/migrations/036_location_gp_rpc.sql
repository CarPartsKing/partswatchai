-- ============================================================
-- 036 — GP RPC functions for Location Performance + Top SKUs
--
-- Root cause of original timeout: both RPCs were being killed
-- by the anon role's 8-second statement timeout when scanning
-- ~1M rows (180-day window of 4.3M-row table).
--
-- Fixes applied (v2):
--   1. Partial covering indexes → index-only scans, no heap access
--   2. SECURITY DEFINER → executes as postgres owner (not anon)
--   3. SET statement_timeout = '60s' → explicit budget per call
--   4. GRANT EXECUTE → anon/authenticated can invoke the function
-- ============================================================

-- Covering index for the location GP aggregate.
-- Partial (WHERE sales IS NOT NULL) matches the RPC's WHERE clause
-- so Postgres can do an index-only scan over the 180-day window.
CREATE INDEX IF NOT EXISTS idx_sdt_gp_covering
    ON sales_detail_transactions (tran_date, location_id, sales, gross_profit)
    WHERE sales IS NOT NULL;

-- Covering index for the top-SKU GP aggregate.
CREATE INDEX IF NOT EXISTS idx_sdt_sku_gp_covering
    ON sales_detail_transactions (tran_date, prod_line_pn, gross_profit, sales, location_id)
    WHERE gross_profit IS NOT NULL AND gross_profit > 0;

-- ── GP summary by location ────────────────────────────────────
-- Returns one row per location: sales and GP for the current 90-day
-- window plus sales for the prior 90-day window (for trend).
-- Used by _build_location_performance to replace 27 round-trips.
CREATE OR REPLACE FUNCTION get_all_locations_gp_summary(
    p_current_start DATE,
    p_prior_start   DATE,
    p_prior_end     DATE
)
RETURNS TABLE (
    location_id     TEXT,
    sales_90d       NUMERIC,
    gp_90d          NUMERIC,
    prior_sales_90d NUMERIC
)
LANGUAGE sql STABLE SECURITY DEFINER
SET statement_timeout = '60s'
AS $$
    SELECT
        location_id::TEXT,
        COALESCE(SUM(sales)        FILTER (WHERE tran_date >= p_current_start), 0),
        COALESCE(SUM(gross_profit) FILTER (WHERE tran_date >= p_current_start), 0),
        COALESCE(SUM(sales)        FILTER (WHERE tran_date BETWEEN p_prior_start AND p_prior_end), 0)
    FROM sales_detail_transactions
    WHERE tran_date >= p_prior_start
      AND sales IS NOT NULL
    GROUP BY location_id
$$;

GRANT EXECUTE ON FUNCTION get_all_locations_gp_summary(DATE, DATE, DATE) TO anon, authenticated;

-- ── Top 10 SKUs by gross profit ───────────────────────────────
-- Returns top 10 SKUs ranked by GP for a rolling 90-day window.
-- Used by _build_top_skus to replace sku_master avg_weekly_units sort.
CREATE OR REPLACE FUNCTION get_top_skus_by_gp(
    p_start_date DATE
)
RETURNS TABLE (
    sku_id         TEXT,
    total_gp       NUMERIC,
    total_sales    NUMERIC,
    location_count BIGINT
)
LANGUAGE sql STABLE SECURITY DEFINER
SET statement_timeout = '60s'
AS $$
    SELECT
        prod_line_pn::TEXT,
        SUM(gross_profit)           AS total_gp,
        SUM(sales)                  AS total_sales,
        COUNT(DISTINCT location_id) AS location_count
    FROM sales_detail_transactions
    WHERE tran_date    >= p_start_date
      AND prod_line_pn IS NOT NULL
      AND gross_profit IS NOT NULL
      AND gross_profit  > 0
    GROUP BY prod_line_pn
    ORDER BY total_gp DESC
    LIMIT 10
$$;

GRANT EXECUTE ON FUNCTION get_top_skus_by_gp(DATE) TO anon, authenticated;
