-- Returns GP/sales summary for ALL active locations in one server-side call.
-- Used by _build_location_performance to avoid 27 per-location round-trips.
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
LANGUAGE sql STABLE AS $$
    SELECT
        location_id::TEXT,
        COALESCE(SUM(sales)         FILTER (WHERE tran_date >= p_current_start), 0),
        COALESCE(SUM(gross_profit)  FILTER (WHERE tran_date >= p_current_start), 0),
        COALESCE(SUM(sales)         FILTER (WHERE tran_date BETWEEN p_prior_start AND p_prior_end), 0)
    FROM sales_detail_transactions
    WHERE tran_date >= p_prior_start
      AND sales IS NOT NULL
    GROUP BY location_id
$$;

-- Returns top 10 SKUs ranked by gross profit for a rolling 90-day window.
-- Used by _build_top_skus to replace the sku_master avg_weekly_units sort.
CREATE OR REPLACE FUNCTION get_top_skus_by_gp(
    p_start_date DATE
)
RETURNS TABLE (
    sku_id         TEXT,
    total_gp       NUMERIC,
    total_sales    NUMERIC,
    location_count BIGINT
)
LANGUAGE sql STABLE AS $$
    SELECT
        prod_line_pn::TEXT,
        SUM(gross_profit)            AS total_gp,
        SUM(sales)                   AS total_sales,
        COUNT(DISTINCT location_id)  AS location_count
    FROM sales_detail_transactions
    WHERE tran_date    >= p_start_date
      AND prod_line_pn IS NOT NULL
      AND gross_profit IS NOT NULL
      AND gross_profit  > 0
    GROUP BY prod_line_pn
    ORDER BY total_gp DESC
    LIMIT 10
$$;
