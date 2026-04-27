-- =============================================================================
-- 051_dead_stock_summary_rpc.sql
--
-- Creates get_dead_stock_summary(p_report_date DATE) -> json
--
-- Replaces the 231K-row full-table paginate in dashboard/_build_dead_stock()
-- with a single server-side aggregation + top-10 fetch.  Typical runtime
-- is <200ms vs 30-40s for the Python-side paginate.
--
-- Return shape:
--   {
--     "kpis": {
--       "capital_at_risk":     numeric,
--       "clean_capital":       numeric,   -- data_conflict IS NOT TRUE rows only
--       "liquidate_count":     integer,
--       "liquidate_value":     numeric,
--       "markdown_count":      integer,
--       "markdown_value":      numeric,
--       "transfer_count":      integer,   -- action_type = 'TRANSFER'
--       "data_conflict_count": integer,
--       "total_positions":     integer
--     },
--     "top10": [
--       { sku_id, location_id, classification, action, action_type,
--         data_conflict, total_inv_value, qty_on_hand, days_since_sale,
--         sale_frequency, abc_class, supplier_id },
--       ...                               -- up to 10 rows, LIQUIDATE only,
--                                         -- ordered by total_inv_value DESC
--     ]
--   }
--
-- Idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

CREATE OR REPLACE FUNCTION get_dead_stock_summary(p_report_date DATE)
RETURNS json
LANGUAGE sql
STABLE
SECURITY DEFINER
AS $$
WITH
agg AS (
    SELECT
        COALESCE(SUM(total_inv_value)
            FILTER (WHERE classification IN ('LIQUIDATE','MARKDOWN')), 0)          AS capital_at_risk,
        COALESCE(SUM(total_inv_value)
            FILTER (WHERE classification IN ('LIQUIDATE','MARKDOWN')
                      AND data_conflict IS NOT TRUE), 0)                            AS clean_capital,
        COUNT(*) FILTER (WHERE classification = 'LIQUIDATE')                        AS liquidate_count,
        COALESCE(SUM(total_inv_value) FILTER (WHERE classification = 'LIQUIDATE'), 0) AS liquidate_value,
        COUNT(*) FILTER (WHERE classification = 'MARKDOWN')                         AS markdown_count,
        COALESCE(SUM(total_inv_value) FILTER (WHERE classification = 'MARKDOWN'), 0)  AS markdown_value,
        COUNT(*) FILTER (WHERE UPPER(action_type::text) = 'TRANSFER')               AS transfer_count,
        COUNT(*) FILTER (WHERE data_conflict IS TRUE)                               AS data_conflict_count,
        COUNT(*)                                                                     AS total_positions
    FROM dead_stock_recommendations
    WHERE report_date = p_report_date
),
top10 AS (
    SELECT COALESCE(json_agg(t), '[]'::json) AS rows
    FROM (
        SELECT
            sku_id, location_id, classification, action, action_type,
            data_conflict, total_inv_value, qty_on_hand, days_since_sale,
            sale_frequency, abc_class, supplier_id
        FROM dead_stock_recommendations
        WHERE report_date = p_report_date
          AND classification = 'LIQUIDATE'
        ORDER BY total_inv_value DESC NULLS LAST
        LIMIT 10
    ) t
)
SELECT json_build_object(
    'kpis', row_to_json(agg),
    'top10', top10.rows
)
FROM agg, top10
$$;

-- Allow the dashboard role (authenticated / anon) to call this function.
GRANT EXECUTE ON FUNCTION get_dead_stock_summary(DATE) TO authenticated, anon;
