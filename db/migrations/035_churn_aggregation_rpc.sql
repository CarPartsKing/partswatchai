-- =============================================================================
-- 035_churn_aggregation_rpc.sql
--
-- Server-side aggregation function for churn_detection.py.
--
-- get_churn_buckets() groups sales_detail_transactions for a single location
-- into one row per customer, computing baseline and comparison spend totals
-- that churn_detection.py classifies into CHURNED / DECLINING / STABLE.
--
-- Running aggregation in Postgres (rather than fetching raw rows to Python)
-- avoids Supabase statement timeouts on high-volume locations — the
-- idx_sdt_loc_date composite index on (location_id, tran_date) makes the
-- GROUP BY cheap.
--
-- Parameters
-- ----------
-- p_location_id      location to aggregate (equality filter)
-- p_window_start     start of the 2-year lookback window
-- p_baseline_end     end of the 18-month baseline period
-- p_comparison_start start of the 90-day comparison period (= effective_today - 90d)
--
-- Returns one row per distinct ship_to value with:
--   customer_id        — ship_to value
--   baseline_sales     — sum of sales in [p_window_start, p_baseline_end]
--   baseline_tx        — count of rows   in [p_window_start, p_baseline_end]
--   comparison_sales   — sum of sales in [p_comparison_start, max(tran_date)]
--   last_purchase_date — max tran_date across the full window
--
-- Safe to re-run — CREATE OR REPLACE.
-- =============================================================================

CREATE OR REPLACE FUNCTION get_churn_buckets(
    p_location_id      TEXT,
    p_window_start     DATE,
    p_baseline_end     DATE,
    p_comparison_start DATE
)
RETURNS TABLE (
    customer_id        TEXT,
    baseline_sales     NUMERIC,
    baseline_tx        BIGINT,
    comparison_sales   NUMERIC,
    last_purchase_date DATE
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        ship_to::TEXT,
        COALESCE(
            SUM(sales) FILTER (WHERE tran_date BETWEEN p_window_start AND p_baseline_end),
            0
        )                                                              AS baseline_sales,
        COUNT(*)     FILTER (WHERE tran_date BETWEEN p_window_start AND p_baseline_end)
                                                                       AS baseline_tx,
        COALESCE(
            SUM(sales) FILTER (WHERE tran_date >= p_comparison_start),
            0
        )                                                              AS comparison_sales,
        MAX(tran_date)                                                 AS last_purchase_date
    FROM  sales_detail_transactions
    WHERE location_id = p_location_id
      AND tran_date   >= p_window_start
      AND ship_to     IS NOT NULL
      AND sales       IS NOT NULL
    GROUP BY ship_to
$$;
