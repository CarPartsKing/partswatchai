-- =============================================================================
-- 040_churn_rpc_v3.sql
--
-- Fix: get_churn_buckets() was timing out on large locations because the
-- salesman_id subquery (JOIN on sales_transactions, ROW_NUMBER window fn)
-- ran inside the same statement as the 2-year aggregation.
--
-- Solution: split salesman attribution into a dedicated lightweight RPC.
--
--   1. get_churn_buckets() — v3: removes salesman CTEs entirely.
--      Returns 7 columns (salesman_id dropped).
--      The two-CTE structure collapses to a single parsed CTE + one
--      GROUP BY, removing the cross-table JOIN that caused the timeout.
--
--   2. get_salesman_map() — new.
--      Groups sales_detail_transactions by (ship_to, salesman_id) in the
--      comparison window only.  Python calls this after get_churn_buckets,
--      parses customer_id from ship_to, and picks the highest-count salesman
--      per account before writing rows.
--
-- All statements are idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

-- =============================================================================
-- 1. get_churn_buckets() v3 — salesman JOIN removed
-- =============================================================================

CREATE OR REPLACE FUNCTION get_churn_buckets(
    p_location_id      TEXT,
    p_window_start     DATE,
    p_baseline_end     DATE,
    p_comparison_start DATE
)
RETURNS TABLE (
    customer_id        TEXT,
    is_commercial      BOOLEAN,
    baseline_sales     NUMERIC,
    baseline_tx        BIGINT,
    baseline_months    BIGINT,
    comparison_sales   NUMERIC,
    last_purchase_date DATE
)
LANGUAGE sql
STABLE
AS $$
    -- Step 1: parse ship_to into (customer_id, is_commercial) and filter
    WITH parsed AS (
        SELECT
            TRIM(SPLIT_PART(ship_to, '-', 1))           AS customer_id,
            TRIM(SPLIT_PART(ship_to, '-', 2)) <> ''     AS is_commercial,
            tran_date,
            sales
        FROM sales_detail_transactions
        WHERE location_id = p_location_id
          AND tran_date   >= p_window_start
          AND ship_to     IS NOT NULL
          AND sales       IS NOT NULL
    )
    -- Step 2: aggregate per parsed customer_id
    SELECT
        customer_id,
        BOOL_OR(is_commercial)                                                  AS is_commercial,
        COALESCE(
            SUM(sales) FILTER (
                WHERE tran_date BETWEEN p_window_start AND p_baseline_end
            ), 0
        )                                                                       AS baseline_sales,
        COUNT(*) FILTER (
            WHERE tran_date BETWEEN p_window_start AND p_baseline_end
        )                                                                       AS baseline_tx,
        COUNT(DISTINCT DATE_TRUNC('month', tran_date)) FILTER (
            WHERE tran_date BETWEEN p_window_start AND p_baseline_end
        )                                                                       AS baseline_months,
        COALESCE(
            SUM(sales) FILTER (WHERE tran_date >= p_comparison_start),
            0
        )                                                                       AS comparison_sales,
        MAX(tran_date)                                                          AS last_purchase_date
    FROM parsed
    WHERE customer_id <> ''
    GROUP BY customer_id
$$;


-- get_salesman_map() is not implemented here.
-- salesman_id is extracted from [Counterman].[Counterman].[Counterman] in the
-- Sales Detail cube and stored directly on sales_detail_transactions
-- (migration 041).  get_churn_buckets() can join it from there once the
-- column is populated; for now salesman_id is written as NULL by the pipeline.
