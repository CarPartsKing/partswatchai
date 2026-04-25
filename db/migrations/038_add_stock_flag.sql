-- =============================================================================
-- 038_add_stock_flag.sql
--
-- Adds stock_flag to sales_detail_transactions and fixes the OPSL analysis
-- to use it instead of a non-existent tran_code = 'OPSL'.
--
-- Background: OPSL is not a tran code in AutoCube_DTR_23160.  The Sales
-- Detail cube exposes outside purchases via [Sales Detail].[Stock Flag]:
--   Y = sold from stock
--   N = sourced from outside (the real "OPSL" signal)
-- The 61 rows with tran_code = 'OPSL' in the table were spurious test data.
--
-- All statements are idempotent — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Add the column
-- ---------------------------------------------------------------------------
ALTER TABLE sales_detail_transactions
    ADD COLUMN IF NOT EXISTS stock_flag VARCHAR(1);

-- ---------------------------------------------------------------------------
-- 2. Delete the spurious OPSL test rows
-- ---------------------------------------------------------------------------
DELETE FROM sales_detail_transactions WHERE tran_code = 'OPSL';

-- ---------------------------------------------------------------------------
-- 3. Index on stock_flag for filtered queries
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_sdt_stock_flag
    ON sales_detail_transactions (stock_flag);

-- ---------------------------------------------------------------------------
-- 4. Replace the old OPSL partial covering index (was WHERE tran_code='OPSL',
--    which never matched real data) with one that covers the real signal.
-- ---------------------------------------------------------------------------
DROP INDEX IF EXISTS idx_sdt_opsl_covering;

CREATE INDEX IF NOT EXISTS idx_sdt_opsl_covering
    ON sales_detail_transactions (tran_date, location_id, prod_line_pn, sales, gross_profit)
    WHERE stock_flag = 'N' AND tran_code = 'SL';

-- ---------------------------------------------------------------------------
-- 5. Update get_opsl_summary RPC — filter stock_flag = 'N' instead of
--    tran_code = 'OPSL'.  DROP + CREATE to change the WHERE clause in the
--    partial index dependency cleanly.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_opsl_summary(
    p_start_date DATE
)
RETURNS TABLE (
    prod_line_pn      TEXT,
    location_id       TEXT,
    opsl_count        BIGINT,
    total_opsl_sales  NUMERIC,
    total_opsl_gp     NUMERIC,
    last_opsl_date    DATE
)
LANGUAGE sql STABLE SECURITY DEFINER
SET statement_timeout = '60s'
AS $$
    SELECT
        prod_line_pn::TEXT,
        location_id::TEXT,
        COUNT(*)                       AS opsl_count,
        COALESCE(SUM(sales), 0)        AS total_opsl_sales,
        COALESCE(SUM(gross_profit), 0) AS total_opsl_gp,
        MAX(tran_date)                 AS last_opsl_date
    FROM sales_detail_transactions
    WHERE stock_flag    = 'N'
      AND tran_code     = 'SL'
      AND tran_date    >= p_start_date
      AND prod_line_pn IS NOT NULL
    GROUP BY prod_line_pn, location_id
$$;

GRANT EXECUTE ON FUNCTION get_opsl_summary(DATE) TO anon, authenticated;
