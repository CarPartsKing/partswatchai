-- =============================================================================
-- 041_add_salesman_to_sales_detail.sql
--
-- Adds salesman_id to sales_detail_transactions.
--
-- Source: [Counterman].[Counterman].[Counterman] from the AutoCube Sales
-- Detail cube.  The extract (extract_sales_detail.py) is updated in the same
-- release to pull this dimension and populate the column going forward.
--
-- Existing rows will have salesman_id = NULL until the next full 2-year
-- backfill runs (trigger manually via:
--   python -m extract.extract_sales_detail --lookback-days 730
-- or let the scheduled nightly run cover it incrementally over ~730 nights).
--
-- All statements are idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

ALTER TABLE sales_detail_transactions
    ADD COLUMN IF NOT EXISTS salesman_id VARCHAR(20);

CREATE INDEX IF NOT EXISTS idx_sdt_salesman
    ON sales_detail_transactions (salesman_id)
    WHERE salesman_id IS NOT NULL;
