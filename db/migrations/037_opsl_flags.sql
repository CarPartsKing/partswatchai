-- =============================================================================
-- 037_opsl_flags.sql
--
-- OPSL (Outside Purchase to fill a customer order) analysis.
-- Every OPSL row = a part the store didn't stock, sourced from an external
-- vendor.  Each event represents lost margin vs. stocking it locally.
--
-- This migration:
--   1. Partial covering index on sales_detail_transactions for OPSL queries
--   2. get_opsl_summary() RPC — server-side GROUP BY to avoid timeouts
--   3. opsl_flags output table
--
-- All statements are idempotent — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Index
-- ---------------------------------------------------------------------------
-- Partial index covers only OPSL rows (small fraction of the table).
-- Lets get_opsl_summary do an index-only scan over the 90-day window
-- without touching the main heap.
CREATE INDEX IF NOT EXISTS idx_sdt_opsl_covering
    ON sales_detail_transactions (tran_date, location_id, prod_line_pn, sales, gross_profit)
    WHERE tran_code = 'OPSL';

-- ---------------------------------------------------------------------------
-- RPC
-- ---------------------------------------------------------------------------
-- Returns one row per (prod_line_pn, location_id) for OPSL transactions
-- in the requested window.  SECURITY DEFINER + explicit timeout avoids
-- the anon role's 8-second statement budget.
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
        COUNT(*)                    AS opsl_count,
        COALESCE(SUM(sales), 0)     AS total_opsl_sales,
        COALESCE(SUM(gross_profit), 0) AS total_opsl_gp,
        MAX(tran_date)              AS last_opsl_date
    FROM sales_detail_transactions
    WHERE tran_code      = 'OPSL'
      AND tran_date     >= p_start_date
      AND prod_line_pn  IS NOT NULL
    GROUP BY prod_line_pn, location_id
$$;

GRANT EXECUTE ON FUNCTION get_opsl_summary(DATE) TO anon, authenticated;

-- ---------------------------------------------------------------------------
-- Output table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS opsl_flags (
    id                       BIGSERIAL       PRIMARY KEY,

    -- Natural key
    prod_line_pn             VARCHAR(100)    NOT NULL,
    location_id              VARCHAR(20)     NOT NULL,

    -- Aggregates
    opsl_count               INTEGER         NOT NULL DEFAULT 0,
    total_opsl_sales         NUMERIC(14, 4)  NOT NULL DEFAULT 0,
    total_opsl_gp            NUMERIC(14, 4)  NOT NULL DEFAULT 0,
    avg_gp_pct               NUMERIC(7, 4),
    estimated_margin_recovery NUMERIC(14, 4),

    -- Derived fields
    flag                     VARCHAR(10)     NOT NULL
                                 CHECK (flag IN ('HIGH', 'MEDIUM', 'LOW')),
    last_opsl_date           DATE,
    in_reorder_queue         BOOLEAN         NOT NULL DEFAULT FALSE,

    -- Housekeeping
    run_date                 DATE            NOT NULL,
    created_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (prod_line_pn, location_id)
);

CREATE INDEX IF NOT EXISTS idx_opsl_location
    ON opsl_flags (location_id);
CREATE INDEX IF NOT EXISTS idx_opsl_flag
    ON opsl_flags (flag);
CREATE INDEX IF NOT EXISTS idx_opsl_run_date
    ON opsl_flags (run_date);
CREATE INDEX IF NOT EXISTS idx_opsl_margin_recovery
    ON opsl_flags (estimated_margin_recovery DESC);

ALTER TABLE opsl_flags DISABLE ROW LEVEL SECURITY;
