-- =============================================================================
-- 029_understocking_report.sql
--
-- Chronic Understocking Report — answers the owner's question:
--   "What should each store be stocking more of instead of waiting for
--    transfers?"
--
-- This migration creates ONE table.  The aggregation runs in Python
-- (engine/understocking.py) using the same paginated supabase-py
-- pattern as ml/dead_stock.py — a server-side Postgres function was
-- tried first but exceeded PostgREST's statement_timeout (57014) on
-- the production cube (~3M snapshots × 100K SKU×location pairs).
--
-- A SKU is "chronically understocked" at a location when its on-hand
-- quantity sat below the reorder point for >=30% of observed snapshot
-- days in the lookback window (target 90 days; engine adapts to
-- whatever snapshot history actually exists).
-- =============================================================================

CREATE TABLE IF NOT EXISTS understocking_report (
    id                          BIGSERIAL    PRIMARY KEY,
    report_date                 DATE         NOT NULL,
    location_id                 VARCHAR(20)  NOT NULL,
    location_name               VARCHAR(60),
    sku_id                      VARCHAR(50)  NOT NULL,
    sku_description             TEXT,
    days_observed               INT          NOT NULL,
    days_below_reorder          INT          NOT NULL,
    stockout_days_pct           NUMERIC(5,4) NOT NULL,    -- 0.0000–1.0000
    avg_daily_demand            NUMERIC(12,4) NOT NULL,   -- avg_weekly_units / 7
    current_min_qty             NUMERIC(12,2) NOT NULL,   -- current reorder_point
    suggested_min_qty           NUMERIC(12,2) NOT NULL,   -- avg_daily × (lead_time + buffer)
    min_qty_gap                 NUMERIC(12,2) NOT NULL,   -- suggested − current
    unit_cost                   NUMERIC(12,2),
    inventory_value_at_risk     NUMERIC(14,2) NOT NULL,   -- avg_daily × unit_cost × 21
    transfer_recommended_count  INT          NOT NULL DEFAULT 0,
    priority_score              NUMERIC(16,4) NOT NULL,   -- stockout_pct × avg_daily × unit_cost
    created_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    -- Run-completion marker.  Inserts during a run leave this NULL;
    -- the engine sets it to NOW() ONCE all batches for `report_date`
    -- are successfully written.  Dashboard filters
    -- `run_completed_at IS NOT NULL` so consumers never read a
    -- mid-write partial report.  See engine/understocking.py.
    run_completed_at            TIMESTAMPTZ,
    UNIQUE (report_date, location_id, sku_id)
);

-- Backfill if the column is being added on a re-apply over an older
-- (pre-completion-marker) version of this table.  Safe no-op when the
-- column already exists with the same definition.
ALTER TABLE understocking_report
    ADD COLUMN IF NOT EXISTS run_completed_at TIMESTAMPTZ;

-- Backfill legacy rows that pre-date the publication marker so the
-- dashboard is not blank between applying this migration and the next
-- engine run.  Sets run_completed_at = created_at for any row where
-- it's still NULL — i.e. anything written by an older engine version.
-- Idempotent: subsequent re-applies are no-ops (no NULLs left).
UPDATE understocking_report
   SET run_completed_at = created_at
 WHERE run_completed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_understock_report_date
    ON understocking_report (report_date DESC);
CREATE INDEX IF NOT EXISTS idx_understock_loc
    ON understocking_report (location_id, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_understock_priority
    ON understocking_report (report_date DESC, priority_score DESC);

ALTER TABLE understocking_report DISABLE ROW LEVEL SECURITY;

-- If a previous run created the obsolete server-side aggregation
-- function (the first design that timed out), drop it so reapplying
-- this migration leaves a clean slate.  Safe no-op when absent.
DROP FUNCTION IF EXISTS fn_run_understocking_report(
    text[], int, int, numeric, numeric, int, boolean
);
