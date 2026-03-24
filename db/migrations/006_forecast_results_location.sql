-- =============================================================================
-- Migration 006: Add location_id to forecast_results
-- Run this in the Supabase SQL Editor before using ml/forecast_rolling.py
-- Safe to re-run — all operations are IF NOT EXISTS / IF EXISTS.
-- =============================================================================

-- 1. Add the new column. Existing rows (Prophet / LightGBM output that
--    does not have a location breakdown) default to 'ALL' to indicate a
--    network-wide forecast.
ALTER TABLE forecast_results
    ADD COLUMN IF NOT EXISTS location_id TEXT NOT NULL DEFAULT 'ALL';

-- 2. Drop the old unique constraint (created without location_id).
--    The auto-generated constraint name matches Postgres naming convention.
ALTER TABLE forecast_results
    DROP CONSTRAINT IF EXISTS
        forecast_results_sku_id_forecast_date_model_type_run_date_key;

-- 3. New unique index — now includes location_id so rolling_avg rows
--    written per-location do not collide with network-level Prophet rows.
CREATE UNIQUE INDEX IF NOT EXISTS uidx_forecast_sku_loc_date_model_run
    ON forecast_results (sku_id, location_id, forecast_date, model_type, run_date);

-- 4. Index for location-level dashboard queries.
CREATE INDEX IF NOT EXISTS idx_forecast_location_id
    ON forecast_results (location_id);

COMMENT ON COLUMN forecast_results.location_id IS
    '''ALL'' = network-level forecast (Prophet/LightGBM). '
    'A specific LOC-nnn = per-location rolling average.';
