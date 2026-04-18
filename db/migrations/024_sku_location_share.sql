-- =====================================================================
-- Migration 024 — sku_location_share
-- =====================================================================
-- Purpose
--   Persist each location's share of a SKU's network-wide demand so the
--   reorder engine can scale a single 'ALL'-location network forecast
--   down to per-location predictions on the fly.  This is the storage
--   companion of the network-level rolling fallback added to
--   ml/forecast_rolling.py.
--
-- Why a separate table (not an extra column on forecast_results)?
--   * forecast_results is keyed by (sku_id, location_id, forecast_date,
--     model_type, run_date) — a per-day grain that would explode shares
--     into 30× duplication per (sku, location).
--   * Shares change slowly (driven by 91-day rolling sales mix) and are
--     queried in bulk by the reorder engine.  A small (sku, location,
--     run_date) table is cheap to fetch and keeps forecast_results pure.
--
-- Granularity
--   One row per (sku_id, location_id, run_date).  Multiple historical
--   runs are retained so we can audit shifting demand mix over time;
--   reorder.py always reads the latest run_date per SKU.
--
-- Share semantics
--   share = effective_demand_at_location / sum(effective_demand_all_locations)
--   Computed over the same 91-day lookback used by the rolling forecast,
--   on anomaly-, residual-, warranty-excluded transactions.  Shares for a
--   given SKU sum to ≤ 1.0 (locations with zero sales are not stored).
--
-- Idempotency
--   Safe to re-run; ON CONFLICT DO UPDATE keeps the row in step with the
--   latest forecast run.
-- =====================================================================

CREATE TABLE IF NOT EXISTS sku_location_share (
    sku_id        TEXT          NOT NULL,
    location_id   TEXT          NOT NULL,
    share         NUMERIC(8, 6) NOT NULL CHECK (share >= 0 AND share <= 1),
    run_date      DATE          NOT NULL,
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (sku_id, location_id, run_date)
);

-- Reorder engine fetches the latest run_date per SKU; this composite
-- index covers the (sku_id, run_date DESC) lookup pattern used in
-- engine.reorder._fetch_location_shares.
CREATE INDEX IF NOT EXISTS idx_sku_location_share_sku_run
    ON sku_location_share (sku_id, run_date DESC);

-- Allows efficient "give me all shares for the most recent run" queries.
CREATE INDEX IF NOT EXISTS idx_sku_location_share_run
    ON sku_location_share (run_date DESC);

COMMENT ON TABLE sku_location_share IS
    'Per-location demand share of a SKU''s network total over the rolling '
    'forecast lookback.  Used by engine/reorder.py to scale network-level '
    'forecasts (location_id=''ALL'') down to per-location predictions.';
