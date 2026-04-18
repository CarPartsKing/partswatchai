-- =============================================================================
-- Migration 023 — dead_stock_recommendations table
--
-- Persists the per-position output of `ml/dead_stock.py` so the dashboard can
-- display LIQUIDATE / MARKDOWN candidates and the assistant can surface them
-- in chat context.  Before this migration the dead-stock pipeline only set
-- the boolean `sku_master.is_dead_stock` flag and wrote a JSON report — the
-- ranked list and per-position values were never queryable from Supabase.
--
-- GRAIN: one row per (report_date, sku_id, location_id).  The pipeline
-- writes only LIQUIDATE and MARKDOWN classifications (skipping MONITOR /
-- HEALTHY) so the table stays small relative to the full inventory grid.
--
-- IDEMPOTENCY: every statement uses IF NOT EXISTS — safe to re-run.
-- =============================================================================

CREATE TABLE IF NOT EXISTS dead_stock_recommendations (
    report_date         DATE        NOT NULL,
    sku_id              TEXT        NOT NULL,
    location_id         TEXT        NOT NULL,
    classification      TEXT        NOT NULL,   -- LIQUIDATE | MARKDOWN
    action              TEXT        NOT NULL,   -- WRITEOFF | RETURN | MARKDOWN | LIQUIDATE
    dead_stock_score    NUMERIC     NOT NULL,
    total_inv_value     NUMERIC     NOT NULL,
    qty_on_hand         NUMERIC     NOT NULL,
    unit_cost           NUMERIC,
    days_since_sale     INTEGER,
    sale_frequency      NUMERIC,
    abc_class           TEXT,
    supplier_id         TEXT,
    part_category       TEXT,
    sub_category        TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (report_date, sku_id, location_id)
);

-- The dashboard always queries "latest report_date" then filters; speeds that up.
CREATE INDEX IF NOT EXISTS idx_dead_stock_recs_report_date
    ON dead_stock_recommendations (report_date DESC);

-- "Top liquidate by dollar value" is the single most-used dashboard query.
CREATE INDEX IF NOT EXISTS idx_dead_stock_recs_liquidate_value
    ON dead_stock_recommendations (report_date, total_inv_value DESC)
    WHERE classification = 'LIQUIDATE';
