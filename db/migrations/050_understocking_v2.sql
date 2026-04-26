-- =============================================================================
-- 050_understocking_v2.sql
--
-- Extends understocking_report and reorder_recommendations with three
-- capabilities added by the engine/understocking.py v2 upgrade:
--
--   understocking_report:
--     financial_severity NUMERIC(14,2) DEFAULT 0
--         GP-weighted urgency score per row:
--           short_pct × avg_gp_per_unit × avg_daily_demand
--         Rows are sorted by this value (double_confirmed rows first).
--
--     avg_gp_pct NUMERIC(6,4)
--         Average gross profit per unit (sum GP / sum qty_ship) from
--         sales_detail_transactions over the last 90 days, per prod_line_pn.
--         Used as the per-unit GP component of financial_severity.
--
--     network_flag BOOLEAN DEFAULT FALSE
--         TRUE when this SKU+location is part of a network-level trigger
--         (5+ locations simultaneously short_pct > 0.5 on the same SKU).
--         A single DC bulk-purchase rec is written to reorder_recommendations
--         when this fires.
--
--     double_confirmed BOOLEAN DEFAULT FALSE
--         TRUE when the SKU+location also appears in opsl_flags with
--         flag IN ('HIGH', 'MEDIUM').  Double-confirmed rows sort to the
--         top of the output (before financial_severity).
--
--   reorder_recommendations:
--     source VARCHAR(50)
--         Identifies the generating engine.  Set to 'NETWORK_UNDERSTOCKING'
--         for DC bulk-purchase rows emitted by the network trigger.
--
--     notes TEXT
--         Free-text context.  For NETWORK_UNDERSTOCKING rows: lists every
--         affected location and its short percentage.
--
-- Idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. understocking_report — new columns
-- ---------------------------------------------------------------------------
ALTER TABLE understocking_report
    ADD COLUMN IF NOT EXISTS financial_severity NUMERIC(14, 2) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_gp_pct         NUMERIC(6,  4),
    ADD COLUMN IF NOT EXISTS network_flag       BOOLEAN       NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS double_confirmed   BOOLEAN       NOT NULL DEFAULT FALSE;

-- Dashboard: primary sort (double-confirmed first, then by financial severity)
CREATE INDEX IF NOT EXISTS idx_understock_financial_severity
    ON understocking_report (report_date DESC, double_confirmed DESC, financial_severity DESC);

-- Dashboard: network-trigger items
CREATE INDEX IF NOT EXISTS idx_understock_network_flag
    ON understocking_report (report_date DESC, network_flag)
    WHERE network_flag = TRUE;

-- Dashboard: double-confirmed items
CREATE INDEX IF NOT EXISTS idx_understock_double_confirmed
    ON understocking_report (report_date DESC, double_confirmed)
    WHERE double_confirmed = TRUE;

-- ---------------------------------------------------------------------------
-- 2. reorder_recommendations — source + notes for network trigger rows
-- ---------------------------------------------------------------------------
-- ADD COLUMN IF NOT EXISTS is a no-op when the column already exists with a
-- narrower type (e.g. VARCHAR(20) added by a prior migration or manual DDL).
-- The ALTER COLUMN below widens source to VARCHAR(50) unconditionally so the
-- full 'NETWORK_UNDERSTOCKING' value fits.  Safe on an empty column.
ALTER TABLE reorder_recommendations
    ADD COLUMN IF NOT EXISTS source VARCHAR(50),
    ADD COLUMN IF NOT EXISTS notes  TEXT;

-- Widen source in case it pre-existed as VARCHAR(20).
ALTER TABLE reorder_recommendations
    ALTER COLUMN source TYPE VARCHAR(50);

CREATE INDEX IF NOT EXISTS idx_reorder_source
    ON reorder_recommendations (source)
    WHERE source IS NOT NULL;
