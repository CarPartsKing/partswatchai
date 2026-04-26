-- =============================================================================
-- 049_sku_master_gp.sql
--
-- Extends the schema for GP-based ABC classification (ml/classify.py):
--
--   sku_master.gp_12m NUMERIC(14,4)
--       Rolling 12-month gross profit sum (unweighted) from
--       sales_detail_transactions (tran_code IN ('SL','SL-I')).
--       Populated nightly by ml/classify.py alongside abc_class.
--
--   sku_location_class — one row per (prod_line_pn, location_id)
--       Stores independent ABC ranking per location derived from
--       location-specific weighted GP.  xyz_class propagated from
--       sku_master for join convenience.  sku_id uses prod_line_pn
--       format (not a FK — the compound key differs from sku_master.sku_id).
--
-- Idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. gp_12m on sku_master
-- ---------------------------------------------------------------------------
ALTER TABLE sku_master
    ADD COLUMN IF NOT EXISTS gp_12m NUMERIC(14, 4);

CREATE INDEX IF NOT EXISTS idx_sku_master_gp_12m
    ON sku_master (gp_12m DESC NULLS LAST)
    WHERE abc_class IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 2. sku_location_class — per-location ABC ranking
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sku_location_class (
    sku_id      VARCHAR(100) NOT NULL,
    location_id VARCHAR(20)  NOT NULL,
    abc_class   CHAR(1)      CHECK (abc_class IN ('A', 'B', 'C')),
    xyz_class   CHAR(1)      CHECK (xyz_class IN ('X', 'Y', 'Z')),
    gp_12m      NUMERIC(14, 4),
    sales_12m   NUMERIC(14, 4),
    run_date    DATE         NOT NULL,

    PRIMARY KEY (sku_id, location_id)
);

-- Dashboard: all A-class items at a given location
CREATE INDEX IF NOT EXISTS idx_sku_loc_class_location
    ON sku_location_class (location_id, abc_class);

-- Lookup by SKU across locations
CREATE INDEX IF NOT EXISTS idx_sku_loc_class_sku
    ON sku_location_class (sku_id, location_id);

-- Dashboard: high-value A items (top GP) per location
CREATE INDEX IF NOT EXISTS idx_sku_loc_class_gp
    ON sku_location_class (location_id, gp_12m DESC NULLS LAST)
    WHERE abc_class = 'A';

ALTER TABLE sku_location_class DISABLE ROW LEVEL SECURITY;
