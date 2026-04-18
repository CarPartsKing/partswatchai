-- =============================================================================
-- 021_unit_cost.sql — real per-unit cost from the Autologue Product cube
--
-- Until now the dead-stock classifier (ml/dead_stock.py) fell back to a
-- $10.00 placeholder cost for every SKU, producing meaningless capital-at-
-- risk dollar figures (the well-known "$7.8M" run).  The Product cube
-- exposes the real cost per SKU as [Measures].[Unit Cost], and the per-
-- location on-hand value as [Measures].[Ext Cost On Hand].
--
-- This migration adds:
--   1. inventory_snapshots.unit_cost  — per-snapshot, per-location cost
--      (so historical ext-cost-on-hand can be reconstructed).
--   2. sku_master.unit_cost           — most recently observed cost for
--      the SKU; queried by ml/dead_stock.py as the primary cost source,
--      with received-PO unit_cost as fallback.
--
-- All statements are idempotent (IF NOT EXISTS) — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- inventory_snapshots.unit_cost
-- ---------------------------------------------------------------------------
ALTER TABLE inventory_snapshots
    ADD COLUMN IF NOT EXISTS unit_cost NUMERIC(12,4);

COMMENT ON COLUMN inventory_snapshots.unit_cost IS
  'Per-unit cost at snapshot time, sourced from Product cube [Measures].[Unit Cost]. '
  'When null at extraction we derive it as Ext Cost On Hand / qty_on_hand if both '
  'are present and qty_on_hand > 0.';

-- ---------------------------------------------------------------------------
-- sku_master.unit_cost
-- ---------------------------------------------------------------------------
ALTER TABLE sku_master
    ADD COLUMN IF NOT EXISTS unit_cost NUMERIC(12,4);

COMMENT ON COLUMN sku_master.unit_cost IS
  'Most recent per-unit cost observed across any location (Product cube). '
  'ml/dead_stock.py prefers this over received-PO unit_cost so capital-at-'
  'risk reflects current replacement cost rather than historical PO prices.';

CREATE INDEX IF NOT EXISTS idx_sku_master_unit_cost
    ON sku_master (unit_cost)
    WHERE unit_cost IS NOT NULL;
