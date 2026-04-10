-- =============================================================================
-- 013_sku_master_product_cube.sql — Enrich sku_master with Product cube fields
-- =============================================================================

ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS primary_supplier_id VARCHAR(50);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS warehouse_aisle     VARCHAR(20);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS warehouse_bin       VARCHAR(20);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS warehouse_zone      VARCHAR(20);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS is_core_item        BOOLEAN DEFAULT FALSE;
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS is_stocked          BOOLEAN DEFAULT TRUE;
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS supplier_code       VARCHAR(50);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS own_class_1         VARCHAR(50);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS own_class_2         VARCHAR(50);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS own_class_3         VARCHAR(50);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS own_class_4         VARCHAR(50);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS own_class_5         VARCHAR(50);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS product_line        VARCHAR(50);
ALTER TABLE sku_master ADD COLUMN IF NOT EXISTS movement_code       VARCHAR(20);

CREATE INDEX IF NOT EXISTS idx_sku_supplier   ON sku_master (primary_supplier_id);
CREATE INDEX IF NOT EXISTS idx_sku_core       ON sku_master (is_core_item) WHERE is_core_item = TRUE;
CREATE INDEX IF NOT EXISTS idx_sku_stocked    ON sku_master (is_stocked);
CREATE INDEX IF NOT EXISTS idx_sku_prod_line  ON sku_master (product_line);
