-- =============================================================================
-- 014_sales_transactions_flags.sql — Sales detail flags from Product cube
-- =============================================================================

ALTER TABLE sales_transactions ADD COLUMN IF NOT EXISTS is_backorder       BOOLEAN DEFAULT FALSE;
ALTER TABLE sales_transactions ADD COLUMN IF NOT EXISTS is_core_return     BOOLEAN DEFAULT FALSE;
ALTER TABLE sales_transactions ADD COLUMN IF NOT EXISTS is_warranty        BOOLEAN DEFAULT FALSE;
ALTER TABLE sales_transactions ADD COLUMN IF NOT EXISTS is_price_override  BOOLEAN DEFAULT FALSE;
ALTER TABLE sales_transactions ADD COLUMN IF NOT EXISTS counterman         VARCHAR(50);

CREATE INDEX IF NOT EXISTS idx_sales_warranty  ON sales_transactions (is_warranty) WHERE is_warranty = TRUE;
CREATE INDEX IF NOT EXISTS idx_sales_backorder ON sales_transactions (is_backorder) WHERE is_backorder = TRUE;
