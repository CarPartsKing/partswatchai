-- =============================================================================
-- 033_sales_detail_transactions.sql
--
-- Sales Detail Transactions — line-item sales pulled from the Autocube
-- Sales Detail cube, filtered to tran codes SL, SL-I, and OPSL.
--
-- Populated by extract/extract_sales_detail.py (rolling 2-year lookback).
-- transaction_id is a deterministic SHA-1 hash of the natural key
-- (tran_date | location_id | tran_code | ship_to | prod_line_pn).
--
-- prod_line_pn is the combined product-line + part-number identifier from
-- [Product].[Prod Line PN].[Prod Line PN] in the Sales Detail cube.
-- It is NOT a FK to sku_master — the compound key format differs from the
-- bare part numbers in sku_master.sku_id.
--
-- All statements are idempotent — safe to re-run.
-- =============================================================================

CREATE TABLE IF NOT EXISTS sales_detail_transactions (
    id              BIGSERIAL      PRIMARY KEY,
    transaction_id  VARCHAR(120)   NOT NULL UNIQUE,
    tran_date       DATE           NOT NULL,
    location_id     VARCHAR(20)    NOT NULL,
    tran_code       VARCHAR(20)    NOT NULL,
    ship_to         VARCHAR(200),
    prod_line_pn    VARCHAR(100),
    qty_ship        NUMERIC(12,4),
    sales           NUMERIC(14,4),
    gross_profit    NUMERIC(14,4),
    backorder_qty   NUMERIC(12,4),
    created_at      TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sdt_tran_date    ON sales_detail_transactions (tran_date);
CREATE INDEX IF NOT EXISTS idx_sdt_location_id  ON sales_detail_transactions (location_id);
CREATE INDEX IF NOT EXISTS idx_sdt_tran_code    ON sales_detail_transactions (tran_code);
CREATE INDEX IF NOT EXISTS idx_sdt_prod_line_pn ON sales_detail_transactions (prod_line_pn);
CREATE INDEX IF NOT EXISTS idx_sdt_ship_to      ON sales_detail_transactions (ship_to);
CREATE INDEX IF NOT EXISTS idx_sdt_loc_date     ON sales_detail_transactions (location_id, tran_date);
CREATE INDEX IF NOT EXISTS idx_sdt_pn_date      ON sales_detail_transactions (prod_line_pn, tran_date);

ALTER TABLE sales_detail_transactions DISABLE ROW LEVEL SECURITY;
