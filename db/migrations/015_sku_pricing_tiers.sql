-- =============================================================================
-- 015_sku_pricing_tiers.sql — Price tier matrix from Product cube
-- =============================================================================

CREATE TABLE IF NOT EXISTS sku_pricing_tiers (
    id              BIGSERIAL     PRIMARY KEY,
    sku_id          VARCHAR(50)   NOT NULL REFERENCES sku_master(sku_id),
    price_tier      VARCHAR(10)   NOT NULL,
    price_value     NUMERIC(12,4),
    gp_margin_pct   NUMERIC(8,4),
    unit_cost       NUMERIC(12,4),
    snapshot_date   DATE          NOT NULL DEFAULT CURRENT_DATE,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (sku_id, price_tier, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_pricing_sku_id  ON sku_pricing_tiers (sku_id);
CREATE INDEX IF NOT EXISTS idx_pricing_tier    ON sku_pricing_tiers (price_tier);
CREATE INDEX IF NOT EXISTS idx_pricing_date    ON sku_pricing_tiers (snapshot_date);
