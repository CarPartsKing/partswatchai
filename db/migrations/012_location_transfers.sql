-- =============================================================================
-- 012_location_transfers.sql — Inter-location transfer history from Product cube
-- =============================================================================

CREATE TABLE IF NOT EXISTS location_transfers (
    id                 BIGSERIAL     PRIMARY KEY,
    transfer_id        VARCHAR(100)  UNIQUE,
    sku_id             VARCHAR(50)   NOT NULL REFERENCES sku_master(sku_id),
    from_location      VARCHAR(20)   NOT NULL,
    to_location        VARCHAR(20)   NOT NULL,
    transfer_date      DATE          NOT NULL,
    qty_transferred    NUMERIC(12,4) NOT NULL DEFAULT 0,
    transfer_cost      NUMERIC(14,4),
    created_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_xfer_sku_id   ON location_transfers (sku_id);
CREATE INDEX IF NOT EXISTS idx_xfer_from     ON location_transfers (from_location);
CREATE INDEX IF NOT EXISTS idx_xfer_to       ON location_transfers (to_location);
CREATE INDEX IF NOT EXISTS idx_xfer_date     ON location_transfers (transfer_date);
CREATE INDEX IF NOT EXISTS idx_xfer_sku_date ON location_transfers (sku_id, transfer_date);
