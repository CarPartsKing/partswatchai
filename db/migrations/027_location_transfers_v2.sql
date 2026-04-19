-- =============================================================================
-- 027_location_transfers_v2.sql
--
-- Rebuild location_transfers to capture inter-location transfers via the
-- Sales Detail cube using Tran Code "-I" (INTERCO) variants instead of the
-- broken [Transfer Locs] dimension (empty in AutoCube_DTR_23160).
--
-- The previous schema (migration 012) used a synthetic transfer_id and a
-- from/to pair derived from [Transfer Locs] members that never populated.
-- Table has always been empty, so it is safe to drop and recreate.
-- =============================================================================

DROP TABLE IF EXISTS location_transfers CASCADE;

CREATE TABLE location_transfers (
    id                 BIGSERIAL     PRIMARY KEY,
    transfer_date      DATE          NOT NULL,
    sku_id             VARCHAR(50)   NOT NULL REFERENCES sku_master(sku_id),
    from_location_id   VARCHAR(20)   NOT NULL,
    to_location_id     VARCHAR(20)   NOT NULL,
    tran_code          VARCHAR(20)   NOT NULL,
    qty_transferred    NUMERIC(12,4) NOT NULL DEFAULT 0,
    transfer_value     NUMERIC(14,2),
    created_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (transfer_date, sku_id, from_location_id, to_location_id, tran_code)
);

CREATE INDEX idx_xfer_v2_date          ON location_transfers (transfer_date);
CREATE INDEX idx_xfer_v2_sku           ON location_transfers (sku_id);
CREATE INDEX idx_xfer_v2_from          ON location_transfers (from_location_id);
CREATE INDEX idx_xfer_v2_to            ON location_transfers (to_location_id);
CREATE INDEX idx_xfer_v2_tran_code     ON location_transfers (tran_code);
CREATE INDEX idx_xfer_v2_sku_date      ON location_transfers (sku_id, transfer_date);

ALTER TABLE location_transfers DISABLE ROW LEVEL SECURITY;
