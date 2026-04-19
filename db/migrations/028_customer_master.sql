-- =============================================================================
-- 028_customer_master.sql
--
-- Customer dimension table populated from the Sales Summary cube, which
-- exposes [Customer].[Salesman] / [Cust Type] / [Status] — attributes that
-- the Sales Detail cube does NOT carry. Used by ml/churn.py for salesman
-- routing and by other engines that need customer-level attributes.
-- =============================================================================

CREATE TABLE IF NOT EXISTS customer_master (
    customer_id       VARCHAR(50)  PRIMARY KEY,
    salesman_id       VARCHAR(50),
    customer_type     VARCHAR(50),
    customer_status   VARCHAR(20),
    last_updated      DATE         NOT NULL DEFAULT CURRENT_DATE,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_custmaster_salesman ON customer_master (salesman_id);
CREATE INDEX IF NOT EXISTS idx_custmaster_type     ON customer_master (customer_type);
CREATE INDEX IF NOT EXISTS idx_custmaster_status   ON customer_master (customer_status);

ALTER TABLE customer_master DISABLE ROW LEVEL SECURITY;
