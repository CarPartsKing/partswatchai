-- =============================================================================
-- 001_initial_schema.sql — partswatch-ai initial database schema
--
-- Run this once in the Supabase SQL Editor:
--   Dashboard → SQL Editor → New Query → paste → Run
--
-- All statements are idempotent (IF NOT EXISTS) — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. sku_master — must be created first (all other tables FK to sku_id)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sku_master (
    id                        BIGSERIAL    PRIMARY KEY,
    sku_id                    VARCHAR(50)  NOT NULL UNIQUE,
    description               TEXT,
    brand                     VARCHAR(100),
    part_category             VARCHAR(100),
    sub_category              VARCHAR(100),
    unit_of_measure           VARCHAR(20)  DEFAULT 'EA',
    abc_class                 VARCHAR(1)   CHECK (abc_class IN ('A','B','C')),
    is_active                 BOOLEAN      NOT NULL DEFAULT TRUE,
    is_dead_stock             BOOLEAN      NOT NULL DEFAULT FALSE,
    -- 0.0 (no sensitivity) to 1.0 (high) — populated by ML nightly
    weather_sensitivity_score NUMERIC(5,4) DEFAULT 0.0,
    last_sale_date            DATE,
    avg_weekly_units          NUMERIC(10,4),
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at                TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sku_master_abc_class  ON sku_master (abc_class);
CREATE INDEX IF NOT EXISTS idx_sku_master_brand      ON sku_master (brand);
CREATE INDEX IF NOT EXISTS idx_sku_master_category   ON sku_master (part_category);
CREATE INDEX IF NOT EXISTS idx_sku_master_active     ON sku_master (is_active);
CREATE INDEX IF NOT EXISTS idx_sku_master_last_sale  ON sku_master (last_sale_date);

-- ---------------------------------------------------------------------------
-- 2. sales_transactions — raw line-item sales from PartsWatch
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sales_transactions (
    id                     BIGSERIAL     PRIMARY KEY,
    transaction_id         VARCHAR(80)   UNIQUE,
    sku_id                 VARCHAR(50)   NOT NULL REFERENCES sku_master(sku_id),
    location_id            VARCHAR(20)   NOT NULL,
    transaction_date       DATE          NOT NULL,
    qty_sold               NUMERIC(12,4) NOT NULL DEFAULT 0,
    unit_price             NUMERIC(12,4),
    total_revenue          NUMERIC(14,4),
    -- TRUE when qty_on_hand was 0 at the time of the sale
    is_stockout            BOOLEAN       NOT NULL DEFAULT FALSE,
    -- ML-imputed demand estimate when a stockout suppressed true demand
    lost_sales_imputation  NUMERIC(12,4) DEFAULT 0,
    created_at             TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sales_sku_id   ON sales_transactions (sku_id);
CREATE INDEX IF NOT EXISTS idx_sales_date     ON sales_transactions (transaction_date);
CREATE INDEX IF NOT EXISTS idx_sales_location ON sales_transactions (location_id);
CREATE INDEX IF NOT EXISTS idx_sales_sku_date ON sales_transactions (sku_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_sales_stockout ON sales_transactions (is_stockout) WHERE is_stockout = TRUE;

-- ---------------------------------------------------------------------------
-- 3. inventory_snapshots — nightly on-hand snapshot per SKU per location
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inventory_snapshots (
    id             BIGSERIAL     PRIMARY KEY,
    sku_id         VARCHAR(50)   NOT NULL REFERENCES sku_master(sku_id),
    location_id    VARCHAR(20)   NOT NULL,
    snapshot_date  DATE          NOT NULL,
    qty_on_hand    NUMERIC(12,4) NOT NULL DEFAULT 0,
    qty_on_order   NUMERIC(12,4) NOT NULL DEFAULT 0,
    -- auto-computed: TRUE when qty_on_hand <= 0
    is_stockout    BOOLEAN GENERATED ALWAYS AS (qty_on_hand <= 0) STORED,
    reorder_point  NUMERIC(12,4),
    reorder_qty    NUMERIC(12,4),
    created_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (sku_id, location_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_inv_sku_id        ON inventory_snapshots (sku_id);
CREATE INDEX IF NOT EXISTS idx_inv_snapshot_date ON inventory_snapshots (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_inv_location      ON inventory_snapshots (location_id);
CREATE INDEX IF NOT EXISTS idx_inv_sku_date      ON inventory_snapshots (sku_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_inv_stockout      ON inventory_snapshots (is_stockout) WHERE is_stockout = TRUE;

-- ---------------------------------------------------------------------------
-- 4. purchase_orders — PO lines with generated performance fields
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS purchase_orders (
    id                     BIGSERIAL     PRIMARY KEY,
    po_number              VARCHAR(80)   NOT NULL,
    line_number            INTEGER       NOT NULL DEFAULT 1,
    sku_id                 VARCHAR(50)   NOT NULL REFERENCES sku_master(sku_id),
    supplier_id            VARCHAR(50)   NOT NULL,
    location_id            VARCHAR(20)   NOT NULL,
    po_date                DATE          NOT NULL,
    expected_delivery_date DATE,
    actual_delivery_date   DATE,
    qty_ordered            NUMERIC(12,4) NOT NULL DEFAULT 0,
    qty_received           NUMERIC(12,4) DEFAULT 0,
    unit_cost              NUMERIC(12,4),
    status                 VARCHAR(20)   NOT NULL DEFAULT 'open'
                               CHECK (status IN ('open','received','partial','cancelled')),
    -- positive = late, negative = early; NULL until delivered
    lead_time_variance     NUMERIC(8,2) GENERATED ALWAYS AS (
        CASE
            WHEN actual_delivery_date IS NOT NULL AND expected_delivery_date IS NOT NULL
            THEN (actual_delivery_date - expected_delivery_date)::NUMERIC
            ELSE NULL
        END
    ) STORED,
    -- 0–1 ratio; NULL until any receipt posted
    fill_rate_pct          NUMERIC(6,4) GENERATED ALWAYS AS (
        CASE
            WHEN qty_ordered > 0 AND qty_received IS NOT NULL
            THEN LEAST(qty_received / qty_ordered, 1.0)
            ELSE NULL
        END
    ) STORED,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (po_number, line_number)
);

CREATE INDEX IF NOT EXISTS idx_po_sku_id     ON purchase_orders (sku_id);
CREATE INDEX IF NOT EXISTS idx_po_supplier   ON purchase_orders (supplier_id);
CREATE INDEX IF NOT EXISTS idx_po_date       ON purchase_orders (po_date);
CREATE INDEX IF NOT EXISTS idx_po_location   ON purchase_orders (location_id);
CREATE INDEX IF NOT EXISTS idx_po_status     ON purchase_orders (status);
CREATE INDEX IF NOT EXISTS idx_po_open       ON purchase_orders (status) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_po_sku_status ON purchase_orders (sku_id, status);

-- ---------------------------------------------------------------------------
-- 5. weather_log — daily NE Ohio conditions (sourced from Open-Meteo)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather_log (
    id                      BIGSERIAL    PRIMARY KEY,
    log_date                DATE         NOT NULL UNIQUE,
    temp_max_f              NUMERIC(6,2),
    temp_min_f              NUMERIC(6,2),
    precipitation_in        NUMERIC(6,3),
    snowfall_in             NUMERIC(6,3),
    -- rolling count of consecutive days with min temp ≤ 32°F
    consecutive_freeze_days INTEGER      NOT NULL DEFAULT 0,
    -- TRUE on the day after a thaw following ≥2 consecutive freeze days
    -- strong demand signal for suspension/steering/wheel parts (pothole season)
    freeze_thaw_cycle       BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_weather_date         ON weather_log (log_date);
CREATE INDEX IF NOT EXISTS idx_weather_freeze_thaw  ON weather_log (freeze_thaw_cycle) WHERE freeze_thaw_cycle = TRUE;
CREATE INDEX IF NOT EXISTS idx_weather_consec_freeze ON weather_log (consecutive_freeze_days);

-- ---------------------------------------------------------------------------
-- 6. forecast_results — output rows from all three ML models
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_results (
    id             BIGSERIAL     PRIMARY KEY,
    sku_id         VARCHAR(50)   NOT NULL REFERENCES sku_master(sku_id),
    forecast_date  DATE          NOT NULL,
    model_type     VARCHAR(20)   NOT NULL
                       CHECK (model_type IN ('prophet','lightgbm','rolling_avg')),
    predicted_qty  NUMERIC(12,4) NOT NULL,
    lower_bound    NUMERIC(12,4),
    upper_bound    NUMERIC(12,4),
    confidence_pct NUMERIC(5,4),
    run_date       DATE          NOT NULL,
    created_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (sku_id, forecast_date, model_type, run_date)
);

CREATE INDEX IF NOT EXISTS idx_forecast_sku_id     ON forecast_results (sku_id);
CREATE INDEX IF NOT EXISTS idx_forecast_date       ON forecast_results (forecast_date);
CREATE INDEX IF NOT EXISTS idx_forecast_model_type ON forecast_results (model_type);
CREATE INDEX IF NOT EXISTS idx_forecast_run_date   ON forecast_results (run_date);
CREATE INDEX IF NOT EXISTS idx_forecast_sku_date   ON forecast_results (sku_id, forecast_date);
CREATE INDEX IF NOT EXISTS idx_forecast_sku_model  ON forecast_results (sku_id, model_type, forecast_date);

-- ---------------------------------------------------------------------------
-- 7. supplier_scores — rolling performance scorecard per supplier
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS supplier_scores (
    id                    BIGSERIAL    PRIMARY KEY,
    supplier_id           VARCHAR(50)  NOT NULL,
    supplier_name         VARCHAR(150),
    score_date            DATE         NOT NULL,
    avg_lead_time_days    NUMERIC(8,2),
    lead_time_variance_avg NUMERIC(8,2),
    fill_rate_pct         NUMERIC(6,4),
    on_time_delivery_pct  NUMERIC(6,4),
    -- composite 0–100 score computed by the reorder engine (higher = better)
    composite_score       NUMERIC(6,2),
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (supplier_id, score_date)
);

CREATE INDEX IF NOT EXISTS idx_supplier_id        ON supplier_scores (supplier_id);
CREATE INDEX IF NOT EXISTS idx_supplier_date      ON supplier_scores (score_date);
CREATE INDEX IF NOT EXISTS idx_supplier_composite ON supplier_scores (composite_score);
