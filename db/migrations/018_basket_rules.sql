-- =============================================================================
-- Migration 018: basket_rules table for co-purchase association rules.
-- Run this in the Supabase SQL Editor.
-- Safe to re-run — all statements are idempotent.
-- =============================================================================

CREATE TABLE IF NOT EXISTS basket_rules (
    id                BIGSERIAL    PRIMARY KEY,
    antecedent_sku    VARCHAR(50)  NOT NULL,
    consequent_sku    VARCHAR(50)  NOT NULL,
    support           NUMERIC(8,6) NOT NULL,
    confidence        NUMERIC(6,4) NOT NULL,
    lift              NUMERIC(8,4) NOT NULL,
    rule_date         DATE         NOT NULL DEFAULT CURRENT_DATE,
    transaction_count INTEGER      NOT NULL DEFAULT 0,
    UNIQUE (antecedent_sku, consequent_sku, rule_date)
);

CREATE INDEX IF NOT EXISTS idx_basket_antecedent ON basket_rules (antecedent_sku);
CREATE INDEX IF NOT EXISTS idx_basket_consequent ON basket_rules (consequent_sku);
CREATE INDEX IF NOT EXISTS idx_basket_lift       ON basket_rules (lift DESC);
CREATE INDEX IF NOT EXISTS idx_basket_rule_date  ON basket_rules (rule_date DESC);

ALTER TABLE basket_rules DISABLE ROW LEVEL SECURITY;

-- ---------------------------------------------------------------------------
-- Expand recommendation_type CHECK to allow 'basket_triggered'
-- ---------------------------------------------------------------------------
ALTER TABLE reorder_recommendations
    DROP CONSTRAINT IF EXISTS reorder_recommendations_recommendation_type_check;

ALTER TABLE reorder_recommendations
    ADD CONSTRAINT reorder_recommendations_recommendation_type_check
    CHECK (recommendation_type IN ('po', 'transfer', 'basket_triggered'));

ALTER TABLE reorder_recommendations
    ALTER COLUMN recommendation_type TYPE VARCHAR(20);
