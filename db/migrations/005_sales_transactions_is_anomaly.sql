-- =============================================================================
-- Migration 005: Add is_anomaly column to sales_transactions
-- Run this in the Supabase SQL Editor before using ml/anomaly.py
-- Safe to re-run — ADD COLUMN IF NOT EXISTS is idempotent.
-- =============================================================================

ALTER TABLE sales_transactions
    ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN NOT NULL DEFAULT FALSE;

-- Partial index — only indexes the TRUE rows, which is what forecasting
-- pipelines query. Keeps the index tiny even at millions of rows.
CREATE INDEX IF NOT EXISTS idx_sales_is_anomaly
    ON sales_transactions (is_anomaly)
    WHERE is_anomaly = TRUE;

COMMENT ON COLUMN sales_transactions.is_anomaly IS
    'TRUE = flagged by Isolation Forest as statistically anomalous. '
    'Forecasting models exclude these rows from training data.';
