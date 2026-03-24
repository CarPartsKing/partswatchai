-- =============================================================================
-- Migration 004: Add risk_flag column to supplier_scores
-- Run this in the Supabase SQL Editor before using transform/derive.py
-- Safe to re-run — ADD COLUMN IF NOT EXISTS is idempotent.
-- =============================================================================

ALTER TABLE supplier_scores
    ADD COLUMN IF NOT EXISTS risk_flag TEXT
        CHECK (risk_flag IN ('green', 'amber', 'red'));

COMMENT ON COLUMN supplier_scores.risk_flag IS
    'Composite score threshold: green > 80, amber 60–80, red < 60';
