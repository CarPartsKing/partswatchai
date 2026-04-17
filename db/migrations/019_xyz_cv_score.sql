-- migration 019 — XYZ coefficient-of-variation score
--
-- Adds the cv_score column to sku_master so the actual CV computed during
-- XYZ classification is persisted alongside the resulting xyz_class.  This
-- lets the dashboard, alerts, and ad-hoc queries reason about HOW erratic
-- a SKU is (e.g. find the most/least predictable items inside a class)
-- without re-running the full classification pass.
--
-- Range: 0.00 (perfectly stable) to ~50+ (extremely erratic).
-- Stored as NUMERIC(8,4) — 4 decimal places of precision on values up
-- to 9999.9999, more than enough for any realistic CV.
--
-- NULL semantics: cv_score is NULL when the SKU does not have enough
-- weeks of sales history to compute a meaningful CV (xyz_class is also
-- NULL in that case — surfaced as "UNKNOWN" in pipeline logs).
--
-- RUN IN SUPABASE SQL EDITOR before the next pipeline execution.

ALTER TABLE sku_master
    ADD COLUMN IF NOT EXISTS cv_score NUMERIC(8,4);

COMMENT ON COLUMN sku_master.cv_score IS
    'Coefficient of variation of weekly sales over the XYZ lookback window '
    '(population std / mean). Persisted alongside xyz_class by '
    'transform/derive.py::derive_xyz_classification. NULL = insufficient '
    'history (UNKNOWN class).';

-- Lightweight index for ranking queries ("top-10 most erratic A-class SKUs")
CREATE INDEX IF NOT EXISTS idx_sku_master_cv_score
    ON sku_master (cv_score)
    WHERE cv_score IS NOT NULL;
