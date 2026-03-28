-- migration 010 — XYZ demand-variability classification
--
-- Adds xyz_class and abc_xyz_class columns to sku_master.
-- Populated nightly by transform/derive.py (derive_xyz_classification).
--
-- xyz_class   : single letter — X (consistent), Y (variable), Z (erratic)
-- abc_xyz_class : combined two-letter class — AX, BZ, CY, etc.
--
-- RUN IN SUPABASE SQL EDITOR before first pipeline execution.

ALTER TABLE sku_master
    ADD COLUMN IF NOT EXISTS xyz_class     CHAR(1)
        CHECK (xyz_class IN ('X', 'Y', 'Z')),
    ADD COLUMN IF NOT EXISTS abc_xyz_class CHAR(2)
        CHECK (abc_xyz_class IN (
            'AX','AY','AZ',
            'BX','BY','BZ',
            'CX','CY','CZ'
        ));

COMMENT ON COLUMN sku_master.xyz_class IS
    'Demand-variability class based on coefficient of variation of weekly sales. '
    'X = CV < 0.5 (consistent), Y = 0.5 ≤ CV < 1.0 (variable), '
    'Z = CV ≥ 1.0 (erratic). Set nightly by transform/derive.py.';

COMMENT ON COLUMN sku_master.abc_xyz_class IS
    'Combined ABC revenue rank + XYZ demand variability (e.g. AX, BZ, CY). '
    'Used as a categorical feature in LightGBM and for safety stock multipliers.';

CREATE INDEX IF NOT EXISTS idx_sku_master_xyz_class
    ON sku_master (xyz_class);

CREATE INDEX IF NOT EXISTS idx_sku_master_abc_xyz_class
    ON sku_master (abc_xyz_class);
