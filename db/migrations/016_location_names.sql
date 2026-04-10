-- =============================================================================
-- Migration 016: Add location_name to locations table, add location_name
-- to alerts table, mark retired locations, flag LOC-021 data quality issue.
--
-- Run this in the Supabase SQL Editor.  Safe to re-run — all IF NOT EXISTS
-- or ON CONFLICT DO NOTHING.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Add location_name + is_active columns to locations table
-- ---------------------------------------------------------------------------
ALTER TABLE locations
    ADD COLUMN IF NOT EXISTS location_name VARCHAR(60);

ALTER TABLE locations
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

COMMENT ON COLUMN locations.location_name IS
    'Human-readable store name from the Product cube (e.g. BROOKPARK, MAIN DC).';

COMMENT ON COLUMN locations.is_active IS
    'FALSE for closed/retired locations — excluded from forecasting and reorder.';


-- ---------------------------------------------------------------------------
-- 2. Populate location names for all 27 active locations
-- ---------------------------------------------------------------------------
UPDATE locations SET location_name = 'BROOKPARK'        WHERE location_id = 'LOC-001';
UPDATE locations SET location_name = 'NOLMSTEAD'        WHERE location_id = 'LOC-002';
UPDATE locations SET location_name = 'S.EUCLID'         WHERE location_id = 'LOC-003';
UPDATE locations SET location_name = 'CLARK AUTO'       WHERE location_id = 'LOC-004';
UPDATE locations SET location_name = 'PARMA'            WHERE location_id = 'LOC-005';
UPDATE locations SET location_name = 'MEDINA'           WHERE location_id = 'LOC-006';
UPDATE locations SET location_name = 'BOARDMAN'         WHERE location_id = 'LOC-007';
UPDATE locations SET location_name = 'ELYRIA'           WHERE location_id = 'LOC-008';
UPDATE locations SET location_name = 'AKRON-GRANT'      WHERE location_id = 'LOC-009';
UPDATE locations SET location_name = 'MIDWAY CROSSINGS' WHERE location_id = 'LOC-010';
UPDATE locations SET location_name = 'ERIE ST'          WHERE location_id = 'LOC-011';
UPDATE locations SET location_name = 'MAYFIELD'         WHERE location_id = 'LOC-012';
UPDATE locations SET location_name = 'CANTON'           WHERE location_id = 'LOC-013';
UPDATE locations SET location_name = 'JUNIATA'          WHERE location_id = 'LOC-015';
UPDATE locations SET location_name = 'ARCHWOOD'         WHERE location_id = 'LOC-016';
UPDATE locations SET location_name = 'EUCLID'           WHERE location_id = 'LOC-017';
UPDATE locations SET location_name = 'WARREN'           WHERE location_id = 'LOC-018';
UPDATE locations SET location_name = 'ROOTSTOWN'        WHERE location_id = 'LOC-020';
UPDATE locations SET location_name = 'INTERNET'         WHERE location_id = 'LOC-021';
UPDATE locations SET location_name = 'MENTOR'           WHERE location_id = 'LOC-024';
UPDATE locations SET location_name = 'MAIN DC'          WHERE location_id = 'LOC-025';
UPDATE locations SET location_name = 'COPLEY'           WHERE location_id = 'LOC-026';
UPDATE locations SET location_name = 'CHARDON'          WHERE location_id = 'LOC-027';
UPDATE locations SET location_name = 'STRONGSVILLE'     WHERE location_id = 'LOC-028';
UPDATE locations SET location_name = 'MIDDLEBURG'       WHERE location_id = 'LOC-029';
UPDATE locations SET location_name = 'PERRY'            WHERE location_id = 'LOC-032';
UPDATE locations SET location_name = 'CRYSTAL'          WHERE location_id = 'LOC-033';


-- ---------------------------------------------------------------------------
-- 3. Insert retired/closed locations as is_active = FALSE
--    (gap location numbers: 14, 19, 22, 23, 30, 31)
--    These may not exist yet — use ON CONFLICT to skip if present.
-- ---------------------------------------------------------------------------
INSERT INTO locations (location_id, location_tier, fill_rate_score, revenue_score,
                       sku_breadth_score, return_rate_score, composite_tier_score,
                       classified_date, location_name, is_active)
VALUES
    ('LOC-014', 3, 0, 0, 0, 0, 0, CURRENT_DATE, 'RETIRED-014', FALSE),
    ('LOC-019', 3, 0, 0, 0, 0, 0, CURRENT_DATE, 'RETIRED-019', FALSE),
    ('LOC-022', 3, 0, 0, 0, 0, 0, CURRENT_DATE, 'RETIRED-022', FALSE),
    ('LOC-023', 3, 0, 0, 0, 0, 0, CURRENT_DATE, 'RETIRED-023', FALSE),
    ('LOC-030', 3, 0, 0, 0, 0, 0, CURRENT_DATE, 'RETIRED-030', FALSE),
    ('LOC-031', 3, 0, 0, 0, 0, 0, CURRENT_DATE, 'RETIRED-031', FALSE)
ON CONFLICT (location_id) DO UPDATE SET
    is_active     = FALSE,
    location_name = EXCLUDED.location_name;


-- ---------------------------------------------------------------------------
-- 4. Add location_name column to alerts table
-- ---------------------------------------------------------------------------
ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS location_name VARCHAR(60);

COMMENT ON COLUMN alerts.location_name IS
    'Human-readable location name stored alongside location_id for display.';


-- ---------------------------------------------------------------------------
-- 5. Backfill existing alerts with location names
-- ---------------------------------------------------------------------------
UPDATE alerts a
SET location_name = l.location_name
FROM locations l
WHERE a.location_id = l.location_id
  AND a.location_name IS NULL
  AND l.location_name IS NOT NULL;


-- ---------------------------------------------------------------------------
-- 6. Flag LOC-021 INTERNET negative on-hand as a data quality issue
-- ---------------------------------------------------------------------------
INSERT INTO data_quality_issues (
    issue_type, table_name, column_name, severity,
    description, affected_count, checked_at
)
VALUES (
    'negative_inventory',
    'inventory_snapshots',
    'qty_on_hand',
    'warning',
    'INTERNET location (LOC-021) shows negative on-hand inventory (-1,795 units) — possible oversell or receiving discrepancy',
    1,
    NOW()
)
ON CONFLICT DO NOTHING;
