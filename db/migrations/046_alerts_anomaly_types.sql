-- =============================================================================
-- 046_alerts_anomaly_types.sql
--
-- Extend the alerts.alert_type CHECK constraint to include two new types
-- produced by the upgraded ml/anomaly.py pipeline:
--
--   volume_anomaly — Isolation Forest-detected day where a SKU's network-wide
--                    daily quantity is statistically anomalous (HIGH or LOW).
--                    severity='warning' for HIGH (possible duplicate order /
--                    data-entry error); severity='info' for LOW (supply issue
--                    or lost account).
--
--   gp_anomaly     — SKU+location whose 7-day average GP% has dropped more
--                    than 15 percentage points below its 90-day baseline.
--                    Signals a pricing error, unauthorised discount, or margin
--                    erosion.  severity='warning'.
--
-- Approach: drop the existing inline CHECK, recreate it with the two new
-- values appended.  All existing alert rows remain valid — the new constraint
-- is a superset of the old one.
--
-- Idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

ALTER TABLE alerts DROP CONSTRAINT IF EXISTS alerts_alert_type_check;

ALTER TABLE alerts ADD CONSTRAINT alerts_alert_type_check
    CHECK (alert_type IN (
        'CRITICAL_STOCKOUT',
        'LOW_SUPPLY',
        'FREEZE_ALERT',
        'SUPPLIER_RISK',
        'DEAD_STOCK',
        'TRANSFER_OPPORTUNITY',
        'FORECAST_ACCURACY_DROP',
        'volume_anomaly',
        'gp_anomaly'
    ));
