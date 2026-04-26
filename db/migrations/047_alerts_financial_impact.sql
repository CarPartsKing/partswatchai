-- =============================================================================
-- 047_alerts_financial_impact.sql
--
-- Extends the alerts table with four capabilities:
--
--   financial_impact NUMERIC(14,2) DEFAULT 0
--       Dollar value at stake for each alert.  Populated by engine/alerts.py:
--         CRITICAL_STOCKOUT / LOW_SUPPLY : qty_to_order × unit_cost
--         SUPPLIER_RISK                  : qty_ordered × unit_cost
--         CHURN_RISK                     : baseline_monthly_spend × 3
--         OPSL_GAP                       : estimated_margin_recovery
--       Alerts are sorted descending by financial_impact within severity tier.
--
--   days_active INTEGER DEFAULT 1
--   first_seen_date DATE
--       Deduplication counter.  When engine/alerts.py finds an existing open
--       (resolved=FALSE) alert with the same alert_key from the previous
--       DEDUP_LOOKBACK_DAYS (30) days, it increments days_active on that row
--       instead of creating a new one.  first_seen_date is set on first insert
--       and never changed.
--
--   resolved BOOLEAN DEFAULT FALSE
--   resolved_date DATE
--       Auto-resolve flag.  After writing alerts, the engine scans open alerts
--       of resolvable types and marks resolved=TRUE / resolved_date=today when
--       the underlying condition is no longer detected.  Dashboard queries
--       must filter resolved=FALSE.
--
--   customer_id VARCHAR(200)
--       Context field for CHURN_RISK alerts (Autologue account number).
--
-- Also extends the alert_type CHECK constraint with two new types:
--   CHURN_RISK  — customer moved to AT_RISK / CHURNED / LOST
--   OPSL_GAP    — HIGH OPSL flag not yet in reorder queue
--
-- Idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. New columns
-- ---------------------------------------------------------------------------
ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS financial_impact  NUMERIC(14,2) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS days_active       INTEGER       NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS first_seen_date   DATE,
    ADD COLUMN IF NOT EXISTS resolved          BOOLEAN       NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS resolved_date     DATE,
    ADD COLUMN IF NOT EXISTS customer_id       VARCHAR(200);

-- Backfill first_seen_date for existing rows so the column is never NULL
-- after the migration runs.
UPDATE alerts
   SET first_seen_date = alert_date
 WHERE first_seen_date IS NULL;

-- ---------------------------------------------------------------------------
-- 2. Extend alert_type CHECK to include CHURN_RISK and OPSL_GAP
-- ---------------------------------------------------------------------------
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
        'gp_anomaly',
        'CHURN_RISK',
        'OPSL_GAP'
    ));

-- ---------------------------------------------------------------------------
-- 3. Indexes
-- ---------------------------------------------------------------------------

-- Dashboard default view: open alerts sorted by date (most recent first)
CREATE INDEX IF NOT EXISTS idx_alerts_resolved
    ON alerts (resolved, alert_date DESC);

-- Within-severity financial ranking for the dashboard priority panel
CREATE INDEX IF NOT EXISTS idx_alerts_financial_impact
    ON alerts (severity, financial_impact DESC NULLS LAST)
    WHERE resolved = FALSE;

-- Churn alert lookup by customer
CREATE INDEX IF NOT EXISTS idx_alerts_customer_id
    ON alerts (customer_id)
    WHERE customer_id IS NOT NULL;

-- Days-active sort for the "chronic issues" view
CREATE INDEX IF NOT EXISTS idx_alerts_days_active
    ON alerts (days_active DESC, first_seen_date)
    WHERE resolved = FALSE;
