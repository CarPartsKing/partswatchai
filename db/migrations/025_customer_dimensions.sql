-- =============================================================================
-- 025_customer_dimensions.sql — capture per-transaction customer identity
--
-- BACKGROUND
-- ----------
-- Until now `sales_transactions` had no customer identifier — every row carried
-- only sku/location/invoice/date.  This blocked any customer-level analytics:
--
--   * Customer churn scoring (ml/churn.py) — cannot group by customer.
--   * Rep-level alerting (route at-risk accounts to the Salesman responsible).
--   * Wholesale-vs-retail segmentation of demand patterns.
--
-- The Sales Detail cube exposes the customer at:
--   [Customer].[Cust No].[Cust No]      — canonical Autologue account number
--   [Customer].[Cust Type].[Cust Type]  — wholesale / retail / fleet / etc.
--   [Customer].[Status].[Status]        — active / inactive / on-hold
--   [Customer].[Salesman].[Salesman]    — outside-rep ownership
--
-- All four dimensions were verified empirically against the AutoCube_DTR_23160
-- catalog on 2026-04-19 via `python -m extract.autocube_pull --test`.
--
-- After applying this migration, the next extract run (--mode incremental or
-- --mode full) will populate the new columns.  Legacy rows extracted before
-- this migration will have NULL in all four columns — that is intentional;
-- ml/churn.py treats NULL customer_id as an unscorable row and skips it.
--
-- IDEMPOTENCY: every statement uses IF NOT EXISTS / IF EXISTS — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- sales_transactions.customer_id
--   Primary grouping key for churn scoring.  VARCHAR(50) matches the width of
--   the existing sku_id / location_id text columns; Autologue Cust No values
--   observed are short numeric strings (≤ 10 chars) but we leave headroom for
--   alphanumeric account schemes.
-- ---------------------------------------------------------------------------
ALTER TABLE sales_transactions
    ADD COLUMN IF NOT EXISTS customer_id VARCHAR(50);

COMMENT ON COLUMN sales_transactions.customer_id IS
  'Autologue customer account number from the Sales Detail cube '
  '([Customer].[Cust No]).  Functionally determined by invoice_number — '
  'each invoice is for exactly one customer.  NULL for legacy rows '
  'extracted before migration 025 applied; ml/churn.py skips NULL rows.';

-- ---------------------------------------------------------------------------
-- sales_transactions.customer_type
--   Slowly-changing attribute denormalized onto each transaction so churn.py
--   can segment risk (e.g. wholesale customers warrant different thresholds
--   than retail walk-ins) without joining a separate customer_master table.
-- ---------------------------------------------------------------------------
ALTER TABLE sales_transactions
    ADD COLUMN IF NOT EXISTS customer_type VARCHAR(50);

COMMENT ON COLUMN sales_transactions.customer_type IS
  'Customer segmentation from [Customer].[Cust Type] — wholesale, retail, '
  'fleet, etc.  Denormalized onto each transaction; the value reflects the '
  'customer''s type AT EXTRACT TIME, not at transaction time.';

-- ---------------------------------------------------------------------------
-- sales_transactions.customer_status
--   Active / inactive / on-hold flag.  ml/churn.py uses this to skip
--   already-inactive accounts (an "inactive" customer isn't churning — they
--   are already gone — so scoring them inflates the at-risk list).
-- ---------------------------------------------------------------------------
ALTER TABLE sales_transactions
    ADD COLUMN IF NOT EXISTS customer_status VARCHAR(20);

COMMENT ON COLUMN sales_transactions.customer_status IS
  'Customer account status from [Customer].[Status] — typically Active, '
  'Inactive, On Hold, COD, etc.  ml/churn.py filters to Active accounts '
  'before scoring so already-gone customers do not pollute the at-risk list.';

-- ---------------------------------------------------------------------------
-- sales_transactions.customer_salesman
--   Outside-rep ownership.  Lets churn alerts be routed to the specific
--   salesperson responsible for the at-risk account rather than a generic
--   inbox.  Stored as a code/name string from the cube — matches whatever
--   format Autologue exposes (commonly a 3-4 char rep code).
-- ---------------------------------------------------------------------------
ALTER TABLE sales_transactions
    ADD COLUMN IF NOT EXISTS customer_salesman VARCHAR(50);

COMMENT ON COLUMN sales_transactions.customer_salesman IS
  'Outside-rep code/name from [Customer].[Salesman] — the salesperson who '
  '"owns" the customer relationship.  Used by engine/alerts.py to route '
  'churn-risk alerts to the responsible rep.';

-- ---------------------------------------------------------------------------
-- Index for customer-grouped queries
--   ml/churn.py paginates sales_transactions by customer_id and aggregates
--   recency / frequency / monetary signals per customer.  This composite
--   index covers (customer_id, transaction_date DESC) which is the access
--   pattern for "give me this customer''s last N transactions" lookups.
--   The WHERE clause keeps the index small by excluding legacy NULL rows.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_sales_transactions_customer
    ON sales_transactions (customer_id, transaction_date DESC)
    WHERE customer_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Project rule (see replit.md): every new-table migration disables RLS so the
-- service-role client is never blocked by an empty policy set.  This migration
-- only ADDS COLUMNS to an existing table (no CREATE TABLE), so no RLS toggle
-- is required — sales_transactions already has RLS disabled from migration 001.
-- ---------------------------------------------------------------------------
