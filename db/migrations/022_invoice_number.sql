-- =============================================================================
-- 022_invoice_number.sql — capture real invoice numbers for basket analysis
--
-- BACKGROUND
-- ----------
-- Until now `sales_transactions.transaction_id` was synthesized as
--   AC-{date}-{sku}-{location}
-- which collapsed every line of every invoice for a given (date, sku, loc)
-- tuple into a single row.  Apriori basket analysis needs to know which line
-- items shared the same physical invoice — and with the synthetic ID every
-- "basket" was inherently single-item, producing zero rules.
--
-- The Sales Detail cube exposes the real invoice number at:
--   [Sales Detail].[Invoice Nbr].[Invoice Nbr]   (cardinality ≈ 600k)
--
-- This migration adds the column to store it.  After the migration is applied,
-- run `python -m extract.autocube_pull --mode historical` to repopulate.
--
-- TRANSACTION_ID SEMANTICS
-- ------------------------
-- The extract changes `_generate_transaction_id()` to a 5-part key whenever an
-- invoice number is present:
--   AC-{date}-{sku}-{location}-INV{invoice}   (NEW, per-invoice line grain)
--   AC-{date}-{sku}-{location}                (LEGACY, pre-migration rows)
--
-- New rows therefore upsert on entirely different keys than the existing
-- aggregated rows.  To prevent a long-lived transition window where legacy
-- rows AND new per-invoice rows both feed the forecast pipeline (double-
-- counting demand), _load_to_supabase() deletes legacy NULL-invoice rows
-- for each chunk's date range AFTER upserting that chunk's new rows
-- (delete-after-upsert is intentional — it guarantees no date range is ever
-- left empty, even on process crash).  Worst case after an interrupted run:
-- one chunk's date range is briefly double-counted until the next attempt
-- or the optional final-sweep DELETE at the bottom of this file.
--
-- RUNBOOK FOR REPOPULATION
-- ------------------------
-- After applying this migration, run the historical re-extract with the
-- new --no-resume flag so every chunk is processed (the auto-resume default
-- would skip chunks whose dates already have rows, leaving most history
-- with NULL invoice_number):
--
--   python -m extract.autocube_pull --mode historical --no-resume
--
-- IDEMPOTENCY: every statement uses IF NOT EXISTS / IF EXISTS — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- sales_transactions.invoice_number
-- ---------------------------------------------------------------------------
ALTER TABLE sales_transactions
    ADD COLUMN IF NOT EXISTS invoice_number TEXT;

COMMENT ON COLUMN sales_transactions.invoice_number IS
  'Real invoice/document number from the Autologue Sales Detail cube '
  '([Sales Detail].[Invoice Nbr]).  Multiple sales_transactions rows that '
  'share the same invoice_number represent line items on the same physical '
  'invoice — this is the grouping key used by ml/basket.py for co-purchase '
  'association rule mining.  NULL for legacy rows extracted before '
  'migration 022 applied.';

-- ---------------------------------------------------------------------------
-- Index for basket analysis grouping queries
--   ml/basket.py paginates sales_transactions by transaction_date and groups
--   the results by invoice_number.  This composite index covers both the
--   keyset cursor and the grouping key, with a NOT NULL filter so the index
--   stays small (excludes legacy rows that have no invoice number).
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_sales_transactions_invoice
    ON sales_transactions (transaction_date, invoice_number)
    WHERE invoice_number IS NOT NULL;

-- ---------------------------------------------------------------------------
-- OPTIONAL FINAL SWEEP — only needed if the re-extract leaves any NULL rows
-- ---------------------------------------------------------------------------
-- _load_to_supabase() deletes legacy NULL-invoice rows per chunk as it goes,
-- so by the time the full re-extract completes there should already be zero
-- rows with invoice_number IS NULL.  If a stray remains (e.g. for a date the
-- re-extract skipped), this final sweep guarantees a clean state:
--
--   DELETE FROM sales_transactions WHERE invoice_number IS NULL;
--
-- Run AFTER the re-extract finishes successfully, never before — running it
-- prematurely would empty the table.
