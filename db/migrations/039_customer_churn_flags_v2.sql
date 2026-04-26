-- =============================================================================
-- 039_customer_churn_flags_v2.sql
--
-- Schema and RPC upgrades for customer_churn_flags:
--
--   1. NEW COLUMNS
--      customer_id semantics change: now stores the parsed Autologue account
--      number ("10015") rather than the raw ship_to string
--      ("10015-ASAP AUTOMOTIVE PAINT").  The table is TRUNCATED below because
--      old rows used the full ship_to as the PK — they will never be matched
--      by new upserts that use the short numeric form.  The nightly pipeline
--      regenerates the table completely each run, so no data is lost.
--
--      salesman_id          VARCHAR(20)  — rep most frequently on this acct
--      is_commercial        BOOLEAN      — True when ship_to has a name after '-'
--      days_since_last_purchase INTEGER  — effective_today − last_purchase_date
--      risk_segment         VARCHAR(20)  — churned-account tier (AT_RISK /
--                                         CHURNED / LOST); NULL for DECLINING
--                                         and STABLE rows
--
--   2. FLAG CONSTRAINT
--      CHURNED is now one of three tiers.  New allowed values:
--        AT_RISK  — zero spend last 90 days, last purchase 90–180 days ago
--        CHURNED  — zero spend last 90 days, last purchase 180–365 days ago
--        LOST     — zero spend last 90 days, last purchase 365+ days ago
--        DECLINING / STABLE unchanged
--
--   3. UPDATED RPC  get_churn_buckets()
--      Parses customer_id from ship_to, aggregates baseline_months for the
--      minimum-active-months quality gate, and joins sales_transactions to
--      surface the most-frequent salesman in the comparison window.
--
-- All statements are idempotent — safe to re-run.
-- Supabase project: pytxjsuwhkzrffvzrelw
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Widen customer_id to accommodate old rows during transition (no-op if
--    already VARCHAR(200)).  We do NOT narrow it here — the existing data
--    is cleared by TRUNCATE below; a future migration can narrow if needed.
-- ---------------------------------------------------------------------------

-- 2. New columns
ALTER TABLE customer_churn_flags
    ADD COLUMN IF NOT EXISTS salesman_id               VARCHAR(20),
    ADD COLUMN IF NOT EXISTS is_commercial             BOOLEAN,
    ADD COLUMN IF NOT EXISTS days_since_last_purchase  INTEGER,
    ADD COLUMN IF NOT EXISTS risk_segment              VARCHAR(20);

-- 3. Replace flag CHECK constraint to permit the three churned tiers
ALTER TABLE customer_churn_flags
    DROP CONSTRAINT IF EXISTS customer_churn_flags_flag_check;

ALTER TABLE customer_churn_flags
    ADD CONSTRAINT customer_churn_flags_flag_check
        CHECK (flag IN ('AT_RISK', 'CHURNED', 'LOST', 'DECLINING', 'STABLE'));

-- 4. risk_segment constraint (NULL for DECLINING / STABLE rows)
ALTER TABLE customer_churn_flags
    DROP CONSTRAINT IF EXISTS customer_churn_flags_risk_segment_check;

ALTER TABLE customer_churn_flags
    ADD CONSTRAINT customer_churn_flags_risk_segment_check
        CHECK (risk_segment IS NULL
            OR risk_segment IN ('AT_RISK', 'CHURNED', 'LOST'));

-- 5. Indexes for new columns
CREATE INDEX IF NOT EXISTS idx_ccf_risk_segment
    ON customer_churn_flags (risk_segment)
    WHERE risk_segment IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ccf_salesman
    ON customer_churn_flags (salesman_id)
    WHERE salesman_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ccf_is_commercial
    ON customer_churn_flags (is_commercial, flag);

-- 6. Clear stale rows whose customer_id = full ship_to string.
--    The table is rebuilt every night, so this is safe.
TRUNCATE TABLE customer_churn_flags;

-- =============================================================================
-- 7. Updated RPC — get_churn_buckets()
--
-- Changes from migration 035:
--   • Parses customer_id = SPLIT_PART(ship_to, '-', 1) and groups on that,
--     collapsing multiple ship_to spellings for the same account into one row.
--   • Returns is_commercial — TRUE when ship_to has a non-blank name after '-'.
--   • Returns baseline_months — distinct calendar months with activity in the
--     baseline window; Python uses this for the MIN_ACTIVE_MONTHS quality gate.
--   • Returns salesman_id — most-frequent customer_salesman from
--     sales_transactions in the comparison window (LEFT JOIN so churned
--     customers with no recent activity get NULL, not dropped).
--
-- Parameters unchanged from migration 035.
-- =============================================================================

CREATE OR REPLACE FUNCTION get_churn_buckets(
    p_location_id      TEXT,
    p_window_start     DATE,
    p_baseline_end     DATE,
    p_comparison_start DATE
)
RETURNS TABLE (
    customer_id            TEXT,
    is_commercial          BOOLEAN,
    baseline_sales         NUMERIC,
    baseline_tx            BIGINT,
    baseline_months        BIGINT,
    comparison_sales       NUMERIC,
    last_purchase_date     DATE,
    salesman_id            TEXT
)
LANGUAGE sql
STABLE
AS $$
    -- ── Step 1: parse ship_to into (customer_id, is_commercial) ──────────────
    WITH parsed AS (
        SELECT
            -- Everything before the first '-' is the Autologue account number.
            -- SPLIT_PART returns the whole string when '-' is absent, so
            -- purely-numeric ship_to values (no company name) are handled
            -- correctly: customer_id = full value, is_commercial = FALSE.
            TRIM(SPLIT_PART(ship_to, '-', 1))           AS customer_id,
            -- Non-blank text after the first '-' → commercial account.
            TRIM(SPLIT_PART(ship_to, '-', 2)) <> ''     AS is_commercial,
            tran_date,
            sales
        FROM sales_detail_transactions
        WHERE location_id = p_location_id
          AND tran_date   >= p_window_start
          AND ship_to     IS NOT NULL
          AND sales       IS NOT NULL
    ),

    -- ── Step 2: aggregate per parsed customer_id ─────────────────────────────
    base_agg AS (
        SELECT
            customer_id,
            -- If the same account appeared with and without a company name,
            -- treat it as commercial if any row says so.
            BOOL_OR(is_commercial)                                          AS is_commercial,
            COALESCE(
                SUM(sales) FILTER (
                    WHERE tran_date BETWEEN p_window_start AND p_baseline_end
                ), 0
            )                                                               AS baseline_sales,
            COUNT(*) FILTER (
                WHERE tran_date BETWEEN p_window_start AND p_baseline_end
            )                                                               AS baseline_tx,
            -- Distinct calendar months active in the baseline period.
            -- Python's MIN_ACTIVE_MONTHS gate requires >= 3.
            COUNT(DISTINCT DATE_TRUNC('month', tran_date)) FILTER (
                WHERE tran_date BETWEEN p_window_start AND p_baseline_end
            )                                                               AS baseline_months,
            COALESCE(
                SUM(sales) FILTER (WHERE tran_date >= p_comparison_start),
                0
            )                                                               AS comparison_sales,
            MAX(tran_date)                                                  AS last_purchase_date
        FROM parsed
        WHERE customer_id <> ''
        GROUP BY customer_id
    ),

    -- ── Step 3: most-frequent salesman per customer in comparison window ─────
    -- Sourced from sales_transactions (which carries [Customer].[Salesman]
    -- from migration 025).  Uses a two-level CTE so COUNT() aggregates in
    -- the inner query before ROW_NUMBER() ranks in the outer one.
    salesman_freq AS (
        SELECT
            TRIM(customer_id)   AS customer_id,
            customer_salesman,
            COUNT(*)            AS tx_count
        FROM sales_transactions
        WHERE location_id       = p_location_id
          AND transaction_date  >= p_comparison_start
          AND customer_id       IS NOT NULL
          AND TRIM(customer_id) <> ''
          AND customer_salesman IS NOT NULL
          AND TRIM(customer_salesman) <> ''
        GROUP BY TRIM(customer_id), customer_salesman
    ),
    salesman_ranked AS (
        SELECT
            customer_id,
            customer_salesman,
            ROW_NUMBER() OVER (
                PARTITION BY customer_id
                ORDER BY tx_count DESC
            )                   AS rn
        FROM salesman_freq
    )

    -- ── Step 4: join and return ───────────────────────────────────────────────
    SELECT
        ba.customer_id,
        ba.is_commercial,
        ba.baseline_sales,
        ba.baseline_tx,
        ba.baseline_months,
        ba.comparison_sales,
        ba.last_purchase_date,
        sr.customer_salesman    AS salesman_id
    FROM base_agg            ba
    LEFT JOIN salesman_ranked sr
           ON sr.customer_id = ba.customer_id
          AND sr.rn = 1
$$;
