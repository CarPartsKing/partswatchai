-- =============================================================================
-- 029_understocking_report.sql
--
-- Chronic Understocking Report — answers the owner's question:
--   "What should each store be stocking more of instead of waiting for
--    transfers?"
--
-- This migration creates BOTH:
--   1. understocking_report TABLE         — one row per (location, SKU) per
--                                            report_date, top-N per location.
--   2. fn_run_understocking_report(...)   — Postgres function that does the
--                                            entire aggregation server-side
--                                            and upserts results.  Called
--                                            from engine/understocking.py via
--                                            supabase-py's .rpc().
--
-- A SKU is "chronically understocked" at a location when its on-hand
-- quantity sat below the reorder point for >=30% of observed snapshot
-- days in the lookback window (target 90 days; engine adapts to whatever
-- snapshot history actually exists).
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1) Report table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS understocking_report (
    id                          BIGSERIAL    PRIMARY KEY,
    report_date                 DATE         NOT NULL,
    location_id                 VARCHAR(20)  NOT NULL,
    location_name               VARCHAR(60),
    sku_id                      VARCHAR(50)  NOT NULL,
    sku_description             TEXT,
    days_observed               INT          NOT NULL,
    days_below_reorder          INT          NOT NULL,
    stockout_days_pct           NUMERIC(5,4) NOT NULL,    -- 0.0000–1.0000
    avg_daily_demand            NUMERIC(12,4) NOT NULL,
    current_min_qty             NUMERIC(12,2) NOT NULL,   -- current reorder_point
    suggested_min_qty           NUMERIC(12,2) NOT NULL,   -- avg_daily × (lead_time + 14)
    min_qty_gap                 NUMERIC(12,2) NOT NULL,   -- suggested − current
    unit_cost                   NUMERIC(12,2),
    inventory_value_at_risk     NUMERIC(14,2) NOT NULL,
    transfer_recommended_count  INT          NOT NULL DEFAULT 0,
    priority_score              NUMERIC(16,4) NOT NULL,   -- ranking key
    created_at                  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (report_date, location_id, sku_id)
);

CREATE INDEX IF NOT EXISTS idx_understock_report_date
    ON understocking_report (report_date DESC);
CREATE INDEX IF NOT EXISTS idx_understock_loc
    ON understocking_report (location_id, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_understock_priority
    ON understocking_report (report_date DESC, priority_score DESC);

ALTER TABLE understocking_report DISABLE ROW LEVEL SECURITY;


-- ---------------------------------------------------------------------------
-- 2) Aggregation function — does everything in one round-trip
--
-- Returns a single-row summary so the engine can log + verify counts.
-- All heavy lifting (joins, aggregation, ranking, upsert) stays in Postgres.
--
-- p_excluded_locs     text[]    locations to drop (retired, INTERNET, MAIN DC)
-- p_lookback_days     int       snapshot/sales window
-- p_transfer_days     int       lookback for confirming transfer-rec count
-- p_stockout_pct      numeric   0.30 → flag when below_reorder ≥ 30% of days
-- p_lead_buffer_days  numeric   lead_time + buffer (e.g. 7 + 14 = 21)
-- p_top_n             int       rows to keep per location
-- p_dry_run           boolean   if true, compute + return summary but skip writes
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION fn_run_understocking_report(
    p_excluded_locs    TEXT[]   DEFAULT ARRAY['LOC-014','LOC-019','LOC-022',
                                               'LOC-023','LOC-030','LOC-031',
                                               'LOC-021','LOC-025'],
    p_lookback_days    INT      DEFAULT 90,
    p_transfer_days    INT      DEFAULT 30,
    p_stockout_pct     NUMERIC  DEFAULT 0.30,
    p_lead_buffer_days NUMERIC  DEFAULT 21.0,
    p_top_n            INT      DEFAULT 20,
    p_dry_run          BOOLEAN  DEFAULT FALSE
)
RETURNS TABLE (
    report_date              DATE,
    rows_built               INT,
    rows_persisted           INT,
    locations_with_chronic   INT,
    total_value_at_risk      NUMERIC,
    actual_window_days       INT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_today           DATE    := CURRENT_DATE;
    v_built           INT     := 0;
    v_persisted       INT     := 0;
    v_locs            INT     := 0;
    v_value           NUMERIC := 0;
    v_window_days     INT     := 0;
BEGIN
    -- Build the candidate set into a temp table.  Local to this txn.
    DROP TABLE IF EXISTS _us_tmp;
    CREATE TEMP TABLE _us_tmp ON COMMIT DROP AS
    WITH
    snap_stats AS (
        SELECT
            s.location_id,
            s.sku_id,
            COUNT(*)                                                    AS days_observed,
            SUM(CASE WHEN s.reorder_point > 0
                      AND s.qty_on_hand < s.reorder_point
                     THEN 1 ELSE 0 END)                                 AS days_below_reorder,
            (ARRAY_AGG(s.reorder_point ORDER BY s.snapshot_date DESC))[1] AS reorder_point,
            (ARRAY_AGG(s.unit_cost     ORDER BY s.snapshot_date DESC))[1] AS unit_cost
        FROM inventory_snapshots s
        WHERE s.snapshot_date >= (CURRENT_DATE - p_lookback_days)
          AND s.location_id <> ALL(p_excluded_locs)
        GROUP BY s.location_id, s.sku_id
    ),
    window_days AS (
        SELECT
            location_id,
            COUNT(DISTINCT snapshot_date) AS observed_days
        FROM inventory_snapshots
        WHERE snapshot_date >= (CURRENT_DATE - p_lookback_days)
          AND location_id <> ALL(p_excluded_locs)
        GROUP BY location_id
    ),
    sales_stats AS (
        SELECT
            location_id,
            sku_id,
            SUM(GREATEST(qty_sold, 0))::numeric AS total_qty
        FROM sales_transactions
        WHERE transaction_date >= (CURRENT_DATE - p_lookback_days)
          AND location_id <> ALL(p_excluded_locs)
        GROUP BY location_id, sku_id
    ),
    transfer_stats AS (
        SELECT
            location_id,
            sku_id,
            COUNT(*) AS xfer_count
        FROM reorder_recommendations
        WHERE recommendation_type = 'transfer'
          AND recommendation_date >= (CURRENT_DATE - p_transfer_days)
        GROUP BY location_id, sku_id
    ),
    combined AS (
        SELECT
            ss.location_id,
            ss.sku_id,
            ss.days_observed,
            ss.days_below_reorder,
            (ss.days_below_reorder::numeric / ss.days_observed)             AS pct,
            COALESCE(sa.total_qty, 0) / GREATEST(wd.observed_days, 1)       AS avg_daily,
            ss.reorder_point                                                AS current_min,
            (COALESCE(sa.total_qty, 0) / GREATEST(wd.observed_days, 1))
                * p_lead_buffer_days                                        AS suggested_min,
            COALESCE(ss.unit_cost, 0)                                       AS unit_cost,
            ss.days_below_reorder
                * (COALESCE(sa.total_qty, 0) / GREATEST(wd.observed_days, 1))
                * COALESCE(ss.unit_cost, 0)                                 AS value_at_risk,
            (ss.days_below_reorder::numeric / ss.days_observed)
                * (COALESCE(sa.total_qty, 0) / GREATEST(wd.observed_days, 1))
                * COALESCE(ss.unit_cost, 0)                                 AS priority,
            COALESCE(t.xfer_count, 0)                                       AS xfer_count
        FROM snap_stats ss
        JOIN window_days wd USING (location_id)
        LEFT JOIN sales_stats    sa ON sa.location_id = ss.location_id AND sa.sku_id = ss.sku_id
        LEFT JOIN transfer_stats t  ON t.location_id  = ss.location_id AND t.sku_id  = ss.sku_id
        WHERE ss.reorder_point > 0
          AND ss.days_observed > 0
          AND (ss.days_below_reorder::numeric / ss.days_observed) >= p_stockout_pct
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY location_id ORDER BY priority DESC) AS rn
        FROM combined
        WHERE priority > 0
    )
    SELECT
        r.location_id,
        r.sku_id,
        sm.description AS sku_description,
        r.days_observed,
        r.days_below_reorder,
        r.pct,
        r.avg_daily,
        r.current_min,
        r.suggested_min,
        r.unit_cost,
        r.value_at_risk,
        r.xfer_count,
        r.priority
    FROM ranked r
    LEFT JOIN sku_master sm ON sm.sku_id = r.sku_id
    WHERE r.rn <= p_top_n;

    GET DIAGNOSTICS v_built = ROW_COUNT;

    -- Summary metrics
    SELECT
        COUNT(DISTINCT location_id),
        COALESCE(SUM(value_at_risk), 0)
    INTO v_locs, v_value
    FROM _us_tmp;

    SELECT COALESCE(MAX(observed_days), 0) INTO v_window_days
    FROM (
        SELECT COUNT(DISTINCT snapshot_date) AS observed_days
        FROM inventory_snapshots
        WHERE snapshot_date >= (CURRENT_DATE - p_lookback_days)
          AND location_id <> ALL(p_excluded_locs)
        GROUP BY location_id
    ) w;

    -- Persist (unless dry-run).  Same-day reruns must FULLY REPLACE
    -- today's rows: if a SKU dropped out of the top-N or a location
    -- now has zero chronic SKUs on a rerun, the old rows must not
    -- linger.  Delete-then-insert inside this function is atomic
    -- (single statement boundary in the calling .rpc()).
    IF NOT p_dry_run THEN
        DELETE FROM understocking_report WHERE report_date = v_today;
    END IF;

    IF NOT p_dry_run AND v_built > 0 THEN
        WITH location_names(loc_id, loc_name) AS (
            VALUES
                ('LOC-001','BROOKPARK'), ('LOC-002','NOLMSTEAD'),
                ('LOC-003','S.EUCLID'),  ('LOC-004','CLARK AUTO'),
                ('LOC-005','PARMA'),     ('LOC-006','MEDINA'),
                ('LOC-007','BOARDMAN'),  ('LOC-008','ELYRIA'),
                ('LOC-009','AKRON-GRANT'),('LOC-010','MIDWAY CROSSINGS'),
                ('LOC-011','ERIE ST'),   ('LOC-012','MAYFIELD'),
                ('LOC-013','CANTON'),    ('LOC-015','JUNIATA'),
                ('LOC-016','ARCHWOOD'),  ('LOC-017','EUCLID'),
                ('LOC-018','WARREN'),    ('LOC-020','ROOTSTOWN'),
                ('LOC-021','INTERNET'),  ('LOC-024','MENTOR'),
                ('LOC-025','MAIN DC'),   ('LOC-026','COPLEY'),
                ('LOC-027','CHARDON'),   ('LOC-028','STRONGSVILLE'),
                ('LOC-029','MIDDLEBURG'),('LOC-032','PERRY'),
                ('LOC-033','CRYSTAL')
        )
        INSERT INTO understocking_report (
            report_date, location_id, location_name, sku_id, sku_description,
            days_observed, days_below_reorder, stockout_days_pct,
            avg_daily_demand, current_min_qty, suggested_min_qty, min_qty_gap,
            unit_cost, inventory_value_at_risk, transfer_recommended_count,
            priority_score
        )
        SELECT
            v_today,
            t.location_id,
            ln.loc_name,
            t.sku_id,
            LEFT(COALESCE(t.sku_description, ''), 500),
            t.days_observed,
            t.days_below_reorder,
            ROUND(t.pct::numeric, 4),
            ROUND(t.avg_daily::numeric, 4),
            ROUND(t.current_min::numeric, 2),
            ROUND(t.suggested_min::numeric, 2),
            ROUND((t.suggested_min - t.current_min)::numeric, 2),
            ROUND(t.unit_cost::numeric, 2),
            ROUND(t.value_at_risk::numeric, 2),
            t.xfer_count,
            ROUND(t.priority::numeric, 4)
        FROM _us_tmp t
        LEFT JOIN location_names ln ON ln.loc_id = t.location_id
        ON CONFLICT (report_date, location_id, sku_id) DO UPDATE SET
            location_name              = EXCLUDED.location_name,
            sku_description            = EXCLUDED.sku_description,
            days_observed              = EXCLUDED.days_observed,
            days_below_reorder         = EXCLUDED.days_below_reorder,
            stockout_days_pct          = EXCLUDED.stockout_days_pct,
            avg_daily_demand           = EXCLUDED.avg_daily_demand,
            current_min_qty            = EXCLUDED.current_min_qty,
            suggested_min_qty          = EXCLUDED.suggested_min_qty,
            min_qty_gap                = EXCLUDED.min_qty_gap,
            unit_cost                  = EXCLUDED.unit_cost,
            inventory_value_at_risk    = EXCLUDED.inventory_value_at_risk,
            transfer_recommended_count = EXCLUDED.transfer_recommended_count,
            priority_score             = EXCLUDED.priority_score;

        GET DIAGNOSTICS v_persisted = ROW_COUNT;
    END IF;

    RETURN QUERY SELECT
        v_today, v_built, v_persisted, v_locs, v_value, v_window_days;
END;
$$;
