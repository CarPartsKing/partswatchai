"""
db_setup.py — Create all 7 partswatch-ai database tables and indexes.

Connects directly to the Supabase PostgreSQL database via psycopg2 and runs
idempotent DDL (IF NOT EXISTS everywhere).  Safe to re-run at any time —
existing data and existing tables are never dropped or truncated.

Usage:
    python db_setup.py

Environment variables required (see .env.example):
    SUPABASE_DB_URL — PostgreSQL connection URI from Supabase Project Settings
                      → Database → Connection string (URI mode)
"""

import sys
import time
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as PgConnection

from utils.logging_config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL definitions — one dict per table
# ---------------------------------------------------------------------------

TABLE_DDL: list[dict] = [
    # ------------------------------------------------------------------
    # sku_master — referenced by all other tables; create first
    # ------------------------------------------------------------------
    {
        "name": "sku_master",
        "create": """
            CREATE TABLE IF NOT EXISTS sku_master (
                id                       BIGSERIAL PRIMARY KEY,
                sku_id                   VARCHAR(50)  NOT NULL UNIQUE,
                description              TEXT,
                brand                    VARCHAR(100),
                part_category            VARCHAR(100),
                -- e.g. brake pads, filters, belts — drives weather sensitivity grouping
                sub_category             VARCHAR(100),
                unit_of_measure          VARCHAR(20)  DEFAULT 'EA',
                abc_class                VARCHAR(1)   CHECK (abc_class IN ('A','B','C')),
                is_active                BOOLEAN      NOT NULL DEFAULT TRUE,
                is_dead_stock            BOOLEAN      NOT NULL DEFAULT FALSE,
                -- 0.0 (no sensitivity) to 1.0 (high sensitivity) — set by ML
                weather_sensitivity_score NUMERIC(5,4) DEFAULT 0.0,
                -- last date this SKU appeared in a sales transaction
                last_sale_date           DATE,
                -- average weekly units sold (updated nightly)
                avg_weekly_units         NUMERIC(10,4),
                created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_sku_master_abc_class   ON sku_master (abc_class)",
            "CREATE INDEX IF NOT EXISTS idx_sku_master_brand        ON sku_master (brand)",
            "CREATE INDEX IF NOT EXISTS idx_sku_master_category     ON sku_master (part_category)",
            "CREATE INDEX IF NOT EXISTS idx_sku_master_active       ON sku_master (is_active)",
            "CREATE INDEX IF NOT EXISTS idx_sku_master_last_sale    ON sku_master (last_sale_date)",
        ],
    },

    # ------------------------------------------------------------------
    # sales_transactions — raw line-item sales from PartsWatch
    # ------------------------------------------------------------------
    {
        "name": "sales_transactions",
        "create": """
            CREATE TABLE IF NOT EXISTS sales_transactions (
                id                       BIGSERIAL PRIMARY KEY,
                -- PartsWatch native transaction identifier
                transaction_id           VARCHAR(80)  UNIQUE,
                sku_id                   VARCHAR(50)  NOT NULL REFERENCES sku_master(sku_id),
                location_id              VARCHAR(20)  NOT NULL,
                transaction_date         DATE         NOT NULL,
                qty_sold                 NUMERIC(12,4) NOT NULL DEFAULT 0,
                unit_price               NUMERIC(12,4),
                total_revenue            NUMERIC(14,4),
                -- TRUE when qty_on_hand was 0 at time of sale (lost-sale signal)
                is_stockout              BOOLEAN      NOT NULL DEFAULT FALSE,
                -- ML-imputed demand when a stockout obscured true demand
                lost_sales_imputation    NUMERIC(12,4) DEFAULT 0,
                created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_sales_sku_id        ON sales_transactions (sku_id)",
            "CREATE INDEX IF NOT EXISTS idx_sales_date          ON sales_transactions (transaction_date)",
            "CREATE INDEX IF NOT EXISTS idx_sales_location      ON sales_transactions (location_id)",
            "CREATE INDEX IF NOT EXISTS idx_sales_sku_date      ON sales_transactions (sku_id, transaction_date)",
            "CREATE INDEX IF NOT EXISTS idx_sales_stockout      ON sales_transactions (is_stockout) WHERE is_stockout = TRUE",
        ],
    },

    # ------------------------------------------------------------------
    # inventory_snapshots — nightly on-hand snapshot per SKU per location
    # ------------------------------------------------------------------
    {
        "name": "inventory_snapshots",
        "create": """
            CREATE TABLE IF NOT EXISTS inventory_snapshots (
                id                       BIGSERIAL PRIMARY KEY,
                sku_id                   VARCHAR(50)  NOT NULL REFERENCES sku_master(sku_id),
                location_id              VARCHAR(20)  NOT NULL,
                snapshot_date            DATE         NOT NULL,
                qty_on_hand              NUMERIC(12,4) NOT NULL DEFAULT 0,
                qty_on_order             NUMERIC(12,4) NOT NULL DEFAULT 0,
                -- computed: qty_on_hand = 0
                is_stockout              BOOLEAN GENERATED ALWAYS AS (qty_on_hand <= 0) STORED,
                -- replenishment trigger level (set by reorder engine)
                reorder_point            NUMERIC(12,4),
                -- order quantity recommendation (set by reorder engine)
                reorder_qty              NUMERIC(12,4),
                created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                -- one snapshot per SKU per location per day
                UNIQUE (sku_id, location_id, snapshot_date)
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_inv_sku_id          ON inventory_snapshots (sku_id)",
            "CREATE INDEX IF NOT EXISTS idx_inv_snapshot_date   ON inventory_snapshots (snapshot_date)",
            "CREATE INDEX IF NOT EXISTS idx_inv_location        ON inventory_snapshots (location_id)",
            "CREATE INDEX IF NOT EXISTS idx_inv_sku_date        ON inventory_snapshots (sku_id, snapshot_date)",
            "CREATE INDEX IF NOT EXISTS idx_inv_stockout        ON inventory_snapshots (is_stockout) WHERE is_stockout = TRUE",
        ],
    },

    # ------------------------------------------------------------------
    # purchase_orders — PO header + line items
    # ------------------------------------------------------------------
    {
        "name": "purchase_orders",
        "create": """
            CREATE TABLE IF NOT EXISTS purchase_orders (
                id                       BIGSERIAL PRIMARY KEY,
                po_number                VARCHAR(80)  NOT NULL,
                -- individual line within a PO (po_number + line_number = unique)
                line_number              INTEGER      NOT NULL DEFAULT 1,
                sku_id                   VARCHAR(50)  NOT NULL REFERENCES sku_master(sku_id),
                supplier_id              VARCHAR(50)  NOT NULL,
                location_id              VARCHAR(20)  NOT NULL,
                po_date                  DATE         NOT NULL,
                expected_delivery_date   DATE,
                actual_delivery_date     DATE,
                qty_ordered              NUMERIC(12,4) NOT NULL DEFAULT 0,
                qty_received             NUMERIC(12,4) DEFAULT 0,
                unit_cost                NUMERIC(12,4),
                status                   VARCHAR(20)  NOT NULL DEFAULT 'open'
                                             CHECK (status IN ('open','received','partial','cancelled')),
                -- positive = days late, negative = days early (NULL until delivered)
                lead_time_variance       NUMERIC(8,2) GENERATED ALWAYS AS (
                    CASE
                        WHEN actual_delivery_date IS NOT NULL AND expected_delivery_date IS NOT NULL
                        THEN (actual_delivery_date - expected_delivery_date)::NUMERIC
                        ELSE NULL
                    END
                ) STORED,
                -- 0–1 ratio of qty received vs ordered (NULL until any receipt)
                fill_rate_pct            NUMERIC(6,4) GENERATED ALWAYS AS (
                    CASE
                        WHEN qty_ordered > 0 AND qty_received IS NOT NULL
                        THEN LEAST(qty_received / qty_ordered, 1.0)
                        ELSE NULL
                    END
                ) STORED,
                created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                UNIQUE (po_number, line_number)
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_po_sku_id           ON purchase_orders (sku_id)",
            "CREATE INDEX IF NOT EXISTS idx_po_supplier_id      ON purchase_orders (supplier_id)",
            "CREATE INDEX IF NOT EXISTS idx_po_po_date          ON purchase_orders (po_date)",
            "CREATE INDEX IF NOT EXISTS idx_po_location         ON purchase_orders (location_id)",
            "CREATE INDEX IF NOT EXISTS idx_po_status           ON purchase_orders (status)",
            "CREATE INDEX IF NOT EXISTS idx_po_open             ON purchase_orders (status) WHERE status = 'open'",
            "CREATE INDEX IF NOT EXISTS idx_po_sku_status       ON purchase_orders (sku_id, status)",
        ],
    },

    # ------------------------------------------------------------------
    # weather_log — daily NE Ohio conditions (from Open-Meteo)
    # ------------------------------------------------------------------
    {
        "name": "weather_log",
        "create": """
            CREATE TABLE IF NOT EXISTS weather_log (
                id                       BIGSERIAL PRIMARY KEY,
                log_date                 DATE         NOT NULL UNIQUE,
                -- temperatures in Fahrenheit
                temp_max_f               NUMERIC(6,2),
                temp_min_f               NUMERIC(6,2),
                -- precipitation total in inches
                precipitation_in         NUMERIC(6,3),
                snowfall_in              NUMERIC(6,3),
                -- how many consecutive days the min temp has been ≤ 32°F
                consecutive_freeze_days  INTEGER      NOT NULL DEFAULT 0,
                -- TRUE on days following a thaw after ≥2 consecutive freeze days
                -- (strong predictor for pothole-season part demand)
                freeze_thaw_cycle        BOOLEAN      NOT NULL DEFAULT FALSE,
                created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_weather_log_date         ON weather_log (log_date)",
            "CREATE INDEX IF NOT EXISTS idx_weather_freeze_thaw       ON weather_log (freeze_thaw_cycle) WHERE freeze_thaw_cycle = TRUE",
            "CREATE INDEX IF NOT EXISTS idx_weather_consec_freeze     ON weather_log (consecutive_freeze_days)",
        ],
    },

    # ------------------------------------------------------------------
    # forecast_results — output from all three ML models
    # ------------------------------------------------------------------
    {
        "name": "forecast_results",
        "create": """
            CREATE TABLE IF NOT EXISTS forecast_results (
                id                       BIGSERIAL PRIMARY KEY,
                sku_id                   VARCHAR(50)  NOT NULL REFERENCES sku_master(sku_id),
                -- week starting date this forecast covers
                forecast_date            DATE         NOT NULL,
                -- which model produced this row
                model_type               VARCHAR(20)  NOT NULL
                                             CHECK (model_type IN ('prophet','lightgbm','rolling_avg')),
                predicted_qty            NUMERIC(12,4) NOT NULL,
                -- prediction interval bounds (NULL for rolling_avg)
                lower_bound              NUMERIC(12,4),
                upper_bound              NUMERIC(12,4),
                -- model confidence 0–1 (NULL if model doesn't emit it)
                confidence_pct           NUMERIC(5,4),
                -- date the forecast run executed
                run_date                 DATE         NOT NULL,
                created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                -- one forecast per SKU per week per model per run
                UNIQUE (sku_id, forecast_date, model_type, run_date)
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_forecast_sku_id       ON forecast_results (sku_id)",
            "CREATE INDEX IF NOT EXISTS idx_forecast_date         ON forecast_results (forecast_date)",
            "CREATE INDEX IF NOT EXISTS idx_forecast_model_type   ON forecast_results (model_type)",
            "CREATE INDEX IF NOT EXISTS idx_forecast_run_date     ON forecast_results (run_date)",
            "CREATE INDEX IF NOT EXISTS idx_forecast_sku_date     ON forecast_results (sku_id, forecast_date)",
            "CREATE INDEX IF NOT EXISTS idx_forecast_sku_model    ON forecast_results (sku_id, model_type, forecast_date)",
        ],
    },

    # ------------------------------------------------------------------
    # supplier_scores — rolling performance scorecard per supplier
    # ------------------------------------------------------------------
    {
        "name": "supplier_scores",
        "create": """
            CREATE TABLE IF NOT EXISTS supplier_scores (
                id                       BIGSERIAL PRIMARY KEY,
                supplier_id              VARCHAR(50)  NOT NULL,
                supplier_name            VARCHAR(150),
                -- date this score snapshot was computed
                score_date               DATE         NOT NULL,
                -- rolling average lead time in calendar days
                avg_lead_time_days       NUMERIC(8,2),
                -- average of abs(lead_time_variance) across all PO lines in window
                lead_time_variance_avg   NUMERIC(8,2),
                -- rolling fill rate: sum(qty_received) / sum(qty_ordered)
                fill_rate_pct            NUMERIC(6,4),
                -- share of PO lines delivered on or before expected date
                on_time_delivery_pct     NUMERIC(6,4),
                -- composite 0–100 score (higher = better) — formula in reorder engine
                composite_score          NUMERIC(6,2),
                created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                UNIQUE (supplier_id, score_date)
            )
        """,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_supplier_supplier_id  ON supplier_scores (supplier_id)",
            "CREATE INDEX IF NOT EXISTS idx_supplier_score_date   ON supplier_scores (score_date)",
            "CREATE INDEX IF NOT EXISTS idx_supplier_composite    ON supplier_scores (composite_score)",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db_connection() -> PgConnection:
    """Open a psycopg2 connection to the Supabase PostgreSQL database.

    Returns:
        An open psycopg2 connection with autocommit disabled.

    Raises:
        EnvironmentError: If SUPABASE_DB_URL is not configured.
        psycopg2.OperationalError: If the connection attempt fails.
    """
    from config import SUPABASE_DB_URL

    log.info("Connecting to Supabase PostgreSQL …")
    conn = psycopg2.connect(SUPABASE_DB_URL, connect_timeout=15)
    conn.autocommit = False
    log.info("Connection established.")
    return conn


def table_exists(cursor, table_name: str) -> bool:
    """Check if a table already exists in the public schema.

    Args:
        cursor:     Active psycopg2 cursor.
        table_name: Table name to look up.

    Returns:
        True if the table exists in public schema, False otherwise.
    """
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name   = %s
        )
        """,
        (table_name,),
    )
    return cursor.fetchone()[0]


def create_table(cursor, table_def: dict) -> tuple[bool, int]:
    """Execute CREATE TABLE IF NOT EXISTS and all associated indexes.

    Args:
        cursor:    Active psycopg2 cursor.
        table_def: Dict with keys 'name', 'create', and 'indexes'.

    Returns:
        Tuple of (was_new: bool, index_count: int).
    """
    name = table_def["name"]
    already_existed = table_exists(cursor, name)

    # Create the table
    cursor.execute(table_def["create"])

    # Create all indexes
    for idx_ddl in table_def.get("indexes", []):
        cursor.execute(idx_ddl)

    return (not already_existed), len(table_def.get("indexes", []))


# ---------------------------------------------------------------------------
# Main setup routine
# ---------------------------------------------------------------------------

def run_setup() -> int:
    """Create all tables and indexes.  Idempotent — safe to re-run.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    log.info("=" * 60)
    log.info("partswatch-ai — Database Setup")
    log.info("=" * 60)

    conn: PgConnection | None = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        new_tables = 0
        total_indexes = 0

        for table_def in TABLE_DDL:
            name = table_def["name"]
            log.info("Processing table: %s …", name)

            t0 = time.perf_counter()
            is_new, idx_count = create_table(cursor, table_def)
            elapsed = time.perf_counter() - t0

            status = "CREATED" if is_new else "EXISTS"
            log.info(
                "  [%s] %s  (%d index%s, %.0f ms)",
                status,
                name,
                idx_count,
                "es" if idx_count != 1 else "",
                elapsed * 1000,
            )

            if is_new:
                new_tables += 1
            total_indexes += idx_count

        # Commit all DDL in one transaction
        conn.commit()
        log.info("-" * 60)
        log.info(
            "Setup complete.  %d new table(s) created, %d total index(es) applied.",
            new_tables,
            total_indexes,
        )
        log.info("=" * 60)
        return 0

    except Exception as exc:
        log.error("Database setup failed: %s", exc, exc_info=True)
        if conn:
            try:
                conn.rollback()
                log.info("Transaction rolled back.")
            except Exception as rb_exc:
                log.error("Rollback also failed: %s", rb_exc)
        return 1

    finally:
        if conn:
            try:
                conn.close()
                log.info("Database connection closed.")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_setup())
