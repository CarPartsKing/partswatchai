"""
extract/partswatch_pull.py — PartsWatch data extraction and load pipeline.

ARCHITECTURE
------------
All data sources implement a common DataSource interface (connect / extract /
close).  Switching from CSV files today to an ODBC connection or REST API
tomorrow requires changing exactly ONE environment variable — PARTSWATCH_SOURCE.
Nothing else in the codebase changes.

DATA SOURCES
    csv   — reads CSV or Excel files from a local folder (default)
    odbc  — direct ODBC connection to PartsWatch DB (stub; ready to implement)
    api   — REST API connection to PartsWatch (stub; ready to implement)

COLUMN MAPPING
    config/partswatch_column_map.json maps our schema field names to whatever
    PartsWatch actually calls them in the export.  Missing columns are logged
    as warnings and skipped — the pipeline never crashes on a missing column.

DATASETS LOADED (in FK-dependency order)
    sku_master            → auto-registers any new SKUs seen in other datasets
    sales_transactions    → on_conflict: transaction_id
    inventory_snapshots   → on_conflict: sku_id, location_id, snapshot_date
    purchase_orders       → on_conflict: po_number, line_number

GENERATED COLUMNS
    PostgreSQL GENERATED ALWAYS columns are stripped before every upsert:
    - inventory_snapshots.is_stockout
    - purchase_orders.lead_time_variance
    - purchase_orders.fill_rate_pct

USAGE
    # Validate sample data without writing to database
    python -m extract.partswatch_pull --test-mode

    # Full load (uses PARTSWATCH_SOURCE env var, default = csv)
    python -m extract.partswatch_pull

SWITCHING DATA SOURCES
    CSV  → set PARTSWATCH_SOURCE=csv  and PARTSWATCH_CSV_PATH=/path/to/exports
    ODBC → set PARTSWATCH_SOURCE=odbc and PARTSWATCH_ODBC_DSN=...
    API  → set PARTSWATCH_SOURCE=api  and PARTSWATCH_API_URL=... PARTSWATCH_API_KEY=...
"""

import abc
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from utils.logging_config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

# Processing order respects FK dependencies: sku_master must be loaded first
DATASETS: list[str] = [
    "sku_master",
    "sales_transactions",
    "inventory_snapshots",
    "purchase_orders",
]

# PostgreSQL GENERATED ALWAYS columns — must never appear in upsert payloads
GENERATED_COLS: dict[str, set[str]] = {
    "inventory_snapshots": {"is_stockout"},
    "purchase_orders":     {"lead_time_variance", "fill_rate_pct"},
}

# Conflict-resolution column(s) per table (used by Supabase upsert)
UPSERT_ON_CONFLICT: dict[str, str] = {
    "sku_master":           "sku_id",
    "sales_transactions":   "transaction_id",
    "inventory_snapshots":  "sku_id,location_id,snapshot_date",
    "purchase_orders":      "po_number,line_number",
}

# Maximum rows per Supabase upsert call — keeps payloads well under PostgREST limits
BATCH_SIZE: int = 500


# ---------------------------------------------------------------------------
# Abstract DataSource interface
# ---------------------------------------------------------------------------

class DataSource(abc.ABC):
    """Abstract interface for all PartsWatch data sources.

    All concrete implementations must honour this contract so that the
    rest of the pipeline can treat them interchangeably.

    Supports the context-manager protocol — use with `with` statements:

        with get_data_source() as source:
            rows = source.extract("sales_transactions")
    """

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish a connection to the data source.

        Raises:
            ConnectionError: If the source cannot be reached.
            FileNotFoundError: If a required file or path is missing.
            NotImplementedError: If the concrete class is a stub.
        """

    @abc.abstractmethod
    def extract(self, dataset: str) -> list[dict]:
        """Return all rows for the named dataset using original source column names.

        Args:
            dataset: Logical dataset name (e.g. "sales_transactions").
                     The column map translates source names to schema names.

        Returns:
            List of dicts keyed by the source's own column names.
            Returns an empty list if the dataset is not available — never raises.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the data source."""

    def __enter__(self) -> "DataSource":
        """Connect on entry to a with-block."""
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        """Close on exit from a with-block regardless of exceptions."""
        self.close()


# ---------------------------------------------------------------------------
# CSV / Excel implementation
# ---------------------------------------------------------------------------

class CSVDataSource(DataSource):
    """Read datasets from a folder of CSV or Excel files.

    File discovery (checked in this order for each dataset name):
        {folder}/{dataset}.csv
        {folder}/{dataset}.xlsx
        {folder}/{dataset}.xls

    The first match wins.  If no file is found the dataset is skipped with
    a warning — the pipeline continues with the remaining datasets.

    Args:
        folder_path: Path to the directory containing the export files.
    """

    def __init__(self, folder_path: str) -> None:
        self.folder = Path(folder_path)
        self._connected: bool = False

    def connect(self) -> None:
        """Verify the folder exists and is readable."""
        if not self.folder.exists():
            raise FileNotFoundError(
                f"CSV source folder not found: {self.folder.resolve()}\n"
                f"Set PARTSWATCH_CSV_PATH to the folder containing your export files."
            )
        if not self.folder.is_dir():
            raise NotADirectoryError(
                f"PARTSWATCH_CSV_PATH must be a directory, not a file: {self.folder.resolve()}"
            )
        self._connected = True
        log.info("CSVDataSource connected → %s", self.folder.resolve())

    def extract(self, dataset: str) -> list[dict]:
        """Find and read the file for the given dataset name.

        Args:
            dataset: Logical dataset name (e.g. "sales_transactions").

        Returns:
            List of row dicts with original source column names.
        """
        if not self._connected:
            raise RuntimeError("Call connect() before extract().")

        for ext in (".csv", ".xlsx", ".xls"):
            candidate = self.folder / f"{dataset}{ext}"
            if candidate.exists():
                log.info("Reading %-30s ← %s", dataset, candidate.name)
                return self._read_file(candidate)

        log.warning(
            "No file found for dataset '%s' in %s (tried .csv .xlsx .xls) — skipping.",
            dataset,
            self.folder,
        )
        return []

    def _read_file(self, path: Path) -> list[dict]:
        """Read a CSV or Excel file and return a list of row dicts.

        All values are initially read as strings to avoid pandas type-inference
        surprises (e.g. leading-zero part numbers becoming integers).
        Type coercion to bool/int/float is applied later in clean_row().

        Args:
            path: Absolute or relative path to the file.

        Returns:
            List of row dicts with stripped column names and string values.

        Raises:
            Exception: Re-raises any pandas read error after logging it.
        """
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
            else:
                df = pd.read_excel(path, dtype=str, keep_default_na=False)

            # Strip whitespace from column headers and string cell values
            df.columns = df.columns.str.strip()
            str_cols = [c for c in df.columns if df[c].dtype == object]
            for col in str_cols:
                df[col] = df[col].str.strip()

            log.info("  → %d rows, %d columns", len(df), len(df.columns))
            return df.to_dict(orient="records")

        except Exception as exc:
            log.error("Failed to read %s: %s", path, exc, exc_info=True)
            raise

    def close(self) -> None:
        """Mark the source as disconnected."""
        self._connected = False
        log.debug("CSVDataSource closed.")


# ---------------------------------------------------------------------------
# ODBC stub
# ---------------------------------------------------------------------------

class ODBCDataSource(DataSource):
    """Direct ODBC connection to the PartsWatch database.

    NOT YET IMPLEMENTED.

    When your Autologue representative provides the ODBC DSN or connection
    string, implement this class and set PARTSWATCH_SOURCE=odbc.
    No other files need to change.

    Args:
        connection_string: ODBC DSN or full connection string.
    """

    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string

    def connect(self) -> None:
        raise NotImplementedError(
            "ODBCDataSource is not yet implemented.\n\n"
            "When Autologue provides an ODBC connection string:\n"
            "  1. pip install pyodbc\n"
            "  2. Set PARTSWATCH_ODBC_DSN in your .env file\n"
            "  3. Implement connect() and extract() in ODBCDataSource\n"
            "  4. Set PARTSWATCH_SOURCE=odbc\n\n"
            "No other changes to the pipeline are required."
        )

    def extract(self, dataset: str) -> list[dict]:
        raise NotImplementedError("Call connect() first.")

    def close(self) -> None:
        pass  # No-op until implemented


# ---------------------------------------------------------------------------
# REST API stub
# ---------------------------------------------------------------------------

class APIDataSource(DataSource):
    """REST API connection to a PartsWatch or Autologue API endpoint.

    NOT YET IMPLEMENTED.

    When Autologue provides API credentials, implement this class and
    set PARTSWATCH_SOURCE=api.  No other files need to change.

    Args:
        base_url: Root URL of the PartsWatch API.
        api_key:  Authentication token or API key.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def connect(self) -> None:
        raise NotImplementedError(
            "APIDataSource is not yet implemented.\n\n"
            "When Autologue provides REST API credentials:\n"
            "  1. Set PARTSWATCH_API_URL and PARTSWATCH_API_KEY in your .env\n"
            "  2. Implement connect() and extract() in APIDataSource\n"
            "  3. Set PARTSWATCH_SOURCE=api\n\n"
            "No other changes to the pipeline are required."
        )

    def extract(self, dataset: str) -> list[dict]:
        raise NotImplementedError("Call connect() first.")

    def close(self) -> None:
        pass  # No-op until implemented


# ---------------------------------------------------------------------------
# Data source factory
# ---------------------------------------------------------------------------

def get_data_source() -> DataSource:
    """Return the DataSource implementation selected by PARTSWATCH_SOURCE.

    Reads the PARTSWATCH_SOURCE environment variable and returns the
    appropriate concrete DataSource instance.

    Returns:
        A DataSource instance (not yet connected — call connect() or use
        it as a context manager).

    Raises:
        ValueError: If PARTSWATCH_SOURCE contains an unrecognised value.
    """
    from config import PARTSWATCH_SOURCE, PARTSWATCH_CSV_PATH

    source = PARTSWATCH_SOURCE.lower().strip()
    log.info("Data source: %s", source)

    if source == "csv":
        return CSVDataSource(PARTSWATCH_CSV_PATH)

    if source == "odbc":
        from config import PARTSWATCH_ODBC_DSN
        return ODBCDataSource(PARTSWATCH_ODBC_DSN)

    if source == "api":
        from config import PARTSWATCH_API_URL, PARTSWATCH_API_KEY
        return APIDataSource(PARTSWATCH_API_URL, PARTSWATCH_API_KEY)

    raise ValueError(
        f"Unknown PARTSWATCH_SOURCE='{source}'. Valid values: csv | odbc | api"
    )


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def load_column_map() -> dict:
    """Load and return the partswatch_column_map.json config file.

    Returns:
        Dict of {dataset: {schema_col: source_col}} mappings.

    Raises:
        FileNotFoundError: If the config file is missing.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    from config import PARTSWATCH_COLUMN_MAP_PATH

    path = Path(PARTSWATCH_COLUMN_MAP_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Column map not found: {path.resolve()}\n"
            "Expected: config/partswatch_column_map.json"
        )

    with path.open(encoding="utf-8") as fh:
        cmap = json.load(fh)

    # Strip the meta comment key if present
    cmap.pop("_comment", None)

    log.info("Column map loaded from %s (%d datasets).", path, len(cmap))
    return cmap


def apply_column_map(
    rows: list[dict],
    dataset: str,
    column_map: dict,
) -> list[dict]:
    """Rename source columns to schema column names using the column map.

    - Missing source columns are logged as warnings once, then skipped.
    - Extra source columns not in the map are silently dropped.
    - Returns rows as-is (with a warning) if the dataset has no mapping entry.

    Args:
        rows:       Raw rows with source column names.
        dataset:    Logical dataset name (e.g. "sales_transactions").
        column_map: Full mapping dict loaded from partswatch_column_map.json.

    Returns:
        List of dicts keyed by schema column names.
    """
    if not rows:
        return []

    if dataset not in column_map:
        log.warning(
            "No column map entry for '%s' — returning rows with original column names.",
            dataset,
        )
        return rows

    mapping: dict[str, str] = column_map[dataset]  # {schema_col: source_col}
    source_cols = set(rows[0].keys())

    # Warn once for each schema field whose source column is absent
    for schema_col, source_col in mapping.items():
        if source_col not in source_cols:
            log.warning(
                "Dataset '%s': source column '%s' (→ '%s') not in file — field will be NULL.",
                dataset,
                source_col,
                schema_col,
            )

    mapped_rows = []
    for row in rows:
        new_row: dict[str, Any] = {}
        for schema_col, source_col in mapping.items():
            if source_col in row:
                new_row[schema_col] = row[source_col]
        mapped_rows.append(new_row)

    return mapped_rows


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------

def _coerce_value(value: Any) -> Any:
    """Auto-coerce a string value to bool, int, float, or None.

    All values arrive as strings when read with pandas dtype=str.
    PostgreSQL/PostgREST requires actual JSON booleans and numbers
    for typed columns — string representations are rejected.

    Coercion rules (applied in order):
        1. None or empty string  → None
        2. 'TRUE'/'FALSE' (case-insensitive) → bool
        3. Pure integer string   → int
        4. Numeric string        → float
        5. Anything else         → string as-is

    Args:
        value: Raw value from the source file.

    Returns:
        Coerced Python value.
    """
    if value is None or value == "":
        return None

    if isinstance(value, str):
        upper = value.upper()
        # Only treat unambiguous text booleans as bool — never bare "0"/"1"
        # (those are integers that could also be line numbers, quantities, etc.)
        if upper in ("TRUE", "YES"):
            return True
        if upper in ("FALSE", "NO"):
            return False
        # Try integer then float
        try:
            as_int = int(value)
            return as_int
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass

    return value


def clean_row(row: dict, dataset: str) -> dict | None:
    """Coerce types, remove GENERATED columns, and drop empty rows.

    Args:
        row:     Single row dict with schema column names.
        dataset: Logical dataset name — used to look up GENERATED_COLS.

    Returns:
        Cleaned row dict, or None if the row is completely empty after cleaning.
    """
    # Strip GENERATED ALWAYS columns — PostgreSQL computes these automatically
    for gen_col in GENERATED_COLS.get(dataset, set()):
        row.pop(gen_col, None)

    # Coerce all values
    cleaned = {k: _coerce_value(v) for k, v in row.items()}

    # Drop the row if every value is None (blank CSV line)
    if all(v is None for v in cleaned.values()):
        return None

    return cleaned


# ---------------------------------------------------------------------------
# Supabase upsert helpers
# ---------------------------------------------------------------------------

def _upsert_batch(client: Any, table: str, rows: list[dict], on_conflict: str) -> None:
    """Upsert one batch of rows into a Supabase table.

    Args:
        client:      Active Supabase client.
        table:       Target table name.
        rows:        List of row dicts (already cleaned).
        on_conflict: Comma-separated conflict column string.

    Raises:
        Exception: Propagates any Supabase API error after logging it.
    """
    try:
        client.table(table).upsert(rows, on_conflict=on_conflict).execute()
    except Exception as exc:
        log.error("Upsert failed — table '%s': %s", table, exc, exc_info=True)
        raise


def upsert_dataset(client: Any, table: str, rows: list[dict]) -> int:
    """Upsert all rows for a table, processing them in BATCH_SIZE chunks.

    Args:
        client: Active Supabase client.
        table:  Target table name.
        rows:   Full list of cleaned row dicts.

    Returns:
        Total number of rows upserted.

    Raises:
        ValueError: If no on_conflict key is defined for the table.
        Exception:  Propagates Supabase errors after logging.
    """
    if not rows:
        log.info("%-30s — no rows to upsert.", table)
        return 0

    on_conflict = UPSERT_ON_CONFLICT.get(table)
    if not on_conflict:
        raise ValueError(f"No on_conflict key defined for table '{table}'.")

    total = len(rows)
    upserted = 0
    log.info("Upserting %d rows → %s …", total, table)

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        _upsert_batch(client, table, batch, on_conflict)
        upserted += len(batch)
        log.info("  %d / %d rows written.", upserted, total)

    return upserted


# ---------------------------------------------------------------------------
# Auto-register missing SKUs (FK safety net)
# ---------------------------------------------------------------------------

def register_new_skus(client: Any, all_rows: dict[str, list[dict]]) -> None:
    """Upsert skeleton sku_master rows for any sku_id not yet in the database.

    Prevents FK constraint violations when sales/inventory/PO data arrives
    before a complete sku_master export has been processed.  The ML pipeline
    will later populate abc_class, weather_sensitivity_score, etc.

    Args:
        client:   Active Supabase client.
        all_rows: Dict of {dataset: [rows]} for all loaded datasets.
    """
    seen_skus: set[str] = set()
    for dataset, rows in all_rows.items():
        if dataset == "sku_master":
            continue
        for row in rows:
            sku = row.get("sku_id")
            if sku and str(sku) not in ("", "None"):
                seen_skus.add(str(sku))

    if not seen_skus:
        return

    existing_resp = (
        client.table("sku_master")
        .select("sku_id")
        .in_("sku_id", list(seen_skus))
        .execute()
    )
    existing = {r["sku_id"] for r in (existing_resp.data or [])}
    missing  = seen_skus - existing

    if not missing:
        log.info("All %d referenced SKUs already exist in sku_master.", len(seen_skus))
        return

    log.info(
        "Auto-registering %d new SKU(s) in sku_master (abc_class to be set by ML pipeline) …",
        len(missing),
    )
    skeleton_rows = [{"sku_id": sku, "is_active": True} for sku in sorted(missing)]
    _upsert_batch(client, "sku_master", skeleton_rows, on_conflict="sku_id")
    log.info("Auto-registered %d skeleton SKU record(s).", len(missing))


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(test_mode: bool = False) -> int:
    """Extract, transform, and load all PartsWatch datasets into Supabase.

    Processing order:
        1. Extract all datasets from the configured source.
        2. Apply column mapping (schema names replace source names).
        3. Clean each row (type coercion, strip GENERATED columns).
        4. If test_mode: print summary and exit — nothing is written.
        5. Auto-register any new SKUs (FK safety net).
        6. Upsert datasets in FK-dependency order.

    Args:
        test_mode: When True, validates and reports without writing to the DB.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    try:
        column_map = load_column_map()
        all_rows: dict[str, list[dict]] = {}

        with get_data_source() as source:
            for dataset in DATASETS:
                raw   = source.extract(dataset)
                if not raw:
                    all_rows[dataset] = []
                    continue

                mapped  = apply_column_map(raw, dataset, column_map)
                cleaned = [r for row in mapped if (r := clean_row(row, dataset)) is not None]

                all_rows[dataset] = cleaned
                log.info("Dataset '%-25s' — %d rows ready.", dataset, len(cleaned))

        # ── Test mode ──────────────────────────────────────────────────────────
        if test_mode:
            log.info("=" * 60)
            log.info("TEST MODE — validation only, nothing written to database.")
            log.info("=" * 60)
            for dataset in DATASETS:
                rows = all_rows.get(dataset, [])
                log.info("  %-30s %d rows", dataset, len(rows))
                if rows:
                    log.info("  Sample: %s", rows[0])
            log.info("=" * 60)
            log.info("Column map is valid. Run without --test-mode to load data.")
            return 0

        # ── Write mode ─────────────────────────────────────────────────────────
        from db.connection import get_client
        client = get_client()

        # Insert skeleton sku_master records for any unknown SKU IDs
        register_new_skus(client, all_rows)

        totals: dict[str, int] = {}
        for dataset in DATASETS:
            rows = all_rows.get(dataset, [])
            totals[dataset] = upsert_dataset(client, dataset, rows)

        log.info("=" * 60)
        log.info("Pipeline complete.")
        for dataset in DATASETS:
            log.info("  %-30s %d rows upserted", dataset, totals[dataset])
        log.info("=" * 60)
        return 0

    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI arguments and run the pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="partswatch-ai: extract PartsWatch data into Supabase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  PARTSWATCH_SOURCE   csv | odbc | api  (default: csv)\n"
            "  PARTSWATCH_CSV_PATH folder containing export files (default: sample_data)\n"
        ),
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Parse and validate data without writing to the database.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("partswatch-ai — partswatch_pull  test_mode=%s", args.test_mode)
    log.info("=" * 60)

    return run_pipeline(test_mode=args.test_mode)


if __name__ == "__main__":
    sys.exit(main())
