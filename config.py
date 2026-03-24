"""
config.py — Centralized configuration for partswatch-ai.

Loads all environment variables from .env and exposes them as typed
attributes. Import this module wherever settings are needed instead of
calling os.getenv() directly.
"""

import os
import logging
from dotenv import load_dotenv

# Load .env file before anything else reads env vars
load_dotenv()


def _require(key: str) -> str:
    """Return the value of a required environment variable.

    Raises:
        EnvironmentError: If the variable is not set or is empty.
    """
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Check your .env file against .env.example."
        )
    return value


def _optional(key: str, default: str = "") -> str:
    """Return the value of an optional environment variable with a default."""
    return os.getenv(key, default).strip()


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL: str = _require("SUPABASE_URL")
SUPABASE_KEY: str = _require("SUPABASE_KEY")

# Direct PostgreSQL connection string — required for DDL (CREATE TABLE/INDEX).
# Format: postgresql://postgres:[password]@[host]:5432/postgres
# Find this in Supabase → Project Settings → Database → Connection string (URI).
SUPABASE_DB_URL: str = _require("SUPABASE_DB_URL")

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = _require("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Open-Meteo weather (NE Ohio / Cleveland defaults)
# ---------------------------------------------------------------------------
WEATHER_LAT: float = float(_optional("WEATHER_LAT", "41.4993"))
WEATHER_LON: float = float(_optional("WEATHER_LON", "-81.6944"))

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
LOG_LEVEL: str = _optional("LOG_LEVEL", "INFO").upper()
ENVIRONMENT: str = _optional("ENVIRONMENT", "development").lower()
IS_PRODUCTION: bool = ENVIRONMENT == "production"

# ---------------------------------------------------------------------------
# Business constants
# ---------------------------------------------------------------------------
ABC_CLASS_A_TOP_N: int = 10_000    # Top 10K SKUs → Prophet
ABC_CLASS_B_TOP_N: int = 40_000    # Next 30K SKUs → LightGBM
# Remaining 160K → rolling average + de-list classifier

ROLLING_AVERAGE_WEEKS: int = 13    # C-class rolling window

# Open-Meteo endpoint
OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"

# Expected database tables
EXPECTED_TABLES: list[str] = [
    "sales_transactions",
    "inventory_snapshots",
    "purchase_orders",
    "sku_master",
    "weather_log",
    "forecast_results",
    "supplier_scores",
]

# Logging format
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

if __name__ == "__main__":
    # Quick sanity-check: print non-secret config values
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    log = logging.getLogger("config")
    log.info("SUPABASE_URL    : %s", SUPABASE_URL)
    log.info("WEATHER_LAT/LON : %s / %s", WEATHER_LAT, WEATHER_LON)
    log.info("LOG_LEVEL       : %s", LOG_LEVEL)
    log.info("ENVIRONMENT     : %s", ENVIRONMENT)
    log.info("Config loaded successfully.")
