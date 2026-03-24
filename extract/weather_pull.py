"""
extract/weather_pull.py — Pull Open-Meteo weather data into Supabase.

WHAT THIS SCRIPT DOES
---------------------
1. Fetches up to 3 years of historical daily weather for Strongsville, OH
   from the Open-Meteo Archive API (free, no key required).
2. Fetches the 14-day forecast from the Open-Meteo Forecast API.
3. Calculates two derived fields on the full sorted date series:
     - consecutive_freeze_days: running count of days where temp_min_f < 32.
       Resets to 0 the first day temp_min_f >= 32.
     - freeze_thaw_cycle: TRUE on any day where the temperature range spans
       32°F (temp_min_f < 32 AND temp_max_f >= 32), meaning the thermometer
       crossed freezing within the same 24-hour period.  Strong pothole-season
       demand signal for suspension / steering / wheel-end parts.
4. Upserts all rows into weather_log in batches (on conflict: log_date).
   Historical rows have is_forecast=FALSE; forecast rows have is_forecast=TRUE.

PREREQUISITES
-------------
Run db/migrations/002_weather_log_is_forecast.sql in the Supabase SQL Editor
before executing this script for the first time.

USAGE
-----
    # Full historical backfill + 14-day forecast (run once on setup)
    python -m extract.weather_pull

    # Incremental update only — yesterday + 14-day forecast (run nightly)
    python -m extract.weather_pull --mode incremental
"""

import argparse
import sys
from datetime import date, timedelta
from typing import Any

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from utils.logging_config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Open-Meteo returns temperatures in the unit we request; precipitation too.
# Snowfall is returned in cm regardless of precipitation_unit, so we convert.
_CM_TO_IN: float = 0.393701

# Batch size for Supabase upserts — keeps payload size well under PostgREST limits
_UPSERT_BATCH_SIZE: int = 500

# Daily variables to request from both archive and forecast endpoints
_DAILY_VARS: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
]


# ---------------------------------------------------------------------------
# API fetch helpers
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    before_sleep=before_sleep_log(log, __import__("logging").WARNING),
    reraise=True,
)
def _get(url: str, params: dict) -> dict:
    """Execute a GET request with retry on transient network failures.

    Args:
        url:    Full endpoint URL.
        params: Query-string parameters.

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.HTTPError: On non-2xx response after all retries exhausted.
        requests.Timeout:   If the server does not respond within 30 seconds.
    """
    log.debug("GET %s params=%s", url, params)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_historical(start_date: date, end_date: date) -> list[dict]:
    """Fetch daily archive data from Open-Meteo for a date range.

    Args:
        start_date: First day to retrieve (inclusive).
        end_date:   Last day to retrieve (inclusive).

    Returns:
        List of raw row dicts with keys:
            log_date, temp_max_f, temp_min_f,
            precipitation_in, snowfall_in, is_forecast.
    """
    from config import (
        OPEN_METEO_ARCHIVE_URL,
        WEATHER_LAT,
        WEATHER_LON,
        WEATHER_TIMEZONE,
    )

    log.info(
        "Fetching historical weather %s → %s …",
        start_date.isoformat(),
        end_date.isoformat(),
    )

    data = _get(
        OPEN_METEO_ARCHIVE_URL,
        {
            "latitude":          WEATHER_LAT,
            "longitude":         WEATHER_LON,
            "start_date":        start_date.isoformat(),
            "end_date":          end_date.isoformat(),
            "daily":             ",".join(_DAILY_VARS),
            "temperature_unit":  "fahrenheit",
            "precipitation_unit": "inch",
            "timezone":          WEATHER_TIMEZONE,
        },
    )

    rows = _parse_daily_response(data, is_forecast=False)
    log.info("Fetched %d historical rows.", len(rows))
    return rows


def fetch_forecast() -> list[dict]:
    """Fetch the 14-day daily forecast from Open-Meteo.

    Returns:
        List of raw row dicts with is_forecast=True.
    """
    from config import (
        OPEN_METEO_FORECAST_URL,
        WEATHER_LAT,
        WEATHER_LON,
        WEATHER_TIMEZONE,
        WEATHER_FORECAST_DAYS,
    )

    log.info("Fetching %d-day forecast …", WEATHER_FORECAST_DAYS)

    data = _get(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude":           WEATHER_LAT,
            "longitude":          WEATHER_LON,
            "daily":              ",".join(_DAILY_VARS),
            "temperature_unit":   "fahrenheit",
            "precipitation_unit": "inch",
            "forecast_days":      WEATHER_FORECAST_DAYS,
            "timezone":           WEATHER_TIMEZONE,
        },
    )

    rows = _parse_daily_response(data, is_forecast=True)
    log.info("Fetched %d forecast rows.", len(rows))
    return rows


def _parse_daily_response(data: dict, is_forecast: bool) -> list[dict]:
    """Convert an Open-Meteo daily response dict into flat row dicts.

    Snowfall from Open-Meteo is always in cm even when precipitation_unit=inch,
    so we convert cm → inches here.

    Args:
        data:        Raw JSON dict from the Open-Meteo API.
        is_forecast: Whether these rows should be marked as forecast data.

    Returns:
        List of dicts ready for derived-field calculation and DB upsert.

    Raises:
        KeyError: If the expected 'daily' key is absent from the response.
        ValueError: If the date list and variable lists have mismatched lengths.
    """
    try:
        daily = data["daily"]
    except KeyError as exc:
        raise KeyError(
            f"Unexpected Open-Meteo response structure — 'daily' key missing. "
            f"Got keys: {list(data.keys())}"
        ) from exc

    dates        = daily["time"]
    temp_max     = daily["temperature_2m_max"]
    temp_min     = daily["temperature_2m_min"]
    precip       = daily["precipitation_sum"]
    snowfall_cm  = daily["snowfall_sum"]

    if not (len(dates) == len(temp_max) == len(temp_min) == len(precip) == len(snowfall_cm)):
        raise ValueError(
            "Open-Meteo returned arrays of mismatched lengths — cannot parse response."
        )

    rows = []
    for i, day_str in enumerate(dates):
        rows.append(
            {
                "log_date":        day_str,          # "YYYY-MM-DD" string
                "temp_max_f":      _safe_float(temp_max[i]),
                "temp_min_f":      _safe_float(temp_min[i]),
                "precipitation_in": _safe_float(precip[i]),
                # Open-Meteo snowfall is cm; convert to inches
                "snowfall_in":     _safe_float(
                    snowfall_cm[i] * _CM_TO_IN if snowfall_cm[i] is not None else None
                ),
                "is_forecast":     is_forecast,
                # Derived fields populated later by calculate_derived_fields()
                "consecutive_freeze_days": 0,
                "freeze_thaw_cycle":       False,
            }
        )
    return rows


def _safe_float(value: Any) -> float | None:
    """Return a rounded float or None if the value is missing.

    Open-Meteo uses None for missing observations (e.g. snowfall on a rain day).

    Args:
        value: Raw value from the API (float, int, or None).

    Returns:
        Float rounded to 4 decimal places, or None.
    """
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Derived field calculation
# ---------------------------------------------------------------------------

def calculate_derived_fields(rows: list[dict]) -> list[dict]:
    """Compute consecutive_freeze_days and freeze_thaw_cycle for every row.

    Rows MUST be sorted ascending by log_date before calling this function.
    Both fields are computed in a single O(n) pass over the sorted list.

    consecutive_freeze_days logic:
        - Increments by 1 each day where temp_min_f < 32°F.
        - Resets to 0 the first day temp_min_f >= 32°F.

    freeze_thaw_cycle logic:
        - TRUE on any day where the daily temperature range spans the freezing
          point: temp_min_f < 32 AND temp_max_f >= 32.
        - Captures within-day freeze/thaw transitions — the strongest demand
          signal for potholes (suspension, steering, wheel-end SKUs).

    Args:
        rows: List of row dicts sorted by log_date ascending.

    Returns:
        The same list with consecutive_freeze_days and freeze_thaw_cycle
        populated in-place.
    """
    streak = 0  # running count of consecutive freeze days

    for row in rows:
        t_min = row.get("temp_min_f")
        t_max = row.get("temp_max_f")

        # ---- consecutive_freeze_days ----------------------------------------
        if t_min is not None and t_min < 32.0:
            streak += 1
        else:
            streak = 0
        row["consecutive_freeze_days"] = streak

        # ---- freeze_thaw_cycle ----------------------------------------------
        # Day's temperature range spans 32°F → thermometer crossed freezing
        if t_min is not None and t_max is not None:
            row["freeze_thaw_cycle"] = (t_min < 32.0) and (t_max >= 32.0)
        else:
            row["freeze_thaw_cycle"] = False

    return rows


# ---------------------------------------------------------------------------
# Supabase upsert
# ---------------------------------------------------------------------------

def upsert_weather_rows(rows: list[dict]) -> int:
    """Upsert weather rows into the weather_log table in batches.

    Uses on_conflict='log_date' so re-running the script safely overwrites
    stale forecast rows when they later become historical observations.

    Args:
        rows: List of fully-populated row dicts (after derive fields).

    Returns:
        Total number of rows upserted.

    Raises:
        Exception: Propagates any Supabase client error after logging it.
    """
    from db.connection import get_client

    client   = get_client()
    total    = len(rows)
    upserted = 0

    log.info("Upserting %d rows into weather_log …", total)

    for batch_start in range(0, total, _UPSERT_BATCH_SIZE):
        batch = rows[batch_start : batch_start + _UPSERT_BATCH_SIZE]
        try:
            client.table("weather_log").upsert(
                batch, on_conflict="log_date"
            ).execute()
            upserted += len(batch)
            log.info(
                "  Upserted rows %d–%d of %d.",
                batch_start + 1,
                batch_start + len(batch),
                total,
            )
        except Exception as exc:
            log.error(
                "Upsert failed on batch starting at row %d: %s",
                batch_start,
                exc,
                exc_info=True,
            )
            raise

    log.info("Upsert complete — %d/%d rows written.", upserted, total)
    return upserted


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_backfill() -> int:
    """Pull 3 years of history + 14-day forecast and load into weather_log.

    Fetches the archive in a single API call, then appends the forecast.
    Deduplicates on log_date so overlapping dates (today appears in both)
    are handled at the upsert layer.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    from config import WEATHER_HISTORY_YEARS

    try:
        today     = date.today()
        # Open-Meteo archive typically lags by ~5 days; use yesterday to be safe
        hist_end  = today - timedelta(days=1)
        hist_start = date(today.year - WEATHER_HISTORY_YEARS, today.month, today.day)

        # --- Fetch ---
        historical_rows = fetch_historical(hist_start, hist_end)
        forecast_rows   = fetch_forecast()

        # --- Merge and sort by date ascending ---
        # Forecast rows for dates already in history will be overwritten by
        # the upsert's on_conflict clause, so order matters less than correctness.
        all_rows = historical_rows + forecast_rows
        all_rows.sort(key=lambda r: r["log_date"])

        # --- Remove true duplicates (same log_date appearing twice) ---
        # Keep the historical row when a date appears in both sets, since a
        # measured observation is more accurate than a forecast.
        seen: dict[str, dict] = {}
        for row in all_rows:
            d = row["log_date"]
            # Prefer historical (is_forecast=False) over forecast rows
            if d not in seen or (seen[d]["is_forecast"] and not row["is_forecast"]):
                seen[d] = row
        deduped = sorted(seen.values(), key=lambda r: r["log_date"])

        log.info(
            "Total unique dates after dedup: %d (%d historical, %d forecast-only).",
            len(deduped),
            sum(1 for r in deduped if not r["is_forecast"]),
            sum(1 for r in deduped if r["is_forecast"]),
        )

        # --- Derived fields (requires sorted order) ---
        deduped = calculate_derived_fields(deduped)

        # --- Upsert ---
        upsert_weather_rows(deduped)

        # --- Summary ---
        freeze_days  = sum(1 for r in deduped if r["consecutive_freeze_days"] > 0)
        thaw_events  = sum(1 for r in deduped if r["freeze_thaw_cycle"])
        log.info("Summary: %d freeze days, %d freeze-thaw events in loaded range.", freeze_days, thaw_events)

        return 0

    except Exception as exc:
        log.error("Full backfill failed: %s", exc, exc_info=True)
        return 1


def run_incremental() -> int:
    """Pull yesterday's observation and the fresh 14-day forecast only.

    Designed for nightly cron execution after the initial backfill.
    Recalculates consecutive_freeze_days by reading recent rows from the DB
    to establish the running streak before yesterday's date.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    try:
        from db.connection import get_client

        today     = date.today()
        yesterday = today - timedelta(days=1)

        # --- Fetch yesterday's observation ---
        log.info("Incremental mode — fetching %s + 14-day forecast.", yesterday.isoformat())
        new_rows = fetch_historical(yesterday, yesterday) + fetch_forecast()

        # --- Read the last 40 days from DB to correctly seed the freeze streak ---
        # 40 days is more than enough to cover any realistic consecutive-freeze run.
        client      = get_client()
        lookback    = yesterday - timedelta(days=40)
        result      = (
            client.table("weather_log")
            .select("log_date,temp_min_f,temp_max_f,consecutive_freeze_days")
            .gte("log_date", lookback.isoformat())
            .lt("log_date", yesterday.isoformat())
            .order("log_date", desc=False)
            .execute()
        )
        prior_rows: list[dict] = result.data or []
        log.info("Loaded %d prior rows to seed freeze streak.", len(prior_rows))

        # Seed the streak from the most recent prior row
        seed_streak = 0
        if prior_rows:
            seed_streak = prior_rows[-1].get("consecutive_freeze_days", 0) or 0

        # Merge prior rows + new rows, sort, derive — then keep only new_rows to upsert
        combined = prior_rows + new_rows
        combined.sort(key=lambda r: r["log_date"])

        # We need to set initial streak state before processing combined rows.
        # Walk through prior rows first to reach the seeded streak, then continue
        # with new rows — but calculate_derived_fields handles the full pass.
        # Reset streak to 0 and let the function recompute from the combined set.
        derived_all = calculate_derived_fields(combined)

        # Keep only the newly fetched dates for upsert
        new_dates = {r["log_date"] for r in new_rows}
        to_upsert = [r for r in derived_all if r["log_date"] in new_dates]

        upsert_weather_rows(to_upsert)
        return 0

    except Exception as exc:
        log.error("Incremental update failed: %s", exc, exc_info=True)
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI arguments and dispatch to the appropriate run mode.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Pull Open-Meteo weather data into Supabase weather_log."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help=(
            "full: 3-year backfill + 14-day forecast (default, run once on setup). "
            "incremental: yesterday + 14-day forecast (run nightly)."
        ),
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("partswatch-ai — weather_pull  mode=%s", args.mode)
    log.info("=" * 60)

    if args.mode == "incremental":
        return run_incremental()
    else:
        return run_full_backfill()


if __name__ == "__main__":
    sys.exit(main())
