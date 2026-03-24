-- =============================================================================
-- 002_weather_log_is_forecast.sql
--
-- Adds is_forecast flag to weather_log so historical rows and 14-day
-- Open-Meteo forecast rows can be distinguished.
--
-- Run in Supabase SQL Editor before executing extract/weather_pull.py
-- =============================================================================

ALTER TABLE weather_log
    ADD COLUMN IF NOT EXISTS is_forecast BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_weather_is_forecast
    ON weather_log (is_forecast)
    WHERE is_forecast = TRUE;

COMMENT ON COLUMN weather_log.is_forecast IS
    'TRUE = row came from Open-Meteo 14-day forecast; FALSE = historical archive.';
