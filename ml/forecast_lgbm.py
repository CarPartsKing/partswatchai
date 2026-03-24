"""
ml/forecast_lgbm.py — LightGBM gradient-boosting demand forecast for B-class SKUs.

Runs nightly after forecast_rolling.py.  Trains one LightGBM model per
(SKU, location) pair with at least MIN_TRAIN_DAYS days of sales history, then
generates FORECAST_HORIZON days of forward predictions.

FEATURES (12 total)
    Calendar:  day_of_week, week_of_year, month
    Lags:      lag_7, lag_14, lag_28       (demand N days before target)
    Rolling:   rolling_mean_7, rolling_mean_28
    Weather:   temp_min_f, freeze_thaw_cycle, snowfall_in, consecutive_freeze_days

LAG STRATEGY FOR FORECAST PERIOD
    Lag dates that fall within the known historical window use actual demand.
    Lag dates that fall inside the 30-day forecast horizon (possible only for
    lag_7 on days 8-30, lag_14 on days 15-30, lag_28 on days 29-30) fall back
    to rolling_mean_28 computed from the training data rather than propagating
    prediction error recursively.

UNCERTAINTY BOUNDS
    One model is trained on the first (n - VAL_WINDOW) days.  The last
    VAL_WINDOW days act as a hold-out.  RMSE on the hold-out is used to build
    ±1.28σ bounds (80 % prediction interval):
        lower_bound = max(0, predicted − 1.28 × val_rmse)
        upper_bound = predicted + 1.28 × val_rmse

EARLY STOPPING
    LightGBM's early stopping terminates training when validation RMSE has
    not improved for EARLY_STOPPING_ROUNDS consecutive iterations, preventing
    overfitting on small SKU histories.

WEATHER FALLBACK
    Open-Meteo provides 7-day forecasts stored in weather_log alongside historical
    actuals.  For forecast dates beyond the available weather window the model uses
    climate-mean values computed from all historical weather records (log_date ≤ today).

PERFORMANCE
    All data is fetched in three bulk queries (SKUs, transactions, weather).
    Everything else — grouping, feature engineering, training, prediction —
    runs in memory.  No per-SKU or per-location round-trips.  LightGBM
    releases Python's GIL during training, so a drop-in switch to
    concurrent.futures.ProcessPoolExecutor is the natural scaling path for
    the full 30 K B-class catalog.

COUNTS
    B-class SKU count and active (SKU, location) pairs are always queried
    live.  Nothing is hardcoded.

PREREQUISITE
    Migration 006 (location_id on forecast_results) must already be applied.

USAGE
    python -m ml.forecast_lgbm            # full run
    python -m ml.forecast_lgbm --dry-run  # compute, do not write
"""

import argparse
import ctypes
import math
import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# NixOS: pre-load libgomp.so.1 (GCC OpenMP runtime) before LightGBM imports.
# LightGBM's shared library links against libgomp dynamically; NixOS does not
# add the GCC lib directory to the system linker path automatically.
# If this path becomes stale after a system update, set the environment variable:
#     GOMP_LIB_DIR=/nix/store/<new-hash>-gcc-<version>-lib/lib
# ---------------------------------------------------------------------------
_GOMP_DIR = os.environ.get(
    "GOMP_LIB_DIR",
    "/nix/store/bmi5znnqk4kg2grkrhk6py0irc8phf6l-gcc-14.2.1.20250322-lib/lib",
)
try:
    # RTLD_GLOBAL makes libgomp's symbols visible to subsequently loaded shared
    # libraries (lib_lightgbm.so).  Without this flag the pre-load has no effect
    # because the linker resolves lib_lightgbm.so's dependencies independently.
    ctypes.CDLL(os.path.join(_GOMP_DIR, "libgomp.so.1"), mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass  # Not on NixOS, or path stale — LightGBM will fail below with a clear message

import lightgbm as lgb
import numpy as np

from utils.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum calendar days from first sale to yesterday required to train
MIN_TRAIN_DAYS: int = 60

# Most-recent days reserved for hold-out validation (early stopping + RMSE)
VAL_WINDOW: int = 14

# Days of forward predictions to generate per (SKU, location)
FORECAST_HORIZON: int = 30

# Prediction interval coverage (±1.28σ ≈ 80 %)
CONFIDENCE_PCT: float = 0.80
UNCERTAINTY_MULTIPLIER: float = 1.28   # σ multiplier for 80 % PI

# Model label in forecast_results.model_type
MODEL_TYPE: str = "lightgbm"

# Supabase pagination cap
_PAGE_SIZE: int = 1000

# Upsert batch size
BATCH_SIZE: int = 500

# LightGBM hyperparameters — tuned for speed on 60–730-row histories
LGBM_PARAMS: dict = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       15,
    "learning_rate":    0.10,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.80,
    "bagging_freq":     5,
    "min_child_samples": 5,
    "n_estimators":     500,    # hard ceiling; early stopping applies first
    "verbose":          -1,
}
EARLY_STOPPING_ROUNDS: int = 20

# Ordered feature names — must match _build_feature_row output exactly
FEATURE_NAMES: list[str] = [
    "day_of_week",
    "week_of_year",
    "month",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "temp_min_f",
    "freeze_thaw_cycle",
    "snowfall_in",
    "consecutive_freeze_days",
]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_all(client: Any, table: str, select: str = "*",
               filters: dict | None = None) -> list[dict]:
    """Paginate through a Supabase table and return all rows.

    Args:
        client:  Active Supabase client.
        table:   Table name.
        select:  PostgREST column selector.
        filters: Optional dict of {column: exact_value} equality filters.

    Returns:
        All matching rows as a list of dicts.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        q = client.table(table).select(select)
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)
        page = q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def _fetch_b_class_skus(client: Any) -> list[str]:
    """Return all current B-class sku_ids from sku_master (live count)."""
    rows = _fetch_all(client, "sku_master", "sku_id", filters={"abc_class": "B"})
    skus = [r["sku_id"] for r in rows if r.get("sku_id")]
    log.info("  B-class SKUs found in sku_master: %d", len(skus))
    return skus


def _fetch_transactions(client: Any, cutoff: str) -> list[dict]:
    """Bulk-fetch all non-anomaly transactions on or after cutoff."""
    rows: list[dict] = []
    offset = 0
    select = (
        "sku_id,location_id,transaction_date,"
        "qty_sold,lost_sales_imputation,is_stockout,is_anomaly"
    )
    while True:
        page = (
            client.table("sales_transactions")
            .select(select)
            .gte("transaction_date", cutoff)
            .eq("is_anomaly", False)
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def _fetch_weather(client: Any, cutoff: str) -> list[dict]:
    """Fetch all weather_log rows from cutoff onwards (historical + forecast)."""
    rows: list[dict] = []
    offset = 0
    select = (
        "log_date,temp_min_f,freeze_thaw_cycle,snowfall_in,"
        "consecutive_freeze_days"
    )
    while True:
        page = (
            client.table("weather_log")
            .select(select)
            .gte("log_date", cutoff)
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_demand_map(
    tx_rows: list[dict],
    b_class_set: set[str],
) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate transaction rows to a daily demand map per (SKU, location).

    Effective demand:
        - Normal sale    → qty_sold
        - Stockout sale  → lost_sales_imputation (or 0 if null)

    Only B-class SKUs are included.

    Returns:
        {(sku_id, location_id): {date_str: demand_float}}
    """
    demand: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for r in tx_rows:
        sku = r.get("sku_id", "")
        loc = r.get("location_id", "")
        if not sku or not loc or sku not in b_class_set:
            continue
        date_str = str(r.get("transaction_date", ""))[:10]
        if not date_str:
            continue
        is_stockout = r.get("is_stockout", False)
        if is_stockout:
            eff = float(r.get("lost_sales_imputation") or 0)
        else:
            eff = float(r.get("qty_sold") or 0)
        demand[(sku, loc)][date_str] += eff
    return {k: dict(v) for k, v in demand.items()}


def _build_weather_map(weather_rows: list[dict]) -> dict[str, dict]:
    """Build a date-keyed weather lookup dict.

    Returns:
        {date_str: {temp_min_f, freeze_thaw_cycle, snowfall_in, consecutive_freeze_days}}
    """
    wmap: dict[str, dict] = {}
    for r in weather_rows:
        d = str(r.get("log_date", ""))[:10]
        if d:
            wmap[d] = {
                "temp_min_f":               float(r.get("temp_min_f") or 32.0),
                "freeze_thaw_cycle":        float(bool(r.get("freeze_thaw_cycle", False))),
                "snowfall_in":              float(r.get("snowfall_in") or 0.0),
                "consecutive_freeze_days":  float(r.get("consecutive_freeze_days") or 0),
            }
    return wmap


def _compute_fallback_weather(weather_rows: list[dict]) -> dict:
    """Compute climate-mean weather values from historical rows.

    Used for forecast dates that extend beyond the available 7-day weather
    forecast window.  Historical rows are those with log_date <= today.
    """
    today_str = date.today().isoformat()
    hist = [r for r in weather_rows if str(r.get("log_date", ""))[:10] <= today_str]
    if not hist:
        return {"temp_min_f": 45.0, "freeze_thaw_cycle": 0.0,
                "snowfall_in": 0.0, "consecutive_freeze_days": 0.0}
    return {
        "temp_min_f": float(
            sum(float(r.get("temp_min_f") or 32) for r in hist) / len(hist)
        ),
        "freeze_thaw_cycle": float(
            sum(1 for r in hist if r.get("freeze_thaw_cycle")) / len(hist)
        ),
        "snowfall_in": float(
            sum(float(r.get("snowfall_in") or 0) for r in hist) / len(hist)
        ),
        "consecutive_freeze_days": float(
            sum(float(r.get("consecutive_freeze_days") or 0) for r in hist) / len(hist)
        ),
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_feature_row(
    target: date,
    demand_by_date: dict[str, float],
    weather_by_date: dict[str, dict],
    fallback_weather: dict,
    fallback_demand: float,
    today: date,
) -> list[float]:
    """Compute the 12 feature values for one (SKU, location, date) observation.

    For lag dates that fall inside the forecast horizon (i.e. after today),
    fallback_demand (rolling_mean_28 of the training series) is used instead
    of an actual or recursive-prediction value.  This avoids error propagation
    while still capturing the SKU's typical demand level.

    Args:
        target:          The date being featurised.
        demand_by_date:  Historical demand dict {date_str: demand}.
        weather_by_date: Weather lookup dict {date_str: weather_dict}.
        fallback_weather: Climate-mean values for out-of-window dates.
        fallback_demand: Substitute demand for lag dates in the future.
        today:           The run date (first day of the forecast horizon).

    Returns:
        List of 12 floats in FEATURE_NAMES order.
    """
    def get_demand(d: date) -> float:
        if d >= today:
            return fallback_demand
        return demand_by_date.get(d.isoformat(), 0.0)

    # Calendar features
    dow = float(target.weekday())
    woy = float(target.isocalendar()[1])
    mon = float(target.month)

    # Lag features
    lag_7  = get_demand(target - timedelta(days=7))
    lag_14 = get_demand(target - timedelta(days=14))
    lag_28 = get_demand(target - timedelta(days=28))

    # Rolling means (look-back only; uses get_demand for each day)
    w7  = [get_demand(target - timedelta(days=k)) for k in range(1, 8)]
    w28 = [get_demand(target - timedelta(days=k)) for k in range(1, 29)]
    rolling_mean_7  = sum(w7)  / len(w7)
    rolling_mean_28 = sum(w28) / len(w28)

    # Weather features
    w = weather_by_date.get(target.isoformat(), fallback_weather)
    temp  = w.get("temp_min_f", fallback_weather["temp_min_f"])
    freeze = w.get("freeze_thaw_cycle", fallback_weather["freeze_thaw_cycle"])
    snow   = w.get("snowfall_in", fallback_weather["snowfall_in"])
    cfd    = w.get("consecutive_freeze_days", fallback_weather["consecutive_freeze_days"])

    return [
        dow, woy, mon,
        lag_7, lag_14, lag_28,
        rolling_mean_7, rolling_mean_28,
        float(temp), float(freeze), float(snow), float(cfd),
    ]


def _build_matrices(
    history_dates: list[date],
    demand_by_date: dict[str, float],
    weather_by_date: dict[str, dict],
    fallback_weather: dict,
    fallback_demand: float,
    today: date,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and target vector y for a list of history dates.

    Args:
        history_dates:  Sorted list of calendar dates to featurise.
        demand_by_date: Historical demand dict.
        weather_by_date: Weather lookup.
        fallback_weather: Out-of-window weather fallback.
        fallback_demand: Fallback demand for future lag lookups.
        today:          Run date.

    Returns:
        (X, y) where X has shape (n, 12) and y has shape (n,).
    """
    rows = []
    targets = []
    for d in history_dates:
        rows.append(
            _build_feature_row(
                d, demand_by_date, weather_by_date,
                fallback_weather, fallback_demand, today,
            )
        )
        targets.append(demand_by_date.get(d.isoformat(), 0.0))
    return np.array(rows, dtype=float), np.array(targets, dtype=float)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[lgb.Booster, float]:
    """Fit LightGBM with early stopping and return (model, val_rmse).

    Args:
        X_train, y_train: Training feature matrix and targets.
        X_val, y_val:     Validation feature matrix and targets.

    Returns:
        Fitted LightGBM Booster and RMSE on the validation set.
    """
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain,
                          feature_name=FEATURE_NAMES)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=-1),   # silence per-round output
    ]

    params = {k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"}
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[dval],
        callbacks=callbacks,
    )

    val_preds = model.predict(X_val)
    val_rmse  = float(math.sqrt(
        sum((p - a) ** 2 for p, a in zip(val_preds, y_val)) / len(y_val)
    ))
    return model, val_rmse


# ---------------------------------------------------------------------------
# Forecast generation
# ---------------------------------------------------------------------------

def _generate_forecast(
    sku_id: str,
    location_id: str,
    model: lgb.Booster,
    demand_by_date: dict[str, float],
    weather_by_date: dict[str, dict],
    fallback_weather: dict,
    fallback_demand: float,
    val_rmse: float,
    today: date,
    run_date_str: str,
) -> list[dict]:
    """Generate FORECAST_HORIZON rows of predictions with uncertainty bounds.

    Prediction interval: predicted ± UNCERTAINTY_MULTIPLIER × val_rmse
    Lower bound is floored at 0 (demand cannot be negative).

    Args:
        sku_id, location_id: Identifiers.
        model:               Fitted LightGBM booster.
        demand_by_date:      Historical demand dict for lag lookups.
        weather_by_date:     Weather lookup.
        fallback_weather:    Out-of-window weather fallback.
        fallback_demand:     Fallback for future lag lookups.
        val_rmse:            Validation RMSE used as uncertainty measure.
        today:               First forecast date.
        run_date_str:        ISO run date string.

    Returns:
        List of FORECAST_HORIZON forecast row dicts.
    """
    forecast_dates = [today + timedelta(days=k) for k in range(FORECAST_HORIZON)]
    feat_rows = [
        _build_feature_row(
            d, demand_by_date, weather_by_date,
            fallback_weather, fallback_demand, today,
        )
        for d in forecast_dates
    ]
    X_future = np.array(feat_rows, dtype=float)
    preds    = model.predict(X_future)

    half_width = UNCERTAINTY_MULTIPLIER * val_rmse

    result = []
    for d, pred in zip(forecast_dates, preds):
        predicted = round(float(pred), 4)
        result.append({
            "sku_id":        sku_id,
            "location_id":   location_id,
            "forecast_date": d.isoformat(),
            "model_type":    MODEL_TYPE,
            "predicted_qty": predicted,
            "lower_bound":   round(max(0.0, predicted - half_width), 4),
            "upper_bound":   round(predicted + half_width, 4),
            "confidence_pct": CONFIDENCE_PCT,
            "run_date":      run_date_str,
        })
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_forecast(dry_run: bool = False) -> int:
    """Run the full LightGBM forecast pipeline.

    Args:
        dry_run: When True, trains models and logs results without writing
                 any rows to forecast_results.

    Returns:
        Exit code 0 on success, 1 on unrecoverable error.
    """
    from db.connection import get_client
    client       = get_client()
    t_start      = time.perf_counter()
    today        = date.today()
    run_date_str = today.isoformat()
    # Fetch 2 years of history — gives enough lag/rolling signal and keeps
    # training sets small for per-SKU models (≤ 730 rows)
    tx_cutoff = (today - timedelta(days=730)).isoformat()

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")

    log.info("Training data window: %s → %s", tx_cutoff, (today - timedelta(days=1)).isoformat())
    log.info("Forecast horizon:     %d days  (%s → %s)", FORECAST_HORIZON,
             today.isoformat(), (today + timedelta(days=FORECAST_HORIZON - 1)).isoformat())
    log.info("-" * 60)

    # ── 1. Fetch B-class SKUs (live count — never hardcoded) ──────────────
    log.info("Fetching B-class SKUs from sku_master …")
    b_skus = _fetch_b_class_skus(client)
    if not b_skus:
        log.warning("No B-class SKUs found — nothing to forecast.")
        return 0
    b_class_set = set(b_skus)

    # ── 2. Bulk-fetch transactions ─────────────────────────────────────────
    log.info("Fetching non-anomaly transactions since %s …", tx_cutoff)
    t0 = time.perf_counter()
    tx_rows = _fetch_transactions(client, tx_cutoff)
    log.info("  Fetched %d transaction(s) in %.2fs.", len(tx_rows), time.perf_counter() - t0)

    # ── 3. Bulk-fetch weather ─────────────────────────────────────────────
    # Fetch historical + forecast weather; the forecast rows cover future dates
    log.info("Fetching weather_log …")
    weather_rows = _fetch_weather(client, tx_cutoff)
    log.info("  Fetched %d weather row(s).", len(weather_rows))

    # ── 4. Build in-memory data structures ───────────────────────────────
    demand_map      = _build_demand_map(tx_rows, b_class_set)
    weather_by_date = _build_weather_map(weather_rows)
    fallback_weather = _compute_fallback_weather(weather_rows)

    # SKUs with any sales in the window
    active_pairs = sorted(demand_map.keys())
    skus_with_sales = {sku for sku, _ in active_pairs}
    skus_no_sales   = len(b_class_set) - len(skus_with_sales)

    log.info(
        "  Active (SKU, location) pairs: %d  (across %d of %d B-class SKUs)",
        len(active_pairs), len(skus_with_sales), len(b_skus),
    )
    if skus_no_sales:
        log.info("  %d B-class SKU(s) had no sales in the 2-year window.", skus_no_sales)
    log.info("-" * 60)

    # ── 5. Train one model per (SKU, location) ────────────────────────────
    processed   = 0
    skipped     = 0
    total_rows  = 0
    total_iters = 0
    fc_buffer: list[dict] = []

    for sku_id, location_id in active_pairs:
        demand_by_date = demand_map[(sku_id, location_id)]

        # Determine calendar span from first sale to yesterday
        all_dates_str = sorted(demand_by_date.keys())
        first_dt = date.fromisoformat(all_dates_str[0])
        last_dt  = date.fromisoformat(all_dates_str[-1])
        span_days = (last_dt - first_dt).days + 1

        if span_days < MIN_TRAIN_DAYS:
            log.info(
                "  SKIP  %-12s  %-10s  %d calendar day span (need >= %d)",
                sku_id, location_id, span_days, MIN_TRAIN_DAYS,
            )
            skipped += 1
            continue

        # Build full calendar range (first_dt … yesterday), filling zeros
        history_dates = [
            first_dt + timedelta(days=k)
            for k in range((min(last_dt, today - timedelta(days=1)) - first_dt).days + 1)
        ]

        if len(history_dates) < MIN_TRAIN_DAYS + VAL_WINDOW:
            log.info(
                "  SKIP  %-12s  %-10s  only %d usable history days",
                sku_id, location_id, len(history_dates),
            )
            skipped += 1
            continue

        # Compute fallback_demand = rolling_mean_28 of the most recent training data
        recent_demand = [
            demand_by_date.get(d.isoformat(), 0.0)
            for d in history_dates[-28:]
        ]
        fallback_demand = sum(recent_demand) / len(recent_demand)

        # Train / validation split: keep last VAL_WINDOW days for hold-out
        train_dates = history_dates[:-VAL_WINDOW]
        val_dates   = history_dates[-VAL_WINDOW:]

        X_train, y_train = _build_matrices(
            train_dates, demand_by_date, weather_by_date,
            fallback_weather, fallback_demand, today,
        )
        X_val, y_val = _build_matrices(
            val_dates, demand_by_date, weather_by_date,
            fallback_weather, fallback_demand, today,
        )

        try:
            t_model = time.perf_counter()
            model, val_rmse = _train(X_train, y_train, X_val, y_val)
            train_ms = (time.perf_counter() - t_model) * 1000
            n_iters = model.num_trees()
        except Exception as exc:
            log.error(
                "  ERROR %-12s  %-10s  training failed: %s",
                sku_id, location_id, exc,
            )
            skipped += 1
            continue

        fc_rows = _generate_forecast(
            sku_id, location_id, model,
            demand_by_date, weather_by_date, fallback_weather,
            fallback_demand, val_rmse, today, run_date_str,
        )

        mean_pred = sum(r["predicted_qty"] for r in fc_rows) / len(fc_rows)
        log.info(
            "  OK    %-12s  %-10s  span=%3dd  train=%3d  val=%2d  "
            "iters=%3d  rmse=%.2f  pred_avg=%.2f  (%.0fms)",
            sku_id, location_id, span_days,
            len(train_dates), len(val_dates),
            n_iters, val_rmse, mean_pred, train_ms,
        )

        fc_buffer.extend(fc_rows)
        processed   += 1
        total_rows  += len(fc_rows)
        total_iters += n_iters

        # Flush buffer in BATCH_SIZE chunks
        while len(fc_buffer) >= BATCH_SIZE:
            batch = fc_buffer[:BATCH_SIZE]
            fc_buffer = fc_buffer[BATCH_SIZE:]
            if not dry_run:
                client.table("forecast_results").upsert(
                    batch,
                    on_conflict="sku_id,location_id,forecast_date,model_type,run_date",
                ).execute()

    # Final flush
    if fc_buffer:
        if not dry_run:
            client.table("forecast_results").upsert(
                fc_buffer,
                on_conflict="sku_id,location_id,forecast_date,model_type,run_date",
            ).execute()

    # ── 6. Summary ────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    avg_iters = round(total_iters / processed, 1) if processed else 0

    log.info("=" * 60)
    log.info("LightGBM forecast complete  (%.2fs)", elapsed)
    log.info("  B-class SKUs queried:              %d", len(b_skus))
    log.info("  B-class SKUs with no sales:        %d", skus_no_sales)
    log.info("  (SKU, location) pairs processed:   %d", processed)
    log.info("  (SKU, location) pairs skipped:     %d", skipped)
    log.info("  Avg boosting iterations (stopped): %.1f", avg_iters)
    log.info("  Forecast rows written:             %d", total_rows)
    if dry_run:
        log.info("  (DRY RUN — no writes were made)")
    log.info("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI arguments and run the LightGBM forecast pipeline."""
    parser = argparse.ArgumentParser(
        description="partswatch-ai: LightGBM demand forecast for B-class SKUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prerequisite: migration 006 (location_id on forecast_results)\n"
            "must already be applied.\n\n"
            "Nightly pipeline order:\n"
            "  extract/partswatch_pull.py  →  transform/clean.py\n"
            "  →  transform/derive.py  →  ml/anomaly.py\n"
            "  →  ml/forecast_rolling.py  →  ml/forecast_lgbm.py\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Train and compute forecasts without writing to the database.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("partswatch-ai — ml.forecast_lgbm")
    log.info(
        "  min_train=%dd  val_window=%dd  horizon=%dd  "
        "lr=%.2f  leaves=%d  early_stop=%d",
        MIN_TRAIN_DAYS, VAL_WINDOW, FORECAST_HORIZON,
        LGBM_PARAMS["learning_rate"], LGBM_PARAMS["num_leaves"],
        EARLY_STOPPING_ROUNDS,
    )
    log.info("=" * 60)

    return run_forecast(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
