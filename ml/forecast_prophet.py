"""ml/forecast_prophet.py — Prophet + XGBoost ensemble demand forecast for A-class SKUs.

Designed to run on a LOCAL LAPTOP (Windows/Mac), not on Replit.
Replit does not have enough compute for 28K+ Prophet models.

PIPELINE POSITION
-----------------
Runs weekly (typically Sunday night on a local machine).  Consumes:
  - sku_master           (A-class SKU list, dynamic count)
  - sales_transactions   (history, excluding anomalies + residual demand)
  - weather_log          (regressors for Prophet + XGBoost)
  - locations            (tier for blending + XGBoost feature)

Writes to:
  - forecast_results     (model_type = 'prophet')

ENSEMBLE
--------
  Final forecast = PROPHET_WEIGHT × Prophet yhat + XGBOOST_WEIGHT × XGBoost pred
  Bounds come from Prophet's yhat_lower / yhat_upper (95% interval).

MODEL CACHING
-------------
  Trained models are saved to MODEL_CACHE_DIR/{prophet,xgboost}/
  On incremental reruns, only SKUs with new sales data since last
  training are retrained.  Forecast-only mode skips training entirely.

USAGE
-----
  python ml/forecast_prophet.py --sku XT12QULV          # single SKU test
  python ml/forecast_prophet.py --mode full              # train all from scratch
  python ml/forecast_prophet.py --mode incremental       # only retrain stale
  python ml/forecast_prophet.py --mode forecast-only     # cached models, new forecasts
  python ml/forecast_prophet.py --mode full --dry-run    # no DB writes
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

PROPHET_WEIGHT: float = 0.60
XGBOOST_WEIGHT: float = 0.40
MIN_HISTORY_DAYS: int = 90
BATCH_SIZE: int = 50
CHANGEPOINT_PRIOR: float = 0.05
FORECAST_HORIZON_DAYS: int = 30
MODEL_CACHE_DIR: str = "models"
CONFIDENCE_PCT: float = 0.95
MODEL_TYPE: str = "prophet"

_PAGE_SIZE: int = 1_000
_BATCH_WRITE: int = 500
_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 5.0

XYZ_MAP: dict[str, float] = {"X": 0.0, "Y": 1.0, "Z": 2.0}
TIER3_BLEND_WEIGHT: float = 0.40
LOW_QUALITY_THRESHOLD: float = 0.50

XGBOOST_FEATURE_NAMES: list[str] = [
    "day_of_week", "month", "week_of_year",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "temp_min_f", "snowfall_in",
    "consecutive_freeze_days", "freeze_thaw_cycle",
    "abc_class_enc", "location_tier_enc",
]


def _get_fresh_client() -> Any:
    return get_client()


def _upsert_with_retry(client_holder: list, table: str, rows: list, on_conflict: str):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client_holder[0].table(table).upsert(
                rows, on_conflict=on_conflict,
            ).execute()
            return
        except Exception as exc:
            err_name = type(exc).__name__
            err_str = str(exc)
            is_conn_err = any(k in err_name + err_str for k in (
                "ConnectionTerminated", "ConnectionError", "RemoteProtocolError",
                "ReadTimeout", "ConnectTimeout", "PoolTimeout",
            ))
            if is_conn_err and attempt < _MAX_RETRIES:
                log.warning(
                    "  DB connection error (attempt %d/%d): %s — reconnecting …",
                    attempt, _MAX_RETRIES, err_name,
                )
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _get_fresh_client()
                continue
            raise


def _fetch_all(client: Any, table: str, select: str = "*",
               filters: dict | None = None) -> list[dict]:
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


def _fetch_chunked_by_date(
    client: Any, table: str, select: str, date_col: str,
    since: str, until: str | None = None, chunk_days: int = 7,
    extra_eq: dict[str, Any] | None = None,
) -> list[dict]:
    from datetime import date as _date, timedelta as _td
    start = _date.fromisoformat(since)
    end = _date.fromisoformat(until) if until else _date.today()
    all_rows: list[dict] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + _td(days=chunk_days - 1), end)
        offset = 0
        while True:
            q = client.table(table).select(select)
            q = q.gte(date_col, chunk_start.isoformat())
            q = q.lte(date_col, chunk_end.isoformat())
            if extra_eq:
                for col, val in extra_eq.items():
                    q = q.eq(col, val)
            page = q.range(offset, offset + _PAGE_SIZE - 1).execute().data or []
            all_rows.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        chunk_start = chunk_end + _td(days=1)
        if all_rows and len(all_rows) % 100_000 < (chunk_days * 7000):
            log.info("    … streamed %d rows so far from %s", len(all_rows), table)
    log.info("    … streamed %d total rows from %s", len(all_rows), table)
    return all_rows


def _fetch_a_class_skus(client: Any) -> list[dict]:
    rows = _fetch_all(client, "sku_master", "sku_id,abc_class,abc_xyz_class",
                      filters={"abc_class": "A"})
    active = [r for r in rows if r.get("sku_id")]
    log.info("  A-class SKUs found in sku_master: %d", len(active))
    return active


def _fetch_transactions(client: Any, cutoff: str) -> list[dict]:
    select = (
        "sku_id,location_id,transaction_date,"
        "qty_sold,lost_sales_imputation,is_stockout,is_anomaly"
    )
    return _fetch_chunked_by_date(
        client, "sales_transactions", select,
        date_col="transaction_date", since=cutoff,
        extra_eq={"is_anomaly": False, "is_residual_demand": False},
    )


def _fetch_weather(client: Any, cutoff: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    select = (
        "log_date,temp_min_f,freeze_thaw_cycle,snowfall_in,"
        "consecutive_freeze_days"
    )
    while True:
        page = (
            client.table("weather_log").select(select)
            .gte("log_date", cutoff)
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute().data or []
        )
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def _fetch_location_tiers(client: Any) -> dict[str, int]:
    rows = _fetch_all(client, "locations", "location_id,location_tier")
    return {r["location_id"]: int(r.get("location_tier") or 2) for r in rows}


def _fetch_demand_quality(client: Any) -> dict[tuple[str, str], float]:
    rows = _fetch_all(client, "sku_location_demand_quality",
                      "sku_id,location_id,demand_quality_score")
    return {
        (r["sku_id"], r["location_id"]): float(r.get("demand_quality_score") or 1.0)
        for r in rows
    }


def _build_demand_map(
    tx_rows: list[dict], a_class_set: set[str],
) -> dict[tuple[str, str], dict[str, float]]:
    demand: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for r in tx_rows:
        sku = r.get("sku_id", "")
        loc = r.get("location_id", "")
        if not sku or not loc or sku not in a_class_set:
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
    wmap: dict[str, dict] = {}
    for r in weather_rows:
        d = str(r.get("log_date", ""))[:10]
        if d:
            wmap[d] = {
                "temp_min_f":              float(r.get("temp_min_f") or 32.0),
                "freeze_thaw_cycle":       float(bool(r.get("freeze_thaw_cycle", False))),
                "snowfall_in":             float(r.get("snowfall_in") or 0.0),
                "consecutive_freeze_days": float(r.get("consecutive_freeze_days") or 0),
            }
    return wmap


def _compute_fallback_weather(weather_rows: list[dict]) -> dict:
    today_str = date.today().isoformat()
    hist = [r for r in weather_rows if str(r.get("log_date", ""))[:10] <= today_str]
    if not hist:
        return {"temp_min_f": 45.0, "freeze_thaw_cycle": 0.0,
                "snowfall_in": 0.0, "consecutive_freeze_days": 0.0}
    return {
        "temp_min_f": sum(float(r.get("temp_min_f") or 32) for r in hist) / len(hist),
        "freeze_thaw_cycle": sum(1 for r in hist if r.get("freeze_thaw_cycle")) / len(hist),
        "snowfall_in": sum(float(r.get("snowfall_in") or 0) for r in hist) / len(hist),
        "consecutive_freeze_days": sum(float(r.get("consecutive_freeze_days") or 0) for r in hist) / len(hist),
    }


def _model_path(model_dir: str, model_type: str, sku_id: str, location_id: str) -> Path:
    p = Path(model_dir) / model_type
    p.mkdir(parents=True, exist_ok=True)
    safe_sku = sku_id.replace("/", "_").replace("\\", "_")
    safe_loc = location_id.replace("/", "_").replace("\\", "_")
    return p / f"{safe_sku}_{safe_loc}.pkl"


def _save_model(path: Path, model: Any):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _load_model(path: Path) -> Any | None:
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _build_prophet_df(
    demand_by_date: dict[str, float],
    weather_by_date: dict[str, dict],
    fallback_weather: dict,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    dates = pd.date_range(min_date, max_date, freq="D")
    records = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        y = demand_by_date.get(ds, 0.0)
        w = weather_by_date.get(ds, fallback_weather)
        records.append({
            "ds": d,
            "y": y,
            "temp_min_f": float(w.get("temp_min_f", fallback_weather["temp_min_f"])),
            "snowfall_in": float(w.get("snowfall_in", fallback_weather["snowfall_in"])),
            "consecutive_freeze_days": float(w.get("consecutive_freeze_days", fallback_weather["consecutive_freeze_days"])),
            "freeze_thaw_cycle": float(w.get("freeze_thaw_cycle", fallback_weather["freeze_thaw_cycle"])),
        })
    return pd.DataFrame(records)


def _build_future_df(
    today: date,
    weather_by_date: dict[str, dict],
    fallback_weather: dict,
) -> pd.DataFrame:
    dates = [today + timedelta(days=k) for k in range(FORECAST_HORIZON_DAYS)]
    records = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        w = weather_by_date.get(ds, fallback_weather)
        records.append({
            "ds": pd.Timestamp(d),
            "temp_min_f": float(w.get("temp_min_f", fallback_weather["temp_min_f"])),
            "snowfall_in": float(w.get("snowfall_in", fallback_weather["snowfall_in"])),
            "consecutive_freeze_days": float(w.get("consecutive_freeze_days", fallback_weather["consecutive_freeze_days"])),
            "freeze_thaw_cycle": float(w.get("freeze_thaw_cycle", fallback_weather["freeze_thaw_cycle"])),
        })
    return pd.DataFrame(records)


def _build_xgb_features(
    target: date,
    demand_by_date: dict[str, float],
    weather_by_date: dict[str, dict],
    fallback_weather: dict,
    fallback_demand: float,
    today: date,
    abc_class_enc: float = 0.0,
    location_tier_enc: float = 2.0,
) -> list[float]:
    def get_demand(d: date) -> float:
        if d >= today:
            return fallback_demand
        return demand_by_date.get(d.isoformat(), 0.0)

    dow = float(target.weekday())
    mon = float(target.month)
    woy = float(target.isocalendar()[1])

    lag_7 = get_demand(target - timedelta(days=7))
    lag_14 = get_demand(target - timedelta(days=14))
    lag_28 = get_demand(target - timedelta(days=28))

    w7 = [get_demand(target - timedelta(days=k)) for k in range(1, 8)]
    w28 = [get_demand(target - timedelta(days=k)) for k in range(1, 29)]
    rolling_mean_7 = sum(w7) / len(w7)
    rolling_mean_28 = sum(w28) / len(w28)

    w = weather_by_date.get(target.isoformat(), fallback_weather)

    return [
        dow, mon, woy,
        lag_7, lag_14, lag_28,
        rolling_mean_7, rolling_mean_28,
        float(w.get("temp_min_f", fallback_weather["temp_min_f"])),
        float(w.get("snowfall_in", fallback_weather["snowfall_in"])),
        float(w.get("consecutive_freeze_days", fallback_weather["consecutive_freeze_days"])),
        float(w.get("freeze_thaw_cycle", fallback_weather["freeze_thaw_cycle"])),
        abc_class_enc,
        location_tier_enc,
    ]


def _train_prophet(
    df: pd.DataFrame,
) -> Any:
    if Prophet is None:
        raise ImportError("prophet is not installed")

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=CHANGEPOINT_PRIOR,
        interval_width=CONFIDENCE_PCT,
    )
    m.add_regressor("temp_min_f")
    m.add_regressor("snowfall_in")
    m.add_regressor("consecutive_freeze_days")
    m.add_regressor("freeze_thaw_cycle")
    m.fit(df)
    return m


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Any:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.10,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        verbosity=0,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _train_and_forecast_pair(
    sku_id: str,
    location_id: str,
    demand_by_date: dict[str, float],
    weather_by_date: dict[str, dict],
    fallback_weather: dict,
    today: date,
    run_date_str: str,
    loc_tier: int,
    mode: str,
    model_cache_dir: str,
) -> tuple[list[dict], dict]:
    stats: dict[str, Any] = {
        "trained": False, "skipped_reason": None,
        "prophet_mape": None, "xgb_mape": None, "train_seconds": 0.0,
    }

    sorted_dates = sorted(demand_by_date.keys())
    if not sorted_dates:
        stats["skipped_reason"] = "no_history"
        return [], stats

    min_date_str = sorted_dates[0]
    max_date_str = sorted_dates[-1]
    min_date = date.fromisoformat(min_date_str)
    max_date = date.fromisoformat(max_date_str)
    span_days = (max_date - min_date).days + 1

    if span_days < MIN_HISTORY_DAYS:
        stats["skipped_reason"] = f"insufficient_history ({span_days}d < {MIN_HISTORY_DAYS}d)"
        return [], stats

    prophet_path = _model_path(model_cache_dir, "prophet", sku_id, location_id)
    xgb_path = _model_path(model_cache_dir, "xgboost", sku_id, location_id)

    prophet_model = None
    xgb_model = None

    if mode == "forecast-only":
        prophet_model = _load_model(prophet_path)
        xgb_model = _load_model(xgb_path)
        if not prophet_model:
            stats["skipped_reason"] = "no_cached_prophet_model"
            return [], stats
    elif mode == "incremental":
        prophet_model = _load_model(prophet_path)
        xgb_model = _load_model(xgb_path)
        if prophet_model and xgb_model:
            cache_mtime = prophet_path.stat().st_mtime
            newest_sale = max_date_str
            cache_date = date.fromtimestamp(cache_mtime).isoformat()
            if newest_sale <= cache_date:
                stats["skipped_reason"] = "model_still_fresh"
                pass
            else:
                prophet_model = None
                xgb_model = None

    t_train = time.perf_counter()

    if prophet_model is None and Prophet is not None:
        try:
            df_prophet = _build_prophet_df(
                demand_by_date, weather_by_date, fallback_weather,
                min_date_str, max_date_str,
            )
            prophet_model = _train_prophet(df_prophet)
            _save_model(prophet_path, prophet_model)
            stats["trained"] = True
        except Exception as exc:
            log.error("  Prophet training failed for %s/%s: %s", sku_id, location_id, exc)
            stats["skipped_reason"] = f"prophet_train_error: {exc}"
            return [], stats

    val_days = min(14, span_days // 5)
    all_dates = [min_date + timedelta(days=k) for k in range(span_days)]
    fallback_demand_vals = list(demand_by_date.values())
    fallback_demand = sum(fallback_demand_vals) / max(len(fallback_demand_vals), 1)

    if xgb_model is None and XGBRegressor is not None:
        try:
            train_dates = all_dates[:-val_days] if val_days > 0 else all_dates
            val_dates_list = all_dates[-val_days:] if val_days > 0 else []

            X_all = np.array([
                _build_xgb_features(
                    d, demand_by_date, weather_by_date, fallback_weather,
                    fallback_demand, today,
                    abc_class_enc=0.0,
                    location_tier_enc=float(loc_tier),
                )
                for d in train_dates
            ])
            y_all = np.array([
                demand_by_date.get(d.isoformat(), 0.0) for d in train_dates
            ])

            if len(val_dates_list) >= 2:
                X_val = np.array([
                    _build_xgb_features(
                        d, demand_by_date, weather_by_date, fallback_weather,
                        fallback_demand, today,
                        abc_class_enc=0.0,
                        location_tier_enc=float(loc_tier),
                    )
                    for d in val_dates_list
                ])
                y_val = np.array([
                    demand_by_date.get(d.isoformat(), 0.0) for d in val_dates_list
                ])
            else:
                X_val = X_all[-5:]
                y_val = y_all[-5:]

            xgb_model = _train_xgboost(X_all, y_all, X_val, y_val)
            _save_model(xgb_path, xgb_model)
            stats["trained"] = True
        except Exception as exc:
            log.warning("  XGBoost training failed for %s/%s: %s — Prophet only", sku_id, location_id, exc)
            xgb_model = None

    stats["train_seconds"] = time.perf_counter() - t_train

    future_df = _build_future_df(today, weather_by_date, fallback_weather)

    prophet_preds = None
    prophet_lower = None
    prophet_upper = None
    if prophet_model is not None:
        try:
            fc = prophet_model.predict(future_df)
            prophet_preds = fc["yhat"].values
            prophet_lower = fc["yhat_lower"].values
            prophet_upper = fc["yhat_upper"].values
        except Exception as exc:
            log.warning("  Prophet forecast failed for %s/%s: %s", sku_id, location_id, exc)

    xgb_preds = None
    if xgb_model is not None:
        try:
            forecast_dates = [today + timedelta(days=k) for k in range(FORECAST_HORIZON_DAYS)]
            X_future = np.array([
                _build_xgb_features(
                    d, demand_by_date, weather_by_date, fallback_weather,
                    fallback_demand, today,
                    abc_class_enc=0.0,
                    location_tier_enc=float(loc_tier),
                )
                for d in forecast_dates
            ])
            xgb_preds = xgb_model.predict(X_future)
        except Exception as exc:
            log.warning("  XGBoost forecast failed for %s/%s: %s", sku_id, location_id, exc)

    if prophet_preds is None and xgb_preds is None:
        stats["skipped_reason"] = "both_models_failed_forecast"
        return [], stats

    fc_rows = []
    for i in range(FORECAST_HORIZON_DAYS):
        d = today + timedelta(days=i)

        if prophet_preds is not None and xgb_preds is not None:
            predicted = PROPHET_WEIGHT * float(prophet_preds[i]) + XGBOOST_WEIGHT * float(xgb_preds[i])
        elif prophet_preds is not None:
            predicted = float(prophet_preds[i])
        else:
            predicted = float(xgb_preds[i])

        lower = float(prophet_lower[i]) if prophet_lower is not None else max(0.0, predicted * 0.7)
        upper = float(prophet_upper[i]) if prophet_upper is not None else predicted * 1.3

        fc_rows.append({
            "sku_id":         sku_id,
            "location_id":    location_id,
            "forecast_date":  d.isoformat(),
            "model_type":     MODEL_TYPE,
            "predicted_qty":  round(max(0.0, predicted), 4),
            "lower_bound":    round(max(0.0, lower), 4),
            "upper_bound":    round(max(0.0, upper), 4),
            "confidence_pct": CONFIDENCE_PCT,
            "run_date":       run_date_str,
        })

    return fc_rows, stats


def run_forecast(
    mode: str = "full",
    dry_run: bool = False,
    single_sku: str | None = None,
) -> int:
    t_start = time.perf_counter()
    today = date.today()
    run_date_str = today.isoformat()
    lookback_start = (today - timedelta(days=730)).isoformat()

    log.info("=" * 60)
    log.info("PROPHET + XGBOOST ENSEMBLE FORECAST — A-class SKUs")
    log.info("  Mode: %s  Dry-run: %s", mode, dry_run)
    log.info("  Prophet weight: %.0f%%  XGBoost weight: %.0f%%",
             PROPHET_WEIGHT * 100, XGBOOST_WEIGHT * 100)
    log.info("  Min history: %dd  Horizon: %dd  Cache: %s",
             MIN_HISTORY_DAYS, FORECAST_HORIZON_DAYS, MODEL_CACHE_DIR)
    if single_sku:
        log.info("  Single SKU test: %s", single_sku)
    log.info("=" * 60)

    if Prophet is None:
        log.error("prophet package is not installed. pip install prophet")
        return 1

    client = _get_fresh_client()
    client_holder = [client]

    log.info("Fetching A-class SKUs …")
    a_class_rows = _fetch_a_class_skus(client)
    if single_sku:
        a_class_rows = [r for r in a_class_rows if r["sku_id"] == single_sku]
        if not a_class_rows:
            a_class_rows = [{"sku_id": single_sku, "abc_class": "A"}]
            log.warning("  SKU %s not found as A-class — forcing single-SKU run.", single_sku)

    a_class_set = {r["sku_id"] for r in a_class_rows}
    xyz_map_local = {r["sku_id"]: r.get("abc_xyz_class", "AY") for r in a_class_rows}

    log.info("Fetching transactions (since %s) …", lookback_start)
    tx_rows = _fetch_transactions(client, lookback_start)

    log.info("Fetching weather data …")
    weather_rows = _fetch_weather(client, lookback_start)
    weather_by_date = _build_weather_map(weather_rows)
    fallback_weather = _compute_fallback_weather(weather_rows)

    log.info("Fetching location tiers …")
    loc_tiers = _fetch_location_tiers(client)

    log.info("Fetching demand quality scores …")
    try:
        dq_scores = _fetch_demand_quality(client)
    except Exception:
        dq_scores = {}

    log.info("Building demand map …")
    demand_map = _build_demand_map(tx_rows, a_class_set)
    del tx_rows

    sku_loc_pairs = sorted(demand_map.keys())
    if single_sku:
        sku_loc_pairs = [(s, l) for s, l in sku_loc_pairs if s == single_sku]

    log.info("  (SKU, location) pairs with data: %d", len(sku_loc_pairs))
    skus_no_sales = len(a_class_set) - len({s for s, _ in sku_loc_pairs})
    log.info("  A-class SKUs with no sales data: %d", skus_no_sales)

    total_batches = math.ceil(len(sku_loc_pairs) / BATCH_SIZE) if sku_loc_pairs else 0
    processed = 0
    skipped = 0
    trained = 0
    total_fc_rows = 0
    fc_buffer: list[dict] = []
    fastest_time = float("inf")
    slowest_time = 0.0
    mape_values: list[float] = []

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(sku_loc_pairs))
        batch_pairs = sku_loc_pairs[batch_start:batch_end]

        elapsed_so_far = time.perf_counter() - t_start
        if batch_idx > 0:
            avg_per_batch = elapsed_so_far / batch_idx
            est_remaining = avg_per_batch * (total_batches - batch_idx)
            est_min = est_remaining / 60
        else:
            est_min = 0

        log.info(
            "[PROPHET] Batch %d/%d — %d SKUs complete — elapsed %.0fm — est %.0fm remaining",
            batch_idx + 1, total_batches, processed,
            elapsed_so_far / 60, est_min,
        )

        for sku_id, loc_id in batch_pairs:
            demand_by_date = demand_map.get((sku_id, loc_id), {})
            loc_tier = loc_tiers.get(loc_id, 2)
            dq_score = dq_scores.get((sku_id, loc_id), 1.0)

            t_pair = time.perf_counter()

            fc_rows, stats = _train_and_forecast_pair(
                sku_id, loc_id, demand_by_date,
                weather_by_date, fallback_weather,
                today, run_date_str, loc_tier, mode, MODEL_CACHE_DIR,
            )

            pair_time = time.perf_counter() - t_pair

            if not fc_rows:
                skipped += 1
                reason = stats.get("skipped_reason", "unknown")
                log.debug("  SKIP  %-12s %-10s  %s", sku_id, loc_id, reason)
                continue

            if loc_tier == 3 and dq_score < LOW_QUALITY_THRESHOLD:
                regional_pairs = [
                    (s, l) for s, l in demand_map.keys()
                    if s == sku_id and loc_tiers.get(l, 2) in (1, 2)
                ]
                if regional_pairs:
                    regional_demands = []
                    for s, l in regional_pairs:
                        vals = list(demand_map[(s, l)].values())
                        if vals:
                            regional_demands.append(sum(vals) / len(vals))
                    if regional_demands:
                        regional_avg = sum(regional_demands) / len(regional_demands)
                        fc_rows = [
                            {
                                **r,
                                "predicted_qty": round(
                                    TIER3_BLEND_WEIGHT * regional_avg
                                    + (1.0 - TIER3_BLEND_WEIGHT) * r["predicted_qty"],
                                    4,
                                ),
                            }
                            for r in fc_rows
                        ]

            processed += 1
            total_fc_rows += len(fc_rows)
            if stats.get("trained"):
                trained += 1
            fastest_time = min(fastest_time, pair_time)
            slowest_time = max(slowest_time, pair_time)

            mean_pred = sum(r["predicted_qty"] for r in fc_rows) / len(fc_rows)
            log.info(
                "  OK    %-12s %-10s  span=%3dd  pred_avg=%.2f  (%.1fs)",
                sku_id, loc_id,
                len(demand_by_date),
                mean_pred, pair_time,
            )

            fc_buffer.extend(fc_rows)

            while len(fc_buffer) >= _BATCH_WRITE:
                batch = fc_buffer[:_BATCH_WRITE]
                fc_buffer = fc_buffer[_BATCH_WRITE:]
                if not dry_run:
                    _upsert_with_retry(
                        client_holder, "forecast_results", batch,
                        "sku_id,location_id,forecast_date,model_type,run_date",
                    )

    if fc_buffer:
        if not dry_run:
            _upsert_with_retry(
                client_holder, "forecast_results", fc_buffer,
                "sku_id,location_id,forecast_date,model_type,run_date",
            )

    elapsed = time.perf_counter() - t_start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    log.info("=" * 60)
    log.info("Prophet + XGBoost ensemble forecast complete")
    log.info("  SKUs trained:                      %d", trained)
    log.info("  SKUs skipped (insufficient/cached): %d", skipped)
    log.info("  (SKU, loc) pairs processed:         %d", processed)
    log.info("  Forecast rows written:              %d", total_fc_rows)
    if fastest_time < float("inf"):
        log.info("  Fastest SKU:                       %.1f seconds", fastest_time)
        log.info("  Slowest SKU:                       %.1f seconds", slowest_time)
    log.info("  Total runtime:                     %dh %dm", hours, minutes)
    if dry_run:
        log.info("  (DRY RUN — no writes were made)")
    log.info("=" * 60)

    if single_sku:
        _show_single_sku_chart(single_sku, demand_map, today)

    return 0


def _show_single_sku_chart(sku_id: str, demand_map: dict, today: date):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for (s, loc), demand in demand_map.items():
            if s != sku_id:
                continue
            dates = sorted(demand.keys())
            vals = [demand[d] for d in dates]
            plt.figure(figsize=(12, 4))
            plt.plot(dates[-90:], vals[-90:], label=f"{sku_id} @ {loc}")
            plt.title(f"Last 90 Days — {sku_id} @ {loc}")
            plt.ylabel("Daily Demand")
            plt.legend()
            plt.tight_layout()
            chart_path = f"models/chart_{sku_id}_{loc}.png"
            plt.savefig(chart_path, dpi=100)
            plt.close()
            log.info("  Chart saved: %s", chart_path)
            break
    except ImportError:
        log.info("  matplotlib not available — skipping chart.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="partswatch-ai: Prophet + XGBoost ensemble forecast for A-class SKUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  full           Train all A-class SKUs from scratch\n"
            "  incremental    Only retrain SKUs with new data since last run\n"
            "  forecast-only  Use cached models, just generate new 30-day forecasts\n\n"
            "Single SKU test:\n"
            "  python ml/forecast_prophet.py --sku XT12QULV\n"
        ),
    )
    parser.add_argument(
        "--mode", choices=["full", "incremental", "forecast-only"],
        default="full",
        help="Training mode (default: full).",
    )
    parser.add_argument(
        "--sku", type=str, default=None,
        help="Single SKU to train/forecast (test mode).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Compute forecasts but do not write to the database.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from config import LOG_LEVEL
        setup_logging(LOG_LEVEL)
    except (ImportError, EnvironmentError):
        setup_logging("INFO")

    args = _parse_args()
    return run_forecast(
        mode=args.mode,
        dry_run=args.dry_run,
        single_sku=args.sku,
    )


if __name__ == "__main__":
    sys.exit(main())
