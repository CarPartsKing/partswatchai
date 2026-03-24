# partswatch-ai — Main Project

## Purpose
Central Python project for the partswatch-ai inventory and purchasing intelligence system.
Acts as shared foundation code (config, DB connection, logging utilities) and system health checker.

## Business Context
- Two-step automotive aftermarket distributor, ~$100M revenue, 23 NE Ohio locations
- ~200,000 active SKUs across A/B/C tiers
- PartsWatch (Autologue) management system + RockAuto fulfillment partnership

## System Architecture — 6 Repls, 1 Supabase DB
| Repl | Role |
|------|------|
| partswatch-pipeline | Nightly CSV extraction from PartsWatch |
| partswatch-ml-nightly | LightGBM forecasting (B-class SKUs) |
| partswatch-reorder | Converts forecasts → PO recommendations |
| partswatch-basket | Weekly basket analysis + accuracy tracking |
| partswatch-assistant | Claude-powered purchasing chat (always-on) |
| partswatch-dashboard | Daily brief + alerts UI (always-on) |
| **partswatch-ai** | **This repo — shared foundation + health check** |

## SKU Tiers
- **A-class** (top 10K): Prophet + NE Ohio weather regressors (runs on laptop weekly)
- **B-class** (next 30K): LightGBM nightly via GitHub Actions
- **C-class** (remaining ~160K): 13-week rolling average + de-list classifier

## Tech Stack
- Python 3.11
- Supabase (PostgreSQL shared across all Repls)
- Anthropic Claude API (purchasing assistant)
- Open-Meteo (free NE Ohio weather — no key required)
- LightGBM, Prophet, scikit-learn, mlxtend, Isolation Forest
- Rich (console output), tenacity (retry logic), python-dotenv

## Database Tables
- `sales_transactions`
- `inventory_snapshots`
- `purchase_orders`
- `sku_master`
- `weather_log`
- `forecast_results`
- `supplier_scores`

## Key Derived Fields
| Field | Description |
|-------|-------------|
| `is_stockout` | qty_on_hand = 0 |
| `lost_sales_imputation` | Estimated demand during stockout |
| `lead_time_variance` | Days late per PO line |
| `fill_rate_pct` | qty_received / qty_ordered |
| `abc_class` | A/B/C based on sales velocity |
| `consecutive_freeze_days` | NE Ohio weather signal |
| `freeze_thaw_cycle` | Pothole season predictor |
| `weather_sensitivity_score` | Per-SKU category sensitivity |

## Project Structure
```
partswatch-ai/
├── main.py                          # System health check entry point
├── config.py                        # All env-var config in one place
├── .env.example                     # Secret template (copy → .env)
├── config/
│   └── partswatch_column_map.json   # Maps schema fields → PartsWatch export names
├── db/
│   ├── __init__.py
│   └── connection.py                # Supabase singleton + retry helpers
├── extract/
│   ├── __init__.py
│   ├── weather_pull.py              # Open-Meteo → weather_log (1,110 rows live)
│   └── partswatch_pull.py          # PartsWatch → 4 Supabase tables
├── sample_data/                     # Realistic test CSVs (15 SKUs, 5 locations)
│   ├── sku_master.csv
│   ├── sales_transactions.csv       # 55 transactions Jan–Mar 2026
│   ├── inventory_snapshots.csv      # 45 rows (15 SKUs × 3 locations, Mar 23)
│   └── purchase_orders.csv          # 20 PO lines, 5 POs, mix of statuses
├── utils/
│   ├── __init__.py
│   └── logging_config.py            # Shared timestamped logger
└── models/                          # ML model wrappers (to be built)
```

## PartsWatch Pipeline — Switching Data Sources
Change ONE environment variable — nothing else in the codebase changes:
| `PARTSWATCH_SOURCE` | What it does |
|---------------------|-------------|
| `csv` (default) | Reads CSV/Excel files from `PARTSWATCH_CSV_PATH` folder |
| `odbc` | ODBC connection (stub — ready to implement when Autologue provides DSN) |
| `api` | REST API (stub — ready to implement when Autologue provides credentials) |

When real PartsWatch exports arrive from Autologue:
1. Update column values in `config/partswatch_column_map.json`
2. Drop real export files into `PARTSWATCH_CSV_PATH`
3. Run `python -m extract.partswatch_pull`

## Environment Variables (.env)
| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase service role key |
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `PARTSWATCH_SOURCE` | No | csv \| odbc \| api (default: csv) |
| `PARTSWATCH_CSV_PATH` | No | Folder with export files (default: sample_data) |
| `PARTSWATCH_ODBC_DSN` | No | ODBC connection string (only if SOURCE=odbc) |
| `PARTSWATCH_API_URL` | No | API base URL (only if SOURCE=api) |
| `PARTSWATCH_API_KEY` | No | API key (only if SOURCE=api) |
| `WEATHER_LAT` | No | NE Ohio latitude (default: 41.4993) |
| `WEATHER_LON` | No | NE Ohio longitude (default: -81.6944) |
| `LOG_LEVEL` | No | INFO / DEBUG / WARNING (default: INFO) |
| `ENVIRONMENT` | No | development / production (default: development) |

## Database Tables (created by db_setup.py)
Run `python db_setup.py` once after adding secrets to create all tables.
All DDL is idempotent (IF NOT EXISTS) — safe to re-run.

| Table | Key fields |
|-------|-----------|
| `sku_master` | sku_id, abc_class, weather_sensitivity_score |
| `sales_transactions` | sku_id, location_id, transaction_date, is_stockout (computed), lost_sales_imputation |
| `inventory_snapshots` | sku_id, location_id, snapshot_date, qty_on_hand, is_stockout (generated) |
| `purchase_orders` | po_number, sku_id, supplier_id, lead_time_variance (generated), fill_rate_pct (generated) |
| `weather_log` | log_date, consecutive_freeze_days, freeze_thaw_cycle |
| `forecast_results` | sku_id, forecast_date, model_type (prophet/lightgbm/rolling_avg) |
| `supplier_scores` | supplier_id, score_date, composite_score |

## Coding Standards
- Full docstrings on every function
- try/except with logging on all external calls
- Timestamps on all log lines via `utils/logging_config.get_logger(__name__)`
- Secrets via .env only — never hardcoded
- Production-quality, no skeletons or placeholders
