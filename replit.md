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
├── main.py                  # System health check entry point
├── config.py                # All env-var config in one place
├── .env.example             # Secret template (copy → .env)
├── db/
│   ├── __init__.py
│   └── connection.py        # Supabase singleton + retry helpers
├── utils/
│   ├── __init__.py
│   └── logging_config.py    # Shared timestamped logger
└── models/                  # ML model wrappers (to be built)
```

## Environment Variables (.env)
| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase service role key |
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `WEATHER_LAT` | No | NE Ohio latitude (default: 41.4993) |
| `WEATHER_LON` | No | NE Ohio longitude (default: -81.6944) |
| `LOG_LEVEL` | No | INFO / DEBUG / WARNING (default: INFO) |
| `ENVIRONMENT` | No | development / production (default: development) |

## Coding Standards
- Full docstrings on every function
- try/except with logging on all external calls
- Timestamps on all log lines via `utils/logging_config.get_logger(__name__)`
- Secrets via .env only — never hardcoded
- Production-quality, no skeletons or placeholders
