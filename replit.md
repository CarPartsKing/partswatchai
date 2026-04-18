# partswatch-ai — Main Project

## Purpose
Central Python project for the partswatch-ai inventory and purchasing intelligence system.
Acts as shared foundation code (config, DB connection, logging utilities) and system health checker.

## Business Context
- Two-step automotive aftermarket distributor, ~$100M revenue, 23 NE Ohio locations
- ~200,000 active SKUs across A/B/C tiers
- PartsWatch (Autologue) management system + RockAuto fulfillment partnership

## Roadmap

### Phase 1 — Complete (core pipeline + dashboard)
- Datatron data extraction (current system pre-PartsWatch)
- PartsWatch extraction ready (switches via `PARTSWATCH_SOURCE` env var)
- Weather pipeline — 3 years NE Ohio data
- Data cleaning — 7 checks + data_quality_issues table
- Derived fields — supplier scores, ABC class, XYZ class (abc_xyz_class combined), weather sensitivity
- Anomaly detection — Isolation Forest
- LightGBM forecasting — B-class SKUs
- Rolling average forecasting — C-class SKUs
- Reorder engine — transfers + PO recommendations
- Alerts engine — 7 alert types, idempotent nightly run
- AI purchasing assistant — Claude multi-turn chat with live context builder
- Morning dashboard — Flask server + dark web UI, 7 live data sections, 5-min auto-refresh

### Phase 2 — Next priorities (confirmed by ownership)

**1. Location demand quality classification** ← *COMPLETE*
- Third-call store problem: downstream stores distort upstream demand signals
- `is_residual_demand` flag on `sales_transactions`
- `sku_location_demand_quality` table (0.0–1.0 score per SKU×location pair)
- Location tier 1/2/3 classification (fill rate 40%, revenue 30%, SKU breadth 20%, return rate 10%)
- Residual demand excluded from forecast training; Tier 3 forecasts blended toward regional baseline
- Migration: `db/migrations/009_location_classify.sql` (written, pending Supabase apply)

**2. L/R part pair grouping**
- Part pairing logic needed — L/R suffix SKUs (e.g. headlights, mirrors, control arms) are treated as separate SKUs but represent the same repair job. Build logic to group L/R pairs and aggregate demand for forecasting accuracy. Example: SKU-1234L and SKU-1234R should be forecasted as a pair.

**3. Prophet A-class forecasting** ← *COMPLETE* (see `ml/forecast_prophet.py`)
- Runs weekly on LOCAL LAPTOP (not Replit — too compute-intensive for 28K+ Prophet models)
- Prophet + XGBoost ensemble: 60%/40% weighted average
- Weather regressors: `temp_min_f`, `snowfall_in`, `consecutive_freeze_days`, `freeze_thaw_cycle`
- Model caching: `models/prophet/` and `models/xgboost/` — incremental mode only retrains stale
- Modes: `--mode full` | `--mode incremental` | `--mode forecast-only` | `--sku XXXX` (single test)
- NOT in main.py pipeline stages (laptop-only); writes to `forecast_results` with `model_type='prophet'`

**4. Min/Max Qty comparison (buyer-set vs AI-generated)**
- Product cube contains buyer-set Min Qty and Max Qty per SKU per location from Datatron. When loaded, compare AI-generated reorder points against these buyer-set minimums. Significant differences (>50% variance) should be flagged for review — either the AI found a better level or the buyer knows something the data doesn't capture.

**5. Ohio VIO data**
- Vehicle registrations by county and model year
- 6-county NE Ohio market: Cuyahoga, Summit, Lorain, Medina, Lake, Geauga

**6. ACES/PIES fitment data**
- Maps every SKU to vehicle applications
- Cross-reference with VIO for active vehicle population per SKU in our market
- Highest ROI Phase 2 addition

**7. PartsTech search data**
- Leading demand signal — searches before orders
- Need to confirm API access via seller account
- If accessible: primary demand indicator

**8. Additional ML models**
- Basket analysis (mlxtend) — co-purchase signals ← *COMPLETE* (see `ml/basket.py`)
- Customer churn predictor
- Dead stock classifier ← *COMPLETE* (see `ml/dead_stock.py`)
- Forecast accuracy feedback loop ← *COMPLETE* (see `ml/accuracy.py`)
- What-if scenario engine

### Phase 3 — Strategy layer (months 6–12)
- Dynamic pricing recommendations
- Supplier negotiation briefs (Claude API)
- Workforce scheduling integration
- Full forecast accuracy feedback loop
- Year-one KPI review: fill rate, inventory turns, excess inventory reduction

### Year 1 Reveal Plan
| Quarter | Goal |
|---------|------|
| Q1–Q2 | Build and validate quietly |
| Q3 | Polish, compile ROI data, soft intro to 1–2 trusted senior buyers |
| Q4 | Full purchasing team rollout — framed as tool that makes team more powerful, not replacement |

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

## Autocube Data Pipeline
- **Source**: AutoCube OLAP (SSAS) via XMLA/SOAP at `/msmdpump.dll` with NTLM auth
- **Historical Load**: Complete — 8.46M sales_transactions + 317K SKUs (Jul 2022 → Apr 2026)
- **Incremental**: `python -m extract.autocube_pull --mode incremental` (previous day)
- **Historical**: `python -m extract.autocube_pull --mode historical` (weekly chunks with resume)
- **Data Cleaning**: Scientific notation → float, MM/DD/YYYY → ISO, location codes "25-CPW - DC" → LOC-025
- **Deduplication**: Same SKU×location×date aggregated (sums qty/rev, latest price)
- **SKU Auto-populate**: New SKUs auto-inserted as stubs into sku_master during extract
- **Progress Tracking**: `/tmp/historical_progress.json` enables resume on restart
- **Runner**: `extract/historical_runner.py` — designed to run as a Replit workflow

## Database Tables
- `sales_transactions` — 8.46M rows (Jul 2022 → present, from Autocube)
- `inventory_snapshots`
- `purchase_orders`
- `sku_master` — 317K SKUs (auto-populated from Autocube + seed data)
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
| `is_anomaly` | Isolation Forest outlier flag — excluded from forecast training |
| `is_warranty` | Warranty transaction flag (TODO) — exclude from forecast training same as is_anomaly |

## Project Structure
```
partswatch-ai/
├── main.py                          # Pipeline orchestrator (9 stages, dry-run, single-stage)
├── .github/
│   └── workflows/
│       ├── nightly.yml              # GitHub Actions: runs full pipeline at 2am EST daily
│       └── weekly.yml               # GitHub Actions: weekly jobs at 11pm EST Sunday
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
│   ├── partswatch_pull.py          # PartsWatch → 4 Supabase tables
│   └── autocube_pull.py            # Autocube XMLA/SOAP OLAP extraction (NTLM auth)
├── sample_data/                     # Realistic test CSVs (15 SKUs, 5 locations)
│   ├── sku_master.csv
│   ├── sales_transactions.csv       # 55 transactions Jan–Mar 2026
│   ├── inventory_snapshots.csv      # 45 rows (15 SKUs × 3 locations, Mar 23)
│   └── purchase_orders.csv          # 20 PO lines, 5 POs, mix of statuses
├── utils/
│   ├── __init__.py
│   └── logging_config.py            # Shared timestamped logger
├── ml/
│   ├── __init__.py
│   ├── anomaly.py                   # Isolation Forest — flags anomalous sales days
│   ├── accuracy.py                  # Forecast accuracy measurement — MAPE/MAE/bias/hit-rate
│   ├── forecast_rolling.py          # 13-week rolling avg — C-class SKU forecasts
│   └── forecast_lgbm.py             # LightGBM gradient-boosted demand forecast — B-class SKUs
├── engine/
│   ├── __init__.py
│   ├── transfer.py                  # Pure-computation inter-location excess detector (no DB I/O)
│   ├── reorder.py                   # Recommendation engine → reorder_recommendations table
│   └── alerts.py                    # 7 alert types → alerts table (idempotent nightly run)
├── assistant/
│   ├── __init__.py
│   ├── context_builder.py           # 7-section live Supabase context (~800 tokens) for Claude
│   └── claude_api.py                # PurchasingAssistant class — multi-turn Claude chat
├── dashboard/
│   ├── server.py                    # Flask server — serves UI + /api/dashboard JSON endpoint
│   └── index.html                   # Dark web dashboard — 7 panels, 5-min auto-refresh
└── models/                          # ML model wrappers (to be built)
```

## Pipeline Orchestrator (main.py)
| Command | What it does |
|---------|-------------|
| `python main.py` | Full 9-stage nightly pipeline |
| `python main.py --dry-run` | All stages in test mode — no DB writes (derive stage skipped) |
| `python main.py --stage forecast_lgbm` | Single named stage (any of the 9 keys) |
| `python main.py --weekly` | Weekly jobs: dead_stock (live); basket_analysis + accuracy_report (Phase 2) |
| `python main.py --health` | System health check (config, DB, weather API) |

**Stage keys**: `extract`, `clean`, `derive`, `location_classify`, `anomaly`, `forecast_rolling`, `forecast_lgbm`, `reorder`, `alerts`

**Dry-run confirmed**: 8/8 active stages OK in 9.2s; `derive` skipped (no dry-run support)

## Full Pipeline Run Status (Apr 10, 2026)
| Stage | Status | Time | Notes |
|-------|--------|------|-------|
| derive (lost sales) | ✅ | 0.7s | 2 stockout rows imputed |
| derive (ABC) | ✅ | 186s | 317,738 SKUs classified (A=28K, B=42K, C=247K) |
| derive (XYZ) | ⏭️ Skipped | 0.1s | Migration 010 pending — apply in Supabase SQL Editor |
| derive (supplier scores) | ✅ | 0.2s | 5 suppliers scored |
| derive (weather sensitivity) | ✅ | 571s | 12 SKUs updated (sample categories only) |
| derive (SKU metrics) | ✅ | 346s | 226,740 SKUs with last_sale_date + avg_weekly_units |
| location_classify | ⚠️ Partial | 567s | Tier classification OK; demand quality step timed out |
| anomaly | ❌ | 772s | 502 Bad Gateway after 2M+ rows (Supabase rate limit) |
| forecast_rolling | ✅ | 151s | 840 forecasts for 28 C-class SKU-location pairs |
| forecast_lgbm | ❌ | 1421s | Statement timeout fetching 2yr B-class data |
| reorder | ✅ | 45s | Recommendations generated |
| alerts | ✅ | 1s | 7 supplier risk warnings |

**Root cause of failures**: Supabase REST API pagination (1000 rows/page) is too slow for 2M+ row fetches.
**Fix for production**: Use direct Postgres connection (service role key) or Supabase Edge Functions for heavy aggregation.

## GitHub Actions
- **nightly.yml**: `0 7 * * *` (7am UTC = 2am EST) — full pipeline
- **weekly.yml**: `0 4 * * 1` (4am UTC Monday = 11pm EST Sunday) — weekly analysis + nightly re-run
- **Required Secrets**: `SUPABASE_URL`, `SUPABASE_KEY`, `ANTHROPIC_API_KEY`
- **Optional Secrets**: `PARTSWATCH_SOURCE`, `WEATHER_LAT`, `WEATHER_LON`
- Both support manual dispatch with dry-run and single-stage inputs

## Dashboard
- **URL**: served from port 5000 via `python dashboard/server.py` (workflow: "Start application")
- **Endpoint**: `GET /api/dashboard` — returns all sections as JSON, parallelized via ThreadPoolExecutor
- **Sections**: Network KPIs, Weather (freeze warnings), Alert Summary, Critical Alerts, Top Reorder Recommendations, **Dead Stock — Capital at Risk**, Supplier Health, Inventory Health, Inter-Location Transfers, Forecast Accuracy, Top SKUs, Anomaly Summary, Location Tiers, Pipeline Status
- **Dead Stock panel** (added 2026-04-18): KPI strip (capital_at_risk / liquidate_count+value / markdown_count+value / total_positions in red/amber), top-10 LIQUIDATE candidates by inventory value, "Export Full Liquidation List" button → `GET /api/dead-stock/export.csv`. Backed by `dead_stock_recommendations` table (migration 023) which `ml/dead_stock.py` populates per-day with normalized action codes (WRITEOFF/RETURN/MARKDOWN/LIQUIDATE). CSV output has CWE-1236 formula-injection guard.
- **Refresh**: JS auto-refreshes every 5 minutes with countdown; manual refresh button
- **Status**: Green dot = live, amber = loading, red = offline (retries in 30s)

## PartsWatch Pipeline — Switching Data Sources
Change ONE environment variable — nothing else in the codebase changes:
| `PARTSWATCH_SOURCE` | What it does |
|---------------------|-------------|
| `datatron` | **Current** — Datatron extraction (pre-PartsWatch system) |
| `csv` | Reads CSV/Excel exports from `PARTSWATCH_CSV_PATH` folder (when PartsWatch goes live) |
| `api` | REST API (stub — ready to implement when Autologue provides credentials) |
| `autocube` | **Autocube OLAP** — XMLA/SOAP via NTLM auth against Autologue's SSAS cube |

When real PartsWatch exports arrive from Autologue:
1. Update column values in `config/partswatch_column_map.json`
2. Drop real export files into `PARTSWATCH_CSV_PATH`
3. Run `python -m extract.partswatch_pull`

## Autocube OLAP Integration
Live connection to Autologue's Autocube data warehouse via XMLA/SOAP protocol.

**Connection (verified):**
- Endpoint: `https://autocubedata.autologue.com/msmdpump.dll`
- Auth: NTLM (`AUTOCUBE\CPW001`)
- Catalog: `AutoCube_DTR_23160`
- Cubes: Product, **Sales Detail** (primary), Sales Summary

**Key Dimensions (Sales Detail cube):**
| Dimension | Hierarchy | Levels |
|-----------|-----------|--------|
| Sales Date | Invoice Date | Year → Qtr → Month → Inv Date |
| Product | Prod Code | flat (part number) |
| Product | Prod Line PN | Prod Line → Prod Code |
| Product | Prod PN Loc | Prod Line → Subline → Product ID |
| Product | Vendor 1 | flat (primary vendor) |
| Location | Loc | flat (store code) |
| Customer | Customer | flat |
| Customer | Cust No | flat |
| Tran Code | Tran Code | flat |

**Key Measures (Sales Detail):**
- Sales: Qty Ship, Unit Price, Ext Price, Unit Cost, Ext Cost, Gross Profit, Gross Profit %
- Inventory: On Hand Qty, Min Qty, Max Qty, Stock Qty, Qty On Order, Ext Cost On Hand
- Trending: MTD/YTD/Prev 12 Mo for Qty, Sales, Cost, GP$ — all with LY comparison and % change
- Period: 1M Ago through 6M Ago for Qty, Sales, Cost, GP$

**Column map:** `config/autocube_column_map.json` — maps cube dimension/measure names to PartsWatch schema fields

**MDX Queries (hardcoded, no dynamic construction):**
- `MDX_FULL_SALES` — full extract: date × product × location with 7 core measures
- `MDX_INCREMENTAL_DAY` — single-day extract by date key

**Security:**
- `query_validator()` blocks INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/EXEC/GRANT/REVOKE
- Only hardcoded MDX templates used (date_key is the only parameterized value)

**Secrets required:** `AUTOCUBE_SERVER`, `AUTOCUBE_USER`, `AUTOCUBE_PASSWORD`, `AUTOCUBE_CATALOG`

## Location Mapping (Product Cube — 27 locations)
| Cube ID | LOC Format | Name | On-Hand Qty | Notes |
|---------|-----------|------|-------------|-------|
| 1 | LOC-001 | BROOKPARK | 51,414 | |
| 2 | LOC-002 | NOLMSTEAD | 30,253 | |
| 3 | LOC-003 | S. EUCLID | 33,111 | |
| 4 | LOC-004 | CLARK AUTO | 55,663 | |
| 5 | LOC-005 | PARMA | 36,035 | |
| 6 | LOC-006 | MEDINA | 33,122 | |
| 7 | LOC-007 | BOARDMAN | 38,375 | |
| 8 | LOC-008 | ELYRIA | 477,242 | Warehouse/hub |
| 9 | LOC-009 | AKRON-GRANT | 29,973 | |
| 10 | LOC-010 | MIDWAY CROSSINGS | 207,697 | Regional hub |
| 11 | LOC-011 | ERIE ST | 54,887 | |
| 12 | LOC-012 | MAYFIELD | 29,326 | |
| 13 | LOC-013 | CANTON | 38,267 | |
| 15 | LOC-015 | JUNIATA | 94,019 | |
| 16 | LOC-016 | ARCHWOOD | 49,177 | |
| 17 | LOC-017 | EUCLID | 38,375 | |
| 18 | LOC-018 | WARREN | 31,377 | |
| 20 | LOC-020 | ROOTSTOWN | 31,978 | |
| 21 | LOC-021 | INTERNET | -1,795 | E-commerce/virtual |
| 24 | LOC-024 | MENTOR | 25,740 | |
| 25 | LOC-025 | DC | 585,402 | Main distribution center |
| 26 | LOC-026 | COPLEY | 31,582 | |
| 27 | LOC-027 | CHARDON | 24,967 | |
| 28 | LOC-028 | STRONGSVILLE | 30,590 | |
| 29 | LOC-029 | MIDDLEBURG | 226,741 | Regional hub |
| 32 | LOC-032 | PERRY | 100,139 | |
| 33 | LOC-033 | CRYSTAL | 28,615 | |

**Gap numbers** (not in Product cube): 14, 19, 22, 23, 30, 31 — closed/retired locations (is_active=FALSE in migration 016).
**Special locations**: LOC-021 (INTERNET) has negative on-hand (-1,795 units) — flagged as data_quality_issue (severity=warning). LOC-025 (MAIN DC) is the main distribution center with 585K items.

**Migration 016** (`db/migrations/016_location_names.sql`): Adds `location_name` and `is_active` to `locations` table, adds `location_name` to `alerts` table, populates all 27 location names, inserts retired locations as is_active=FALSE, backfills existing alert location names, inserts LOC-021 DQI entry. Dashboard and context_builder use LOCATION_NAMES fallback mapping so they work before and after migration is applied.

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
| `AUTOCUBE_SERVER` | No | Autocube XMLA server URL (only if SOURCE=autocube) |
| `AUTOCUBE_USER` | No | Autocube NTLM username (DOMAIN\\user format) |
| `AUTOCUBE_PASSWORD` | No | Autocube NTLM password |
| `AUTOCUBE_CATALOG` | No | Autocube catalog name |
| `AUTOCUBE_CUBE` | No | Cube name (default: Sales Detail) |
| `AUTOCUBE_XMLA_PATH` | No | XMLA endpoint path (default: /msmdpump.dll) |
| `LOG_LEVEL` | No | INFO / DEBUG / WARNING (default: INFO) |
| `ENVIRONMENT` | No | development / production (default: development) |

## Database Tables (created by db_setup.py)
Run `python db_setup.py` once after adding secrets to create all tables.
All DDL is idempotent (IF NOT EXISTS) — safe to re-run.

| Table | Key fields | Status |
|-------|-----------|--------|
| `sku_master` | sku_id, abc_class, weather_sensitivity_score | Live |
| `sales_transactions` | sku_id, location_id, transaction_date, is_stockout (computed), lost_sales_imputation, is_anomaly, is_residual_demand | Live (`is_residual_demand` added by migration 009) |
| `inventory_snapshots` | sku_id, location_id, snapshot_date, qty_on_hand, is_stockout (generated) | Live |
| `purchase_orders` | po_number, sku_id, supplier_id, lead_time_variance (generated), fill_rate_pct (generated) | Live |
| `weather_log` | log_date, consecutive_freeze_days, freeze_thaw_cycle | Live |
| `forecast_results` | sku_id, forecast_date, model_type (prophet/lightgbm/rolling_avg) | Live |
| `supplier_scores` | supplier_id, score_date, composite_score, risk_flag | Live |
| `reorder_recommendations` | sku_id, location_id, recommendation_date, qty_to_order, recommendation_type (po/transfer), urgency, is_approved | Live |
| `alerts` | alert_id, alert_type, sku_id, location_id, location_name, severity, is_acknowledged | Live (migration 008, 016 adds location_name) |
| `accuracy_reports` | report_date, model_type, abc_class, avg_mape, avg_mae, bias, hit_rate_20pct | Pending migration 011 |
| `locations` | location_id, location_name, is_active, location_tier (1/2/3), composite_score, fill_rate, revenue_rank, sku_breadth, return_rate | Pending migrations 009, 016 |
| `sku_location_demand_quality` | sku_id, location_id, demand_quality_score (0.0–1.0), residual_event_count, quality_tier | Pending migration 009 |

## Nightly Pipeline Execution Order
Run stages in this order — each depends on the output of the prior stage:
```
python -m extract.partswatch_pull      # Load raw PartsWatch data → 4 tables
python -m transform.clean              # Null-fill, type coerce, range-clamp
python -m transform.derive             # Lost sales, ABC class, supplier scores, weather sensitivity, SKU metrics
python -m transform.location_classify  # Tier 1/2/3 classification, residual demand flags, demand quality scores
python -m ml.anomaly                   # Isolation Forest → is_anomaly flag on sales_transactions
python -m ml.forecast_rolling          # 13-week rolling avg for C-class SKUs → forecast_results (Tier 3 blended)
python -m ml.forecast_lgbm             # LightGBM gradient-boosted forecast for B-class SKUs (Tier 3 blended)
python -m engine.reorder               # Convert forecasts → reorder_recommendations (transfers + POs)
python -m engine.alerts                # 7 alert types → alerts table (idempotent, preserves acknowledged)
```

## NixOS / LightGBM libgomp Fix
LightGBM 4.6.0's shared library (`lib_lightgbm.so`) requires `libgomp.so.1` (GCC OpenMP).
On NixOS the linker path is not automatically configured. Two things were done to resolve this:

1. **patchelf** (`patchelf` added to system deps) — adds the correct GCC lib rpath permanently into `lib_lightgbm.so`:
   ```
   patchelf --add-rpath /nix/store/bmi5znnqk4kg2grkrhk6py0irc8phf6l-gcc-14.2.1.20250322-lib/lib \
       .pythonlibs/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.so
   ```
2. **ctypes pre-load guard** in `ml/forecast_lgbm.py` — as a belt-and-suspenders fallback that re-loads the correct `libgomp.so.1` before importing LightGBM, using the path in `_GOMP_DIR` (overridable via `GOMP_LIB_DIR` env var if the GCC hash changes after a `nix-env -u`).

If the nix store hash changes (after a NixOS update), re-run the patchelf command and update `_GOMP_DIR` in `ml/forecast_lgbm.py`.

## Data Accuracy Audit Findings (2026-04-18)

### Cube cost measures — `[Measures].[Unit Cost]` is unusable
Empirical finding against `AutoCube_DTR_23160`: `[Measures].[Unit Cost]` returns
NULL for nearly every Product-cube SKU. Real costs live in:
- `[Unit Cost on Hand]` — true per-unit cost of stock currently on hand (NULL when qty=0)
- `[Cost]` — catalog/master cost, populated for essentially every active SKU
- `[Ext Cost On Hand]` — extended on-hand value, used as last-resort `/qty` derivation

`extract/autocube_product_pull.py` resolves cost in that exact order. Same
substitution applies to the Sales Detail cube as well — to be revisited.

### Sales Detail flag dimensions
The Sales Detail cube exposes per-line flag dimensions (`Warranty Flag`,
`Backorder Flag`, `Core Flag`, `Price Override Flag`) with members `'Y'` / `'N'`
(Price Override returns `'-'` instead of `'N'` for the not-set case — boolean
coercion treats anything ≠ `'Y'` as False). They're now added to the MDX
CROSSJOIN in `extract/autocube_pull.py` (`MDX_FULL_SALES`,
`MDX_INCREMENTAL_DAY`, `MDX_MONTHLY_RANGE`) and stored on
`sales_transactions` (migration 014). Forecast and dead-stock pipelines
exclude `is_warranty=True` rows alongside `is_anomaly` and `is_residual_demand`.

**Important:** when adding new mapped fields to `config/autocube_column_map.json`,
also add the destination column to `_DB_COLUMNS` in `extract/autocube_pull.py`,
or the field will be silently dropped before upsert.

### No real PO source in this AutoCube deployment
Catalog only contains 3 cubes: `Product`, `Sales Detail`, `Sales Summary`.
No PO / Purchase Order cube exists. `purchase_orders` table is sample data
(20 rows, sequential numbering, identical timestamps). No extraction job
can be built until a real source is provisioned.

### Min-Qty reconciliation (engine/reorder.py)
After computing `demand_over_coverage` (the AI-derived ideal on-hand level
that should trigger a reorder cycle), `engine/reorder.py` compares it to the
buyer-set `inventory_snapshots.reorder_point`. If `|rel_diff| > 50%`
(`_RECONCILE_THRESHOLD = 0.5`), the discrepancy is written to
`data_quality_issues` with `issue_type='reorder_threshold_mismatch'` and
severity mapped to `'error'` (>100% gap) or `'warning'` (the CHECK constraint
in migration 003 only permits those two values).

## Coding Standards
- Full docstrings on every function
- try/except with logging on all external calls
- Timestamps on all log lines via `utils/logging_config.get_logger(__name__)`
- Secrets via .env only — never hardcoded
- Production-quality, no skeletons or placeholders
