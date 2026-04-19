#!/usr/bin/env python3
"""main.py — PartsWatch AI pipeline orchestrator.

Runs the complete nightly pipeline in sequence with per-stage timing,
fail-continue semantics, and a structured completion summary.

Usage
-----
    python main.py                          # full 9-stage nightly run
    python main.py --dry-run               # every stage in test mode (no DB writes)
    python main.py --stage forecast_lgbm   # run one named stage only
    python main.py --weekly                # weekly analysis jobs (basket, accuracy, dead-stock)
    python main.py --health                # connectivity health check only
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from utils.logging_config import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Logging + console
# ---------------------------------------------------------------------------

try:
    from config import LOG_LEVEL
    setup_logging(LOG_LEVEL)
except Exception:
    setup_logging("INFO")

log     = get_logger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    """Metadata for one pipeline stage."""
    key:           str          # CLI name used in --stage
    module:        str          # importlib path
    func_name:     str          # function to call in the module
    dry_run_kwarg: str | None   # kwarg name for dry-run, None = not supported
    skip_dry_run:  bool = False # if True, skip entirely in dry-run mode

NIGHTLY_STAGES: list[Stage] = [
    Stage("extract",           "extract.partswatch_pull",     "run_pipeline",
          dry_run_kwarg="test_mode"),
    Stage("clean",             "transform.clean",             "run_checks",
          dry_run_kwarg="dry_run"),
    Stage("derive",            "transform.derive",            "run_derivations",
          dry_run_kwarg=None, skip_dry_run=True),   # no dry-run support
    Stage("location_classify", "transform.location_classify", "run_classify",
          dry_run_kwarg="dry_run"),
    Stage("anomaly",           "ml.anomaly",                  "run_anomaly_detection",
          dry_run_kwarg="dry_run"),
    Stage("forecast_rolling",  "ml.forecast_rolling",         "run_forecast",
          dry_run_kwarg="dry_run"),
    Stage("forecast_lgbm",     "ml.forecast_lgbm",            "run_forecast",
          dry_run_kwarg="dry_run"),
    Stage("reorder",           "engine.reorder",              "run_reorder",
          dry_run_kwarg="dry_run"),
    Stage("alerts",            "engine.alerts",               "run_alerts",
          dry_run_kwarg="dry_run"),
]

WEEKLY_STAGES: list[Stage] = [
    Stage("dead_stock",      "ml.dead_stock",         "run_dead_stock",      "dry_run"),
    Stage("basket_analysis", "ml.basket",             "run_basket",          "dry_run"),
    Stage("churn",           "ml.churn",              "run_churn",           "dry_run"),
    Stage("understocking",   "engine.understocking",  "run_understocking",   "dry_run"),
    Stage("accuracy_report", "ml.accuracy",           "run",                 "dry_run"),
]

# On-demand extract stages — not part of the nightly default loop, but
# individually runnable via `python main.py --stage <key>`.
EXTRACT_STAGES: list[Stage] = [
    Stage("product_extract", "extract.autocube_product_pull",
          "run_inventory_extract", dry_run_kwarg="dry_run"),
]

STAGE_INDEX: dict[str, Stage] = {
    s.key: s for s in NIGHTLY_STAGES + WEEKLY_STAGES + EXTRACT_STAGES
}

# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    key:        str
    skipped:    bool  = False
    success:    bool  = False
    elapsed_s:  float = 0.0
    error:      str   = ""

# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_stage(stage: Stage, dry_run: bool) -> StageResult:
    """Import `stage.module`, call `stage.func_name`, and capture the result.

    Always returns a StageResult — never raises.
    """
    result = StageResult(key=stage.key)

    if dry_run and stage.skip_dry_run:
        log.info("Stage %-22s  SKIPPED (no dry-run support)", stage.key)
        result.skipped = True
        result.success = True       # treat as OK so pipeline continues
        return result

    t0 = time.monotonic()
    log.info("=" * 64)
    log.info("STAGE %-22s  START  %s", stage.key.upper(),
             datetime.now().strftime("%H:%M:%S"))
    log.info("=" * 64)

    try:
        mod  = importlib.import_module(stage.module)
        func = getattr(mod, stage.func_name)

        kwargs: dict[str, Any] = {}
        if dry_run and stage.dry_run_kwarg:
            kwargs[stage.dry_run_kwarg] = True

        rc = func(**kwargs)
        result.elapsed_s = time.monotonic() - t0
        result.success   = (rc == 0)

        status = "OK" if result.success else f"WARN (rc={rc})"
        log.info(
            "STAGE %-22s  %-8s  %.1fs",
            stage.key.upper(), status, result.elapsed_s,
        )

    except Exception as exc:
        result.elapsed_s = time.monotonic() - t0
        result.success   = False
        result.error     = str(exc)
        log.exception(
            "STAGE %-22s  FAILED  %.1fs — %s",
            stage.key.upper(), result.elapsed_s, exc,
        )

    return result


def run_stages(stages: list[Stage], dry_run: bool) -> list[StageResult]:
    """Run a list of stages sequentially, always continuing on failure."""
    results: list[StageResult] = []
    for stage in stages:
        r = run_stage(stage, dry_run)
        results.append(r)
    return results

# ---------------------------------------------------------------------------
# Post-pipeline metrics query
# ---------------------------------------------------------------------------

def query_metrics(today: date, dry_run: bool) -> dict[str, Any]:
    """Query Supabase for today's pipeline output metrics.

    Returns a dict suitable for the completion summary.
    Silently returns an empty dict if the DB is unreachable.
    """
    if dry_run:
        return {}
    try:
        from db.connection import get_client
        client = get_client()
        iso = today.isoformat()

        def _count(table: str, col: str, val: str) -> int:
            try:
                rows = client.table(table).select("*", count="exact").eq(col, val).execute()
                return rows.count or 0
            except Exception:
                return -1

        return {
            "alerts_total":     _count("alerts",                  "alert_date",           iso),
            "alerts_critical":  _count("alerts",                  "alert_date",           iso),   # re-queried below
            "reorders":         _count("reorder_recommendations", "recommendation_date",  iso),
            "forecasts":        _count("forecast_results",        "run_date",             iso),
            "anomalies":        _count("sales_transactions",      "transaction_date",     iso),
        }
    except Exception as exc:
        log.warning("Metrics query failed (non-fatal): %s", exc)
        return {}


def _count_severity(client: Any, today: str, severity: str) -> int:
    try:
        rows = client.table("alerts").select("*", count="exact").eq(
            "alert_date", today).eq("severity", severity).execute()
        return rows.count or 0
    except Exception:
        return -1


def query_metrics_detailed(today: date, dry_run: bool) -> dict[str, Any]:
    """Richer metrics including per-severity alert counts."""
    if dry_run:
        return {}
    try:
        from db.connection import get_client
        client = get_client()
        iso = today.isoformat()

        def _count(table: str, col: str, val: str) -> int:
            try:
                rows = client.table(table).select("*", count="exact").eq(col, val).execute()
                return rows.count or 0
            except Exception:
                return -1

        return {
            "alerts_total":    _count("alerts", "alert_date", iso),
            "alerts_critical": _count_severity(client, iso, "critical"),
            "alerts_warning":  _count_severity(client, iso, "warning"),
            "alerts_info":     _count_severity(client, iso, "info"),
            "reorders":        _count("reorder_recommendations", "recommendation_date", iso),
            "forecasts":       _count("forecast_results", "run_date", iso),
            "locations":       _count("sku_location_demand_quality", "scored_at", iso),
        }
    except Exception as exc:
        log.warning("Detailed metrics query failed (non-fatal): %s", exc)
        return {}

# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

_STAGE_DISPLAY: dict[str, str] = {
    "extract":           "Extract (PartsWatch pull)",
    "clean":             "Clean (quality checks)",
    "derive":            "Derive (scores + ABC class)",
    "location_classify": "Location classify (tiers + quality)",
    "anomaly":           "Anomaly detection (Isolation Forest)",
    "forecast_rolling":  "Forecast — rolling avg (C-class)",
    "forecast_lgbm":     "Forecast — LightGBM (B-class)",
    "reorder":           "Reorder engine",
    "alerts":            "Alerts engine",
    "basket_analysis":   "Basket analysis (weekly)",
    "accuracy_report":   "Forecast accuracy report (weekly)",
    "dead_stock":        "Dead stock — capital-weighted (weekly)",
    "product_extract":   "Product extract (Autocube inventory + master)",
}


def render_summary(
    results:      list[StageResult],
    metrics:      dict[str, Any],
    total_s:      float,
    dry_run:      bool,
    pipeline_tag: str,
) -> int:
    """Print a Rich summary table.  Returns exit code (0 = all OK, 1 = any failed)."""

    # ── Stage table ────────────────────────────────────────────────────────
    tbl = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    tbl.add_column("#",           width=3,  style="dim")
    tbl.add_column("Stage",       width=36)
    tbl.add_column("Status",      width=10)
    tbl.add_column("Time",        width=8,  justify="right")

    any_failed = False
    for i, r in enumerate(results, 1):
        label = _STAGE_DISPLAY.get(r.key, r.key)
        if r.skipped:
            status = "[yellow]SKIPPED[/yellow]"
            elapsed = "—"
        elif r.success:
            status  = "[green]OK[/green]"
            elapsed = f"{r.elapsed_s:.1f}s"
        else:
            status  = "[red]FAILED[/red]"
            elapsed = f"{r.elapsed_s:.1f}s"
            any_failed = True

        tbl.add_row(str(i), label, status, elapsed)

    title_color = "red" if any_failed else "green"
    dry_tag     = "  [yellow]DRY RUN[/yellow]" if dry_run else ""
    border_sty  = "red" if any_failed else "green"
    console.print(Panel(tbl,
        title=f"[bold {title_color}]{pipeline_tag}  —  Pipeline Summary[/bold {title_color}]{dry_tag}",
        border_style=border_sty))

    # ── Metrics table ──────────────────────────────────────────────────────
    if metrics:
        m_tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        m_tbl.add_column("Metric", style="bold", width=32)
        m_tbl.add_column("Value",  width=12, justify="right")

        metric_rows: list[tuple[str, str]] = [
            ("Alerts generated (total)",    str(metrics.get("alerts_total",    "—"))),
            ("  — critical",                str(metrics.get("alerts_critical",  "—"))),
            ("  — warning",                 str(metrics.get("alerts_warning",   "—"))),
            ("  — info",                    str(metrics.get("alerts_info",      "—"))),
            ("Reorder recommendations",     str(metrics.get("reorders",         "—"))),
            ("Forecast rows written",       str(metrics.get("forecasts",        "—"))),
        ]
        for label, val in metric_rows:
            m_tbl.add_row(label, f"[cyan]{val}[/cyan]")

        console.print(Panel(m_tbl, title="Today's Output Metrics", border_style="dim"))

    # ── Overall status ─────────────────────────────────────────────────────
    succeeded = sum(1 for r in results if r.success and not r.skipped)
    failed    = sum(1 for r in results if not r.success and not r.skipped)
    skipped   = sum(1 for r in results if r.skipped)

    status_line = (
        f"[green]{succeeded} succeeded[/green]"
        f"{f'  [red]{failed} failed[/red]' if failed else ''}"
        f"{f'  [yellow]{skipped} skipped[/yellow]' if skipped else ''}"
        f"  |  Total runtime: {total_s:.1f}s"
    )
    console.print(Panel(status_line,
        border_style="red" if any_failed else "green"))

    return 1 if any_failed else 0

# ---------------------------------------------------------------------------
# Health check (reused from original main.py)
# ---------------------------------------------------------------------------

def run_health_check() -> int:
    """Confirm config, DB tables, and weather API are all reachable."""
    from rich.panel import Panel

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(Panel(
        f"[bold cyan]partswatch-ai[/bold cyan] — System Health Check\n[dim]{timestamp}[/dim]",
        border_style="cyan",
    ))

    all_ok = True

    # ── Config ────────────────────────────────────────────────────────────
    try:
        from config import SUPABASE_URL, WEATHER_LAT, WEATHER_LON, LOG_LEVEL, ENVIRONMENT
        console.print(f"  [green]✓[/green]  Config  ENVIRONMENT={ENVIRONMENT}  LOG_LEVEL={LOG_LEVEL}")
        console.print(f"  [green]✓[/green]  Supabase  {SUPABASE_URL}")
        console.print(f"  [green]✓[/green]  Weather coords  lat={WEATHER_LAT} lon={WEATHER_LON}")
    except EnvironmentError as exc:
        console.print(f"  [red]✗[/red]  Config  {exc}")
        all_ok = False

    # ── DB tables ─────────────────────────────────────────────────────────
    try:
        from db.connection import get_client, check_table_exists
        from config import EXPECTED_TABLES
        get_client()
        console.print("  [green]✓[/green]  Supabase connection established")
        for t in EXPECTED_TABLES:
            if check_table_exists(t):
                console.print(f"  [green]✓[/green]  Table  {t}")
            else:
                console.print(f"  [yellow]⚠[/yellow]  Table  {t}  — NOT FOUND (run migrations)")
                all_ok = False
    except Exception as exc:
        console.print(f"  [red]✗[/red]  Database  {exc}")
        all_ok = False

    # ── Weather API ───────────────────────────────────────────────────────
    try:
        import requests
        from config import OPEN_METEO_FORECAST_URL, WEATHER_LAT, WEATHER_LON
        params = {"latitude": WEATHER_LAT, "longitude": WEATHER_LON,
                  "daily": "temperature_2m_max", "forecast_days": 1}
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        console.print(f"  [green]✓[/green]  Open-Meteo  HTTP {resp.status_code}")
    except Exception as exc:
        console.print(f"  [red]✗[/red]  Open-Meteo  {exc}")
        all_ok = False

    if all_ok:
        console.print(Panel("[bold green]All systems operational.[/bold green]", border_style="green"))
        return 0
    else:
        console.print(Panel("[bold yellow]One or more checks failed.[/bold yellow]", border_style="yellow"))
        return 1

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PartsWatch AI pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Run every stage in test mode — no database writes.",
    )
    mode.add_argument(
        "--stage", metavar="STAGE_KEY",
        help="Run a single named stage (e.g. forecast_lgbm).",
    )
    mode.add_argument(
        "--weekly", action="store_true",
        help="Run the weekly analysis jobs instead of the nightly pipeline.",
    )
    mode.add_argument(
        "--health", action="store_true",
        help="Run the system health check and exit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    today = date.today()

    # ── Health check ──────────────────────────────────────────────────────
    if args.health:
        return run_health_check()

    # ── Single-stage mode ─────────────────────────────────────────────────
    if args.stage:
        key = args.stage.lower().strip()
        if key not in STAGE_INDEX:
            valid = ", ".join(sorted(STAGE_INDEX.keys()))
            console.print(f"[red]Unknown stage '{key}'.[/red]  Valid keys: {valid}")
            return 1

        stage = STAGE_INDEX[key]
        log.info("partswatch-ai  single-stage run: %s", key)
        t0 = time.monotonic()
        result = run_stage(stage, dry_run=False)
        total_s = time.monotonic() - t0
        return render_summary([result], {}, total_s, dry_run=False, pipeline_tag="Single Stage")

    # ── Weekly mode ───────────────────────────────────────────────────────
    if args.weekly:
        log.info("partswatch-ai  weekly pipeline  %s", today.isoformat())
        if not WEEKLY_STAGES:
            console.print(Panel(
                "[yellow]Weekly pipeline stages are not yet implemented.[/yellow]\n"
                "Planned: basket analysis, forecast accuracy report, dead-stock classifier.\n"
                "Add stage entries to WEEKLY_STAGES in main.py as each module is built.",
                title="Weekly Pipeline",
                border_style="yellow",
            ))
            return 0
        t0 = time.monotonic()
        results = run_stages(WEEKLY_STAGES, dry_run=args.dry_run)
        total_s = time.monotonic() - t0
        metrics = query_metrics_detailed(today, dry_run=args.dry_run)
        return render_summary(results, metrics, total_s, dry_run=args.dry_run,
                              pipeline_tag="Weekly Pipeline")

    # ── Nightly pipeline ──────────────────────────────────────────────────
    dry_run = args.dry_run
    pipeline_tag = "Nightly Pipeline"

    console.print(Panel(
        f"[bold cyan]partswatch-ai[/bold cyan] — {pipeline_tag}\n"
        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"{'DRY RUN — no writes' if dry_run else 'LIVE — writing to Supabase'}[/dim]",
        border_style="cyan",
    ))
    log.info("partswatch-ai  nightly pipeline  %s  dry_run=%s",
             today.isoformat(), dry_run)

    t0 = time.monotonic()
    results = run_stages(NIGHTLY_STAGES, dry_run=dry_run)
    total_s = time.monotonic() - t0

    metrics = query_metrics_detailed(today, dry_run=dry_run)

    rc = render_summary(results, metrics, total_s, dry_run=dry_run,
                        pipeline_tag=pipeline_tag)

    log.info(
        "Nightly pipeline complete in %.1fs  |  "
        "succeeded=%d  failed=%d  skipped=%d",
        total_s,
        sum(1 for r in results if r.success and not r.skipped),
        sum(1 for r in results if not r.success and not r.skipped),
        sum(1 for r in results if r.skipped),
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
