"""
main.py — partswatch-ai system health check and entry point.

Run this script to verify that all environment variables are loaded,
the Supabase connection is live, and all expected database tables are
accessible.  This is the first thing to run after any environment change.

Usage:
    python main.py
"""

import sys
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)
console = Console()


def check_config() -> dict:
    """Validate that all required configuration values are present.

    Returns:
        Dict with keys 'ok' (bool) and 'details' (list of status strings).
    """
    results = []
    ok = True

    checks = [
        ("SUPABASE_URL", "SUPABASE_KEY"),
        ("ANTHROPIC_API_KEY",),
    ]

    try:
        import config  # noqa: F401 — triggers EnvironmentError if vars missing

        from config import (
            SUPABASE_URL,
            WEATHER_LAT,
            WEATHER_LON,
            LOG_LEVEL,
            ENVIRONMENT,
        )

        results.append(("[green]✓[/green]", "Config", f"ENVIRONMENT={ENVIRONMENT}, LOG_LEVEL={LOG_LEVEL}"))
        results.append(("[green]✓[/green]", "Supabase URL", SUPABASE_URL))
        results.append(("[green]✓[/green]", "Weather coords", f"lat={WEATHER_LAT}, lon={WEATHER_LON}"))

    except EnvironmentError as exc:
        ok = False
        results.append(("[red]✗[/red]", "Config", str(exc)))

    return {"ok": ok, "details": results}


def check_database() -> dict:
    """Attempt to connect to Supabase and probe each expected table.

    Returns:
        Dict with keys 'ok' (bool) and 'details' (list of status tuples).
    """
    results = []
    ok = True

    try:
        from db.connection import get_client, check_table_exists
        from config import EXPECTED_TABLES

        # Force client initialization
        get_client()
        results.append(("[green]✓[/green]", "Connection", "Supabase client initialized"))

        for table in EXPECTED_TABLES:
            if check_table_exists(table):
                results.append(("[green]✓[/green]", "Table", table))
            else:
                ok = False
                results.append(("[yellow]⚠[/yellow]", "Table", f"{table} — NOT FOUND (run migrations)"))

    except EnvironmentError as exc:
        ok = False
        results.append(("[red]✗[/red]", "DB Config", str(exc)))
    except Exception as exc:
        ok = False
        results.append(("[red]✗[/red]", "Connection", str(exc)))

    return {"ok": ok, "details": results}


def check_weather_api() -> dict:
    """Ping Open-Meteo to confirm the weather feed is reachable.

    Returns:
        Dict with keys 'ok' (bool) and 'details' (list of status tuples).
    """
    results = []
    ok = True

    try:
        import requests
        from config import OPEN_METEO_FORECAST_URL, WEATHER_LAT, WEATHER_LON

        params = {
            "latitude": WEATHER_LAT,
            "longitude": WEATHER_LON,
            "daily": "temperature_2m_max",
            "forecast_days": 1,
        }
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        results.append(("[green]✓[/green]", "Open-Meteo", f"HTTP {resp.status_code} — NE Ohio feed reachable"))

    except EnvironmentError as exc:
        ok = False
        results.append(("[red]✗[/red]", "Weather Config", str(exc)))
    except Exception as exc:
        ok = False
        results.append(("[red]✗[/red]", "Open-Meteo", str(exc)))

    return {"ok": ok, "details": results}


def render_results(section: str, check_result: dict) -> None:
    """Render a check section as a Rich table to the console.

    Args:
        section:      Display name of the section (e.g. "Configuration").
        check_result: Dict returned by a check_*() function.
    """
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("Status", width=3)
    tbl.add_column("Category", style="bold", width=18)
    tbl.add_column("Detail")

    for row in check_result["details"]:
        tbl.add_row(*row)

    status_label = "[green]PASS[/green]" if check_result["ok"] else "[red]FAIL[/red]"
    console.print(Panel(tbl, title=f"{section}  {status_label}", border_style="dim"))


def main() -> int:
    """Run all system health checks and print a summary.

    Returns:
        Exit code: 0 if all checks passed, 1 if any check failed.
    """
    try:
        from config import LOG_LEVEL
        setup_logging(LOG_LEVEL)
    except EnvironmentError:
        setup_logging("INFO")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        Panel(
            f"[bold cyan]partswatch-ai[/bold cyan] — System Health Check\n"
            f"[dim]{timestamp}[/dim]",
            border_style="cyan",
        )
    )

    log.info("Starting system health check …")

    checks = [
        ("Configuration", check_config),
        ("Database",      check_database),
        ("Weather API",   check_weather_api),
    ]

    all_ok = True
    for section, fn in checks:
        log.info("Checking %s …", section)
        result = fn()
        render_results(section, result)
        if not result["ok"]:
            all_ok = False

    # Final summary banner
    if all_ok:
        console.print(Panel("[bold green]All systems operational.[/bold green]", border_style="green"))
        log.info("Health check complete — all systems operational.")
        return 0
    else:
        console.print(
            Panel(
                "[bold yellow]One or more checks failed.[/bold yellow]\n"
                "Review the output above, check your .env file, and run database migrations if tables are missing.",
                border_style="yellow",
            )
        )
        log.warning("Health check complete — issues detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
