"""
db_setup.py — Verify all 7 partswatch-ai tables exist in Supabase.

Uses the supabase-py client (SUPABASE_URL + SUPABASE_KEY) to probe each
expected table.  Run this after applying db/migrations/001_initial_schema.sql
in the Supabase SQL Editor to confirm every table is accessible.

Usage:
    python db_setup.py
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from utils.logging_config import get_logger

log = get_logger(__name__)
console = Console()


def verify_tables() -> int:
    """Probe each expected table using the supabase-py client.

    A table is considered accessible if a SELECT LIMIT 1 succeeds without
    a PostgREST PGRST205 (table not found) error.

    Returns:
        0 if all tables are accessible, 1 if any are missing or unreachable.
    """
    from config import EXPECTED_TABLES
    from db.connection import get_client

    log.info("Connecting to Supabase …")

    try:
        client = get_client()
    except Exception as exc:
        log.error("Could not initialise Supabase client: %s", exc)
        console.print(f"[red]Supabase connection failed:[/red] {exc}")
        return 1

    log.info("Verifying %d tables …", len(EXPECTED_TABLES))

    results = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    results.add_column("", width=3)
    results.add_column("Table", style="bold", width=28)
    results.add_column("Status")

    all_ok = True

    for table_name in EXPECTED_TABLES:
        try:
            client.table(table_name).select("*").limit(1).execute()
            results.add_row("[green]✓[/green]", table_name, "[green]accessible[/green]")
            log.info("  OK   — %s", table_name)
        except Exception as exc:
            all_ok = False
            results.add_row("[red]✗[/red]", table_name, f"[red]NOT FOUND[/red]")
            log.warning("  FAIL — %s: %s", table_name, exc)

    status = (
        "[green]All 7 tables verified[/green]"
        if all_ok
        else "[red]Missing tables — run db/migrations/001_initial_schema.sql in Supabase SQL Editor[/red]"
    )
    console.print(Panel(results, title=f"[cyan]partswatch-ai — Table Verification[/cyan]", border_style="dim"))

    if all_ok:
        console.print(Panel("[bold green]Database is ready. All 7 tables are live.[/bold green]", border_style="green"))
        log.info("All tables verified successfully.")
    else:
        console.print(Panel(
            "[yellow]One or more tables are missing.\n\n[/yellow]"
            "Apply [bold]db/migrations/001_initial_schema.sql[/bold] in the Supabase SQL Editor, then re-run this script.",
            border_style="yellow",
        ))
        log.warning("Verification failed — missing tables detected.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(verify_tables())
