"""
db_setup.py — partswatch-ai database setup tool.

WHAT THIS SCRIPT DOES
---------------------
supabase-py wraps PostgREST, which handles SELECT/INSERT/UPDATE/DELETE only.
It cannot execute DDL (CREATE TABLE / CREATE INDEX).  That is a PostgREST
architectural constraint, not a limitation of this project.

The correct Supabase workflow for schema changes is the SQL Editor:
    Dashboard → SQL Editor → New Query → paste SQL → Run

This script therefore does two things:
  1. Reads db/migrations/001_initial_schema.sql and prints it to the console
     so you can copy-paste it into the Supabase SQL Editor.
  2. Connects via supabase-py (SUPABASE_URL + SUPABASE_KEY) and verifies
     that every expected table is accessible after the migration has been run.

Usage:
  Step 1 — Run this script to see the migration SQL and instructions:
              python db_setup.py

  Step 2 — Paste the printed SQL into Supabase SQL Editor and click Run.

  Step 3 — Run this script again to verify all tables are live:
              python db_setup.py --verify
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

from utils.logging_config import get_logger

log = get_logger(__name__)
console = Console()

MIGRATION_FILE = Path(__file__).parent / "db" / "migrations" / "001_initial_schema.sql"


# ---------------------------------------------------------------------------
# Step 1 — Print migration SQL
# ---------------------------------------------------------------------------

def print_migration_sql() -> None:
    """Read the migration file and display it with syntax highlighting.

    Instructs the user to paste the output into the Supabase SQL Editor.
    """
    if not MIGRATION_FILE.exists():
        log.error("Migration file not found: %s", MIGRATION_FILE)
        console.print(f"[red]Migration file not found:[/red] {MIGRATION_FILE}")
        return

    sql_text = MIGRATION_FILE.read_text(encoding="utf-8")

    console.print(
        Panel(
            "[bold]Step 1 of 2 — Copy the SQL below[/bold]\n\n"
            "Go to your Supabase project:\n"
            "  [cyan]Dashboard → SQL Editor → New Query[/cyan]\n\n"
            "Paste everything between the dividers and click [bold green]Run[/bold green].\n"
            "Then re-run this script with [bold]--verify[/bold] to confirm all tables are live.",
            title="[cyan]partswatch-ai — Database Migration[/cyan]",
            border_style="cyan",
        )
    )

    console.print("\n" + "─" * 80)
    console.print(Syntax(sql_text, "sql", theme="monokai", line_numbers=True))
    console.print("─" * 80 + "\n")

    log.info(
        "Migration SQL printed (%d lines). Paste into Supabase SQL Editor and run.",
        len(sql_text.splitlines()),
    )


# ---------------------------------------------------------------------------
# Step 2 — Verify tables via supabase-py
# ---------------------------------------------------------------------------

def verify_tables() -> int:
    """Probe each expected table using the supabase-py client.

    Uses SUPABASE_URL and SUPABASE_KEY only — no raw DB connection required.
    A table is considered accessible if a SELECT LIMIT 1 succeeds.

    Returns:
        0 if all tables are accessible, 1 if any are missing or unreachable.
    """
    from config import EXPECTED_TABLES
    from db.connection import get_client

    log.info("Verifying %d tables via supabase-py …", len(EXPECTED_TABLES))

    try:
        client = get_client()
    except Exception as exc:
        log.error("Could not initialise Supabase client: %s", exc)
        console.print(f"[red]Supabase connection failed:[/red] {exc}")
        return 1

    tbl = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    tbl.add_column("Status", width=4)
    tbl.add_column("Table", style="bold", width=28)
    tbl.add_column("Detail")

    all_ok = True

    for table_name in EXPECTED_TABLES:
        try:
            client.table(table_name).select("*").limit(1).execute()
            tbl.add_row("[green]✓[/green]", table_name, "accessible")
            log.info("  OK  — %s", table_name)
        except Exception as exc:
            all_ok = False
            detail = str(exc)
            tbl.add_row("[red]✗[/red]", table_name, f"[red]{detail}[/red]")
            log.warning("  FAIL — %s: %s", table_name, detail)

    status_label = "[green]All tables accessible[/green]" if all_ok else "[red]Some tables missing — run the migration SQL first[/red]"
    console.print(Panel(tbl, title=f"[cyan]Table Verification[/cyan]  {status_label}", border_style="dim"))

    if all_ok:
        console.print(Panel("[bold green]Database is ready. All 7 tables verified.[/bold green]", border_style="green"))
        log.info("Verification complete — all tables accessible.")
    else:
        console.print(
            Panel(
                "[yellow]One or more tables are missing.[/yellow]\n\n"
                "Run [bold]python db_setup.py[/bold] (without --verify) to print the migration SQL,\n"
                "paste it into [cyan]Supabase → SQL Editor[/cyan], then run this check again.",
                border_style="yellow",
            )
        )
        log.warning("Verification complete — missing tables detected.")

    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse arguments and run either the SQL printer or the verifier.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="partswatch-ai DB setup: print migration SQL or verify tables."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that all 7 tables exist via supabase-py (run after applying migration SQL).",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold cyan]partswatch-ai[/bold cyan] — Database Setup",
            border_style="cyan",
        )
    )

    if args.verify:
        return verify_tables()
    else:
        print_migration_sql()
        return 0


if __name__ == "__main__":
    sys.exit(main())
