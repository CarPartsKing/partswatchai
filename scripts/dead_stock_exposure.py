#!/usr/bin/env python3
"""
scripts/dead_stock_exposure.py

Dead-stock exposure (total_inv_value) at the latest report_date,
grouped by supplier_id + location_id, excluding data_conflict rows.
Top 25 by dollars plus grand total.

Usage:
    python scripts/dead_stock_exposure.py
"""

import sys
import os

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_client


def fetch_all(query_fn, page_size: int = 1000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        resp = query_fn(offset, offset + page_size - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def main() -> None:
    client = get_client()

    # Latest report_date
    resp = (
        client.table("dead_stock_recommendations")
        .select("report_date")
        .order("report_date", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        print("dead_stock_recommendations is empty.")
        return
    report_date = resp.data[0]["report_date"]
    print(f"Latest report_date: {report_date}")

    # Fetch all clean rows for that date
    print("Fetching rows (data_conflict = false) …")
    rows = fetch_all(
        lambda lo, hi: (
            client.table("dead_stock_recommendations")
            .select("supplier_id,location_id,total_inv_value")
            .eq("report_date", report_date)
            .eq("data_conflict", False)
            .range(lo, hi)
        )
    )

    if not rows:
        print("No clean dead-stock rows found for that date.")
        return

    df = pd.DataFrame(rows)
    df["total_inv_value"] = pd.to_numeric(df["total_inv_value"], errors="coerce").fillna(0)
    df["supplier_id"]  = df["supplier_id"].fillna("(unknown)")
    df["location_id"]  = df["location_id"].fillna("(unknown)")

    agg = (
        df.groupby(["supplier_id", "location_id"], dropna=False)
        .agg(total_inv_value=("total_inv_value", "sum"), sku_count=("total_inv_value", "count"))
        .reset_index()
        .sort_values("total_inv_value", ascending=False)
    )

    grand_total   = agg["total_inv_value"].sum()
    total_rows    = len(df)
    total_groups  = len(agg)
    top25         = agg.head(25)
    top25_total   = top25["total_inv_value"].sum()

    console = Console()
    table = Table(
        title=f"Dead-Stock Exposure by Supplier × Location — {report_date}  (top {len(top25)} of {total_groups})",
        box=box.SIMPLE_HEAVY,
        show_footer=True,
        footer_style="bold",
    )

    def fmt_currency(v: float) -> str:
        return f"${v:,.2f}"

    table.add_column("Supplier ID",  footer="TOP 25",                       style="cyan",  no_wrap=True)
    table.add_column("Location ID",  footer="",                             style="white", no_wrap=True)
    table.add_column("Dead-Stock $", footer=fmt_currency(top25_total),      justify="right", style="red",    footer_style="bold red")
    table.add_column("SKU Count",    footer=str(int(top25["sku_count"].sum())), justify="right")

    for _, row in top25.iterrows():
        table.add_row(
            row["supplier_id"],
            row["location_id"],
            fmt_currency(row["total_inv_value"]),
            str(int(row["sku_count"])),
        )

    console.print()
    console.print(table)
    console.print(
        f"[bold]  Grand total (all {total_groups} groups): [red]{fmt_currency(grand_total)}[/red][/bold]"
        f"  across [dim]{total_rows:,} rows[/dim]\n"
    )


if __name__ == "__main__":
    main()
