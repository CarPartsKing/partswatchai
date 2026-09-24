#!/usr/bin/env python3
"""
scripts/vendor_return_allowance.py

Trailing-12-month received purchases by supplier with CPW vendor return
allowance targets (2% and 5% of purchases).

Usage:
    python scripts/vendor_return_allowance.py
"""

import sys
import os
from datetime import date

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_client


def fetch_all(query_fn, page_size: int = 1000) -> list[dict]:
    """Paginate through a Supabase query, returning all rows."""
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

    cutoff = (pd.Timestamp.now() - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    today_str = date.today().strftime("%B %d, %Y")

    print(f"Fetching purchase_orders received since {cutoff} …")
    po_rows = fetch_all(
        lambda lo, hi: (
            client.table("purchase_orders")
            .select("supplier_id,po_number,sku_id,qty_received,unit_cost")
            .gte("po_date", cutoff)
            .gt("qty_received", 0)
            .range(lo, hi)
        )
    )

    if not po_rows:
        print("No received purchase data found in the trailing 12 months.")
        return

    print(f"Fetching supplier_scores for names …")
    ss_rows = fetch_all(
        lambda lo, hi: (
            client.table("supplier_scores")
            .select("supplier_id,supplier_name,score_date")
            .range(lo, hi)
        )
    )

    # Most-recent name per supplier
    ss_df = pd.DataFrame(ss_rows) if ss_rows else pd.DataFrame(columns=["supplier_id", "supplier_name", "score_date"])
    if not ss_df.empty:
        ss_df = (
            ss_df.sort_values("score_date", ascending=False)
            .drop_duplicates("supplier_id")
            [["supplier_id", "supplier_name"]]
        )
    name_map: dict[str, str] = dict(zip(ss_df["supplier_id"], ss_df["supplier_name"])) if not ss_df.empty else {}

    # Aggregate
    po_df = pd.DataFrame(po_rows)
    po_df["qty_received"] = pd.to_numeric(po_df["qty_received"], errors="coerce").fillna(0)
    po_df["unit_cost"]    = pd.to_numeric(po_df["unit_cost"],    errors="coerce").fillna(0)
    po_df["line_spend"]   = po_df["qty_received"] * po_df["unit_cost"]

    agg = (
        po_df.groupby("supplier_id")
        .agg(
            purchases_12m=("line_spend",  "sum"),
            po_count     =("po_number",   "nunique"),
            sku_count    =("sku_id",      "nunique"),
        )
        .reset_index()
        .sort_values("purchases_12m", ascending=False)
    )

    agg["supplier_name"] = agg["supplier_id"].map(name_map).fillna("")
    agg["target_2pct"]   = agg["purchases_12m"] * 0.02
    agg["target_5pct"]   = agg["purchases_12m"] * 0.05

    # Totals
    total_purchases = agg["purchases_12m"].sum()
    total_pos       = agg["po_count"].sum()
    total_skus      = po_df["sku_id"].nunique()  # company-wide distinct SKUs

    # Render table
    console = Console()
    table = Table(
        title=f"Vendor Return Allowance — Trailing 12 Months (as of {today_str})",
        box=box.SIMPLE_HEAVY,
        show_footer=True,
        footer_style="bold",
    )

    def fmt_currency(v: float) -> str:
        return f"${v:,.2f}"

    def fmt_int(v: int) -> str:
        return f"{v:,}"

    table.add_column("Supplier ID",   footer="TOTAL",                              style="cyan",  no_wrap=True)
    table.add_column("Supplier Name", footer="",                                   style="white")
    table.add_column("Purchases 12M", footer=fmt_currency(total_purchases),        justify="right", style="green",  footer_style="bold green")
    table.add_column("PO Count",      footer=fmt_int(int(total_pos)),              justify="right")
    table.add_column("SKU Count",     footer=fmt_int(total_skus),                  justify="right")
    table.add_column("2% Target",     footer=fmt_currency(total_purchases * 0.02), justify="right", style="yellow", footer_style="bold yellow")
    table.add_column("5% Target",     footer=fmt_currency(total_purchases * 0.05), justify="right", style="yellow", footer_style="bold yellow")

    for _, row in agg.iterrows():
        table.add_row(
            row["supplier_id"],
            row["supplier_name"],
            fmt_currency(row["purchases_12m"]),
            fmt_int(int(row["po_count"])),
            fmt_int(int(row["sku_count"])),
            fmt_currency(row["target_2pct"]),
            fmt_currency(row["target_5pct"]),
        )

    console.print()
    console.print(table)
    console.print(
        f"[dim]  Rows: {len(po_rows):,} PO lines | "
        f"Suppliers: {len(agg):,} | "
        f"Cutoff: {cutoff}[/dim]\n"
    )


if __name__ == "__main__":
    main()
