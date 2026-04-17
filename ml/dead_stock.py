"""
ml/dead_stock.py — Capital-weighted dead stock detection and liquidation ranking.

Identifies inventory that is tying up capital with little prospect of near-term
sale.  Unlike a simple "days since last sale" check, every SKU×location pair
is scored by the *dollar value* of the capital tied up, adjusted for how long
it has sat and how infrequently it sells.  The result is a ranked liquidation
priority list: the most expensive, slowest-moving stock appears first.

SCORING
    For every (sku_id, location_id) pair with inventory on hand:

    unit_cost           — most recent received-PO unit cost for the SKU
                          (falls back to DEFAULT_UNIT_COST if no PO exists)
    total_inv_value     — qty_on_hand × unit_cost
    days_since_sale     — calendar days since the last transaction AT THIS
                          location (not the global sku_master value, which
                          would hide locations where the SKU never moves)
    sale_frequency      — distinct transaction dates in the last 365 days
                          at this location  (0 = never sold here in a year)
    dead_stock_score    — total_inv_value
                          × (days_since_sale / LOOKBACK_DAYS)
                          / (sale_frequency + 1)

    Higher score = more capital locked up in slower, older inventory.

CLASSIFICATION (per SKU×location)
    LIQUIDATE  — days_since_sale ≥ 180  AND  (sale_frequency ≤ LOW_FREQ OR
                 total_inv_value ≥ LIQUIDATE_MIN_VALUE)
    MARKDOWN   — 90 ≤ days_since_sale < 180
    MONITOR    — 60 ≤ days_since_sale < 90
    HEALTHY    — days_since_sale < 60  (or actively selling)

    Override to HEALTHY if sale_frequency ≥ HIGH_FREQ_OVERRIDE regardless
    of days — fast-movers that happen to have a temporary gap are not dead.

LIQUIDATION ACTIONS (LIQUIDATE candidates only)
    Return to vendor   — supplier identified in purchase_orders history
    Write off          — total_inv_value < WRITE_OFF_THRESHOLD  and no vendor
    Markdown           — abc_class A/B  or  avg_weekly_units > HIGH_VELOCITY
    Liquidate / delist — everything else

is_dead_stock FLAG
    sku_master.is_dead_stock is set TRUE for any SKU with at least one
    LIQUIDATE location.  It is cleared (FALSE) for SKUs that no longer
    qualify.  This feeds the nightly engine.alerts DEAD_STOCK generator.

WEEKLY REPORT
    A structured summary is printed to console and can optionally be written
    to a JSON file with --report-file <path>.

USAGE
    python -m ml.dead_stock               # full run
    python -m ml.dead_stock --dry-run     # score only, no DB writes
    python -m ml.dead_stock --report-file /tmp/dead_stock_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

setup_logging()
log     = get_logger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOKBACK_DAYS:          int   = 365    # window for sale_frequency count
LIQUIDATE_DAYS:         int   = 180    # days without sale → LIQUIDATE candidate
MARKDOWN_DAYS:          int   = 90     # days without sale → MARKDOWN
MONITOR_DAYS:           int   = 60     # days without sale → MONITOR

LOW_FREQ_THRESHOLD:     int   = 2      # ≤ this many sales/year = low frequency
HIGH_FREQ_OVERRIDE:     int   = 12     # ≥ this many sales/year = always HEALTHY
LIQUIDATE_MIN_VALUE:    float = 50.0   # minimum inv value ($) to flag LIQUIDATE
WRITE_OFF_THRESHOLD:    float = 25.0   # below this ($) with no vendor → write off
HIGH_VELOCITY_UNITS:    float = 1.0    # avg_weekly_units above which → markdown
DEFAULT_UNIT_COST:      float = 10.0   # fallback when no PO cost found

PAGE_SIZE: int = 1_000

# ---------------------------------------------------------------------------
# Retry configuration for transient Supabase errors (57014 statement
# timeout, dropped HTTP/2 streams, read timeouts, …).  Mirrors the pattern
# used in engine/reorder.py and ml/forecast_rolling.py.
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = 5
_RETRY_DELAY: float = 5.0
_RETRYABLE_TOKENS: tuple[str, ...] = (
    "57014",
    "statement timeout",
    "canceling statement",
    "ConnectionTerminated",
    "RemoteProtocolError",
    "ReadTimeout",
    "ReadError",
    "ProtocolError",
    "RemoteDisconnected",
    "Server disconnected",
    "Connection aborted",
)


def _is_retryable_error(exc: Exception) -> bool:
    """True if exc looks like a Supabase timeout / dropped-connection error."""
    blob = type(exc).__name__ + " " + str(exc)
    return any(tok in blob for tok in _RETRYABLE_TOKENS)


def _get_fresh_client() -> Any:
    """Return a brand-new Supabase client (bypasses lru_cache when available).

    Only ImportError / AttributeError fall back to the cached client — any
    other failure (auth, network, config) must surface so we don't paper
    over a real outage by reusing a known-bad cached connection.
    """
    try:
        from db.connection import get_new_client  # type: ignore[attr-defined]
        return get_new_client()
    except (ImportError, AttributeError):
        return get_client()

# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------

CLASS_LIQUIDATE = "LIQUIDATE"
CLASS_MARKDOWN  = "MARKDOWN"
CLASS_MONITOR   = "MONITOR"
CLASS_HEALTHY   = "HEALTHY"

ACTION_RETURN    = "Return to vendor"
ACTION_MARKDOWN  = "Markdown — price reduction"
ACTION_WRITEOFF  = "Write off"
ACTION_LIQUIDATE = "Liquidate / delist"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InventoryPosition:
    """All data needed to score one (sku_id, location_id) pair."""
    sku_id:            str
    location_id:       str
    qty_on_hand:       float
    unit_cost:         float
    days_since_sale:   int          # days since last tx at this location
    sale_frequency:    int          # distinct sale days in last 365 days
    abc_class:         str          # A / B / C
    avg_weekly_units:  float        # from sku_master
    supplier_id:       str | None   # from latest purchase order
    part_category:     str
    sub_category:      str

    @property
    def total_inv_value(self) -> float:
        return self.qty_on_hand * self.unit_cost

    @property
    def dead_stock_score(self) -> float:
        """Capital-weighted staleness score.  Higher = worse."""
        return (
            self.total_inv_value
            * (self.days_since_sale / LOOKBACK_DAYS)
            / (self.sale_frequency + 1)
        )


@dataclass
class ScoredPosition:
    """Classification output for one (sku_id, location_id) pair."""
    sku_id:           str
    location_id:      str
    classification:   str
    action:           str
    dead_stock_score: float
    total_inv_value:  float
    qty_on_hand:      float
    unit_cost:        float
    days_since_sale:  int
    sale_frequency:   int
    abc_class:        str
    avg_weekly_units: float
    supplier_id:      str | None
    part_category:    str
    sub_category:     str


# ---------------------------------------------------------------------------
# Supabase pagination helper
# ---------------------------------------------------------------------------

def _paginate(
    client_holder: list,
    table:  str,
    select: str,
    filters:     dict | None = None,
    gte_filters: dict | None = None,
    eq_bool:     dict | None = None,
    order_col:   str | None  = None,
    order_desc:  bool        = False,
) -> list[dict]:
    """Paginated fetch with retry + reconnect on transient Supabase errors.

    ``client_holder`` is a single-element list; the reference is replaced
    in-place when a 57014 / dropped-connection error forces a reconnect.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page: list[dict] | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                q = client_holder[0].table(table).select(select)
                for col, val in (filters  or {}).items():
                    q = q.eq(col, val)
                for col, val in (gte_filters or {}).items():
                    q = q.gte(col, val)
                for col, val in (eq_bool or {}).items():
                    q = q.eq(col, val)
                if order_col:
                    q = q.order(order_col, desc=order_desc)
                page = q.range(offset, offset + PAGE_SIZE - 1).execute().data or []
                break
            except Exception as exc:
                if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                    log.warning(
                        "  %s fetch retry %d/%d (offset=%d): %s — reconnecting in %.0fs …",
                        table, attempt, _MAX_RETRIES, offset,
                        type(exc).__name__, _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                    client_holder[0] = _get_fresh_client()
                    continue
                raise
        assert page is not None
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _fetch_latest_inventory(client_holder: list) -> dict[tuple[str, str], dict]:
    """Latest inventory snapshot per (sku_id, location_id)."""
    rows = _paginate(client_holder, "inventory_snapshots",
                     "sku_id,location_id,snapshot_date,qty_on_hand")
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["sku_id"], r["location_id"])
        if key not in latest or r["snapshot_date"] > latest[key]["snapshot_date"]:
            latest[key] = r
    return {k: v for k, v in latest.items() if (v.get("qty_on_hand") or 0) > 0}


def _fetch_sku_master(client_holder: list) -> dict[str, dict]:
    """sku_id → sku_master row."""
    rows = _paginate(client_holder, "sku_master",
                     "sku_id,abc_class,last_sale_date,avg_weekly_units,"
                     "is_dead_stock,part_category,sub_category")
    return {r["sku_id"]: r for r in rows}


def _fetch_unit_costs(client_holder: list) -> dict[str, float]:
    """sku_id → most recent unit_cost from received purchase orders."""
    rows = _paginate(client_holder, "purchase_orders",
                     "sku_id,unit_cost,actual_delivery_date",
                     filters={"status": "received"},
                     order_col="actual_delivery_date", order_desc=True)
    costs: dict[str, float] = {}
    for r in rows:
        sid = r.get("sku_id")
        if sid and sid not in costs and r.get("unit_cost") is not None:
            costs[sid] = float(r["unit_cost"])
    return costs


def _fetch_supplier_map(client_holder: list) -> dict[str, str]:
    """sku_id → supplier_id from the most recent received PO."""
    rows = _paginate(client_holder, "purchase_orders",
                     "sku_id,supplier_id,actual_delivery_date",
                     filters={"status": "received"},
                     order_col="actual_delivery_date", order_desc=True)
    suppliers: dict[str, str] = {}
    for r in rows:
        sid = r.get("sku_id")
        sup = r.get("supplier_id")
        if sid and sup and sid not in suppliers:
            suppliers[sid] = sup
    return suppliers


def _fetch_location_sales(
    client_holder: list,
    today:  date,
) -> tuple[dict[tuple[str, str], date], dict[tuple[str, str], int]]:
    """Per-(sku_id, location_id): last sale date and 365-day sale frequency.

    Only the most-recent ``LOOKBACK_DAYS`` (365) of sales_transactions are
    fetched.  Dead stock detection only needs recent sale history; pulling
    the full ~8 M-row history triggers Supabase statement timeouts (57014).
    Pagination is wrapped in retry/reconnect via ``_paginate``.

    Returns
    -------
    last_sale_map  : (sku_id, location_id) → most recent transaction_date
    frequency_map  : (sku_id, location_id) → count of distinct sale days
    """
    cutoff = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    log.info("  Fetching sales_transactions since %s (last %d days) …",
             cutoff, LOOKBACK_DAYS)
    rows   = _paginate(client_holder, "sales_transactions",
                       "sku_id,location_id,transaction_date",
                       gte_filters={"transaction_date": cutoff})
    log.info("  Fetched %d transaction row(s) within lookback window.", len(rows))

    last_sale_map: dict[tuple[str, str], date]  = {}
    freq_dates:    dict[tuple[str, str], set[str]] = defaultdict(set)

    for r in rows:
        key = (r["sku_id"], r["location_id"])
        d   = str(r.get("transaction_date", ""))[:10]
        if not d:
            continue
        freq_dates[key].add(d)
        tx_date = date.fromisoformat(d)
        if key not in last_sale_map or tx_date > last_sale_map[key]:
            last_sale_map[key] = tx_date

    frequency_map = {k: len(v) for k, v in freq_dates.items()}
    return last_sale_map, frequency_map


# ---------------------------------------------------------------------------
# Scoring and classification
# ---------------------------------------------------------------------------

def _classify(pos: InventoryPosition) -> tuple[str, str]:
    """Return (classification, action) for one inventory position."""
    days  = pos.days_since_sale
    freq  = pos.sale_frequency
    value = pos.total_inv_value

    # Fast movers are always healthy regardless of date gap
    if freq >= HIGH_FREQ_OVERRIDE:
        return CLASS_HEALTHY, ""

    if days < MONITOR_DAYS:
        return CLASS_HEALTHY, ""

    if days < MARKDOWN_DAYS:
        return CLASS_MONITOR, ""

    if days < LIQUIDATE_DAYS:
        return CLASS_MARKDOWN, ACTION_MARKDOWN

    # LIQUIDATE candidates — determine recommended action
    # Low total value → write off immediately
    if value < WRITE_OFF_THRESHOLD:
        return CLASS_LIQUIDATE, ACTION_WRITEOFF

    # Supplier exists → return to vendor first choice
    if pos.supplier_id:
        return CLASS_LIQUIDATE, ACTION_RETURN

    # High velocity SKU (was fast before, now slow) → markdown
    if pos.avg_weekly_units >= HIGH_VELOCITY_UNITS or pos.abc_class in ("A", "B"):
        return CLASS_LIQUIDATE, ACTION_MARKDOWN

    return CLASS_LIQUIDATE, ACTION_LIQUIDATE


def score_positions(
    inventory:    dict[tuple[str, str], dict],
    sku_master:   dict[str, dict],
    unit_costs:   dict[str, float],
    suppliers:    dict[str, str],
    last_sale_map: dict[tuple[str, str], date],
    freq_map:      dict[tuple[str, str], int],
    today:         date,
) -> list[ScoredPosition]:
    """Build and score every inventory position.  Returns all positions."""
    results: list[ScoredPosition] = []

    for (sku_id, loc_id), snap in inventory.items():
        sku = sku_master.get(sku_id, {})
        qty = float(snap.get("qty_on_hand") or 0)
        if qty <= 0:
            continue

        unit_cost = unit_costs.get(sku_id, DEFAULT_UNIT_COST)
        last_sale = last_sale_map.get((sku_id, loc_id))
        # Fall back to global last_sale_date if no location-level transaction
        if last_sale is None:
            global_ls = sku.get("last_sale_date")
            last_sale = date.fromisoformat(global_ls) if global_ls else None

        days_since = (today - last_sale).days if last_sale else LOOKBACK_DAYS
        freq       = freq_map.get((sku_id, loc_id), 0)

        pos = InventoryPosition(
            sku_id           = sku_id,
            location_id      = loc_id,
            qty_on_hand      = qty,
            unit_cost        = unit_cost,
            days_since_sale  = days_since,
            sale_frequency   = freq,
            abc_class        = sku.get("abc_class") or "C",
            avg_weekly_units = float(sku.get("avg_weekly_units") or 0),
            supplier_id      = suppliers.get(sku_id),
            part_category    = sku.get("part_category") or "",
            sub_category     = sku.get("sub_category") or "",
        )
        cls, action = _classify(pos)

        results.append(ScoredPosition(
            sku_id           = sku_id,
            location_id      = loc_id,
            classification   = cls,
            action           = action,
            dead_stock_score = pos.dead_stock_score,
            total_inv_value  = pos.total_inv_value,
            qty_on_hand      = qty,
            unit_cost        = unit_cost,
            days_since_sale  = days_since,
            sale_frequency   = freq,
            abc_class        = pos.abc_class,
            avg_weekly_units = pos.avg_weekly_units,
            supplier_id      = pos.supplier_id,
            part_category    = pos.part_category,
            sub_category     = pos.sub_category,
        ))

    # Ranked: highest dead_stock_score first
    results.sort(key=lambda r: r.dead_stock_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def _upsert_with_retry(
    client_holder: list,
    table: str,
    rows: list[dict],
    on_conflict: str,
) -> None:
    """Upsert rows with retry/reconnect on transient Supabase errors."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client_holder[0].table(table).upsert(
                rows, on_conflict=on_conflict,
            ).execute()
            return
        except Exception as exc:
            if _is_retryable_error(exc) and attempt < _MAX_RETRIES:
                log.warning(
                    "  upsert retry %d/%d (%d rows): %s — reconnecting in %.0fs …",
                    attempt, _MAX_RETRIES, len(rows),
                    type(exc).__name__, _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
                client_holder[0] = _get_fresh_client()
                continue
            raise


def _update_is_dead_stock(
    client_holder: list,
    scored:        list[ScoredPosition],
    dry_run:       bool,
) -> tuple[int, int]:
    """Set is_dead_stock = TRUE for LIQUIDATE SKUs; clear for HEALTHY SKUs.

    Returns (set_true_count, set_false_count).
    """
    liquidate_skus: set[str] = {
        s.sku_id for s in scored if s.classification == CLASS_LIQUIDATE
    }
    healthy_skus: set[str] = {
        s.sku_id for s in scored
        if s.classification == CLASS_HEALTHY
        and s.sku_id not in liquidate_skus
    }

    set_true  = [{"sku_id": s, "is_dead_stock": True}  for s in liquidate_skus]
    set_false = [{"sku_id": s, "is_dead_stock": False} for s in healthy_skus]

    if dry_run:
        log.info("DRY RUN — would set is_dead_stock=TRUE  for %d SKU(s)", len(set_true))
        log.info("DRY RUN — would set is_dead_stock=FALSE for %d SKU(s)", len(set_false))
        return len(set_true), len(set_false)

    PAGE = 100
    for batch in [set_true[i:i+PAGE] for i in range(0, len(set_true), PAGE)]:
        _upsert_with_retry(client_holder, "sku_master", batch, "sku_id")
    for batch in [set_false[i:i+PAGE] for i in range(0, len(set_false), PAGE)]:
        _upsert_with_retry(client_holder, "sku_master", batch, "sku_id")

    log.info("is_dead_stock flag updated: TRUE=%d  FALSE=%d", len(set_true), len(set_false))
    return len(set_true), len(set_false)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _cls_badge(cls: str) -> str:
    return {
        CLASS_LIQUIDATE: "[red]LIQUIDATE[/red]",
        CLASS_MARKDOWN:  "[yellow]MARKDOWN[/yellow]",
        CLASS_MONITOR:   "[cyan]MONITOR[/cyan]",
        CLASS_HEALTHY:   "[green]HEALTHY[/green]",
    }.get(cls, cls)


def _render_liquidation_table(positions: list[ScoredPosition], top_n: int = 20) -> None:
    """Print the ranked liquidation priority table to console."""
    candidates = [p for p in positions if p.classification in (CLASS_LIQUIDATE, CLASS_MARKDOWN)]
    if not candidates:
        console.print(Panel("[green]No liquidation candidates found.[/green]",
                            title="Liquidation Priority", border_style="green"))
        return

    tbl = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    tbl.add_column("#",            width=3,  style="dim")
    tbl.add_column("SKU",          width=14)
    tbl.add_column("Location",     width=8)
    tbl.add_column("Class",        width=11)
    tbl.add_column("Inv Value",    width=10, justify="right")
    tbl.add_column("Qty",          width=6,  justify="right")
    tbl.add_column("Unit Cost",    width=9,  justify="right")
    tbl.add_column("Days Idle",    width=9,  justify="right")
    tbl.add_column("Sales/yr",     width=8,  justify="right")
    tbl.add_column("Score",        width=8,  justify="right")
    tbl.add_column("Action",       width=22)

    for i, p in enumerate(candidates[:top_n], 1):
        tbl.add_row(
            str(i),
            p.sku_id,
            p.location_id,
            _cls_badge(p.classification),
            f"${p.total_inv_value:,.2f}",
            f"{int(p.qty_on_hand)}",
            f"${p.unit_cost:.2f}",
            str(p.days_since_sale),
            str(p.sale_frequency),
            f"{p.dead_stock_score:.1f}",
            p.action or "—",
        )

    title = f"Liquidation Priority  ({len(candidates)} candidate(s)"
    if len(candidates) > top_n:
        title += f", showing top {top_n}"
    title += ")"
    console.print(Panel(tbl, title=title, border_style="red"))


def _build_report(
    scored:     list[ScoredPosition],
    today:      date,
    elapsed_s:  float,
    set_true:   int,
    set_false:  int,
    dry_run:    bool,
) -> dict:
    """Build the structured weekly performance report dict."""
    by_class: dict[str, list[ScoredPosition]] = defaultdict(list)
    for p in scored:
        by_class[p.classification].append(p)

    liquidate = by_class[CLASS_LIQUIDATE]
    markdown  = by_class[CLASS_MARKDOWN]
    monitor   = by_class[CLASS_MONITOR]

    total_at_risk = sum(p.total_inv_value for p in liquidate + markdown)
    liquidate_val = sum(p.total_inv_value for p in liquidate)
    markdown_val  = sum(p.total_inv_value for p in markdown)

    action_counts: dict[str, int] = defaultdict(int)
    for p in liquidate:
        action_counts[p.action] += 1

    return {
        "report_date":     today.isoformat(),
        "generated_at":    __import__("datetime").datetime.now().isoformat(),
        "dry_run":         dry_run,
        "elapsed_s":       round(elapsed_s, 2),
        "summary": {
            "total_positions_scored": len(scored),
            "liquidate_count":   len(liquidate),
            "markdown_count":    len(markdown),
            "monitor_count":     len(monitor),
            "healthy_count":     len(by_class[CLASS_HEALTHY]),
            "capital_at_risk":   round(total_at_risk, 2),
            "liquidate_value":   round(liquidate_val, 2),
            "markdown_value":    round(markdown_val, 2),
            "is_dead_stock_set": set_true,
            "is_dead_stock_cleared": set_false,
        },
        "action_breakdown": dict(action_counts),
        "top_liquidate": [
            {
                "sku_id":         p.sku_id,
                "location_id":    p.location_id,
                "total_inv_value": round(p.total_inv_value, 2),
                "days_since_sale": p.days_since_sale,
                "sale_frequency":  p.sale_frequency,
                "action":          p.action,
                "dead_stock_score": round(p.dead_stock_score, 2),
            }
            for p in sorted(liquidate, key=lambda x: x.total_inv_value, reverse=True)[:20]
        ],
    }


def _render_summary_panel(report: dict, dry_run: bool) -> None:
    s = report["summary"]
    dry_tag = "  [yellow]DRY RUN[/yellow]" if dry_run else ""

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column("Metric",  style="bold", width=32)
    tbl.add_column("Value",   justify="right", width=16)

    rows = [
        ("Positions scored",      str(s["total_positions_scored"])),
        ("LIQUIDATE",             f"[red]{s['liquidate_count']}[/red]"),
        ("MARKDOWN",              f"[yellow]{s['markdown_count']}[/yellow]"),
        ("MONITOR",               f"[cyan]{s['monitor_count']}[/cyan]"),
        ("HEALTHY",               f"[green]{s['healthy_count']}[/green]"),
        ("Capital at risk (total)", f"[red]${s['capital_at_risk']:,.2f}[/red]"),
        ("  — Liquidate value",   f"${s['liquidate_value']:,.2f}"),
        ("  — Markdown value",    f"${s['markdown_value']:,.2f}"),
        ("is_dead_stock set TRUE", str(s["is_dead_stock_set"])),
        ("is_dead_stock cleared",  str(s["is_dead_stock_cleared"])),
    ]
    for label, val in rows:
        tbl.add_row(label, val)

    if report["action_breakdown"]:
        tbl.add_row("", "")
        for action, count in report["action_breakdown"].items():
            tbl.add_row(f"  {action}", str(count))

    console.print(Panel(tbl,
        title=f"Dead Stock Summary — {report['report_date']}{dry_tag}",
        border_style="red" if s["liquidate_count"] > 0 else "green"))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_dead_stock(
    dry_run:     bool = False,
    report_file: str | None = None,
) -> int:
    """Execute the full capital-weighted dead stock detection pipeline.

    Args:
        dry_run:     When True, score and classify everything but skip all
                     DB writes and is_dead_stock flag updates.
        report_file: Optional file path to write the JSON performance report.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    t0  = time.monotonic()
    banner = "=" * 64
    log.info(banner)
    log.info("partswatch-ai — ml.dead_stock")
    log.info(
        "  liquidate_days=%d  markdown_days=%d  monitor_days=%d  "
        "low_freq=%d  write_off=$%.0f",
        LIQUIDATE_DAYS, MARKDOWN_DAYS, MONITOR_DAYS,
        LOW_FREQ_THRESHOLD, WRITE_OFF_THRESHOLD,
    )
    log.info(banner)

    if dry_run:
        log.info("DRY RUN — no database writes will be made.")

    today = date.today()
    log.info("Analysis date: %s", today.isoformat())

    # ── Connect ───────────────────────────────────────────────────────
    # Wrap the client in a single-element list so retry helpers can swap
    # in a freshly minted client when Supabase drops the connection.
    try:
        client_holder: list = [_get_fresh_client()]
    except Exception:
        log.exception("Failed to initialise Supabase client.")
        return 1

    # ── Fetch all data ────────────────────────────────────────────────
    log.info("Fetching inventory positions …")
    try:
        inventory    = _fetch_latest_inventory(client_holder)
        sku_master   = _fetch_sku_master(client_holder)
        unit_costs   = _fetch_unit_costs(client_holder)
        suppliers    = _fetch_supplier_map(client_holder)
        last_sale_map, freq_map = _fetch_location_sales(client_holder, today)
    except Exception:
        log.exception("Data fetch failed.")
        return 1

    log.info("  Inventory positions (on-hand > 0):  %d", len(inventory))
    log.info("  SKUs in master:                     %d", len(sku_master))
    log.info("  SKUs with known unit cost:          %d", len(unit_costs))
    log.info("  SKUs with known supplier:           %d", len(suppliers))
    log.info("  SKU×location pairs with sales data: %d", len(freq_map))

    if not inventory:
        log.warning("No inventory positions found — nothing to score.")
        return 0

    # ── Score and classify ────────────────────────────────────────────
    log.info("Scoring %d inventory position(s) …", len(inventory))
    scored = score_positions(
        inventory, sku_master, unit_costs, suppliers,
        last_sale_map, freq_map, today,
    )

    by_class: dict[str, list[ScoredPosition]] = defaultdict(list)
    for p in scored:
        by_class[p.classification].append(p)

    log.info("Classification results:")
    for cls in (CLASS_LIQUIDATE, CLASS_MARKDOWN, CLASS_MONITOR, CLASS_HEALTHY):
        log.info("  %-12s  %d", cls, len(by_class[cls]))

    liquidate_value = sum(p.total_inv_value for p in by_class[CLASS_LIQUIDATE])
    markdown_value  = sum(p.total_inv_value for p in by_class[CLASS_MARKDOWN])
    log.info("Capital at risk:  LIQUIDATE $%.2f  MARKDOWN $%.2f",
             liquidate_value, markdown_value)

    # ── Log top candidates ────────────────────────────────────────────
    liquidate_sorted = sorted(
        by_class[CLASS_LIQUIDATE],
        key=lambda p: p.total_inv_value,
        reverse=True,
    )
    if liquidate_sorted:
        log.info("-" * 64)
        log.info("Top LIQUIDATE candidates (by inventory value):")
        for i, p in enumerate(liquidate_sorted[:10], 1):
            log.info(
                "  %2d. %-14s @%-8s  $%8.2f  %3d days  %2d sales/yr  → %s",
                i, p.sku_id, p.location_id,
                p.total_inv_value, p.days_since_sale,
                p.sale_frequency, p.action,
            )
        log.info("-" * 64)

    # ── Rich console table ────────────────────────────────────────────
    _render_liquidation_table(scored)

    # ── Update is_dead_stock flag ─────────────────────────────────────
    try:
        set_true, set_false = _update_is_dead_stock(client_holder, scored, dry_run)
    except Exception:
        log.exception("is_dead_stock update failed (non-fatal).")
        set_true = set_false = 0

    # ── Build and render summary ──────────────────────────────────────
    elapsed_s = time.monotonic() - t0
    report    = _build_report(scored, today, elapsed_s, set_true, set_false, dry_run)

    _render_summary_panel(report, dry_run)

    # ── Optional JSON report file ─────────────────────────────────────
    if report_file:
        try:
            with open(report_file, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=str)
            log.info("Weekly performance report written → %s", report_file)
        except Exception:
            log.exception("Failed to write report file (non-fatal).")

    log.info(banner)
    log.info(
        "Dead stock analysis complete  (%.1fs)  "
        "LIQUIDATE=%d  MARKDOWN=%d  MONITOR=%d  HEALTHY=%d",
        elapsed_s,
        len(by_class[CLASS_LIQUIDATE]),
        len(by_class[CLASS_MARKDOWN]),
        len(by_class[CLASS_MONITOR]),
        len(by_class[CLASS_HEALTHY]),
    )
    log.info(banner)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capital-weighted dead stock detection and liquidation ranking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Score and classify without writing to the database.")
    p.add_argument("--report-file", metavar="PATH",
                   help="Write the JSON performance report to this file path.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_dead_stock(dry_run=args.dry_run, report_file=args.report_file)


if __name__ == "__main__":
    sys.exit(main())
