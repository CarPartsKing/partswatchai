"""engine/transfer.py — Inter-location stock transfer recommendation logic.

Called by engine/reorder.py before generating any external PO recommendation.
Transfers always take priority over outside purchase orders when a nearby
location holds excess inventory of the same SKU.

This module is **pure computation** — it performs no database I/O.  All data
is received as pre-fetched Python dicts from reorder.py.

EXCESS DETECTION
----------------
A source location is a candidate transfer supplier if, after giving away the
requested quantity, it would still retain enough stock to cover:

    minimum_to_keep = avg_daily_forecast × (avg_lead_time_days
                                            + SAFETY_BUFFER_DAYS
                                            + TRANSFER_EXTRA_PADDING)

where SAFETY_BUFFER_DAYS = 7 (the same reorder buffer used everywhere) and
TRANSFER_EXTRA_PADDING = 7 (an additional week so we never strip a source
location to its own reorder point).

When avg_daily_forecast at the source is zero (no demand), we retain 50 % of
on-hand stock as a safety reserve before allowing a transfer.

The best source is the location with the most days_of_supply_remaining after
the transfer, which minimises the risk of creating a stockout at the source.
"""

from __future__ import annotations

from utils.logging_config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — all overridable by callers via kwargs if needed
# ---------------------------------------------------------------------------

SAFETY_BUFFER_DAYS: int = 7
"""Days of buffer added to avg_lead_time_days in the standard reorder threshold."""

TRANSFER_EXTRA_PADDING: int = 7
"""Additional days a source location must retain *beyond* its own reorder
threshold before any inventory can be transferred away."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_transfer_source(
    sku_id: str,
    needing_location: str,
    qty_needed: float,
    avg_lead_time_days: float,
    inventory_summary: dict[tuple[str, str], dict],
) -> dict | None:
    """Find the best location with excess stock that can fulfil a transfer need.

    Scans all entries in ``inventory_summary`` for the given ``sku_id``
    (excluding ``needing_location``) and returns the candidate with the most
    days of supply remaining after the transfer, or ``None`` if no suitable
    source exists.

    Args:
        sku_id:            SKU that needs restocking.
        needing_location:  Location that is short; excluded from candidates.
        qty_needed:        Units required to cover the reorder at the needing
                           location.  Must be > 0.
        avg_lead_time_days: Supplier lead time for this SKU (used only for
                            logging context; source locations use their own
                            stored avg_lead_time_days from the summary dict).
        inventory_summary: Dict keyed by ``(sku_id, location_id)`` → summary
                           dict containing at minimum:
                             - qty_on_hand           (float)
                             - avg_daily_forecast    (float)
                             - days_of_supply_remaining (float)
                             - avg_lead_time_days    (float)

    Returns:
        A dict with keys ``location_id``, ``transferable_qty``, and
        ``days_of_supply_remaining`` for the winning source, or ``None``
        if no candidate can satisfy the request.
    """
    if qty_needed <= 0:
        return None

    candidates: list[dict] = []

    for (s_sku, s_loc), summary in inventory_summary.items():
        if s_sku != sku_id or s_loc == needing_location:
            continue

        qty_on_hand: float       = float(summary.get("qty_on_hand", 0) or 0)
        avg_daily: float         = float(summary.get("avg_daily_forecast", 0) or 0)
        src_lead_days: float     = float(
            summary.get("avg_lead_time_days", avg_lead_time_days) or avg_lead_time_days
        )

        if avg_daily > 0:
            # Units the source must keep to cover its own demand through
            # its next reorder cycle, plus an extra week of padding.
            minimum_to_keep = avg_daily * (
                src_lead_days + SAFETY_BUFFER_DAYS + TRANSFER_EXTRA_PADDING
            )
            transferable_qty = max(0.0, qty_on_hand - minimum_to_keep)
        else:
            # Zero-demand source: give away at most half the on-hand stock.
            transferable_qty = qty_on_hand * 0.5

        if transferable_qty < qty_needed:
            log.debug(
                "Transfer candidate %s/%s rejected: transferable=%.2f < needed=%.2f",
                sku_id, s_loc, transferable_qty, qty_needed,
            )
            continue

        # Post-transfer days of supply at the source — used to rank candidates.
        remaining_after = qty_on_hand - qty_needed
        src_days_after = (
            remaining_after / avg_daily if avg_daily > 0 else float("inf")
        )

        candidates.append({
            "location_id":              s_loc,
            "transferable_qty":         round(transferable_qty, 4),
            "days_of_supply_remaining": round(
                float(summary.get("days_of_supply_remaining", 0) or 0), 2
            ),
            "days_after_transfer":      src_days_after,
        })

    if not candidates:
        return None

    # Best candidate = most days of supply remaining after the transfer,
    # which minimises the risk of creating a new shortage at the source.
    best = max(candidates, key=lambda c: c["days_after_transfer"])
    log.debug(
        "Transfer source for %s → %s: from %s (%.2f transferable, %.1f days after)",
        sku_id, needing_location,
        best["location_id"], best["transferable_qty"], best["days_after_transfer"],
    )
    return best
