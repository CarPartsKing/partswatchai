"""extract/autocube_transfers_pull.py — Inter-location transfer extraction stub.

DEPLOYMENT STATUS
-----------------
The [Transfer Locs] dimension on the Product cube is structurally present in
AutoCube_DTR_23160 but UNPOPULATED — [Loc From] and [Loc To] return ZERO
members.  All three extraction approaches were tested and return zero rows:

  1. Product cube XF measures (Qty XF, Ext Cost XF, Ext Price XF) with
     [Transfer Locs].[Loc From/To] WHERE slicer — zero rows across 90 days
     for all 27 active locations.

  2. Sales Detail INTERCO tran codes (-I suffix e.g. SL-I, RT-I) — only 6
     tran codes exist in this deployment: SL, RT, CORT, COSL, WREX, WRRT.
     None are INTERCO.

  3. T-prefix invoice numbers (T{from}{to} format) — zero T-prefix invoices
     across 111k rows sampled from the Sales Detail cube.

This module runs as a no-op in the nightly pipeline and exits cleanly.  It
will become functional if Autologue enables transfer tracking in Autocube.

MODES
-----
    python -m extract.autocube_transfers_pull [--dry-run]
        Logs status message and exits 0.  No network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)


def run_transfers_extract(dry_run: bool = False, lookback_days: int = 90) -> int:
    """No-op stub: Transfer Locs unpopulated in this Autocube deployment."""
    log.info("=" * 60)
    log.info("  AUTOCUBE TRANSFERS EXTRACT%s", " [DRY RUN]" if dry_run else "")
    log.info("  Status: Transfer Locs unpopulated in AutoCube_DTR_23160")
    log.info("  [Transfer Locs].[Loc From/To] return 0 members.")
    log.info("  No INTERCO tran codes exist. No T-prefix invoices found.")
    log.info("  0 rows extracted. Skipping Supabase write.")
    log.info("=" * 60)
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Autocube transfer extraction (no-op: Transfer Locs unpopulated)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=90, metavar="N")
    args = parser.parse_args()
    return run_transfers_extract(dry_run=args.dry_run, lookback_days=args.lookback_days)


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
