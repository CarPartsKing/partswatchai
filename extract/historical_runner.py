"""Historical load runner - designed to run as a Replit workflow.

Processes all remaining weekly chunks, one at a time, with memory cleanup
between chunks. Tracks progress in /tmp/historical_progress.json so it
can resume if restarted.
"""
import gc
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract.autocube_pull import (
    _generate_chunk_ranges,
    _HISTORY_START,
    get_client,
    load_column_map,
    MDX_MONTHLY_RANGE,
    _load_to_supabase,
    SecurityError,
    setup_logging,
)

import config
import logging

setup_logging()
log = logging.getLogger(__name__)

PROGRESS_FILE = Path("/tmp/historical_progress.json")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_chunks": [], "total_rows": 0, "failed_chunks": []}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def main():
    progress = load_progress()
    completed = set(progress["completed_chunks"])
    total_rows = progress["total_rows"]
    failed = progress["failed_chunks"]

    end_date = date.today() - timedelta(days=1)
    all_chunks = _generate_chunk_ranges(_HISTORY_START, end_date)

    remaining = [(i, l, s, e) for i, (l, s, e) in enumerate(all_chunks)
                 if i not in completed]

    log.info("=" * 60)
    log.info("  HISTORICAL LOAD RUNNER")
    log.info("  Total chunks: %d | Completed: %d | Remaining: %d",
             len(all_chunks), len(completed), len(remaining))
    log.info("  Rows loaded so far: %d", total_rows)
    log.info("=" * 60)

    if not remaining:
        log.info("All chunks already completed!")
        return 0

    column_map = load_column_map()
    cube = config.AUTOCUBE_CUBE

    client = get_client()
    client.connect()

    t0 = time.perf_counter()

    for pos, (idx, label, start_key, end_key) in enumerate(remaining, 1):
        chunk_t0 = time.perf_counter()
        log.info("[CHUNK %d/%d] #%d: %s (keys %s–%s)",
                 pos, len(remaining), idx, label, start_key, end_key)

        mdx = MDX_MONTHLY_RANGE.format(
            cube=cube, start_key=start_key, end_key=end_key
        )

        try:
            rows = client.execute_mdx(mdx)
        except SecurityError:
            log.exception("Security violation — aborting.")
            return 1
        except Exception:
            log.exception("Failed to extract chunk %s.", label)
            failed.append(label)
            progress["failed_chunks"] = failed
            save_progress(progress)
            continue

        chunk_rows = 0
        if rows:
            try:
                chunk_rows = _load_to_supabase(rows, column_map, dry_run=False)
            except Exception:
                log.exception("Failed to load chunk %s.", label)
                failed.append(label)
                progress["failed_chunks"] = failed
                save_progress(progress)
                del rows
                gc.collect()
                continue

        del rows
        gc.collect()

        total_rows += chunk_rows
        completed.add(idx)
        progress["completed_chunks"] = sorted(completed)
        progress["total_rows"] = total_rows
        save_progress(progress)

        elapsed = time.perf_counter() - t0
        avg = elapsed / pos
        eta = avg * (len(remaining) - pos)

        log.info("[CHUNK %d/%d] Done — %d rows — elapsed %dm %ds — ETA %dm %ds",
                 pos, len(remaining), chunk_rows,
                 int(elapsed // 60), int(elapsed % 60),
                 int(eta // 60), int(eta % 60))

    log.info("=" * 60)
    log.info("  HISTORICAL LOAD COMPLETE")
    log.info("  Total rows loaded: %d", total_rows)
    log.info("  Failed chunks: %d", len(failed))
    if failed:
        log.warning("  Failed: %s", ", ".join(failed))
    log.info("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
