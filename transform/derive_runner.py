"""Derive stage runner - designed to run as a Replit workflow.

Runs each derivation step sequentially, with progress tracking and
memory cleanup between steps. Resumes from where it left off if restarted.

Usage:
    python transform/derive_runner.py          # run all derivations
    python transform/derive_runner.py --resume  # resume from last checkpoint
"""
import gc
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging_config import get_logger, setup_logging

try:
    from config import LOG_LEVEL
    setup_logging(LOG_LEVEL)
except Exception:
    setup_logging("INFO")

log = get_logger(__name__)

PROGRESS_FILE = Path("/tmp/derive_progress.json")

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    log.info("Received signal %d — will stop after current step.", signum)
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "results": {}}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def main():
    from db.connection import get_client
    from transform.derive import DERIVATIONS

    resume = "--resume" in sys.argv
    if not resume and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    progress = load_progress()
    completed = set(progress["completed"])
    client = get_client()

    derivation_names = [name for name, _ in DERIVATIONS]
    remaining = [(name, fn) for name, fn in DERIVATIONS if name not in completed]

    log.info("=" * 60)
    log.info("  DERIVE RUNNER")
    log.info("  Total steps: %d | Completed: %d | Remaining: %d",
             len(derivation_names), len(completed), len(remaining))
    log.info("=" * 60)

    if not remaining:
        log.info("All derivations already completed!")
        for name, info in progress["results"].items():
            log.info("  %-30s %s", name, info.get("result", info.get("error", "?")))
        sys.exit(0)

    for name, fn in remaining:
        if _shutdown:
            log.info("Shutdown requested — stopping. Resume with --resume.")
            break

        log.info("")
        log.info("▶  Starting: %s", name)
        t0 = time.perf_counter()
        try:
            result = fn(client)
            elapsed = time.perf_counter() - t0
            log.info("   ✓ %s completed in %.1fs — %s", name, elapsed, result)

            progress["completed"].append(name)
            progress["results"][name] = {
                "result": str(result),
                "elapsed_s": round(elapsed, 1),
            }
            save_progress(progress)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            log.error("   ✗ %s failed after %.1fs: %s", name, elapsed, e, exc_info=True)
            progress["results"][name] = {
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
            }
            save_progress(progress)

        gc.collect()

    log.info("")
    log.info("=" * 60)
    log.info("  DERIVE RUNNER COMPLETE")
    log.info("=" * 60)
    for name, info in progress["results"].items():
        if "error" in info:
            log.info("  ✗ %-30s ERROR: %s", name, info["error"])
        else:
            log.info("  ✓ %-30s %s (%.1fs)", name, info["result"], info["elapsed_s"])

    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(5)


if __name__ == "__main__":
    main()
