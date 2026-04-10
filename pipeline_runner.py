"""Pipeline runner - runs specified stages as a long-running process.

Usage:
    python pipeline_runner.py derive location_classify anomaly
    python pipeline_runner.py --all
    python pipeline_runner.py --remaining   # resume from last checkpoint
"""
import gc
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.logging_config import get_logger, setup_logging

try:
    from config import LOG_LEVEL
    setup_logging(LOG_LEVEL)
except Exception:
    setup_logging("INFO")

log = get_logger(__name__)

PROGRESS_FILE = Path("/tmp/pipeline_progress.json")

ALL_STAGES = [
    "location_classify",
    "anomaly",
    "forecast_rolling",
    "forecast_lgbm",
    "reorder",
    "alerts",
]

STAGE_RUNNERS = {
    "location_classify": ("transform.location_classify", "run_classify", "dry_run"),
    "anomaly":           ("ml.anomaly", "run_anomaly_detection", "dry_run"),
    "forecast_rolling":  ("ml.forecast_rolling", "run_forecast", "dry_run"),
    "forecast_lgbm":     ("ml.forecast_lgbm", "run_forecast", "dry_run"),
    "reorder":           ("engine.reorder", "run_reorder", "dry_run"),
    "alerts":            ("engine.alerts", "run_alerts", "dry_run"),
}

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    log.info("Received signal %d — will stop after current stage.", signum)
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "results": {}}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def run_stage(name: str) -> dict:
    import importlib
    mod_path, func_name, _ = STAGE_RUNNERS[name]
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, func_name)
    result = fn()
    return {"exit_code": result}


def main():
    args = sys.argv[1:]

    if "--remaining" in args:
        progress = load_progress()
        stages = [s for s in ALL_STAGES if s not in progress["completed"]]
    elif "--all" in args:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        stages = ALL_STAGES[:]
        progress = load_progress()
    else:
        stages = [a for a in args if a in STAGE_RUNNERS]
        if not stages:
            print(f"Usage: python pipeline_runner.py [--all|--remaining|STAGE...]")
            print(f"Available stages: {', '.join(ALL_STAGES)}")
            sys.exit(1)
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        progress = load_progress()

    log.info("=" * 60)
    log.info("  PIPELINE RUNNER")
    log.info("  Stages to run: %s", ", ".join(stages))
    log.info("=" * 60)

    for name in stages:
        if _shutdown:
            log.info("Shutdown requested — stopping.")
            break

        log.info("")
        log.info("▶  Stage: %s", name)
        t0 = time.perf_counter()
        try:
            result = run_stage(name)
            elapsed = time.perf_counter() - t0
            log.info("   ✓ %s completed in %.1fs — %s", name, elapsed, result)
            progress["completed"].append(name)
            progress["results"][name] = {"result": str(result), "elapsed_s": round(elapsed, 1)}
            save_progress(progress)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            log.error("   ✗ %s failed after %.1fs: %s", name, elapsed, e, exc_info=True)
            progress["results"][name] = {"error": str(e), "elapsed_s": round(elapsed, 1)}
            save_progress(progress)

        gc.collect()

    log.info("")
    log.info("=" * 60)
    log.info("  PIPELINE RUNNER COMPLETE")
    log.info("=" * 60)
    for name, info in progress["results"].items():
        if "error" in info:
            log.info("  ✗ %-25s ERROR: %s", name, info["error"])
        else:
            log.info("  ✓ %-25s %s (%.1fs)", name, info["result"], info["elapsed_s"])

    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(3)


if __name__ == "__main__":
    main()
