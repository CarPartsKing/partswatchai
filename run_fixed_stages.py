"""Run the pipeline stages that previously failed due to Supabase timeouts.

Now using _fetch_chunked_by_date() to avoid 502/statement timeout errors.
"""
import sys
import time
import json

sys.path.insert(0, "/home/runner/workspace")

from utils.logging_config import get_logger
log = get_logger(__name__)

PROGRESS_FILE = "/tmp/fixed_stage_progress.json"

def save_progress(results):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run only this stage (anomaly or forecast_lgbm)")
    args = parser.parse_args()

    results = {}
    all_stages = [
        ("anomaly", "ml.anomaly", "run_anomaly_detection"),
        ("forecast_lgbm", "ml.forecast_lgbm", "run_forecast"),
    ]
    stages = [s for s in all_stages if not args.only or s[0] == args.only]

    for name, module_path, func_name in stages:
        log.info("=" * 60)
        log.info("▶  Stage: %s", name)
        log.info("=" * 60)
        t0 = time.time()
        try:
            mod = __import__(module_path, fromlist=[func_name])
            fn = getattr(mod, func_name)
            result = fn()
            elapsed = time.time() - t0
            results[name] = {"exit_code": result, "elapsed_s": round(elapsed, 1)}
            log.info("✓ %s completed in %.1fs — exit_code=%d", name, elapsed, result)
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = {"error": str(e)[:200], "elapsed_s": round(elapsed, 1)}
            log.info("✗ %s failed after %.1fs: %s", name, elapsed, e)
        save_progress(results)

    log.info("=" * 60)
    log.info("ALL STAGES COMPLETE")
    log.info(json.dumps(results, indent=2))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
