"""Wrapper to run the historical Autocube extract as a long-running workflow."""
import sys
from utils.logging_config import setup_logging
setup_logging()

from extract.autocube_pull import run_historical_extract

start_chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rc = run_historical_extract(dry_run=False, start_chunk=start_chunk)
sys.exit(rc)
