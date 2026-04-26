"""
db/connection.py — Supabase client singleton for partswatch-ai.

All modules that need database access should import `get_client()` from
here rather than instantiating their own Supabase clients.  A single
shared client avoids redundant TCP connections.
"""

import logging
from functools import lru_cache
from typing import Optional

from supabase import create_client, Client
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from utils.logging_config import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_client() -> Client:
    """Return the shared Supabase client, creating it on first call.

    Uses lru_cache so the client is instantiated exactly once per
    process regardless of how many modules call this function.

    Returns:
        A ready-to-use supabase.Client instance.

    Raises:
        EnvironmentError: If required env vars are missing.
        Exception: If the Supabase client cannot be created.
    """
    # Import config here so the module can be imported without crashing
    # when env vars are not yet set (e.g., during unit test discovery).
    from config import SUPABASE_URL, SUPABASE_KEY

    log.info("Initializing Supabase client -> %s", SUPABASE_URL)
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    log.info("Supabase client ready.")
    return client


def get_new_client() -> Client:
    """Build a brand-new, uncached Supabase client.

    Use this from long-running jobs to drop a stale HTTP/2 connection
    after a `RemoteProtocolError` / `ConnectionTerminated`. This bypasses
    the `get_client()` lru_cache and also clears it so subsequent
    `get_client()` callers receive the fresh instance.
    """
    from config import SUPABASE_URL, SUPABASE_KEY

    get_client.cache_clear()
    log.info("Building fresh Supabase client (cache cleared) -> %s", SUPABASE_URL)
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def query_with_retry(table: str, select: str = "*", filters: Optional[dict] = None) -> list[dict]:
    """Execute a SELECT query with automatic retry on transient failures.

    Args:
        table:   Supabase table name.
        select:  Column selector string (default "*").
        filters: Optional dict of {column: value} equality filters.

    Returns:
        List of row dicts returned by Supabase.

    Raises:
        Exception: After 3 failed attempts.
    """
    client = get_client()
    query = client.table(table).select(select)

    if filters:
        for col, val in filters.items():
            query = query.eq(col, val)

    response = query.execute()
    return response.data


def check_table_exists(table: str) -> bool:
    """Probe a table by fetching a single row — returns True if accessible.

    Args:
        table: Table name to probe.

    Returns:
        True if the table exists and is readable, False otherwise.
    """
    try:
        client = get_client()
        client.table(table).select("*").limit(1).execute()
        return True
    except Exception as exc:
        log.warning("Table '%s' not accessible: %s", table, exc)
        return False
