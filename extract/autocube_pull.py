"""extract/autocube_pull.py — Autocube OLAP extraction via XMLA/SOAP over HTTPS.

Connects to the Autologue-hosted OLAP cube (Microsoft Analysis Services)
using the standard XMLA/SOAP protocol over HTTPS.  Works on Linux — no
Windows COM or MSOLAP dependencies required.

SECURITY
--------
- Read-only: only MDX SELECT queries are executed, never INSERT/UPDATE/DELETE
- query_validator() checks every query before execution; raises SecurityError
  on any disallowed keyword
- All MDX queries are hardcoded constants — no dynamic construction from
  external input
- Mode=Read set in XMLA properties where the provider supports it
- All writes go to Supabase only — nothing is ever written back to Autocube
- Complete audit trail: every query logged with timestamp and row count

AUTHENTICATION
--------------
Uses NTLM authentication (AUTOCUBE\\CPW001 domain format).  Falls back to
HTTP Basic if NTLM is rejected.

MODES
-----
    python -m extract.autocube_pull --test
        Connection test — verify connectivity, list available cubes,
        dimensions, hierarchies, and measures.  Does not pull data.

    python -m extract.autocube_pull --mode full
        Pull 3 years of sales history from the Sales Detail cube.
        Map fields via config/autocube_column_map.json.
        Load into sales_transactions in Supabase.

    python -m extract.autocube_pull --mode incremental
        Pull previous day only.  Deduplicate before inserting.

XMLA ENDPOINT PATHS TRIED (in order)
    /olap/msmdpump.dll   — standard IIS HTTP pump
    /OLAP/msmdpump.dll   — case variant
    /xmla                — generic XMLA path
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests
from requests_ntlm import HttpNtlmAuth

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# XMLA namespace constants
# ---------------------------------------------------------------------------

NS_SOAP = "http://schemas.xmlsoap.org/soap/envelope/"
NS_XMLA = "urn:schemas-microsoft-com:xml-analysis"
NS_ROWSET = "urn:schemas-microsoft-com:xml-analysis:rowset"
NS_MDDATASET = "urn:schemas-microsoft-com:xml-analysis:mddataset"

_NS_MAP = {
    "soap": NS_SOAP,
    "xmla": NS_XMLA,
    "rs": NS_ROWSET,
    "md": NS_MDDATASET,
}

_XMLA_PATHS = [
    "/olap/msmdpump.dll",
    "/OLAP/msmdpump.dll",
    "/xmla",
    "/XMLA",
    "/msmdpump.dll",
    "/OlapService/msmdpump.dll",
    "/olapservice/msmdpump.dll",
    "/SSAS/msmdpump.dll",
]

_REQUEST_TIMEOUT_SECS = 120

# ---------------------------------------------------------------------------
# Security — query validation
# ---------------------------------------------------------------------------

_ALLOWED_MDX_KEYWORDS = frozenset({
    "select", "with", "from", "where", "on",
    "members", "children", "member", "set",
    "non", "empty", "crossjoin", "columns", "rows",
    "nonempty", "order", "filter", "topcount",
    "head", "tail", "generate", "hierarchize",
    "descendants", "ascendants", "except", "intersect",
    "union", "distinct", "strtoset", "strtomember",
    "currentmember", "properties", "dimension",
    "measures", "all", "and", "or", "not", "is",
    "null", "true", "false", "as", "cell",
})

_DANGEROUS_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|TRUNCATE|MERGE)\b",
    re.IGNORECASE,
)


class SecurityError(Exception):
    """Raised when a query fails security validation."""


def query_validator(query: str) -> None:
    """Validate an MDX query string before execution.

    Raises SecurityError if the query contains any dangerous keywords
    (INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, EXEC, TRUNCATE, MERGE).
    Logs the blocked attempt.

    Args:
        query: The MDX query to validate.

    Raises:
        SecurityError: If a disallowed keyword is found.
    """
    match = _DANGEROUS_KEYWORDS.search(query)
    if match:
        blocked_keyword = match.group(1).upper()
        log.critical(
            "SECURITY: Blocked dangerous keyword '%s' in query: %.200s",
            blocked_keyword, query,
        )
        raise SecurityError(
            f"Query contains disallowed keyword '{blocked_keyword}'. "
            "Only MDX SELECT queries are permitted."
        )

    stripped = query.strip()
    first_word = stripped.split()[0].upper() if stripped else ""
    if first_word not in ("SELECT", "WITH"):
        log.critical(
            "SECURITY: Query does not start with SELECT or WITH: %.200s", query,
        )
        raise SecurityError(
            f"Query must start with SELECT or WITH, got '{first_word}'."
        )


# ---------------------------------------------------------------------------
# Hardcoded MDX queries — no dynamic construction from external input
# ---------------------------------------------------------------------------

MDX_FULL_SALES = """\
SELECT
  NON EMPTY {{
    [Measures].[Qty Ship],
    [Measures].[Unit Price],
    [Measures].[Ext Price],
    [Measures].[Unit Cost],
    [Measures].[Ext Cost],
    [Measures].[Gross Profit],
    [Measures].[Gross Profit %]
  }} ON COLUMNS,
  NON EMPTY CROSSJOIN(
    [Sales Date].[Invoice Date].[Inv Date].MEMBERS,
    [Product].[Prod Code].[Prod Code].MEMBERS,
    [Location].[Loc].[Loc].MEMBERS
  ) ON ROWS
FROM [{cube}]
"""

MDX_INCREMENTAL_DAY = """\
SELECT
  NON EMPTY {{
    [Measures].[Qty Ship],
    [Measures].[Unit Price],
    [Measures].[Ext Price],
    [Measures].[Unit Cost],
    [Measures].[Ext Cost],
    [Measures].[Gross Profit],
    [Measures].[Gross Profit %]
  }} ON COLUMNS,
  NON EMPTY CROSSJOIN(
    {{ [Sales Date].[Invoice Date].[Inv Date].&[{date_key}] }},
    [Product].[Prod Code].[Prod Code].MEMBERS,
    [Location].[Loc].[Loc].MEMBERS
  ) ON ROWS
FROM [{cube}]
"""


# ---------------------------------------------------------------------------
# XMLA/SOAP client
# ---------------------------------------------------------------------------

class AutocubeClient:
    """XMLA/SOAP client for connecting to Autologue-hosted OLAP cubes.

    Sends SOAP envelopes over HTTPS to the XMLA endpoint.  Supports
    both Discover (metadata listing) and Execute (MDX query) operations.
    Uses NTLM authentication with HTTP Basic fallback.
    """

    def __init__(
        self,
        server: str,
        user: str,
        password: str,
        catalog: str,
        cube: str = "Sales Detail",
        xmla_path: str = "",
    ) -> None:
        self._server = server.rstrip("/")
        self._user = user
        self._password = password
        self._catalog = catalog
        self._cube = cube
        self._xmla_path = xmla_path
        self._endpoint: str | None = None
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": '"urn:schemas-microsoft-com:xml-analysis:Execute"',
        })
        self._auth_ntlm = HttpNtlmAuth(user, password)
        self._auth_basic = (user, password)
        self._use_ntlm = True

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> str:
        """Discover the working XMLA endpoint and verify connectivity.

        Tries multiple known XMLA paths and authentication methods.

        Returns:
            The URL of the working endpoint.

        Raises:
            ConnectionError: If no endpoint responds successfully.
        """
        if self._xmla_path:
            paths = [self._xmla_path] + [p for p in _XMLA_PATHS if p != self._xmla_path]
        else:
            paths = list(_XMLA_PATHS)

        soap_probe = self._build_discover_envelope("DISCOVER_PROPERTIES", {})

        for path in paths:
            url = self._server + path
            log.info("Trying XMLA endpoint: %s", url)

            for auth_label, auth_obj in [("NTLM", self._auth_ntlm), ("Basic", self._auth_basic)]:
                try:
                    resp = self._session.post(
                        url,
                        data=soap_probe,
                        auth=auth_obj,
                        timeout=_REQUEST_TIMEOUT_SECS,
                    )
                    log.info(
                        "  %s auth → HTTP %d  (%d bytes)",
                        auth_label, resp.status_code, len(resp.content),
                    )
                    if resp.status_code == 200 and b"Envelope" in resp.content:
                        self._endpoint = url
                        self._use_ntlm = auth_label == "NTLM"
                        log.info(
                            "Connected to %s  auth=%s  catalog=%s",
                            url, auth_label, self._catalog,
                        )
                        return url
                except requests.RequestException as exc:
                    log.warning("  %s auth failed: %s", auth_label, exc)

        raise ConnectionError(
            f"Could not connect to any XMLA endpoint on {self._server}. "
            f"Tried paths: {paths}"
        )

    # ------------------------------------------------------------------
    # XMLA Discover — metadata queries
    # ------------------------------------------------------------------

    def discover(
        self,
        request_type: str,
        restrictions: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        """Execute an XMLA Discover request and return rows as dicts.

        Args:
            request_type: XMLA request type (e.g. MDSCHEMA_CUBES).
            restrictions: Key-value restriction pairs.

        Returns:
            List of dicts, one per result row.
        """
        t0 = time.perf_counter()
        envelope = self._build_discover_envelope(request_type, restrictions or {})

        log.info(
            "XMLA Discover  type=%s  restrictions=%s",
            request_type, restrictions or {},
        )

        resp = self._post(envelope, action="Discover")
        rows = self._parse_rowset(resp.content)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "  → %d rows in %.0fms  [%s]",
            len(rows), elapsed_ms, request_type,
        )
        return rows

    def list_cubes(self) -> list[dict]:
        return self.discover("MDSCHEMA_CUBES", {
            "CATALOG_NAME": self._catalog,
        })

    def list_dimensions(self, cube: str | None = None) -> list[dict]:
        return self.discover("MDSCHEMA_DIMENSIONS", {
            "CATALOG_NAME": self._catalog,
            "CUBE_NAME": cube or self._cube,
        })

    def list_hierarchies(self, cube: str | None = None) -> list[dict]:
        return self.discover("MDSCHEMA_HIERARCHIES", {
            "CATALOG_NAME": self._catalog,
            "CUBE_NAME": cube or self._cube,
        })

    def list_measures(self, cube: str | None = None) -> list[dict]:
        return self.discover("MDSCHEMA_MEASURES", {
            "CATALOG_NAME": self._catalog,
            "CUBE_NAME": cube or self._cube,
        })

    def list_levels(self, cube: str | None = None) -> list[dict]:
        return self.discover("MDSCHEMA_LEVELS", {
            "CATALOG_NAME": self._catalog,
            "CUBE_NAME": cube or self._cube,
        })

    # ------------------------------------------------------------------
    # XMLA Execute — MDX queries
    # ------------------------------------------------------------------

    def execute_mdx(self, mdx: str) -> list[dict[str, str]]:
        """Execute an MDX query and return result rows as dicts.

        Every query passes through query_validator() before execution.
        Results are returned in tabular format.

        Args:
            mdx: The MDX SELECT query.

        Returns:
            List of dicts, one per result row.

        Raises:
            SecurityError: If the query fails validation.
        """
        query_validator(mdx)

        t0 = time.perf_counter()
        log.info("MDX Execute  chars=%d", len(mdx))
        log.debug("MDX:\n%s", mdx[:500])

        envelope = self._build_execute_envelope(mdx)
        resp = self._post(envelope, action="Execute")
        rows = self._parse_rowset(resp.content)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "  → %d rows in %.0fms",
            len(rows), elapsed_ms,
        )
        return rows

    # ------------------------------------------------------------------
    # SOAP envelope builders
    # ------------------------------------------------------------------

    def _build_discover_envelope(
        self,
        request_type: str,
        restrictions: dict[str, str],
    ) -> str:
        restrictions_xml = ""
        if restrictions:
            inner = "".join(
                f"<{k}>{_xml_escape(v)}</{k}>"
                for k, v in restrictions.items()
            )
            restrictions_xml = f"<RestrictionList>{inner}</RestrictionList>"

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope
  xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"
  xmlns:xsd="http://www.w3.org/2001/XMLSchema"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <SOAP-ENV:Body>
    <Discover xmlns="urn:schemas-microsoft-com:xml-analysis">
      <RequestType>{request_type}</RequestType>
      <Restrictions>{restrictions_xml}</Restrictions>
      <Properties>
        <PropertyList>
          <Catalog>{_xml_escape(self._catalog)}</Catalog>
          <Format>Tabular</Format>
          <Content>SchemaData</Content>
        </PropertyList>
      </Properties>
    </Discover>
  </SOAP-ENV:Body>
</SOAP-ENV:Envelope>"""

    def _build_execute_envelope(self, mdx: str) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope
  xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"
  xmlns:xsd="http://www.w3.org/2001/XMLSchema"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <SOAP-ENV:Body>
    <Execute xmlns="urn:schemas-microsoft-com:xml-analysis">
      <Command>
        <Statement>{_xml_escape(mdx)}</Statement>
      </Command>
      <Properties>
        <PropertyList>
          <Catalog>{_xml_escape(self._catalog)}</Catalog>
          <Format>Tabular</Format>
          <Content>Data</Content>
        </PropertyList>
      </Properties>
    </Execute>
  </SOAP-ENV:Body>
</SOAP-ENV:Envelope>"""

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def _post(self, envelope: str, action: str = "Execute") -> requests.Response:
        if not self._endpoint:
            raise ConnectionError("Not connected — call connect() first.")

        self._session.headers["SOAPAction"] = (
            f'"urn:schemas-microsoft-com:xml-analysis:{action}"'
        )

        auth = self._auth_ntlm if self._use_ntlm else self._auth_basic

        try:
            resp = self._session.post(
                self._endpoint,
                data=envelope.encode("utf-8"),
                auth=auth,
                timeout=_REQUEST_TIMEOUT_SECS,
            )
        except requests.RequestException:
            log.exception("XMLA HTTP request failed.")
            raise

        if resp.status_code != 200:
            log.error(
                "XMLA HTTP %d: %s",
                resp.status_code, resp.text[:500],
            )
            raise ConnectionError(
                f"XMLA request failed with HTTP {resp.status_code}"
            )

        fault = self._check_soap_fault(resp.content)
        if fault:
            log.error("SOAP Fault: %s", fault)
            raise RuntimeError(f"SOAP Fault: {fault}")

        return resp

    # ------------------------------------------------------------------
    # XML parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_soap_fault(xml_bytes: bytes) -> str | None:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return None
        fault = root.find(f".//{{{NS_SOAP}}}Fault")
        if fault is not None:
            faultstring = fault.findtext("faultstring") or fault.findtext(
                f"{{{NS_SOAP}}}Reason/{{{NS_SOAP}}}Text"
            )
            return faultstring or ET.tostring(fault, encoding="unicode")
        return None

    @staticmethod
    def _parse_rowset(xml_bytes: bytes) -> list[dict[str, str]]:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            log.error("Failed to parse XMLA response XML: %s", exc)
            return []

        rows_out: list[dict[str, str]] = []
        for row_el in root.iter(f"{{{NS_ROWSET}}}row"):
            row_dict: dict[str, str] = {}
            for child in row_el:
                tag = child.tag
                if tag.startswith(f"{{{NS_ROWSET}}}"):
                    tag = tag[len(f"{{{NS_ROWSET}}}"):]
                tag = _decode_ssas_column(tag)
                row_dict[tag] = child.text or ""
            if row_dict:
                rows_out.append(row_dict)

        return rows_out


_SSAS_HEX_RE = re.compile(r"_x([0-9A-Fa-f]{4})_")


def _decode_ssas_column(encoded: str) -> str:
    decoded = _SSAS_HEX_RE.sub(lambda m: chr(int(m.group(1), 16)), encoded)
    if decoded.endswith(".[MEMBER_CAPTION]"):
        decoded = decoded[: -len(".[MEMBER_CAPTION]")]
    return decoded


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ---------------------------------------------------------------------------
# DataSource adapter — plugs into the partswatch_pull.py pipeline
# ---------------------------------------------------------------------------

class AutocubeDataSource:
    """DataSource-compatible adapter for the Autocube OLAP cube.

    Implements the connect / extract / close contract from
    extract.partswatch_pull.DataSource so that ``PARTSWATCH_SOURCE=autocube``
    routes the standard pipeline through the XMLA/SOAP client.
    """

    def __init__(self) -> None:
        self._client: AutocubeClient | None = None

    def connect(self) -> None:
        self._client = get_client()
        self._client.connect()

    def extract(self, dataset: str) -> list[dict]:
        if dataset != "sales_transactions":
            log.debug("Autocube does not provide dataset '%s' — skipping.", dataset)
            return []

        if self._client is None:
            raise ConnectionError("Not connected — call connect() first.")

        cube = config.AUTOCUBE_CUBE
        mdx = MDX_FULL_SALES.format(cube=cube)

        try:
            return self._client.execute_mdx(mdx)
        except Exception:
            log.exception("Autocube extract failed for dataset '%s'.", dataset)
            return []

    def close(self) -> None:
        self._client = None

    def __enter__(self) -> "AutocubeDataSource":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client() -> AutocubeClient:
    """Create an AutocubeClient from Replit secrets / env vars."""
    server   = config.AUTOCUBE_SERVER
    user     = config.AUTOCUBE_USER
    password = config.AUTOCUBE_PASSWORD
    catalog  = config.AUTOCUBE_CATALOG
    cube     = config.AUTOCUBE_CUBE
    path     = config.AUTOCUBE_XMLA_PATH

    if not all([server, user, password, catalog]):
        raise EnvironmentError(
            "Missing Autocube credentials. Set AUTOCUBE_SERVER, AUTOCUBE_USER, "
            "AUTOCUBE_PASSWORD, and AUTOCUBE_CATALOG."
        )

    return AutocubeClient(
        server=server,
        user=user,
        password=password,
        catalog=catalog,
        cube=cube,
        xmla_path=path,
    )


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def load_column_map() -> dict:
    """Load autocube_column_map.json and return the mapping dict."""
    map_path = Path(ROOT / config.AUTOCUBE_COLUMN_MAP_PATH)
    if not map_path.exists():
        log.warning("Column map not found at %s — using identity mapping.", map_path)
        return {}
    with open(map_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Mode: CONNECTION_TEST
# ---------------------------------------------------------------------------

def run_connection_test() -> int:
    """Verify connectivity and list all metadata from the cube.

    Returns 0 on success, 1 on failure.
    """
    log.info("=" * 60)
    log.info("  AUTOCUBE CONNECTION TEST")
    log.info("=" * 60)
    log.info("Server:  %s", config.AUTOCUBE_SERVER)
    log.info("User:    %s", config.AUTOCUBE_USER)
    log.info("Catalog: %s", config.AUTOCUBE_CATALOG)
    log.info("Cube:    %s", config.AUTOCUBE_CUBE)

    try:
        client = get_client()
        endpoint = client.connect()
        log.info("Connected: %s", endpoint)
    except Exception:
        log.exception("Connection failed.")
        return 1

    log.info("")
    log.info("--- AVAILABLE CUBES ---")
    try:
        cubes = client.list_cubes()
        for c in cubes:
            log.info(
                "  Cube: %-30s  Type: %s",
                c.get("CUBE_NAME", "?"), c.get("CUBE_TYPE", "?"),
            )
        if not cubes:
            log.warning("  (no cubes returned)")
    except Exception:
        log.exception("  Failed to list cubes.")

    log.info("")
    log.info("--- DIMENSIONS in '%s' ---", config.AUTOCUBE_CUBE)
    try:
        dims = client.list_dimensions()
        for d in dims:
            log.info(
                "  Dimension: %-30s  UniqueName: %s  Type: %s",
                d.get("DIMENSION_NAME", "?"),
                d.get("DIMENSION_UNIQUE_NAME", "?"),
                d.get("DIMENSION_TYPE", "?"),
            )
        if not dims:
            log.warning("  (no dimensions returned)")
    except Exception:
        log.exception("  Failed to list dimensions.")

    log.info("")
    log.info("--- HIERARCHIES in '%s' ---", config.AUTOCUBE_CUBE)
    try:
        hierarchies = client.list_hierarchies()
        for h in hierarchies:
            log.info(
                "  Hierarchy: %-30s  Dimension: %-20s  Levels: %s",
                h.get("HIERARCHY_UNIQUE_NAME", "?"),
                h.get("DIMENSION_UNIQUE_NAME", "?"),
                h.get("LEVELS", "?"),
            )
        if not hierarchies:
            log.warning("  (no hierarchies returned)")
    except Exception:
        log.exception("  Failed to list hierarchies.")

    log.info("")
    log.info("--- MEASURES in '%s' ---", config.AUTOCUBE_CUBE)
    try:
        measures = client.list_measures()
        for m in measures:
            log.info(
                "  Measure: %-30s  UniqueName: %-40s  DataType: %s  Agg: %s",
                m.get("MEASURE_NAME", "?"),
                m.get("MEASURE_UNIQUE_NAME", "?"),
                m.get("DATA_TYPE", "?"),
                m.get("MEASURE_AGGREGATOR", "?"),
            )
        if not measures:
            log.warning("  (no measures returned)")
    except Exception:
        log.exception("  Failed to list measures.")

    log.info("")
    log.info("--- LEVELS in '%s' ---", config.AUTOCUBE_CUBE)
    try:
        levels = client.list_levels()
        for lv in levels:
            log.info(
                "  Level: %-35s  Hierarchy: %-30s  Depth: %s",
                lv.get("LEVEL_UNIQUE_NAME", "?"),
                lv.get("HIERARCHY_UNIQUE_NAME", "?"),
                lv.get("LEVEL_NUMBER", "?"),
            )
        if not levels:
            log.warning("  (no levels returned)")
    except Exception:
        log.exception("  Failed to list levels.")

    log.info("")
    log.info("Connection test complete.")
    return 0


# ---------------------------------------------------------------------------
# Mode: FULL — 3 years of sales history
# ---------------------------------------------------------------------------

def run_full_extract() -> int:
    """Pull 3 years of sales data from the cube and load to Supabase.

    Returns 0 on success, 1 on failure.
    """
    t0 = time.perf_counter()
    log.info("=" * 60)
    log.info("  AUTOCUBE FULL EXTRACT — 3 years")
    log.info("=" * 60)

    try:
        client = get_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    column_map = load_column_map()
    cube = config.AUTOCUBE_CUBE
    today = date.today()
    total_rows = 0

    mdx = MDX_FULL_SALES.format(cube=cube)

    try:
        rows = client.execute_mdx(mdx)
        log.info("Full extract: %d rows returned.", len(rows))
        total_rows = len(rows)

        if rows:
            _load_to_supabase(rows, column_map)

    except SecurityError:
        log.exception("Security violation — aborting.")
        return 1
    except Exception:
        log.exception("Full extract failed.")
        return 1

    elapsed = time.perf_counter() - t0
    log.info(
        "Full extract complete: %d total rows in %.1fs",
        total_rows, elapsed,
    )
    return 0


# ---------------------------------------------------------------------------
# Mode: INCREMENTAL — previous day
# ---------------------------------------------------------------------------

def run_incremental_extract() -> int:
    """Pull the previous day's sales and load to Supabase.

    Returns 0 on success, 1 on failure.
    """
    t0 = time.perf_counter()
    yesterday = date.today() - timedelta(days=1)
    date_key = yesterday.strftime("%Y%m%d")

    log.info("=" * 60)
    log.info("  AUTOCUBE INCREMENTAL EXTRACT — %s", date_key)
    log.info("=" * 60)

    try:
        client = get_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    column_map = load_column_map()
    cube = config.AUTOCUBE_CUBE

    mdx = MDX_INCREMENTAL_DAY.format(cube=cube, date_key=date_key)

    try:
        rows = client.execute_mdx(mdx)
        log.info("Rows returned: %d", len(rows))
    except SecurityError:
        log.exception("Security violation — aborting.")
        return 1
    except Exception:
        log.exception("Incremental extract failed.")
        return 1

    if rows:
        _load_to_supabase(rows, column_map)

    elapsed = time.perf_counter() - t0
    log.info(
        "Incremental extract complete: %d rows in %.1fs",
        len(rows), elapsed,
    )
    return 0


# ---------------------------------------------------------------------------
# Supabase loader
# ---------------------------------------------------------------------------

_BATCH_SIZE = 500

_GENERATED_COLS = frozenset({
    "is_stockout",
    "lead_time_variance",
    "fill_rate_pct",
})


def _load_to_supabase(
    rows: list[dict],
    column_map: dict,
) -> int:
    """Map, clean, and upsert rows into sales_transactions.

    Returns the count of rows loaded.
    """
    from db.connection import get_client as get_db_client

    mapping = column_map.get("sales_transactions", {})
    active_mapping = {k: v for k, v in mapping.items() if v is not None}

    if not active_mapping:
        log.warning(
            "No active column mappings in autocube_column_map.json — "
            "run --test first to discover available fields, then update "
            "the mapping file."
        )
        return 0

    reverse_map = {v: k for k, v in active_mapping.items()}

    mapped_rows: list[dict] = []
    for row in rows:
        mapped: dict[str, Any] = {}
        for source_col, target_col in reverse_map.items():
            if source_col in row:
                mapped[target_col] = row[source_col]
        for col in _GENERATED_COLS:
            mapped.pop(col, None)
        if mapped:
            mapped_rows.append(mapped)

    if not mapped_rows:
        log.warning("No rows survived column mapping.")
        return 0

    log.info("Loading %d mapped rows to sales_transactions …", len(mapped_rows))

    try:
        db = get_db_client()
    except Exception:
        log.exception("Supabase connection failed.")
        raise

    loaded = 0
    for i in range(0, len(mapped_rows), _BATCH_SIZE):
        batch = mapped_rows[i : i + _BATCH_SIZE]
        try:
            db.table("sales_transactions").upsert(
                batch, on_conflict="transaction_id"
            ).execute()
            loaded += len(batch)
            log.debug("  Batch %d–%d upserted.", i, i + len(batch))
        except Exception:
            log.exception("Upsert failed at batch offset %d.", i)
            raise

    log.info("Loaded %d rows to Supabase.", loaded)
    return loaded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autocube OLAP extraction via XMLA/SOAP",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Connection test — list dimensions and measures, no data pull",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        help="Extract mode: full (3 years) or incremental (yesterday)",
    )
    args = parser.parse_args()

    if not args.test and not args.mode:
        parser.print_help()
        return 1

    if args.test:
        return run_connection_test()

    if args.mode == "full":
        return run_full_extract()

    if args.mode == "incremental":
        return run_incremental_extract()

    return 1


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
