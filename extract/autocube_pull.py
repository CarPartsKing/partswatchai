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

    python -m extract.autocube_pull --mode transfers-test [--lookback-days N]
        Diagnostic — pull last N days (default 30) with a T-prefix FILTER
        MDX (no Customer dimensions) and report how many transfer invoices
        exist in the cube, unique routes, and total units.  Does not write
        to Supabase.  Run this BEFORE --mode transfers to verify the cube
        actually contains T-pattern invoices.

    python -m extract.autocube_pull --mode transfers [--lookback-days N]
        Pull last N days (default 90) and filter to invoice_number matching
        the transfer pattern "T{from_loc}{to_loc}" (e.g. "T2501").
        Uses a dedicated MDX WITHOUT Customer dimensions so transfer invoices
        (which have no customer) are not silently dropped by NON EMPTY.
        Writes parsed rows to location_transfers with tran_code='ACTUAL'.

WARRANTY FLAG (TODO — Sales Detail cube)
    The Sales Detail cube contains a Warranty Flag dimension. Once captured,
    warranty transactions (is_warranty = TRUE) should be excluded from forecast
    training data using the same exclusion logic as is_anomaly = TRUE.
    Migration backlog: add is_warranty BOOLEAN DEFAULT FALSE to sales_transactions.

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

# Shorter timeout used only during connect() endpoint probing.
# 16 probe attempts (8 paths × 2 auth methods) at the full 120 s would waste
# up to 32 minutes per module when the server is slow/unreachable.  10 s is
# enough to distinguish "port closed fast" from "port open but no XMLA" without
# hanging indefinitely.  Actual MDX data queries still use _REQUEST_TIMEOUT_SECS.
_PROBE_TIMEOUT_SECS = 10

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

_MEASURES_BLOCK = """\
    [Measures].[Qty Ship],
    [Measures].[Unit Price],
    [Measures].[Ext Price],
    [Measures].[Unit Cost],
    [Measures].[Ext Cost],
    [Measures].[Gross Profit],
    [Measures].[Gross Profit %]"""

# ---------------------------------------------------------------------------
# Sales Detail flag dimensions captured alongside the core CROSSJOIN.
# Each member is 'Y' or 'N' (verified empirically 2026-04-18 against the
# AutoCube_DTR_23160 catalog).  Adding them to the CROSSJOIN expands every
# returned row by the cardinality of the flag (≤ 2), so worst-case row count
# grows by 16x.  In practice each invoice line has exactly ONE non-NULL
# combination of flags so NON EMPTY collapses the result back to roughly the
# original cardinality.
#
#   Warranty Flag         — warranty-replacement sales (excluded from forecast)
#   Backorder Flag        — backorder lines (subset of normal demand)
#   Core Flag             — core return lines (negative qty, separate stream)
#   Price Override Flag   — manually-priced lines (kept; useful for margin work)
# ---------------------------------------------------------------------------
_FLAG_AXES = (
    "    [Sales Detail].[Warranty Flag].[Warranty Flag].MEMBERS,\n"
    "    [Sales Detail].[Backorder Flag].[Backorder Flag].MEMBERS,\n"
    "    [Sales Detail].[Core Flag].[Core Flag].MEMBERS,\n"
    "    [Sales Detail].[Price Override Flag].[Price Override Flag].MEMBERS,\n"
    # Invoice number — the real document number per line item.  Adds ~3x to
    # row volume but is mandatory for basket/co-purchase analysis (without it
    # multiple invoices on the same date+sku+location collapse into a single
    # synthetic "transaction" and Apriori finds zero multi-item baskets).
    "    [Sales Detail].[Invoice Nbr].[Invoice Nbr].MEMBERS,\n"
    # Customer dimensions added 2026-04-19 to enable customer churn scoring
    # (ml/churn.py) and rep-level routing of churn alerts.  Each invoice has
    # exactly ONE customer, and Cust Type / Status / Salesman are functionally
    # determined by Cust No, so NON EMPTY collapses these axes back to roughly
    # the same row cardinality as without them (verified against the
    # AutoCube_DTR_23160 catalog 2026-04-19).
    #   Cust No   — canonical Autologue customer account number (PK for churn)
    #   Cust Type — wholesale / retail / fleet / etc. (segmentation)
    #   Status    — active / inactive / on-hold (churn.py skips already-gone)
    #   Salesman  — outside-rep ownership (alert routing)
    "    [Customer].[Cust No].[Cust No].MEMBERS,\n"
    "    [Customer].[Cust Type].[Cust Type].MEMBERS,\n"
    "    [Customer].[Status].[Status].MEMBERS,\n"
    "    [Customer].[Salesman].[Salesman].MEMBERS"
)

MDX_FULL_SALES = (
    "SELECT\n  NON EMPTY {{\n" + _MEASURES_BLOCK + "\n  }} ON COLUMNS,\n"
    "  NON EMPTY CROSSJOIN(\n"
    "    [Sales Date].[Invoice Date].[Inv Date].MEMBERS,\n"
    "    [Product].[Prod Code].[Prod Code].MEMBERS,\n"
    "    [Location].[Loc].[Loc].MEMBERS,\n"
    + _FLAG_AXES + "\n"
    "  ) ON ROWS\n"
    "FROM [{cube}]\n"
)

MDX_INCREMENTAL_DAY = (
    "SELECT\n  NON EMPTY {{\n" + _MEASURES_BLOCK + "\n  }} ON COLUMNS,\n"
    "  NON EMPTY CROSSJOIN(\n"
    "    {{ [Sales Date].[Invoice Date].[Inv Date].&[{date_key}] }},\n"
    "    [Product].[Prod Code].[Prod Code].MEMBERS,\n"
    "    [Location].[Loc].[Loc].MEMBERS,\n"
    + _FLAG_AXES + "\n"
    "  ) ON ROWS\n"
    "FROM [{cube}]\n"
)

MDX_MONTHLY_RANGE = (
    "SELECT\n  NON EMPTY {{\n" + _MEASURES_BLOCK + "\n  }} ON COLUMNS,\n"
    "  NON EMPTY CROSSJOIN(\n"
    "    {{ [Sales Date].[Invoice Date].[Inv Date].&[{start_key}]"
    " : [Sales Date].[Invoice Date].[Inv Date].&[{end_key}] }},\n"
    "    [Product].[Prod Code].[Prod Code].MEMBERS,\n"
    "    [Location].[Loc].[Loc].MEMBERS,\n"
    + _FLAG_AXES + "\n"
    "  ) ON ROWS\n"
    "FROM [{cube}]\n"
)

# ---------------------------------------------------------------------------
# Transfers-specific MDX — NO Customer dimensions.
#
# Inter-store transfers in the Sales Detail cube have invoice numbers of the
# form "T{from_loc}{to_loc}" (e.g. "T2501" = from LOC-025 to LOC-001).
# Because transfer invoices have no associated customer, including Customer
# dimensions in the CROSSJOIN causes NON EMPTY to silently drop them (a
# member combination with a NULL measure is filtered as empty).
#
# MDX_TRANSFERS_RANGE is identical to MDX_MONTHLY_RANGE minus the four
# Customer axes.  This is intentional: we pull all transactions for the
# date window and then filter Python-side to rows whose invoice_number
# starts with 'T'.  A server-side FILTER(... LEFT(member.NAME,1)="T") was
# tried but consistently timed out (>120 s) because SSAS must evaluate the
# LEFT() predicate against every Invoice Nbr member in the cube before
# returning results.  The Python-side filter is essentially free.
#
# Used by both run_transfers_test() and run_transfers_extract().
# ---------------------------------------------------------------------------

_FLAG_AXES_NO_CUSTOMER = (
    "    [Sales Detail].[Warranty Flag].[Warranty Flag].MEMBERS,\n"
    "    [Sales Detail].[Backorder Flag].[Backorder Flag].MEMBERS,\n"
    "    [Sales Detail].[Core Flag].[Core Flag].MEMBERS,\n"
    "    [Sales Detail].[Price Override Flag].[Price Override Flag].MEMBERS,\n"
    "    [Sales Detail].[Invoice Nbr].[Invoice Nbr].MEMBERS"
)

MDX_TRANSFERS_RANGE = (
    "SELECT\n  NON EMPTY {{\n" + _MEASURES_BLOCK + "\n  }} ON COLUMNS,\n"
    "  NON EMPTY CROSSJOIN(\n"
    "    {{ [Sales Date].[Invoice Date].[Inv Date].&[{start_key}]"
    " : [Sales Date].[Invoice Date].[Inv Date].&[{end_key}] }},\n"
    "    [Product].[Prod Code].[Prod Code].MEMBERS,\n"
    "    [Location].[Loc].[Loc].MEMBERS,\n"
    + _FLAG_AXES_NO_CUSTOMER + "\n"
    "  ) ON ROWS\n"
    "FROM [{cube}]\n"
)


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
                        timeout=_PROBE_TIMEOUT_SECS,
                    )
                    log.info(
                        "  %s auth -> HTTP %d  (%d bytes)",
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
            "  -> %d rows in %.0fms  [%s]",
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
            "  -> %d rows in %.0fms",
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

    _missing = [
        name for name, val in (
            ("AUTOCUBE_SERVER",   server),
            ("AUTOCUBE_USER",     user),
            ("AUTOCUBE_PASSWORD", password),
            ("AUTOCUBE_CATALOG",  catalog),
        ) if not val
    ]
    if _missing:
        raise EnvironmentError(
            f"Missing Autocube credential(s): {', '.join(_missing)}. "
            "Ensure each is set as a GitHub Secret and listed under the "
            "workflow job's `env:` block."
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


def run_full_discovery() -> int:
    """Deep metadata discovery — list ALL catalogs, cubes, dimensions, and measures.

    Goes beyond the default catalog to find every database and cube
    available on the SSAS server, including inventory, purchasing,
    and customer cubes we may not have explored yet.

    Returns 0 on success, 1 on failure.
    """
    log.info("=" * 70)
    log.info("  AUTOCUBE FULL METADATA DISCOVERY")
    log.info("=" * 70)
    log.info("Server:  %s", config.AUTOCUBE_SERVER)
    log.info("User:    %s", config.AUTOCUBE_USER)
    log.info("Default Catalog: %s", config.AUTOCUBE_CATALOG)

    try:
        client = get_client()
        endpoint = client.connect()
        log.info("Connected: %s", endpoint)
    except Exception:
        log.exception("Connection failed.")
        return 1

    log.info("")
    log.info("=" * 70)
    log.info("  PHASE 1: DISCOVER ALL CATALOGS (DATABASES)")
    log.info("=" * 70)
    catalogs: list[str] = []
    try:
        cat_rows = client.discover("DBSCHEMA_CATALOGS", {})
        if cat_rows:
            for c in cat_rows:
                name = c.get("CATALOG_NAME", "?")
                catalogs.append(name)
                log.info("  Database: %-40s  Description: %s",
                         name, c.get("DESCRIPTION", ""))
            log.info("  Total databases found: %d", len(catalogs))
        else:
            log.info("  (DBSCHEMA_CATALOGS returned 0 rows — server may not support it)")
            catalogs.append(config.AUTOCUBE_CATALOG)
    except Exception:
        log.warning("  DBSCHEMA_CATALOGS not supported — falling back to default catalog.")
        catalogs.append(config.AUTOCUBE_CATALOG)

    log.info("")
    log.info("=" * 70)
    log.info("  PHASE 2: DISCOVER ALL CUBES IN EACH CATALOG")
    log.info("=" * 70)

    all_cubes: list[tuple[str, str, str]] = []

    for catalog in catalogs:
        log.info("")
        log.info("--- Catalog: %s ---", catalog)
        try:
            cube_rows = client.discover("MDSCHEMA_CUBES", {
                "CATALOG_NAME": catalog,
            })
            if not cube_rows:
                log.info("  (no cubes found)")
                continue
            for c in cube_rows:
                cube_name = c.get("CUBE_NAME", "?")
                cube_type = c.get("CUBE_TYPE", "?")
                desc = c.get("DESCRIPTION", "")
                created = c.get("CREATED_ON", "")
                updated = c.get("LAST_SCHEMA_UPDATE", c.get("LAST_DATA_UPDATE", ""))
                all_cubes.append((catalog, cube_name, cube_type))
                log.info("  Cube: %-35s  Type: %-12s  Created: %s  Updated: %s",
                         cube_name, cube_type, created[:19] if created else "?",
                         updated[:19] if updated else "?")
                if desc:
                    log.info("        Description: %s", desc[:200])
        except Exception:
            log.exception("  Failed to list cubes in catalog '%s'.", catalog)

    log.info("")
    log.info("  Total cubes found across all catalogs: %d", len(all_cubes))

    log.info("")
    log.info("=" * 70)
    log.info("  PHASE 3: DIMENSIONS & MEASURES FOR EACH CUBE")
    log.info("=" * 70)

    for catalog, cube_name, cube_type in all_cubes:
        log.info("")
        log.info("╔══════════════════════════════════════════════════════════════╗")
        log.info("║  Catalog: %-20s  Cube: %-25s ║", catalog[:20], cube_name[:25])
        log.info("╚══════════════════════════════════════════════════════════════╝")

        log.info("")
        log.info("  --- Dimensions ---")
        try:
            dims = client.discover("MDSCHEMA_DIMENSIONS", {
                "CATALOG_NAME": catalog,
                "CUBE_NAME": cube_name,
            })
            if dims:
                for d in dims:
                    log.info("    Dim: %-30s  UniqueName: %-40s  Type: %s",
                             d.get("DIMENSION_NAME", "?"),
                             d.get("DIMENSION_UNIQUE_NAME", "?"),
                             d.get("DIMENSION_TYPE", "?"))
            else:
                log.info("    (no dimensions)")
        except Exception:
            log.exception("    Failed to list dimensions for cube '%s'.", cube_name)

        log.info("")
        log.info("  --- Hierarchies ---")
        try:
            hierarchies = client.discover("MDSCHEMA_HIERARCHIES", {
                "CATALOG_NAME": catalog,
                "CUBE_NAME": cube_name,
            })
            if hierarchies:
                for h in hierarchies:
                    log.info("    Hierarchy: %-35s  Dimension: %-30s",
                             h.get("HIERARCHY_UNIQUE_NAME", "?"),
                             h.get("DIMENSION_UNIQUE_NAME", "?"))
            else:
                log.info("    (no hierarchies)")
        except Exception:
            log.exception("    Failed to list hierarchies for cube '%s'.", cube_name)

        log.info("")
        log.info("  --- Measures ---")
        try:
            measures = client.discover("MDSCHEMA_MEASURES", {
                "CATALOG_NAME": catalog,
                "CUBE_NAME": cube_name,
            })
            if measures:
                for m in measures:
                    log.info("    Measure: %-35s  UniqueName: %-45s  Type: %s  Agg: %s",
                             m.get("MEASURE_NAME", "?"),
                             m.get("MEASURE_UNIQUE_NAME", "?"),
                             m.get("DATA_TYPE", "?"),
                             m.get("MEASURE_AGGREGATOR", "?"))
                log.info("    Total measures: %d", len(measures))
            else:
                log.info("    (no measures)")
        except Exception:
            log.exception("    Failed to list measures for cube '%s'.", cube_name)

        log.info("")
        log.info("  --- Levels ---")
        try:
            levels = client.discover("MDSCHEMA_LEVELS", {
                "CATALOG_NAME": catalog,
                "CUBE_NAME": cube_name,
            })
            if levels:
                for lv in levels:
                    log.info("    Level: %-40s  Hierarchy: %-35s  Depth: %s",
                             lv.get("LEVEL_UNIQUE_NAME", "?"),
                             lv.get("HIERARCHY_UNIQUE_NAME", "?"),
                             lv.get("LEVEL_NUMBER", "?"))
            else:
                log.info("    (no levels)")
        except Exception:
            log.exception("    Failed to list levels for cube '%s'.", cube_name)

    log.info("")
    log.info("=" * 70)
    log.info("  DISCOVERY SUMMARY")
    log.info("=" * 70)
    log.info("  Catalogs (databases): %d", len(catalogs))
    for cat in catalogs:
        cube_names = [cn for ca, cn, _ in all_cubes if ca == cat]
        log.info("    %s: %d cubes — %s", cat, len(cube_names),
                 ", ".join(cube_names) if cube_names else "(none)")
    log.info("  Total cubes: %d", len(all_cubes))
    log.info("")
    log.info("Full discovery complete.")
    return 0


# ---------------------------------------------------------------------------
# Data cleaning — scientific notation + date formatting
# ---------------------------------------------------------------------------

_NUMERIC_FIELDS = frozenset({
    "qty_sold", "unit_price", "total_revenue",
    "cost_per_unit", "ext_cost", "gross_profit", "gross_profit_pct",
})

_DB_COLUMNS = frozenset({
    "transaction_id", "sku_id", "location_id", "transaction_date",
    "qty_sold", "unit_price", "total_revenue",
    # Sales Detail flag dimensions added by migration 014.  Without these
    # in the allow-list the booleans are silently dropped before upsert,
    # leaving every row at the column default (FALSE) and rendering the
    # forecast/dead-stock `is_warranty=False` exclusions a no-op.
    "is_warranty", "is_backorder", "is_core_return", "is_price_override",
    # Real invoice number from migration 022.  MUST be in the allow-list or
    # it is silently dropped before upsert and basket analysis collapses
    # back to single-item baskets.
    "invoice_number",
    # Customer dimensions added by migration 025 (2026-04-19).  MUST be in
    # the allow-list or churn scoring (ml/churn.py) cannot group transactions
    # by customer.  customer_id is the canonical Autologue Cust No;
    # customer_type / customer_status / customer_salesman are slowly-changing
    # attributes denormalized onto each transaction so churn.py can segment
    # and route alerts without an extra join.
    "customer_id", "customer_type", "customer_status", "customer_salesman",
})

_GENERATED_COLS = frozenset({
    "is_stockout", "lead_time_variance", "fill_rate_pct",
})

_BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Transfer invoice detection
# ---------------------------------------------------------------------------

# Invoice numbers follow "T{from_num}{to_num}" — NO spaces, exactly 4 digits
# after the T (two-digit zero-padded location codes), e.g.:
#   T2501  = from LOC-025 to LOC-001
#   T2505  = from LOC-025 to LOC-005
#   T0103  = from LOC-001 to LOC-003
_TRANSFER_INVOICE_RE = re.compile(r'^T(\d{2})(\d{2})$')


def _parse_transfer_invoice(invoice: str) -> tuple[str, str] | None:
    """Parse a transfer invoice number into (from_location_id, to_location_id).

    'T2501' → ('LOC-025', 'LOC-001')
    'T0103' → ('LOC-001', 'LOC-003')
    Returns None if the string does not match the transfer pattern.
    """
    m = _TRANSFER_INVOICE_RE.match((invoice or "").strip())
    if not m:
        return None
    from_num, to_num = m.group(1), m.group(2)
    return f"LOC-{int(from_num):03d}", f"LOC-{int(to_num):03d}"


def clean_numeric(value: Any) -> float | None:
    """Convert a value to float, handling scientific notation.

    Handles: '5.177E1' → 51.77, '51.77' → 51.77, '' → None, None → None.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        log.warning("Cannot convert to float: %r", value)
        return None


def clean_date(value: Any) -> str | None:
    """Convert MM/DD/YYYY → YYYY-MM-DD (ISO 8601).

    Returns None for empty/malformed dates.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        month, day, year = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    log.warning("Unrecognised date format: %r", value)
    return None


def _generate_transaction_id(row: dict[str, Any]) -> str:
    """Generate a deterministic transaction ID.

    When the row carries a real `invoice_number` (post-migration 022) we
    append it so each invoice line gets its own row — required for basket
    analysis to see multi-item invoices.  Without an invoice number we fall
    back to the legacy 4-part key so old rows extracted before migration 022
    keep their original IDs.
    """
    dt = row.get("transaction_date", "")
    sku = row.get("sku_id", "")
    loc = row.get("location_id", "")
    inv = row.get("invoice_number")
    if inv:
        return f"AC-{dt}-{sku}-{loc}-INV{inv}"
    return f"AC-{dt}-{sku}-{loc}"


def _extract_location_code(raw: str) -> str:
    """Extract the numeric location code from '25-CPW - DC' → 'LOC-25'.

    Converts Autocube format (e.g. '1 -CPW - BROOKPARK') to PartsWatch
    location ID format (e.g. 'LOC-001') matching existing DB convention.
    """
    if not raw:
        return raw
    code = raw.split("-", 1)[0].strip()
    try:
        return f"LOC-{int(code):03d}"
    except ValueError:
        return raw


# ---------------------------------------------------------------------------
# Row mapping + cleaning pipeline
# ---------------------------------------------------------------------------

def _map_and_clean_rows(
    rows: list[dict],
    column_map: dict,
) -> list[dict[str, Any]]:
    """Map raw XMLA rows to Supabase schema with data cleaning.

    Applies clean_numeric to numeric fields, clean_date to dates,
    generates transaction_id, and filters to DB-safe columns only.
    Returns list of cleaned, DB-ready dicts.
    """
    mapping = column_map.get("sales_transactions", {})
    active_mapping = {
        k: v for k, v in mapping.items()
        if v is not None and not k.startswith("_")
    }

    if not active_mapping:
        log.warning("No active column mappings — run --test first.")
        return []

    reverse_map = {v: k for k, v in active_mapping.items()}

    cleaned: list[dict[str, Any]] = []
    skipped = 0

    for row in rows:
        mapped: dict[str, Any] = {}
        for source_col, target_col in reverse_map.items():
            if source_col in row:
                mapped[target_col] = row[source_col]

        for col in _GENERATED_COLS:
            mapped.pop(col, None)

        if not mapped:
            skipped += 1
            continue

        if "transaction_date" in mapped:
            iso_date = clean_date(mapped["transaction_date"])
            if iso_date is None:
                log.debug("Skipping row with bad date: %r", mapped.get("transaction_date"))
                skipped += 1
                continue
            mapped["transaction_date"] = iso_date

        for field in _NUMERIC_FIELDS:
            if field in mapped:
                mapped[field] = clean_numeric(mapped[field])

        # Sales Detail flag dimensions arrive as 'Y' / 'N' strings.  Coerce
        # to booleans so the rows match the schema's BOOLEAN columns.  Any
        # value other than 'Y' (case-insensitive) becomes False — covers
        # 'N', None, and unexpected blanks safely.
        for flag_field in ("is_warranty", "is_backorder",
                           "is_core_return", "is_price_override"):
            if flag_field in mapped:
                v = mapped[flag_field]
                mapped[flag_field] = (
                    isinstance(v, str) and v.strip().upper() == "Y"
                )

        if "location_id" in mapped:
            mapped["location_id"] = _extract_location_code(mapped["location_id"])

        mapped["transaction_id"] = _generate_transaction_id(mapped)

        db_row = {k: v for k, v in mapped.items() if k in _DB_COLUMNS}
        if db_row:
            cleaned.append(db_row)
        else:
            skipped += 1

    if skipped:
        log.debug("Skipped %d rows during mapping/cleaning.", skipped)

    return cleaned


# ---------------------------------------------------------------------------
# Supabase loader
# ---------------------------------------------------------------------------

def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rows with the same transaction_id.

    Sums qty_sold, total_revenue.  Takes the latest unit_price per group.
    This handles the case where the cube returns multiple invoice lines
    for the same SKU × location × date combination.
    """
    # Sales Detail flag dimensions are added to the MDX CROSSJOIN, so a
    # single transaction can appear as multiple rows that differ only in
    # flag values.  Use OR semantics when collapsing them so that a TRUE
    # warranty / backorder / core-return / price-override on ANY contributing
    # line propagates to the merged record — never silently lost.
    _BOOL_OR_FIELDS = (
        "is_warranty", "is_backorder", "is_core_return", "is_price_override",
    )
    # Customer fields (migration 025) are functionally determined by
    # invoice_number — every line of the same invoice MUST belong to the
    # same customer with the same Cust Type / Status / Salesman.  We track
    # conflicts here as a data-integrity tripwire: if the cube ever returns
    # rows that share a transaction_id but disagree on a customer field,
    # the first-row-wins behavior would silently pick one and we'd never
    # know.  Counts are logged at WARNING/CRITICAL after the loop.
    _CUSTOMER_FIELDS = (
        "customer_id", "customer_type", "customer_status", "customer_salesman",
    )
    customer_conflicts: dict[str, int] = {f: 0 for f in _CUSTOMER_FIELDS}
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = row.get("transaction_id", "")
        if tid in groups:
            existing = groups[tid]
            existing["qty_sold"] = (existing.get("qty_sold") or 0) + (row.get("qty_sold") or 0)
            existing["total_revenue"] = (existing.get("total_revenue") or 0) + (row.get("total_revenue") or 0)
            if row.get("unit_price") is not None:
                existing["unit_price"] = row["unit_price"]
            for f in _BOOL_OR_FIELDS:
                if row.get(f):
                    existing[f] = True
            # Customer-field invariant check.  Only count a conflict when
            # both sides are non-NULL and differ — a NULL on either side is
            # treated as "no information" rather than a true disagreement.
            for f in _CUSTOMER_FIELDS:
                new_val = row.get(f)
                old_val = existing.get(f)
                if new_val is not None and old_val is not None and new_val != old_val:
                    customer_conflicts[f] += 1
                elif old_val is None and new_val is not None:
                    # Backfill if the first row had NULL but a later row carries a value.
                    existing[f] = new_val
        else:
            groups[tid] = dict(row)

    total_conflicts = sum(customer_conflicts.values())
    if total_conflicts:
        # Conflicts indicate a violation of the invoice→customer 1:1 invariant
        # the customer-dimension extract relies on.  Log loudly so it surfaces
        # in the nightly run summary; first-row-wins still produces a row, but
        # downstream churn scoring may be assigning transactions to the wrong
        # customer/segment/rep.
        log.warning(
            "DEDUP: customer-field conflicts on duplicate transaction_ids: %s "
            "(total=%d). Investigate cube grain — invoice→customer should be 1:1.",
            customer_conflicts, total_conflicts,
        )
    return list(groups.values())


def _load_to_supabase(
    rows: list[dict],
    column_map: dict,
    dry_run: bool = False,
) -> int:
    """Map, clean, deduplicate, and upsert rows into sales_transactions.

    If dry_run is True, maps and cleans but does not write to Supabase.
    Returns the count of rows that would be (or were) loaded.
    """
    cleaned = _map_and_clean_rows(rows, column_map)

    if not cleaned:
        log.warning("No rows survived column mapping and cleaning.")
        return 0

    pre_dedup = len(cleaned)
    cleaned = _deduplicate_rows(cleaned)
    if len(cleaned) < pre_dedup:
        log.info("Deduplicated %d → %d rows (aggregated %d duplicates).",
                 pre_dedup, len(cleaned), pre_dedup - len(cleaned))

    if dry_run:
        log.info("[DRY RUN] %d rows cleaned and ready (not loaded).", len(cleaned))
        for i, r in enumerate(cleaned[:5]):
            log.info("[DRY RUN]   Row %d: %s", i + 1, r)
        return len(cleaned)

    from db.connection import get_client as get_db_client

    log.info("Loading %d cleaned rows to sales_transactions …", len(cleaned))

    try:
        db = get_db_client()
    except Exception:
        log.exception("Supabase connection failed.")
        raise

    unique_skus = sorted({r["sku_id"] for r in cleaned})
    log.info("Ensuring %d unique SKUs exist in sku_master …", len(unique_skus))
    for i in range(0, len(unique_skus), _BATCH_SIZE):
        sku_batch = [{"sku_id": s} for s in unique_skus[i : i + _BATCH_SIZE]]
        try:
            db.table("sku_master").upsert(
                sku_batch, on_conflict="sku_id", ignore_duplicates=True
            ).execute()
        except Exception:
            log.exception("sku_master upsert failed at offset %d.", i)
            raise
    log.info("sku_master populated.")

    loaded = 0
    for i in range(0, len(cleaned), _BATCH_SIZE):
        batch = cleaned[i : i + _BATCH_SIZE]
        try:
            db.table("sales_transactions").upsert(
                batch, on_conflict="transaction_id"
            ).execute()
            loaded += len(batch)
            if (i // _BATCH_SIZE) % 10 == 0:
                log.debug("  Batch %d–%d upserted.", i, i + len(batch))
        except Exception:
            log.exception("Upsert failed at batch offset %d.", i)
            raise

    # ----------------------------------------------------------------
    # Migration 022 transition guardrail (runs AFTER successful upsert).
    #
    # The new 5-part transaction_ids (`AC-{date}-{sku}-{loc}-INV{inv}`) are
    # different keys from the legacy 4-part ids, so the upsert above does
    # NOT replace the pre-022 aggregated rows for the same (date,sku,loc)
    # tuples — both would coexist and downstream forecast/dead_stock would
    # double-count demand.  Delete the legacy NULL-invoice rows in this
    # chunk's date range now that the new rows are safely persisted.
    #
    # Ordering rationale (delete-AFTER-upsert is intentional):
    #   * Crash mid-upsert → no legacy data lost; next chunk retry re-upserts
    #     idempotently, then deletes.
    #   * Crash between upsert-finish and delete → that chunk's date range is
    #     briefly double-counted, but the next chunk attempt re-runs the
    #     delete and self-heals.  Worst case: the optional final-sweep DELETE
    #     in migration 022's runbook cleans up any orphan rows.
    #   * Reversed order (delete-first) would risk an EMPTY date range on
    #     crash mid-upsert — strictly worse.
    # ----------------------------------------------------------------
    have_invoices = any(r.get("invoice_number") for r in cleaned)
    if have_invoices:
        dates_in_chunk = sorted({r["transaction_date"] for r in cleaned
                                 if r.get("transaction_date")})
        if dates_in_chunk:
            d_lo, d_hi = dates_in_chunk[0], dates_in_chunk[-1]
            try:
                resp = (
                    db.table("sales_transactions")
                    .delete(count="exact")
                    .gte("transaction_date", d_lo)
                    .lte("transaction_date", d_hi)
                    .is_("invoice_number", "null")
                    .execute()
                )
                deleted = resp.count or 0
                if deleted:
                    log.info("Deleted %d legacy (NULL invoice_number) rows in "
                             "[%s, %s] after loading invoice-bearing chunk.",
                             deleted, d_lo, d_hi)
            except Exception:
                # Don't re-raise — the new rows are already persisted.  Log
                # and continue; final-sweep DELETE in migration 022 runbook
                # will clean up any stragglers.
                log.exception("Legacy-row cleanup failed for [%s, %s] AFTER "
                              "successful upsert — chunk may briefly "
                              "double-count until next attempt or final sweep.",
                              d_lo, d_hi)

    log.info("Loaded %d rows to Supabase.", loaded)
    return loaded


# ---------------------------------------------------------------------------
# Mode: INCREMENTAL — previous day
# ---------------------------------------------------------------------------

def run_incremental_extract(dry_run: bool = False) -> int:
    """Pull the previous day's sales and load to Supabase.

    Returns 0 on success, 1 on failure.
    """
    t0 = time.perf_counter()
    yesterday = date.today() - timedelta(days=1)
    date_key = yesterday.strftime("%Y%m%d")

    log.info("=" * 60)
    log.info("  AUTOCUBE INCREMENTAL EXTRACT — %s%s",
             date_key, " [DRY RUN]" if dry_run else "")
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
        _load_to_supabase(rows, column_map, dry_run=dry_run)
    else:
        log.info("No data for %s — may be weekend/holiday or data not yet loaded.",
                 yesterday.isoformat())

    elapsed = time.perf_counter() - t0
    log.info("Incremental extract complete: %d rows in %.1fs", len(rows), elapsed)
    return 0


# ---------------------------------------------------------------------------
# Mode: FULL — single-shot (kept for backwards compat, prefer historical)
# ---------------------------------------------------------------------------

def run_full_extract(dry_run: bool = False) -> int:
    """Pull all sales data in one query. May timeout on large datasets.

    For production use, prefer run_historical_extract() which chunks by month.
    Returns 0 on success, 1 on failure.
    """
    t0 = time.perf_counter()
    log.info("=" * 60)
    log.info("  AUTOCUBE FULL EXTRACT%s", " [DRY RUN]" if dry_run else "")
    log.info("=" * 60)

    try:
        client = get_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    column_map = load_column_map()
    cube = config.AUTOCUBE_CUBE

    mdx = MDX_FULL_SALES.format(cube=cube)

    try:
        rows = client.execute_mdx(mdx)
        log.info("Full extract: %d rows returned.", len(rows))
    except SecurityError:
        log.exception("Security violation — aborting.")
        return 1
    except Exception:
        log.exception("Full extract failed.")
        return 1

    loaded = 0
    if rows:
        loaded = _load_to_supabase(rows, column_map, dry_run=dry_run)

    elapsed = time.perf_counter() - t0
    log.info("Full extract complete: %d rows in %.1fs", loaded, elapsed)
    return 0


# ---------------------------------------------------------------------------
# Mode: HISTORICAL — monthly-chunked full load (Jul 2022 → present)
# ---------------------------------------------------------------------------

_HISTORY_START = date(2022, 7, 1)

_CHUNK_DAYS = 7


def _generate_chunk_ranges(
    start: date,
    end: date,
) -> list[tuple[str, str, str]]:
    """Generate (label, start_key, end_key) tuples for weekly chunks.

    Uses 7-day chunks to keep XMLA responses under ~50K rows,
    avoiding OOM on memory-constrained environments.
    Keys are in YYYYMMDD format matching SSAS date member keys.
    """
    chunks: list[tuple[str, str, str]] = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS - 1), end)
        label = f"{current.isoformat()}..{chunk_end.isoformat()}"
        start_key = current.strftime("%Y%m%d")
        end_key = chunk_end.strftime("%Y%m%d")
        chunks.append((label, start_key, end_key))
        current = chunk_end + timedelta(days=1)
    return chunks


def _label_to_months(label: str) -> set[str]:
    """Extract all YYYY-MM values covered by a chunk label.

    A label like '2022-07-29..2022-08-04' spans two months,
    so both '2022-07' and '2022-08' are returned.
    """
    parts = label.split("..")
    months = {parts[0][:7]}
    if len(parts) == 2:
        months.add(parts[1][:7])
    return months


def _detect_resume_chunk(all_chunks: list[tuple[str, str, str]]) -> int:
    """Query Supabase for the latest loaded date and return the chunk index to resume from.

    Returns 0 if no data exists or on error (start from beginning).
    """
    try:
        from db.connection import get_client as get_db_client
        db = get_db_client()
        resp = (
            db.table("sales_transactions")
            .select("transaction_date")
            .order("transaction_date", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            log.info("[RESUME] No existing data in sales_transactions — starting from chunk 1.")
            return 0

        latest_date_str = resp.data[0]["transaction_date"]
        latest_date = date.fromisoformat(latest_date_str[:10])
        log.info("[RESUME] Latest transaction_date in Supabase: %s", latest_date.isoformat())

        for i, (label, start_key, end_key) in enumerate(all_chunks):
            chunk_end = date(int(end_key[:4]), int(end_key[4:6]), int(end_key[6:8]))
            if chunk_end >= latest_date:
                skip = max(0, i - 1)
                log.info("[RESUME] Resuming from chunk %d (re-processing chunk containing %s for safety).",
                         skip + 1, latest_date.isoformat())
                return skip

        log.info("[RESUME] All chunks appear loaded — will re-verify last few.")
        return max(0, len(all_chunks) - 3)

    except Exception as exc:
        log.warning("[RESUME] Could not detect resume point (%s) — starting from chunk 1.", exc)
        return 0


def run_historical_extract(
    dry_run: bool = False,
    start_chunk: int = 0,
    max_chunks: int = 0,
    months_filter: list[str] | None = None,
    auto_resume: bool = True,
) -> int:
    """Pull historical data month by month and load to Supabase.

    Args:
        dry_run: If True, pull and clean but do not write to Supabase.
        months_filter: If provided, only process these months (YYYY-MM format).
                       Used by --mode retry-failed.
        auto_resume: If True and start_chunk is 0, auto-detect resume point
                     from existing Supabase data.

    Returns 0 if all months succeeded, 1 if any failed.
    """
    t0 = time.perf_counter()

    log.info("=" * 60)
    log.info("  AUTOCUBE HISTORICAL EXTRACT%s",
             " [DRY RUN]" if dry_run else "")
    log.info("=" * 60)

    try:
        client = get_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    column_map = load_column_map()
    cube = config.AUTOCUBE_CUBE

    end_date = date.today() - timedelta(days=1)
    all_chunks = _generate_chunk_ranges(_HISTORY_START, end_date)

    if months_filter:
        filter_set = set(months_filter)
        all_chunks = [
            (l, s, e) for l, s, e in all_chunks
            if _label_to_months(l) & filter_set
        ]
        if not all_chunks:
            log.error("None of the requested months %s are in range.", months_filter)
            return 1
        log.info("Retry mode: processing %d chunk(s) for months: %s",
                 len(all_chunks), ", ".join(months_filter))

    if start_chunk > 0:
        all_chunks = all_chunks[start_chunk:]
    elif auto_resume and not months_filter and not dry_run:
        resume_idx = _detect_resume_chunk(all_chunks)
        if resume_idx > 0:
            log.info("[RESUME] Skipping %d already-loaded chunks.", resume_idx)
            all_chunks = all_chunks[resume_idx:]
    if max_chunks > 0:
        all_chunks = all_chunks[:max_chunks]

    if not all_chunks:
        log.info("No chunks to process after filtering (start_chunk=%d, max_chunks=%d).",
                 start_chunk, max_chunks)
        return 0

    total_chunks = len(all_chunks)
    log.info("Processing %d weekly chunks (start_chunk=%d) from %s to %s",
             total_chunks, start_chunk, all_chunks[0][0], all_chunks[-1][0])

    succeeded = 0
    failed_chunks: list[str] = []
    total_rows = 0

    for idx, (label, start_key, end_key) in enumerate(all_chunks, 1):
        chunk_t0 = time.perf_counter()
        log.info("[AUTOCUBE] Chunk %d/%d: %s (keys %s–%s)",
                 idx, total_chunks, label, start_key, end_key)

        mdx = MDX_MONTHLY_RANGE.format(
            cube=cube, start_key=start_key, end_key=end_key
        )

        try:
            rows = client.execute_mdx(mdx)
        except SecurityError:
            log.exception("Security violation in chunk %s — aborting entire run.", label)
            return 1
        except Exception:
            log.exception("Failed to extract chunk %s — continuing.", label)
            failed_chunks.append(label)
            continue

        chunk_rows = 0
        if rows:
            try:
                chunk_rows = _load_to_supabase(rows, column_map, dry_run=dry_run)
            except Exception:
                log.exception("Failed to load chunk %s — continuing.", label)
                failed_chunks.append(label)
                continue

        del rows

        total_rows += chunk_rows
        succeeded += 1
        chunk_elapsed = time.perf_counter() - chunk_t0
        total_elapsed = time.perf_counter() - t0

        avg_per_chunk = total_elapsed / idx
        remaining_est = avg_per_chunk * (total_chunks - idx)
        remaining_min = int(remaining_est // 60)
        remaining_sec = int(remaining_est % 60)

        log.info(
            "[AUTOCUBE] Chunk %d/%d complete — %s — %d rows loaded — "
            "elapsed %dm %ds — estimated %dm %ds remaining",
            idx, total_chunks, label, chunk_rows,
            int(total_elapsed // 60), int(total_elapsed % 60),
            remaining_min, remaining_sec,
        )

    total_elapsed = time.perf_counter() - t0
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    failed_months = sorted(set(
        m for l in failed_chunks for m in _label_to_months(l)
    ))

    log.info("=" * 60)
    log.info("  HISTORICAL EXTRACT COMPLETE")
    log.info("=" * 60)
    log.info("  Total chunks attempted: %d", total_chunks)
    log.info("  Total chunks succeeded: %d", succeeded)
    log.info("  Total chunks failed:    %d", len(failed_chunks))
    log.info("  Total rows loaded:      %d", total_rows)
    log.info("  Total runtime:          %dm %ds", total_min, total_sec)

    if failed_months:
        log.warning("  Failed months to retry: %s", ", ".join(failed_months))
        log.warning("  Retry command: python -m extract.autocube_pull "
                     "--mode retry-failed --months %s", " ".join(failed_months))

    return 1 if failed_chunks else 0


# ---------------------------------------------------------------------------
# Mode: TRANSFERS-TEST — diagnostic smoke-test (no write)
# ---------------------------------------------------------------------------

_TRANSFERS_TEST_DAYS: int = 30


def run_transfers_test(lookback_days: int = _TRANSFERS_TEST_DAYS) -> int:
    """Diagnostic smoke-test — reports whether T-pattern invoices exist in cube.

    Uses MDX_TRANSFERS_RANGE (no Customer dims, no server-side FILTER) in
    7-day chunks.  Filters Python-side for invoice_number starting with 'T'.
    Stops after the first chunk that returns T-invoices so the test completes
    quickly.  Does NOT write to Supabase.  Returns 0 on success, 1 on failure.
    """
    t0 = time.perf_counter()

    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days - 1)
    chunks     = _generate_chunk_ranges(start_date, end_date)

    log.info("=" * 60)
    log.info("  AUTOCUBE TRANSFERS DIAGNOSTIC (no write)")
    log.info("  Window: %s to %s (%d days, %d chunks)",
             start_date.isoformat(), end_date.isoformat(),
             lookback_days, len(chunks))
    log.info("  MDX: no Customer dims; T-filter applied in Python")
    log.info("=" * 60)

    try:
        client = get_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    column_map = load_column_map()
    cube = config.AUTOCUBE_CUBE

    all_transfer_rows: list[dict] = []
    total_raw_rows = 0
    sample_invoices: list[str] = []  # up to 20 invoice numbers for diagnostics

    for idx, (label, start_key, end_key) in enumerate(chunks, 1):
        log.info("[TRANSFERS-TEST] Chunk %d/%d: %s", idx, len(chunks), label)
        mdx = MDX_TRANSFERS_RANGE.format(
            cube=cube, start_key=start_key, end_key=end_key,
        )
        try:
            rows = client.execute_mdx(mdx)
        except SecurityError:
            log.exception("Security violation — aborting.")
            return 1
        except Exception:
            log.exception("Chunk %s failed — skipping.", label)
            continue

        total_raw_rows += len(rows)
        cleaned = _map_and_clean_rows(rows, column_map)
        del rows

        for r in cleaned:
            inv = r.get("invoice_number") or ""
            if _parse_transfer_invoice(inv) is not None:
                all_transfer_rows.append(r)
            # Collect a diverse sample of invoice numbers for diagnostics
            if len(sample_invoices) < 20 and inv and inv not in sample_invoices:
                sample_invoices.append(inv)

        log.info("[TRANSFERS-TEST] Chunk %d: %d raw rows, %d transfer invoices found so far",
                 idx, len(cleaned), len(all_transfer_rows))

        if all_transfer_rows:
            log.info("T-invoices found — stopping early (run --mode transfers for full extract).")
            break

    elapsed = time.perf_counter() - t0
    log.info("Total raw rows scanned: %d  (%.1fs)", total_raw_rows, elapsed)

    if not all_transfer_rows:
        log.warning(
            "RESULT: No T-pattern invoices found in %d days of cube '%s'.",
            lookback_days, cube,
        )
        if sample_invoices:
            log.info("Sample invoice_number values from cube (first 20): %s",
                     sample_invoices)
            non_numeric = [s for s in sample_invoices if not s.isdigit()]
            if non_numeric:
                log.info("Non-numeric invoice samples: %s", non_numeric)
            else:
                log.info("All sampled invoices are purely numeric. "
                         "Transfer invoices may use a different format or cube.")
        log.info("Try: --mode transfers-test --lookback-days 365 for a wider window.")
        return 0

    # Compute and report stats
    invoices: set[str] = set()
    routes: dict[tuple[str, str], int] = {}
    total_units = 0.0

    for r in all_transfer_rows:
        inv = r.get("invoice_number") or ""
        invoices.add(inv)
        parsed = _parse_transfer_invoice(inv)
        if parsed:
            routes[parsed] = routes.get(parsed, 0) + 1
        qty_raw = r.get("qty_sold") or 0
        try:
            total_units += float(qty_raw)
        except (TypeError, ValueError):
            pass

    log.info("=" * 60)
    log.info("  TRANSFERS DIAGNOSTIC RESULTS")
    log.info("  Transfer invoice rows:  %d", len(all_transfer_rows))
    log.info("  Unique invoice numbers: %d", len(invoices))
    log.info("  Unique routes:          %d", len(routes))
    log.info("  Total units:            %.0f", total_units)
    if routes:
        most_common = max(routes, key=lambda k: routes[k])
        log.info("  Most common route:  %s to %s (%d transfers)",
                 most_common[0], most_common[1], routes[most_common])
        log.info("  All routes:")
        for (frm, to), cnt in sorted(routes.items(), key=lambda x: -x[1]):
            log.info("    %s to %s: %d", frm, to, cnt)
    if invoices:
        sample = sorted(invoices)[:10]
        log.info("  Sample invoice numbers: %s", sample)
    log.info("=" * 60)
    log.info("T-invoices confirmed. Run: python -m extract.autocube_pull --mode transfers")
    return 0


# ---------------------------------------------------------------------------
# Mode: TRANSFERS — extract T-pattern invoices → location_transfers
# ---------------------------------------------------------------------------

_TRANSFERS_LOOKBACK_DAYS: int = 90


def run_transfers_extract(
    dry_run: bool = False,
    lookback_days: int = _TRANSFERS_LOOKBACK_DAYS,
) -> int:
    """Pull last `lookback_days` of transactions from Autocube, filter to
    T-pattern invoice numbers, and upsert into location_transfers.

    Uses MDX_TRANSFERS_RANGE which intentionally omits Customer dimensions.
    Transfer invoices (format "T{from}{to}", e.g. "T2501") have no customer,
    so including Customer dims in the CROSSJOIN causes NON EMPTY to silently
    drop them.  T-invoice detection is done Python-side after each chunk.

    tran_code is set to 'ACTUAL' to distinguish real cube data from the
    INFERRED estimates written by engine/reorder.py.

    Returns 0 on success, 1 on failure.
    """
    t0 = time.perf_counter()

    log.info("=" * 60)
    log.info("  AUTOCUBE TRANSFERS EXTRACT — last %d days%s",
             lookback_days, " [DRY RUN]" if dry_run else "")
    log.info("  Pattern: T{from_loc}{to_loc} (e.g. T2501), no Customer dims")
    log.info("=" * 60)

    try:
        client = get_client()
        client.connect()
    except Exception:
        log.exception("Connection failed.")
        return 1

    column_map = load_column_map()
    cube = config.AUTOCUBE_CUBE

    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days - 1)
    chunks     = _generate_chunk_ranges(start_date, end_date)

    log.info("Pulling %d weekly chunks: %s to %s",
             len(chunks), start_date.isoformat(), end_date.isoformat())

    transfer_rows: list[dict] = []
    total_raw_rows = 0
    failed_chunks: list[str] = []

    for idx, (label, start_key, end_key) in enumerate(chunks, 1):
        log.info("[TRANSFERS] Chunk %d/%d: %s", idx, len(chunks), label)
        mdx = MDX_TRANSFERS_RANGE.format(
            cube=cube, start_key=start_key, end_key=end_key,
        )
        try:
            rows = client.execute_mdx(mdx)
        except SecurityError:
            log.exception("Security violation in chunk %s — aborting.", label)
            return 1
        except Exception:
            log.exception("Failed to extract chunk %s — skipping.", label)
            failed_chunks.append(label)
            continue

        total_raw_rows += len(rows)
        cleaned = _map_and_clean_rows(rows, column_map)
        del rows

        for r in cleaned:
            inv = r.get("invoice_number") or ""
            parsed = _parse_transfer_invoice(inv)
            if parsed is None:
                continue
            from_loc_id, to_loc_id = parsed
            tdate = r.get("transaction_date")
            sku   = r.get("sku_id")
            if not tdate or not sku:
                continue
            transfer_rows.append({
                "transfer_date":    tdate,
                "sku_id":           sku,
                "from_location_id": from_loc_id,
                "to_location_id":   to_loc_id,
                "tran_code":        "ACTUAL",
                "qty_transferred":  r.get("qty_sold") or 0,
                "transfer_value":   r.get("total_revenue"),
            })

    log.info("Raw rows pulled from Autocube: %d", total_raw_rows)
    log.info("Total transfer invoices found: %d", len(transfer_rows))

    if not transfer_rows:
        log.warning(
            "No T-pattern transfer invoices found in the last %d days of "
            "cube '%s'.  Expected format: 'T{from_num}{to_num}' (e.g. 'T2501'). "
            "Run --mode transfers-test to diagnose whether transfers exist in "
            "the cube and verify the invoice number format.",
            lookback_days, cube,
        )
        return 0

    # ---------- statistics ----------
    routes: dict[tuple[str, str], int] = {}
    total_units = 0.0
    for r in transfer_rows:
        key = (r["from_location_id"], r["to_location_id"])
        routes[key] = routes.get(key, 0) + 1
        total_units += float(r.get("qty_transferred") or 0)

    most_common = max(routes, key=lambda k: routes[k])
    log.info("Unique routes found: %d", len(routes))
    log.info("Most common route: %s -> %s (%d transfers)",
             most_common[0], most_common[1], routes[most_common])
    log.info("Total units transferred: %.0f", total_units)

    # ---------- write ----------
    if dry_run:
        log.info("DRY RUN — skipping write.  %d rows would be upserted.",
                 len(transfer_rows))
        elapsed = time.perf_counter() - t0
        log.info("Transfers extract complete (dry run): %.1fs", elapsed)
        return 0

    from db.connection import get_client as get_db_client
    try:
        db = get_db_client()
    except Exception:
        log.exception("Supabase connection failed.")
        return 1

    loaded = 0
    for i in range(0, len(transfer_rows), _BATCH_SIZE):
        batch = transfer_rows[i : i + _BATCH_SIZE]
        try:
            db.table("location_transfers").upsert(
                batch,
                on_conflict=(
                    "transfer_date,sku_id,from_location_id,"
                    "to_location_id,tran_code"
                ),
            ).execute()
            loaded += len(batch)
        except Exception:
            log.exception("location_transfers upsert failed at batch %d.", i)
            raise

    elapsed = time.perf_counter() - t0
    log.info("=" * 60)
    log.info("  TRANSFERS EXTRACT COMPLETE")
    log.info("  Rows loaded:    %d", loaded)
    log.info("  Unique routes:  %d", len(routes))
    log.info("  Total units:    %.0f", total_units)
    log.info("  Elapsed:        %.1fs", elapsed)
    if failed_chunks:
        log.warning("  Failed chunks: %s", ", ".join(failed_chunks))
    log.info("=" * 60)
    return 1 if failed_chunks else 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse arguments and dispatch to the appropriate extract mode."""
    parser = argparse.ArgumentParser(
        description="Autocube OLAP extraction via XMLA/SOAP",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Connection test — list dimensions and measures, no data pull",
    )
    parser.add_argument(
        "--discover-all",
        action="store_true",
        help="Deep discovery — list ALL catalogs, cubes, dimensions, and measures across the entire server",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "historical", "retry-failed",
                 "transfers-test", "transfers"],
        help="Extract mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pull and clean data but do not write to Supabase",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        metavar="YYYY-MM",
        help="Specific months to process (used with --mode retry-failed)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=0,
        help="Skip the first N chunks (resume from chunk N)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Process at most N chunks (0 = all remaining)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume — process every chunk in the historical "
             "range from scratch even if rows already exist for those dates. "
             "Required after migration 022 (and any other schema change that "
             "needs all historical rows repopulated with new columns).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=_TRANSFERS_LOOKBACK_DAYS,
        metavar="N",
        help=(
            f"Days of history to pull (default: {_TRANSFERS_TEST_DAYS} for "
            f"transfers-test, {_TRANSFERS_LOOKBACK_DAYS} for transfers)"
        ),
    )
    args = parser.parse_args()

    if not args.test and not args.discover_all and not args.mode:
        parser.print_help()
        return 1

    if args.discover_all:
        return run_full_discovery()

    if args.test:
        return run_connection_test()

    if args.mode == "full":
        return run_full_extract(dry_run=args.dry_run)

    if args.mode == "incremental":
        return run_incremental_extract(dry_run=args.dry_run)

    if args.mode == "historical":
        return run_historical_extract(
            dry_run=args.dry_run,
            start_chunk=args.start_chunk,
            max_chunks=args.max_chunks,
            auto_resume=not args.no_resume,
        )

    if args.mode == "retry-failed":
        if not args.months:
            log.error("--mode retry-failed requires --months YYYY-MM [YYYY-MM ...]")
            return 1
        return run_historical_extract(
            dry_run=args.dry_run,
            start_chunk=args.start_chunk,
            max_chunks=args.max_chunks,
            months_filter=args.months,
            auto_resume=not args.no_resume,
        )

    if args.mode == "transfers-test":
        return run_transfers_test(lookback_days=args.lookback_days)

    if args.mode == "transfers":
        return run_transfers_extract(
            dry_run=args.dry_run,
            lookback_days=args.lookback_days,
        )

    return 1


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
