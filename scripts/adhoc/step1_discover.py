"""Ad-hoc step 1: locate Brookpark + the SL-I tran code across both cubes.

Run from repo root:   python scripts/adhoc/step1_discover.py
Writes:               scripts/adhoc/step1_discover.json
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from extract.autocube_pull import get_client
from utils.logging_config import setup_logging

OUT = ROOT / "scripts" / "adhoc" / "step1_discover.json"


def members(client, cube, dim, hier, lvl=None):
    r = {"CATALOG_NAME": client._catalog, "CUBE_NAME": cube,
         "DIMENSION_UNIQUE_NAME": dim, "HIERARCHY_UNIQUE_NAME": hier}
    if lvl:
        r["LEVEL_UNIQUE_NAME"] = lvl
    try:
        rows = client.discover("MDSCHEMA_MEMBERS", r)
        return [{"name": x.get("MEMBER_NAME"), "unique": x.get("MEMBER_UNIQUE_NAME"),
                 "caption": x.get("MEMBER_CAPTION"), "level": x.get("LEVEL_NUMBER")}
                for x in rows]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    setup_logging()
    out = {}
    client = get_client()
    out["endpoint"] = client.connect()
    out["catalog"] = client._catalog
    out["cubes"] = [c.get("CUBE_NAME") for c in client.list_cubes()]

    for cube in ("Sales Detail", "Product"):
        info = {}
        try:
            hiers = client.list_hierarchies(cube)
            info["hierarchies"] = [
                {"dim": h.get("DIMENSION_UNIQUE_NAME"),
                 "hier": h.get("HIERARCHY_UNIQUE_NAME"),
                 "name": h.get("HIERARCHY_NAME"),
                 "card": h.get("HIERARCHY_CARDINALITY")}
                for h in hiers
            ]
            info["measures"] = [m.get("MEASURE_NAME") for m in client.list_measures(cube)]
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
            out[cube] = info
            continue

        # pull members for anything that smells like tran code / location / transfer
        wanted = {}
        for h in info["hierarchies"]:
            hn = (h.get("name") or "").lower()
            dn = (h.get("dim") or "").lower()
            if any(k in hn or k in dn for k in
                   ("tran", "loc", "branch", "store", "transfer", "whse", "warehouse")):
                wanted[h["hier"]] = members(client, cube, h["dim"], h["hier"])
        info["members"] = wanted
        out[cube] = info

    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        OUT.write_text(json.dumps({"fatal": traceback.format_exc()}, indent=1), encoding="utf-8")
        traceback.print_exc()
        sys.exit(1)
