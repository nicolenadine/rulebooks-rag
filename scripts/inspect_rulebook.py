#!/usr/bin/env python3
"""Inspect raw JSON shape of a parsed rulebook file.

Use this to debug ParseLoader mismatches (e.g. different top-level keys,
nested structure, or field names).

Usage:
    uv run python scripts/inspect_rulebook.py data/parsed/flip7_rulebook_parsed.json
"""

import json
import sys
from pathlib import Path
from pprint import pprint


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/inspect_rulebook.py <path-to-json>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text())

    print("ROOT TYPE:", type(data).__name__)

    if isinstance(data, dict):
        print("\nROOT KEYS:")
        pprint(list(data.keys())[:30])

        for key in list(data.keys())[:15]:
            value = data[key]
            print(f"\nKEY: {key!r} -> {type(value).__name__}")
            if isinstance(value, list):
                print("  LEN:", len(value))
                if value and isinstance(value[0], dict):
                    print("  FIRST ITEM KEYS:", list(value[0].keys())[:25])
                elif value:
                    print("  FIRST ITEM (preview):", repr(value[0])[:200])
            elif isinstance(value, dict):
                print("  DICT KEYS:", list(value.keys()))
                if "blocks" in value and isinstance(value["blocks"], list):
                    print("  .blocks LEN:", len(value["blocks"]))
                    if value["blocks"] and isinstance(value["blocks"][0], dict):
                        print("  FIRST BLOCK KEYS:", list(value["blocks"][0].keys()))
                        print("  FIRST BLOCK (preview):")
                        pprint({k: (v if k != "content" else (v[:150] + "..." if len(str(v)) > 150 else v)) for k, v in value["blocks"][0].items()})

    elif isinstance(data, list):
        print("\nLEN:", len(data))
        if data and isinstance(data[0], dict):
            print("FIRST ITEM KEYS:", list(data[0].keys())[:25])
            print("FIRST ITEM (preview):")
            pprint(data[0])
        elif data:
            print("FIRST ITEM:", repr(data[0])[:300])

    return 0


if __name__ == "__main__":
    sys.exit(main())
