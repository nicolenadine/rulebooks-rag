#!/usr/bin/env python3
"""Initialize the PostgreSQL database with the document graph schema.

Uses settings from src.config (settings.postgres_url). Run after starting
Postgres with: docker compose up -d

Usage:
    uv run python scripts/init_db.py
"""

import sys

from src.config import settings
from src.db.database import Database


def main() -> int:
    """Apply schema to the database. Returns 0 on success, 1 on failure."""
    print(f"Connecting to {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db} ...")
    db = Database()
    try:
        db.init_schema_from_sql()
    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            print("Schema may already be applied. If so, you can ignore this.", file=sys.stderr)
        return 1
    print("Database initialized successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
