"""Revoke sensitive columns from agent_reader and grant SELECT on safe columns only.

Must be run as a superuser (``postgres``). Reads ``SENSITIVE_COLUMNS`` directly
from ``connection.py`` so the grants are always in sync.
"""
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql://postgres:vanna123@localhost:5433/platform_pmo")

# Import the single source of truth
from src.db.connection import SENSITIVE_COLUMNS  # noqa: E402

with ENGINE.begin() as conn:
    for table, sensitive_cols in SENSITIVE_COLUMNS.items():
        rows = conn.execute(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_name = :t AND table_schema = 'public' "
                 "ORDER BY ordinal_position"),
            {"t": table},
        )
        all_cols = [r[0] for r in rows]

        safe_cols = [c for c in all_cols if c not in sensitive_cols]

        column_list = ", ".join(safe_cols)
        conn.execute(text(f"REVOKE ALL ON {table} FROM agent_reader"))
        conn.execute(text(f"GRANT SELECT ({column_list}) ON {table} TO agent_reader"))
        print(f"[OK] {table}: {len(safe_cols)}/{len(all_cols)} columns granted (hidden: {', '.join(sensitive_cols)})")

print("\nDone. Column-level permissions applied.")
