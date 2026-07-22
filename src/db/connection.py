"""PostgreSQL connection management and schema introspection.

Provides a shared engine, session factory, and utilities for executing
raw SQL and fetching table schemas for the LLM.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:poc_password@localhost:5432/agent_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_session():
    """Return a new SQLAlchemy session bound to the shared engine."""
    return SessionLocal()


def execute_sql(sql: str) -> list[dict]:
    """Execute a raw SELECT statement and return rows as dicts.

    Args:
        sql: A valid PostgreSQL SELECT query.

    Returns:
        List of dicts mapping column names to values.

    Raises:
        Exception: Database errors (invalid SQL, connection issues, etc.).
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]


def get_table_schema() -> str:
    """Introspect the public schema and return a human-readable string for the LLM prompt.

    Excludes internal tables (``agent_memory``, ``user_facts``).
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
        )
        tables: dict[str, list[str]] = {}
        for row in result:
            table = row.table_name
            if table not in tables:
                tables[table] = []
            col_info = (
                f"  - {row.column_name} ({row.data_type}"
                f"{', nullable' if row.is_nullable == 'YES' else ', not null'}"
                f"{f', default={row.column_default}' if row.column_default else ''})"
            )
            tables[table].append(col_info)

        schema_str: str = ""
        for tname, cols in tables.items():
            if tname in ("agent_memory", "user_facts"):
                continue
            schema_str += f"Table: {tname}\n" + "\n".join(cols) + "\n\n"
        return schema_str.strip()
