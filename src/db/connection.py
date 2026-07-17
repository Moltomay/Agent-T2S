import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:poc_password@localhost:5432/agent_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_session():
    return SessionLocal()


def execute_sql(sql: str) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]


def get_table_schema() -> str:
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
        tables = {}
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

        schema_str = ""
        for tname, cols in tables.items():
            if tname in ("agent_memory", "user_facts"):
                continue
            schema_str += f"Table: {tname}\n" + "\n".join(cols) + "\n\n"
        return schema_str.strip()
