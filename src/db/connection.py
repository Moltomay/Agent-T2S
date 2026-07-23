"""PostgreSQL connection management and schema introspection.

Provides a shared engine, session factory, and utilities for executing
raw SQL and fetching table schemas for the LLM. Also contains Split-Plane
CTE generation for Row-Level Security (scoping queries by the current
PMO user's accessible projects).
"""

import os
import uuid
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

# ── Split-Plane RLS configuration ──────────────────────────────────────────
# Tables whose rows are scoped by the current user's accessible project IDs.
SCOPED_BY_PROJECT: set[str] = {
    "projects", "milestones", "financial_metrics", "project_statuses",
    "project_team_assignments", "project_partners", "project_temporal_values",
    "projects_history", "milestones_history", "financial_metrics_history",
    "project_statuses_history", "project_team_assignments_history",
    "project_partners_history", "project_temporal_values_history",
    "partners",
}

# Tables scoped by user_id (notifications for the current user only).
SCOPED_BY_USER: set[str] = {"notifications", "notification_preferences"}

# Tables scoped via team membership (users on the same projects).
SCOPED_VIA_TEAM: set[str] = {"team_members", "users"}

# Tables exposed as-is (shared reference data, no sensitive rows).
REFERENCE_TABLES: set[str] = {
    "categories", "project_phases", "project_programs", "project_roles",
    "project_status_types", "milestone_status_types", "user_roles",
    "app_configs", "seed_history", "alembic_version",
}

ALL_SCOPED: set[str] = SCOPED_BY_PROJECT | SCOPED_BY_USER | SCOPED_VIA_TEAM

# Columns stripped from the LLM-visible schema (data privacy).
SENSITIVE_COLUMNS: dict[str, set[str]] = {
    "users": {"phone", "password_hash", "password_reset_token", "password_reset_expires"},
    "team_members": {"phone"},
    "partners": {"address", "phone"},
}


def _is_sensitive(table: str, column: str) -> bool:
    """Check if a column is marked as sensitive and should be hidden from the LLM."""
    return column in SENSITIVE_COLUMNS.get(table, set())


def _safe_columns(table: str, alias: str | None = None) -> str:
    """Return comma-separated list of non-sensitive columns for a table.

    Used by ``build_ctes`` to avoid ``SELECT *`` on tables with
    column-level grants that would be rejected by PostgreSQL.

    Args:
        table: Physical table name (e.g. ``"users"``).
        alias: Optional table alias to prefix each column (e.g. ``"u"``
            produces ``"u.id, u.email, …"``).
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_name = :t AND table_schema = 'public' "
                 "ORDER BY ordinal_position"),
            {"t": table},
        )
        safe = [r[0] for r in rows if r[0] not in SENSITIVE_COLUMNS.get(table, set())]
    prefix = f"{alias}." if alias else ""
    return ", ".join(f"{prefix}{c}" for c in safe)


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


def _quote_ids(ids: list) -> list[str]:
    """Quote a list of values for safe SQL injection.

    Strings (and UUID objects) get single-quoted, numbers stay bare.
    """
    out: list[str] = []
    for v in ids:
        if isinstance(v, str):
            escaped = v.replace("'", "''")
            out.append(f"'{escaped}'")
        elif isinstance(v, uuid.UUID):
            out.append(f"'{v}'")
        else:
            out.append(str(v))
    return out


def build_ctes(project_ids: list, user_id: str) -> str:
    """Build the Split-Plane ``WITH`` block that scopes every table.

    The LLM only sees ``scoped_*`` table names and writes SQL against
    those. This function generates the CTEs that define them.

    Args:
        project_ids: List of project UUIDs (strings) the current user can access.
        user_id: The PMO ``users.id`` for the current user.

    Returns:
        A ``WITH ...`` SQL string, or empty string if no project_ids.
    """
    if not project_ids:
        project_filter: str = "1=0"
        pids: str = ""
    else:
        pids = ", ".join(_quote_ids(project_ids))
        project_filter = f"id IN ({pids})"
    uid: str = _quote_ids([user_id])[0] if user_id else "''"

    ctes: list[str] = [
        # ── Project-scoped ──────────────────────────────────────────────
        f"scoped_projects AS (SELECT * FROM projects WHERE {project_filter} AND is_deleted = false)",
        f"scoped_milestones AS (SELECT m.* FROM milestones m INNER JOIN scoped_projects p ON m.project_id = p.id AND m.is_deleted = false)",
        f"scoped_financial_metrics AS (SELECT f.* FROM financial_metrics f INNER JOIN scoped_projects p ON f.project_id = p.id)",
        f"scoped_project_statuses AS (SELECT ps.* FROM project_statuses ps INNER JOIN scoped_projects p ON ps.project_id = p.id)",
        f"scoped_project_team_assignments AS (SELECT a.* FROM project_team_assignments a INNER JOIN scoped_projects p ON a.project_id = p.id AND a.is_deleted = false)",
        f"scoped_project_partners AS (SELECT pp.* FROM project_partners pp INNER JOIN scoped_projects p ON pp.project_id = p.id)",
        f"scoped_partners AS (SELECT {_safe_columns('partners', 'p')} FROM partners p INNER JOIN scoped_project_partners spp ON p.id = spp.partner_id)",
        f"scoped_project_temporal_values AS (SELECT t.* FROM project_temporal_values t INNER JOIN scoped_projects p ON t.project_id = p.id)",
        # ── History tables (scoped by the same project IDs) ────────────
        f"scoped_projects_history AS (SELECT * FROM projects_history WHERE {project_filter})",
        f"scoped_milestones_history AS (SELECT mh.* FROM milestones_history mh INNER JOIN scoped_projects p ON mh.project_id = p.id)",
        f"scoped_financial_metrics_history AS (SELECT fh.* FROM financial_metrics_history fh INNER JOIN scoped_projects p ON fh.project_id = p.id)",
        f"scoped_project_statuses_history AS (SELECT psh.* FROM project_statuses_history psh INNER JOIN scoped_projects p ON psh.project_id = p.id)",
        f"scoped_project_team_assignments_history AS (SELECT ah.* FROM project_team_assignments_history ah INNER JOIN scoped_projects p ON ah.project_id = p.id)",
        f"scoped_project_partners_history AS (SELECT pph.* FROM project_partners_history pph INNER JOIN scoped_projects p ON pph.project_id = p.id)",
        f"scoped_project_temporal_values_history AS (SELECT th.* FROM project_temporal_values_history th INNER JOIN scoped_projects p ON th.project_id = p.id)",
        # ── Team / user scoped (via project_team_assignments) ───────────
        f"scoped_team_members AS (SELECT DISTINCT {_safe_columns('team_members', 'tm')} FROM team_members tm INNER JOIN scoped_project_team_assignments a ON tm.id = a.team_member_id)",
        f"scoped_users AS (SELECT DISTINCT {_safe_columns('users', 'u')} FROM users u INNER JOIN scoped_team_members tm ON u.id = tm.user_id)",
        # ── User-scoped ────────────────────────────────────────────────
        f"scoped_notifications AS (SELECT * FROM notifications WHERE user_id = {uid} AND is_deleted = false)",
        f"scoped_notification_preferences AS (SELECT * FROM notification_preferences WHERE user_id = {uid})",
    ]

    return "WITH " + ",\n".join(ctes)


def _fetch_schema_rows() -> list[dict]:
    """Return all public-schema column metadata from information_schema."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT table_name, column_name, data_type,
                       is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
        )
        return [dict(r._mapping) for r in result]


def _format_rows_to_schema(rows: list[dict]) -> str:
    """Convert information_schema rows into a human-readable schema string."""
    tables: dict[str, list[str]] = {}
    for row in rows:
        tname: str = row["table_name"]
        cname: str = row["column_name"]
        if _is_sensitive(tname, cname):
            continue
        if tname not in tables:
            tables[tname] = []
        col_info = (
            f"  - {cname} ({row['data_type']}"
            f"{', nullable' if row['is_nullable'] == 'YES' else ', not null'}"
            f"{f', default={row['column_default']}' if row['column_default'] else ''})"
        )
        tables[tname].append(col_info)

    schema_str: str = ""
    for tname, cols in tables.items():
        schema_str += f"Table: {tname}\n" + "\n".join(cols) + "\n\n"
    return schema_str.strip()


def get_table_schema() -> str:
    """Return the full public schema as a human-readable string.

    Excludes internal tables (``agent_memory``, ``user_facts``).
    """
    rows = _fetch_schema_rows()
    rows = [r for r in rows if r["table_name"] not in ("agent_memory", "user_facts")]
    return _format_rows_to_schema(rows)


def get_scoped_schema(project_ids: list, user_id: str) -> str:
    """Return the scoped schema the LLM should see.

    - ``scoped_*`` names for all data tables (filtered by user's access).
    - Original names for reference tables (shared labels).
    - Internal tables (``agent_memory``, ``user_facts``) excluded entirely.
    """
    rows = _fetch_schema_rows()

    # Build display mapping
    scoped_rows: list[dict] = []
    for row in rows:
        tname: str = row["table_name"]
        if tname in ("agent_memory", "user_facts", "alembic_version", "seed_history"):
            continue
        if tname in ALL_SCOPED:
            display_name: str = f"scoped_{tname}"
        elif tname in REFERENCE_TABLES:
            display_name = tname
        else:
            continue  # unknown table — hide it
        scoped_rows.append({**row, "table_name": display_name})

    return _format_rows_to_schema(scoped_rows)


# ── PMO user helper queries ──────────────────────────────────────────────


def get_pmo_users() -> list[dict]:
    """Return all active PMO users for the identity picker.

    Returns:
        List of dicts with keys ``id``, ``display_name``, ``email``.
    """
    rows = execute_sql("""
        SELECT id,
               COALESCE(NULLIF(first_name || ' ' || last_name, ' '), email) AS display_name,
               email
        FROM users
        WHERE is_deleted = false
          AND is_active = true
        ORDER BY display_name
    """)
    return rows


def get_user_project_ids(pmo_user_id: str) -> list[str]:
    """Return the list of project UUIDs accessible by the given PMO user.

    Traverses: ``users → team_members → project_team_assignments → projects``
    """
    uid_quoted: str = _quote_ids([pmo_user_id])[0]
    rows = execute_sql(f"""
        SELECT DISTINCT p.id
        FROM projects p
        INNER JOIN project_team_assignments pta ON p.id = pta.project_id
        INNER JOIN team_members tm ON pta.team_member_id = tm.id
        WHERE tm.user_id = {uid_quoted}
          AND p.is_deleted = false
    """)
    return [r["id"] for r in rows]
