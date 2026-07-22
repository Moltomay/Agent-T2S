"""Hierarchical long-term memory stored in PostgreSQL.

Three levels:
- **Leaf** (level 1): Summarised every 5 turns from raw conversation.
- **Block** (level 2): Rolled up from 4 leafs.
- **Broad** (level 3): Rolled up from 2 blocks.

Inactive entries remain in the database for future semantic search.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Boolean,
    create_engine, func as sa_func, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

MemoryBase = declarative_base()


class MemoryEntry(MemoryBase):
    """Single row in the ``agent_memory`` table."""

    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    turn_count = Column(Integer, default=0)
    level = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    turn_start = Column(Integer, default=0)


class LongTermMemory:
    """CRUD interface for the ``agent_memory`` table and hierarchical rollups."""

    def __init__(self, db_url: str | None = None) -> None:
        from src.db.connection import DATABASE_URL
        url = db_url or DATABASE_URL
        self.engine = create_engine(url)
        MemoryBase.metadata.create_all(bind=self.engine)
        self._migrate()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def _migrate(self) -> None:
        """Add columns that may not exist on older schemas (idempotent)."""
        with self.engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS level INTEGER DEFAULT 1")
            )
            conn.execute(
                text("ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE")
            )
            conn.execute(
                text("ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS turn_start INTEGER DEFAULT 0")
            )
            conn.execute(
                text("ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS user_id VARCHAR(100)")
            )

    def store(
        self, session_id: str, summary: str, turn_count: int,
        level: int = 1, turn_start: int | None = None,
        user_id: str | None = None,
    ) -> int:
        """Persist a memory entry and return its auto-generated id."""
        entry = MemoryEntry(
            session_id=session_id,
            user_id=user_id,
            summary=summary,
            turn_count=turn_count,
            level=level,
            is_active=True,
            turn_start=turn_start or turn_count,
        )
        self.session.add(entry)
        self.session.commit()
        return entry.id

    def get_active_entries(self, session_id: str, level: int | None = None) -> list[MemoryEntry]:
        """Return active (non-rolled-up) entries for a session, optionally filtered by level."""
        query = self.session.query(MemoryEntry).filter(
            MemoryEntry.session_id == session_id,
            MemoryEntry.is_active == True,
        )
        if level is not None:
            query = query.filter(MemoryEntry.level == level)
        return query.order_by(MemoryEntry.turn_start.asc()).all()

    def mark_inactive(self, entry_ids: list[int]) -> None:
        """Set ``is_active = False`` for entries that have been rolled up."""
        self.session.query(MemoryEntry).filter(
            MemoryEntry.id.in_(entry_ids)
        ).update({"is_active": False})
        self.session.commit()

    def get_available_sessions(self, user_id: str | None = None) -> list[dict]:
        """Return session metadata (turn count, summary count, last activity) for the session picker."""
        query = self.session.query(
            MemoryEntry.session_id,
            sa_func.max(MemoryEntry.turn_count).label("turn_count"),
            sa_func.count(MemoryEntry.id).label("summary_count"),
            sa_func.max(MemoryEntry.created_at).label("last_activity"),
        ).filter(MemoryEntry.is_active == True)
        if user_id:
            query = query.filter(MemoryEntry.user_id == user_id)
        rows = query.group_by(MemoryEntry.session_id).order_by(
            sa_func.max(MemoryEntry.created_at).desc()
        ).all()
        return [
            {
                "session_id": r.session_id,
                "turn_count": r.turn_count,
                "summary_count": r.summary_count,
                "last_activity": r.last_activity,
            }
            for r in rows
        ]

    def get_summary_context(self, session_id: str) -> str:
        """Format all active entries into a human-readable context block for the LLM."""
        entries = self.get_active_entries(session_id)
        if not entries:
            return ""
        lines: list[str] = ["Past conversation summaries:"]
        for e in entries:
            prefix = {2: "# Block\n  ", 3: "# Overview\n  "}.get(e.level, "  ")
            ts = e.created_at.strftime("%Y-%m-%d %H:%M") if e.created_at else "?"
            lines.append(f"{prefix}[{ts}] (turns {e.turn_start}-{e.turn_count}) {e.summary}")
        return "\n".join(lines)

    def close(self) -> None:
        """Release the database session."""
        self.session.close()
