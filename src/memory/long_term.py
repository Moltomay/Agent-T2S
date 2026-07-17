from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Boolean,
    create_engine, func as sa_func, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

MemoryBase = declarative_base()


class MemoryEntry(MemoryBase):
    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    turn_count = Column(Integer, default=0)
    level = Column(Integer, default=1)       # 1=leaf, 2=block, 3=broad
    is_active = Column(Boolean, default=True)
    turn_start = Column(Integer, default=0)


class LongTermMemory:
    def __init__(self, db_url: str | None = None):
        from src.db.connection import DATABASE_URL
        url = db_url or DATABASE_URL
        self.engine = create_engine(url)
        MemoryBase.metadata.create_all(bind=self.engine)
        self._migrate()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def _migrate(self):
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

    def store(self, session_id: str, summary: str, turn_count: int,
              level: int = 1, turn_start: int | None = None,
              user_id: str | None = None) -> int:
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
        query = self.session.query(MemoryEntry).filter(
            MemoryEntry.session_id == session_id,
            MemoryEntry.is_active == True,
        )
        if level is not None:
            query = query.filter(MemoryEntry.level == level)
        return query.order_by(MemoryEntry.turn_start.asc()).all()

    def mark_inactive(self, entry_ids: list[int]):
        self.session.query(MemoryEntry).filter(
            MemoryEntry.id.in_(entry_ids)
        ).update({"is_active": False})
        self.session.commit()

    def get_available_sessions(self, user_id: str | None = None) -> list[dict]:
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
        entries = self.get_active_entries(session_id)
        if not entries:
            return ""
        lines = ["Past conversation summaries:"]
        for e in entries:
            prefix = {2: "# Block\n  ", 3: "# Overview\n  "}.get(e.level, "  ")
            ts = e.created_at.strftime("%Y-%m-%d %H:%M") if e.created_at else "?"
            lines.append(f"{prefix}[{ts}] (turns {e.turn_start}-{e.turn_count}) {e.summary}")
        return "\n".join(lines)

    def close(self):
        self.session.close()
