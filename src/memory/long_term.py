from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, create_engine, func as sa_func
)
from sqlalchemy.orm import declarative_base, sessionmaker

MemoryBase = declarative_base()


class MemoryEntry(MemoryBase):
    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    turn_count = Column(Integer, default=0)


class LongTermMemory:
    def __init__(self, db_url: str | None = None):
        from src.db.connection import DATABASE_URL
        url = db_url or DATABASE_URL
        self.engine = create_engine(url)
        MemoryBase.metadata.create_all(bind=self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def store(self, session_id: str, summary: str, turn_count: int):
        entry = MemoryEntry(
            session_id=session_id,
            summary=summary,
            turn_count=turn_count,
        )
        self.session.add(entry)
        self.session.commit()

    def get_recent(self, session_id: str, limit: int = 5) -> list[MemoryEntry]:
        return (
            self.session.query(MemoryEntry)
            .filter(MemoryEntry.session_id == session_id)
            .order_by(MemoryEntry.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_available_sessions(self) -> list[dict]:
        rows = (
            self.session.query(
                MemoryEntry.session_id,
                sa_func.max(MemoryEntry.turn_count).label("turn_count"),
                sa_func.count(MemoryEntry.id).label("summary_count"),
                sa_func.max(MemoryEntry.created_at).label("last_activity"),
            )
            .group_by(MemoryEntry.session_id)
            .order_by(sa_func.max(MemoryEntry.created_at).desc())
            .all()
        )
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
        entries = self.get_recent(session_id)
        if not entries:
            return ""
        lines = ["Past conversation summaries:"]
        for e in reversed(entries):
            lines.append(f"[{e.created_at.strftime('%Y-%m-%d %H:%M')}] {e.summary}")
        return "\n".join(lines)

    def close(self):
        self.session.close()
