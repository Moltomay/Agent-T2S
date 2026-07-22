"""User-level persistent facts — key-value JSONB store scoped by user UUID.

Facts survive across sessions and database restores. Surfaces implicit
preferences to the LLM as injected system context.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

FactsBase = declarative_base()
MAX_USER_FACTS: int = 11


class UserFactsEntry(FactsBase):
    """Single row in the ``user_facts`` table. One row per user."""

    __tablename__ = "user_facts"

    user_id = Column(String(100), primary_key=True)
    facts = Column(JSONB, default=dict)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class UserFactsMemory:
    """CRUD interface for reading, writing, and formatting user facts.

    Uses SQLAlchemy JSONB; in-place dict mutations require explicit
    reassignment (``entry.facts = dict(entry.facts); facts[k] = v``).
    """

    def __init__(self, db_url: str | None = None) -> None:
        from src.db.connection import DATABASE_URL
        url = db_url or DATABASE_URL
        self.engine = create_engine(url)
        FactsBase.metadata.create_all(bind=self.engine)
        self._migrate()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def _migrate(self) -> None:
        """Placeholder for future schema migrations."""

    def get_facts(self, user_id: str) -> dict:
        """Return all facts for a user as a plain dict (empty if none)."""
        entry = self.session.query(UserFactsEntry).filter_by(user_id=user_id).first()
        return dict(entry.facts) if entry else {}

    def set_fact(self, user_id: str, key: str, value: str) -> bool:
        """Store a fact. Returns False if the 11-fact cap is reached (does not overwrite existing keys)."""
        entry = self.session.query(UserFactsEntry).filter_by(user_id=user_id).first()
        if not entry:
            entry = UserFactsEntry(user_id=user_id, facts={})
            self.session.add(entry)
        if len(entry.facts) >= MAX_USER_FACTS and key not in entry.facts:
            return False
        facts = dict(entry.facts)
        facts[key] = value
        entry.facts = facts
        entry.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        return True

    def delete_fact(self, user_id: str, key: str) -> bool:
        """Remove a fact. Returns False if the key does not exist."""
        entry = self.session.query(UserFactsEntry).filter_by(user_id=user_id).first()
        if not entry or key not in entry.facts:
            return False
        facts = dict(entry.facts)
        facts.pop(key, None)
        entry.facts = facts
        entry.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        return True

    def format_facts(self, user_id: str) -> str:
        """Return user facts as a human-readable string for the LLM prompt, or empty string."""
        facts = self.get_facts(user_id)
        if not facts:
            return ""
        lines: list[str] = ["About the user:"]
        for k, v in facts.items():
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    def close(self) -> None:
        """Release the database session."""
        self.session.close()
