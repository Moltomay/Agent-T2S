from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

FactsBase = declarative_base()
MAX_USER_FACTS = 11


class UserFactsEntry(FactsBase):
    __tablename__ = "user_facts"

    user_id = Column(String(100), primary_key=True)
    facts = Column(JSONB, default=dict)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class UserFactsMemory:
    def __init__(self, db_url: str | None = None):
        from src.db.connection import DATABASE_URL

        url = db_url or DATABASE_URL
        self.engine = create_engine(url)
        FactsBase.metadata.create_all(bind=self.engine)
        self._migrate()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def _migrate(self):
        pass

    def get_facts(self, user_id: str) -> dict:
        entry = self.session.query(UserFactsEntry).filter_by(user_id=user_id).first()
        return dict(entry.facts) if entry else {}

    def set_fact(self, user_id: str, key: str, value: str) -> bool:
        entry = self.session.query(UserFactsEntry).filter_by(user_id=user_id).first()
        if not entry:
            entry = UserFactsEntry(user_id=user_id, facts={})
            self.session.add(entry)
        if len(entry.facts) >= MAX_USER_FACTS and key not in entry.facts:
            return False
        entry.facts[key] = value
        entry.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        return True

    def delete_fact(self, user_id: str, key: str) -> bool:
        entry = self.session.query(UserFactsEntry).filter_by(user_id=user_id).first()
        if not entry or key not in entry.facts:
            return False
        del entry.facts[key]
        entry.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        return True

    def format_facts(self, user_id: str) -> str:
        facts = self.get_facts(user_id)
        if not facts:
            return ""
        lines = ["About the user:"]
        for k, v in facts.items():
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    def close(self):
        self.session.close()
