"""Memory orchestration — ties the LLM, short/long-term memory, and session persistence together.

The ``DatabaseAgent`` class is the top-level coordinator. It manages turn
counting, hierarchical memory rollups, session file persistence, and
delegates query processing to ``process_question``.
"""

import uuid
import os

from src.agent.text2sql import process_question
from src.agent.llm_client import chat
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.user_facts import UserFactsMemory
from src.memory import session_log


SESSION_FILE: str = os.path.join(os.path.dirname(__file__), "..", "..", ".session_id")


def _load_session_id() -> str:
    """Read the last active session id from disk, or return empty string."""
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE) as f:
                sid = f.read().strip()
                if sid:
                    return sid
    except Exception:
        pass
    return ""


def _save_session_id(sid: str) -> None:
    """Persist the current session id to disk."""
    try:
        with open(SESSION_FILE, "w") as f:
            f.write(sid)
    except Exception:
        pass


def _generate_session_id() -> str:
    """Create a short random session id and persist it."""
    sid = str(uuid.uuid4())[:8]
    _save_session_id(sid)
    return sid


class DatabaseAgent:
    """Orchestrates conversation turns, memory layers, and session persistence.

    - Routes every user turn through ``process_question`` (ReAct loop).
    - Tracks turn count and triggers leaf memory creation every 5 turns.
    - Handles hierarchical rollup (leafs → blocks → broads).
    - Logs every turn to the session markdown file for ``search_memories``.
    - Optionally enforces Split-Plane RLS via ``pmo_user_id`` / ``project_ids``.
    """

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        pmo_user_id: str | None = None,
        project_ids: list | None = None,
    ) -> None:
        if session_id:
            self.session_id: str = session_id
            _save_session_id(session_id)
        else:
            self.session_id: str = _generate_session_id()
        self.user_id: str | None = user_id
        self.pmo_user_id: str | None = pmo_user_id
        self.project_ids: list | None = project_ids
        self.short_term: ShortTermMemory = ShortTermMemory(max_turns=10)
        self.long_term: LongTermMemory = LongTermMemory()
        self.user_facts: UserFactsMemory | None = UserFactsMemory() if user_id else None
        self.turn_count: int = 0
        self._cached_context: str = ""

    def _load_cached_context(self) -> str:
        """Fetch and cache active long-term memory summaries from the DB."""
        self._cached_context = self.long_term.get_summary_context(self.session_id)
        return self._cached_context

    def process_message(self, user_message: str) -> str:
        """Process one user message: log, route to LLM, reflect, summarise.

        Returns the final natural-language answer string.
        """
        self.short_term.add("user", user_message)
        self.turn_count += 1
        session_log.append_turn(self.session_id, "user", user_message)

        if self.turn_count == 1:
            self._load_cached_context()

        conv_history: str = self.short_term.get_conversation_summary()
        long_context: str = self._cached_context

        result: dict = process_question(
            user_message,
            long_term_context=long_context,
            conversation_history=conv_history,
            user_id=self.user_id,
            user_facts_memory=self.user_facts,
            session_id=self.session_id,
            pmo_user_id=self.pmo_user_id,
            project_ids=self.project_ids,
        )

        answer: str = result.get("answer", "Error: no response from agent.")
        self.short_term.add("assistant", answer)
        session_log.append_turn(self.session_id, "assistant", answer)

        reflections: list[dict] = result.get("reflections", [])
        if reflections:
            for i, ref in enumerate(reflections, 1):
                if ref.get("raw"):
                    print(f"  [Reflection {i}] {ref['raw'].strip()}")
                if ref.get("error"):
                    print(f"  [Error {i}] {ref['error']}")
                if ref.get("sql"):
                    print(f"  [SQL {i}] {ref['sql']}")
            print()

        if self.turn_count % 5 == 0:
            self._store_leaf()

        return answer

    def _format_last_turns(self, n: int = 5) -> str:
        """Return the last *n* turns as a formatted string for summarisation."""
        recent = self.short_term.history[-(n * 2):]
        lines: list[str] = []
        for entry in recent:
            role = "User" if entry["role"] == "user" else "Assistant"
            lines.append(f"{role}: {entry['content']}")
        return "\n".join(lines)

    def _store_leaf(self) -> None:
        """Summarise the last 5 turns and persist as a level-1 memory entry."""
        conversation_block: str = self._format_last_turns(5)
        try:
            leaf_summary: str = chat(
                [
                    {"role": "system", "content": "Summarise the following conversation in 2-3 sentences. Preserve all named entities (people, products), specific values (prices, counts, stock levels), and concrete facts. Do not generalize named entities into generic categories."},
                    {"role": "user", "content": conversation_block},
                ],
                model_key="format",
            )
        except Exception:
            leaf_summary = "(summary unavailable)"
        turn_start: int = max(1, self.turn_count - 4)
        self.long_term.store(
            self.session_id, leaf_summary, self.turn_count,
            level=1, turn_start=turn_start,
            user_id=self.user_id,
        )
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self._cached_context += f"\n  [{now}] (turns {turn_start}-{self.turn_count}) {leaf_summary}"
        self._try_rollup()

    def _try_rollup(self) -> None:
        """Roll active leafs into blocks (every 4) and blocks into broads (every 2)."""
        did_rollup: bool = False
        active_leafs = self.long_term.get_active_entries(self.session_id, level=1)
        while len(active_leafs) >= 4:
            batch = active_leafs[:4]
            texts: list[str] = [e.summary for e in batch]
            try:
                block_summary: str = chat(
                    [
                        {"role": "system", "content": "Combine these summaries into a single summary of 2-3 sentences. Preserve all named entities (people, products), specific values (prices, counts), and concrete facts. Do not generalize named entities into generic categories."},
                        {"role": "user", "content": "\n".join(f"- {t}" for t in texts)},
                    ],
                    model_key="format",
                )
            except Exception:
                break
            child_ids: list[int] = [e.id for e in batch]
            self.long_term.store(
                self.session_id, block_summary, batch[-1].turn_count,
                level=2, turn_start=batch[0].turn_start,
                user_id=self.user_id,
            )
            self.long_term.mark_inactive(child_ids)
            did_rollup = True
            active_leafs = self.long_term.get_active_entries(self.session_id, level=1)

        active_blocks = self.long_term.get_active_entries(self.session_id, level=2)
        while len(active_blocks) >= 2:
            batch = active_blocks[:2]
            texts = [e.summary for e in batch]
            try:
                broad_summary: str = chat(
                    [
                        {"role": "system", "content": "Combine these into one high-level summary of 2-3 sentences. Preserve all named entities (people, products), specific values (prices, counts), and concrete facts. Do not generalize named entities into generic categories."},
                        {"role": "user", "content": "\n".join(f"- {t}" for t in texts)},
                    ],
                    model_key="format",
                )
            except Exception:
                break
            child_ids = [e.id for e in batch]
            self.long_term.store(
                self.session_id, broad_summary, batch[-1].turn_count,
                level=3, turn_start=batch[0].turn_start,
                user_id=self.user_id,
            )
            self.long_term.mark_inactive(child_ids)
            did_rollup = True
            active_blocks = self.long_term.get_active_entries(self.session_id, level=2)

        if did_rollup:
            self._load_cached_context()

    def get_history(self) -> list[dict]:
        """Return the raw short-term conversation history."""
        return self.short_term.history

    def get_long_term_summaries(self) -> list:
        """Return all active memory entries for display (``/memory`` command)."""
        return self.long_term.get_active_entries(self.session_id)
