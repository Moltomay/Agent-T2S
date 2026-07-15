import uuid
import os

from src.agent.text2sql import process_question
from src.agent.llm_client import chat
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory


SESSION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".session_id")


def _load_session_id() -> str:
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE) as f:
                sid = f.read().strip()
                if sid:
                    return sid
    except Exception:
        pass
    return ""


def _save_session_id(sid: str):
    try:
        with open(SESSION_FILE, "w") as f:
            f.write(sid)
    except Exception:
        pass


def _generate_session_id() -> str:
    sid = str(uuid.uuid4())[:8]
    _save_session_id(sid)
    return sid


class DatabaseAgent:
    def __init__(self, session_id: str | None = None):
        if session_id:
            self.session_id = session_id
            _save_session_id(session_id)
        else:
            self.session_id = _generate_session_id()
        self.short_term = ShortTermMemory(max_turns=10)
        self.long_term = LongTermMemory()
        self.turn_count = 0
        self._cached_context = ""

    def _load_cached_context(self) -> str:
        self._cached_context = self.long_term.get_summary_context(self.session_id)
        return self._cached_context

    def process_message(self, user_message: str) -> str:
        self.short_term.add("user", user_message)
        self.turn_count += 1

        if self.turn_count == 1:
            self._load_cached_context()

        conv_history = self.short_term.get_conversation_summary()
        long_context = self._cached_context

        result = process_question(
            user_message,
            long_term_context=long_context,
            conversation_history=conv_history,
        )

        answer = result.get("answer", "Error: no response from agent.")
        self.short_term.add("assistant", answer)

        reflections = result.get("reflections", [])
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
        recent = self.short_term.history[-(n * 2):]
        lines = []
        for entry in recent:
            role = "User" if entry["role"] == "user" else "Assistant"
            lines.append(f"{role}: {entry['content']}")
        return "\n".join(lines)

    def _store_leaf(self):
        conversation_block = self._format_last_turns(5)
        try:
            leaf_summary = chat(
                [
                    {"role": "system", "content": "Summarise the following conversation in 2-3 sentences. Preserve all named entities (people, products), specific values (prices, counts, stock levels), and concrete facts. Do not generalize named entities into generic categories."},
                    {"role": "user", "content": conversation_block},
                ],
                model_key="format",
            )
        except Exception:
            leaf_summary = "(summary unavailable)"
        turn_start = max(1, self.turn_count - 4)
        self.long_term.store(
            self.session_id, leaf_summary, self.turn_count,
            level=1, turn_start=turn_start,
        )
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self._cached_context += f"\n  [{now}] (turns {turn_start}-{self.turn_count}) {leaf_summary}"
        self._try_rollup()

    def _try_rollup(self):
        did_rollup = False
        active_leafs = self.long_term.get_active_entries(self.session_id, level=1)
        while len(active_leafs) >= 4:
            batch = active_leafs[:4]
            texts = [e.summary for e in batch]
            try:
                block_summary = chat(
                    [
                        {"role": "system", "content": "Combine these summaries into a single summary of 2-3 sentences. Preserve all named entities (people, products), specific values (prices, counts), and concrete facts. Do not generalize named entities into generic categories."},
                        {"role": "user", "content": "\n".join(f"- {t}" for t in texts)},
                    ],
                    model_key="format",
                )
            except Exception:
                break
            child_ids = [e.id for e in batch]
            self.long_term.store(
                self.session_id, block_summary, batch[-1].turn_count,
                level=2, turn_start=batch[0].turn_start,
            )
            self.long_term.mark_inactive(child_ids)
            did_rollup = True
            active_leafs = self.long_term.get_active_entries(self.session_id, level=1)

        active_blocks = self.long_term.get_active_entries(self.session_id, level=2)
        while len(active_blocks) >= 2:
            batch = active_blocks[:2]
            texts = [e.summary for e in batch]
            try:
                broad_summary = chat(
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
            )
            self.long_term.mark_inactive(child_ids)
            did_rollup = True
            active_blocks = self.long_term.get_active_entries(self.session_id, level=2)

        if did_rollup:
            self._load_cached_context()

    def get_history(self) -> list[dict]:
        return self.short_term.history

    def get_long_term_summaries(self) -> list:
        return self.long_term.get_active_entries(self.session_id)
