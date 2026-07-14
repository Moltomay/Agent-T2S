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
    sid = str(uuid.uuid4())[:8]
    try:
        with open(SESSION_FILE, "w") as f:
            f.write(sid)
    except Exception:
        pass
    return sid


class DatabaseAgent:
    def __init__(self):
        self.session_id = _load_session_id()
        self.short_term = ShortTermMemory(max_turns=10)
        self.long_term = LongTermMemory()
        self.turn_count = 0

    def process_message(self, user_message: str) -> str:
        self.short_term.add("user", user_message)
        self.turn_count += 1

        conv_history = self.short_term.get_conversation_summary()
        long_context = self.long_term.get_summary_context(self.session_id)

        result = process_question(
            user_message,
            long_term_context=long_context,
            conversation_history=conv_history,
        )

        answer = result.get("answer", "Error: no response from agent.")
        self.short_term.add("assistant", answer)

        if self.turn_count % 5 == 0:
            self._store_summary(user_message, answer)

        return answer

    def _store_summary(self, question: str, answer: str):
        summary = chat([
            {"role": "system", "content": "Summarise in 1-2 sentences."},
            {"role": "user", "content": f"Q: {question}\nA: {answer}"},
        ])
        self.long_term.store(self.session_id, summary, self.turn_count)

    def get_history(self) -> list[dict]:
        return self.short_term.history

    def get_long_term_summaries(self) -> list:
        return self.long_term.get_recent(self.session_id)
