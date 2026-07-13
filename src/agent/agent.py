import uuid
import json
import os

from src.agent.text2sql import query_database
from src.agent.llm_client import chat
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory


SESSION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".session_id")


def _is_memory_question(question: str) -> bool:
    q = question.lower().strip()
    memory_patterns = [
        "what was my last",
        "what did i just",
        "what have we talked",
        "what did we discuss",
        "what was the previous",
        "what was our",
        "do you remember",
        "what did i say",
        "what was my previous",
        "recall my",
        "what is my",
    ]
    return any(p in q for p in memory_patterns)


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


def _format_answer(question: str, result: dict) -> str:
    if not result["success"]:
        return (
            f"Error querying database:\n"
            f"SQL: `{result['sql']}`\n"
            f"Error: {result['error']}"
        )

    sql = result["sql"]
    rows = result["results"]
    count = result["row_count"]

    if count == 0:
        return f"No results.\nSQL: `{sql}`"

    header = f"**Query:** `{sql}`\n**Rows:** {count}\n"
    data = json.dumps(rows[:10], indent=2, default=str)
    truncated = "\n... (truncated)" if len(rows) > 10 else ""
    return header + data + truncated


class DatabaseAgent:
    def __init__(self):
        self.session_id = _load_session_id()
        self.short_term = ShortTermMemory(max_turns=10)
        self.long_term = LongTermMemory()
        self.turn_count = 0

    def process_message(self, user_message: str) -> str:
        self.short_term.add("user", user_message)
        self.turn_count += 1

        if _is_memory_question(user_message):
            answer = self._answer_from_memory(user_message)
        else:
            answer = self._answer_from_database(user_message)

        self.short_term.add("assistant", answer)

        if self.turn_count % 5 == 0:
            self._store_summary(user_message, answer)

        return answer

    def _answer_from_memory(self, question: str) -> str:
        history_text = self.short_term.get_conversation_summary()
        if not history_text:
            return "We haven't had any conversation yet."

        prompt = (
            f"The user asks: '{question}'\n\n"
            f"Answer based on this conversation history:\n{history_text}\n\n"
            "Give a direct, concise answer."
        )
        return chat([
            {"role": "system", "content": "You are a helpful assistant with access to the conversation history."},
            {"role": "user", "content": prompt},
        ])

    def _answer_from_database(self, question: str) -> str:
        conv_history = self.short_term.get_conversation_summary()
        long_context = self.long_term.get_summary_context(self.session_id)

        result = query_database(
            question,
            long_term_context=long_context,
            conversation_history=conv_history,
        )
        return _format_answer(question, result)

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
