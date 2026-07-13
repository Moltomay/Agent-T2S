import uuid
import json

from src.agent.text2sql import query_database
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory


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
        self.session_id = str(uuid.uuid4())[:8]
        self.short_term = ShortTermMemory(max_turns=10)
        self.long_term = LongTermMemory()
        self.turn_count = 0

    def process_message(self, user_message: str) -> str:
        self.short_term.add("user", user_message)
        self.turn_count += 1

        summary_context = self.long_term.get_summary_context(self.session_id)
        result = query_database(user_message, context=summary_context)
        answer = _format_answer(user_message, result)

        self.short_term.add("assistant", answer)

        if self.turn_count % 5 == 0:
            self._store_summary(user_message, answer)

        return answer

    def _store_summary(self, question: str, answer: str):
        from src.agent.llm_client import chat
        summary = chat([
            {"role": "system", "content": "Summarise in 1-2 sentences."},
            {"role": "user", "content": f"Q: {question}\nA: {answer}"},
        ])
        self.long_term.store(self.session_id, summary, self.turn_count)

    def get_history(self) -> list[dict]:
        return self.short_term.history

    def get_long_term_summaries(self) -> list:
        return self.long_term.get_recent(self.session_id)
