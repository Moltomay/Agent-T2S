from collections import deque


class ShortTermMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_context(self) -> str:
        if not self.history:
            return ""
        lines = []
        for entry in self.history:
            prefix = "User" if entry["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {entry['content']}")
        return "\n".join(lines)

    def get_messages(self, system_prompt: str) -> list[dict]:
        return [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]}
            for m in self.history
        ]

    def clear(self):
        self.history = []
