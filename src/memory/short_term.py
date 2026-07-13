import json
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

    def get_last_user_message(self) -> str:
        for entry in reversed(self.history):
            if entry["role"] == "user":
                return entry["content"]
        return ""

    def get_conversation_summary(self) -> str:
        if not self.history:
            return ""
        lines = ["Recent conversation:"]
        for entry in self.history[-6:]:
            speaker = "User" if entry["role"] == "user" else "You (Assistant)"
            content = entry["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def clear(self):
        self.history = []
