"""In-memory conversation buffer — the last N raw turns, RAM-only.

Cleared on process restart. Used to inject recent verbatim context
into the agent prompt every turn.
"""

from collections import deque


class ShortTermMemory:
    """Circular buffer storing the last ``max_turns`` user/assistant pairs."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns: int = max_turns
        self.history: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Append a turn and trim to ``max_turns * 2`` entries."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_context(self) -> str:
        """Return the full history as a formatted string, oldest first."""
        if not self.history:
            return ""
        lines: list[str] = []
        for entry in self.history:
            prefix = "User" if entry["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {entry['content']}")
        return "\n".join(lines)

    def get_last_user_message(self) -> str:
        """Return the most recent user message, or empty string."""
        for entry in reversed(self.history):
            if entry["role"] == "user":
                return entry["content"]
        return ""

    def get_conversation_summary(self) -> str:
        """Return the last 6 turns formatted for the system prompt.

        Long messages are truncated to 200 characters.
        """
        if not self.history:
            return ""
        lines: list[str] = ["Recent conversation:"]
        for entry in self.history[-6:]:
            speaker = "User" if entry["role"] == "user" else "You (Assistant)"
            content = entry["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all entries."""
        self.history = []
