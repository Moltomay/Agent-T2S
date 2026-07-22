"""Persistent session log — every turn is appended to a markdown file.

Provides keyword search over the session file for the search_memories
native tool, and utility to reload recent turns on session resume.
"""

import os
from datetime import datetime


SESSIONS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "..", "sessions")


def _ensure_dir() -> None:
    """Create the sessions/ directory if it does not exist."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def session_path(session_id: str) -> str:
    """Return the absolute path to the markdown file for a given session."""
    return os.path.join(SESSIONS_DIR, f"{session_id}.md")


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to the session markdown file with a timestamp.

    Args:
        session_id: Unique session identifier.
        role: 'user' or 'assistant'.
        content: The message text.
    """
    _ensure_dir()
    path = session_path(session_id)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    label = "User" if role == "user" else "Agent"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"## {ts}\n**{label}:** {content}\n\n")


def load_recent_turns(session_id: str, n: int = 6) -> list[dict]:
    """Read the last *n* user/assistant turns from the session file.

    Used to populate short-term memory when resuming an existing session.

    Returns:
        List of dicts with keys ``role`` and ``content``, newest last.
    """
    path = session_path(session_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    turns: list[dict] = []
    current_role: str | None = None
    current_content: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_role and current_content:
                turns.append({"role": current_role, "content": " ".join(current_content).strip()})
            current_role = None
            current_content = []
        elif line.startswith("**User:**"):
            current_role = "user"
            current_content = [line.split("**User:**", 1)[1].strip()]
        elif line.startswith("**Agent:**"):
            current_role = "assistant"
            current_content = [line.split("**Agent:**", 1)[1].strip()]
        elif current_role and line.strip():
            current_content.append(line.strip())
    if current_role and current_content:
        turns.append({"role": current_role, "content": " ".join(current_content).strip()})
    return turns[-n * 2:]


def search(session_id: str, keyword: str) -> str:
    """Case-insensitive line grep over the session markdown file.

    Returns up to 15 matching lines, or a "not found" message.
    Called by the ``search_memories`` native tool.
    """
    path = session_path(session_id)
    if not os.path.exists(path):
        return "No session history found."
    results: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if keyword.lower() in line.lower():
                results.append(line.strip())
    if not results:
        return f"No matches found for '{keyword}'."
    return "\n".join(results[:15])
