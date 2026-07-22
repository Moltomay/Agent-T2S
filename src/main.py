"""CLI entry point — session picker, persistent user identity, and the main REPL loop.

Manages:
- User UUID persistence (``.user_id`` file)
- Display name capture + persistence via ``UserFactsMemory``
- Session picker scoped by user UUID (``/memory`` tables)
- Interactive loop with ``/history``, ``/memory``, ``/exit`` commands
"""

import sys
import os
import uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv()

OLD_USER_ID_FILE: str = os.path.join(os.path.dirname(__file__), "..", "..", ".user_id")
USER_ID_FILE: str = os.path.join(os.path.dirname(__file__), "..", ".user_id")


def _load_user_id() -> str:
    """Load existing user UUID from disk, or generate and persist a new one.

    Migrates from the old (wrong) path to the correct project-root location.
    """
    try:
        if os.path.exists(OLD_USER_ID_FILE):
            with open(OLD_USER_ID_FILE) as f:
                uid = f.read().strip()
                if uid:
                    with open(USER_ID_FILE, "w") as f2:
                        f2.write(uid)
                    os.remove(OLD_USER_ID_FILE)
                    return uid
    except Exception:
        pass

    try:
        if os.path.exists(USER_ID_FILE):
            with open(USER_ID_FILE) as f:
                uid = f.read().strip()
                if uid:
                    return uid
    except Exception:
        pass
    uid: str = str(uuid.uuid4())
    try:
        with open(USER_ID_FILE, "w") as f:
            f.write(uid)
    except Exception:
        pass
    return uid


def _ensure_display_name(user_id: str) -> str:
    """Prompt for the user's display name on first run; persist via UserFactsMemory."""
    from src.memory.user_facts import UserFactsMemory

    facts = UserFactsMemory()
    existing = facts.get_facts(user_id)
    if "name" in existing:
        return existing["name"]
    print("\n" + "=" * 60)
    name: str = input("  Welcome! What's your name? ").strip()
    if name:
        facts.set_fact(user_id, "name", name)
    else:
        name = "User"
    facts.close()
    return name


def print_banner() -> None:
    """Print the welcome banner with available commands."""
    print("=" * 60)
    print("  Database Agent PoC — Text-to-SQL + Memory")
    print("=" * 60)
    print("  Commands:  /exit  quit  |  /history  show conversation")
    print("             /memory   show long-term summaries")
    print("=" * 60)


def _pick_session(user_id: str) -> str | None:
    """Show existing sessions for this user and let them pick one to resume.

    Returns the session id string, or ``None`` for a new session.
    """
    from src.memory.long_term import LongTermMemory

    long_term = LongTermMemory()
    sessions = long_term.get_available_sessions(user_id=user_id)
    long_term.close()

    if not sessions:
        return None

    print("\nExisting sessions:")
    print("  " + "-" * 55)
    for i, s in enumerate(sessions, 1):
        ts: str = s["last_activity"].strftime("%Y-%m-%d %H:%M") if s["last_activity"] else "?"
        print(f"  {i}. {s['session_id']} — {s['turn_count']} turns, {s['summary_count']} memories, last {ts}")
    print("  " + "-" * 55)
    choice: str = input("  [n]ew session, or pick a number to continue [n]: ").strip().lower()

    if choice == "n" or choice == "":
        return None

    try:
        idx: int = int(choice) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx]["session_id"]
    except ValueError:
        pass

    return None


def _backfill_user_id(user_id: str) -> None:
    """Assign ``user_id`` to any agent_memory rows that lack one (migration helper)."""
    from src.memory.long_term import LongTermMemory
    from sqlalchemy import text

    long_term = LongTermMemory()
    with long_term.engine.begin() as conn:
        conn.execute(
            text("UPDATE agent_memory SET user_id = :uid WHERE user_id IS NULL"),
            {"uid": user_id},
        )
    long_term.close()


def main() -> None:
    """Start the interactive CLI: identity, session picker, then REPL loop."""
    from src.agent.agent import DatabaseAgent

    print("Database ready.")

    user_id: str = _load_user_id()
    _backfill_user_id(user_id)
    display_name: str = _ensure_display_name(user_id)
    session_id: str | None = _pick_session(user_id)
    agent = DatabaseAgent(session_id=session_id, user_id=user_id)
    if session_id:
        print(f"\n  Welcome back, {display_name}! Continuing session: {agent.session_id}\n")
    else:
        print(f"\n  Hello, {display_name}! New session: {agent.session_id}\n")

    print_banner()

    while True:
        try:
            user_input: str = input(f"\n{display_name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/history":
            for entry in agent.get_history():
                role: str = "You" if entry["role"] == "user" else "Agent"
                print(f"\n[{role}]\n{entry['content']}")
            continue

        if user_input.lower() == "/memory":
            summaries = agent.get_long_term_summaries()
            if not summaries:
                print("No long-term memories yet.")
            else:
                for s in summaries:
                    ts: str = s.created_at.strftime("%Y-%m-%d %H:%M")
                    print(f"\n[{ts}] (turn {s.turn_count})\n{s.summary}")
            continue

        response: str = agent.process_message(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
