"""CLI entry point — PMO user identity, session picker, and the main REPL loop.

Manages:
- User UUID persistence (``.user_id`` file) for internal session tracking.
- PMO user picker — select a real user from the PMO ``users`` table.
- Split-Plane RLS — computes accessible project IDs for the chosen PMO user.
- Session picker scoped by UUID (``agent_memory`` table).
- Interactive loop with ``/history``, ``/memory``, ``/exit`` commands.
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


def _pick_pmo_user() -> tuple[str, str, list[str]] | None:
    """Show the list of active PMO users and let the user pick one.

    Returns:
        ``(pmo_user_id, display_name, project_ids)`` or ``None`` to skip scoping.
    """
    from src.db.connection import get_pmo_users, get_user_project_ids

    print("\n" + "=" * 60)
    print("  PMO User Identity")
    print("=" * 60)
    try:
        users = get_pmo_users()
    except Exception as e:
        print(f"  Could not fetch PMO users: {e}")
        print("  Continuing without RLS scoping.")
        return None

    if not users:
        print("  No active PMO users found. Continuing without RLS scoping.")
        return None

    print("  Select your user identity (for data access scoping):")
    print("  " + "-" * 55)
    for i, u in enumerate(users, 1):
        print(f"  {i}. {u['display_name']}  ({u['email']})")
    print("  " + "-" * 55)
    print("  0. Skip RLS scoping (see all data)")
    choice: str = input("  Pick a number [1]: ").strip()

    if choice == "0":
        return None

    try:
        idx: int = int(choice) - 1 if choice else 0
        if 0 <= idx < len(users):
            pmo_user = users[idx]
            print(f"  Computing accessible projects for {pmo_user['display_name']}...")
            project_ids = get_user_project_ids(pmo_user["id"])
            print(f"  {pmo_user['display_name']} has access to {len(project_ids)} project(s).")
            return (pmo_user["id"], pmo_user["display_name"], project_ids)
    except ValueError:
        pass

    return None


def _store_display_name(user_id: str, display_name: str, overwrite: bool = False) -> None:
    """Persist the display name in user_facts for future sessions.

    Args:
        user_id: The app UUID.
        display_name: The name to store.
        overwrite: When True (PMO user picked), overwrites any existing name.
    """
    from src.memory.user_facts import UserFactsMemory

    facts = UserFactsMemory()
    if overwrite:
        facts.set_fact(user_id, "name", display_name)
    else:
        existing = facts.get_facts(user_id)
        if "name" not in existing:
            facts.set_fact(user_id, "name", display_name)
    facts.close()


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
    """Start the interactive CLI: identity, PMO user picker, session picker, then REPL loop."""
    from src.agent.agent import DatabaseAgent

    print("Database ready.")

    user_id: str = _load_user_id()
    _backfill_user_id(user_id)

    # PMO user identity for RLS scoping
    pmo_info = _pick_pmo_user()
    if pmo_info:
        pmo_user_id: str = pmo_info[0]
        display_name: str = pmo_info[1]
        project_ids: list[str] = pmo_info[2]
        _store_display_name(user_id, display_name, overwrite=True)
    else:
        pmo_user_id = None
        project_ids = None
        # Fallback: prompt for display name
        from src.memory.user_facts import UserFactsMemory
        facts = UserFactsMemory()
        existing = facts.get_facts(user_id)
        display_name = existing.get("name", "")
        if not display_name:
            print("\n" + "=" * 60)
            display_name = input("  Welcome! What's your name? ").strip() or "User"
            facts.set_fact(user_id, "name", display_name)
        facts.close()

    session_id: str | None = _pick_session(user_id)
    agent = DatabaseAgent(
        session_id=session_id,
        user_id=user_id,
        pmo_user_id=pmo_user_id,
        pmo_user_name=display_name,
        project_ids=project_ids,
    )
    if session_id:
        print(f"\n  Welcome back, {display_name}! Continuing session: {agent.session_id}\n")
    else:
        print(f"\n  Hello, {display_name}! New session: {agent.session_id}\n")

    if pmo_user_id:
        print(f"  RLS scoping active — {len(project_ids)} project(s) accessible.\n")

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
