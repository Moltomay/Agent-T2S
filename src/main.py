import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv()


def print_banner():
    print("=" * 60)
    print("  Database Agent PoC — Text-to-SQL + Long/Short-Term Memory")
    print("=" * 60)
    print("  Commands:  /exit  quit  |  /history  show conversation")
    print("             /memory   show long-term summaries")
    print("=" * 60)


def _pick_session() -> str | None:
    from src.memory.long_term import LongTermMemory

    long_term = LongTermMemory()
    sessions = long_term.get_available_sessions()
    long_term.close()

    if not sessions:
        return None

    print("\nExisting sessions:")
    print("  " + "-" * 55)
    for i, s in enumerate(sessions, 1):
        ts = s["last_activity"].strftime("%Y-%m-%d %H:%M") if s["last_activity"] else "?"
        print(f"  {i}. {s['session_id']} — {s['turn_count']} turns, {s['summary_count']} memories, last {ts}")
    print("  " + "-" * 55)
    choice = input("  [n]ew session, or pick a number to continue [n]: ").strip().lower()

    if choice == "n" or choice == "":
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx]["session_id"]
    except ValueError:
        pass

    return None


def main():
    from src.db.seed import seed_database
    from src.agent.agent import DatabaseAgent

    print("Initialising database...")
    seed_database()
    print("Database ready.")

    session_id = _pick_session()
    agent = DatabaseAgent(session_id=session_id)
    if session_id:
        print(f"Continuing session: {agent.session_id}\n")
    else:
        print(f"New session: {agent.session_id}\n")

    print_banner()

    while True:
        try:
            user_input = input("\nYou: ").strip()
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
                role = "You" if entry["role"] == "user" else "Agent"
                print(f"\n[{role}]\n{entry['content']}")
            continue

        if user_input.lower() == "/memory":
            summaries = agent.get_long_term_summaries()
            if not summaries:
                print("No long-term memories yet.")
            else:
                for s in summaries:
                    ts = s.created_at.strftime("%Y-%m-%d %H:%M")
                    print(f"\n[{ts}] (turn {s.turn_count})\n{s.summary}")
            continue

        response = agent.process_message(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
