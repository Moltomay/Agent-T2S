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


def main():
    from src.db.seed import seed_database
    from src.agent.agent import DatabaseAgent

    print("Initialising database...")
    seed_database()
    print("Database ready.")

    agent = DatabaseAgent()
    print(f"Session ID: {agent.session_id}\n")

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

        print("\nAgent: ", end="", flush=True)
        response = agent.process_message(user_input)
        print(response)


if __name__ == "__main__":
    main()
