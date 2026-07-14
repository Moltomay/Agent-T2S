import re
import json

from src.agent.llm_client import chat
from src.db.connection import get_table_schema, execute_sql


SQL_SYSTEM_PROMPT = """You are a PostgreSQL expert. Convert natural language questions into SQL queries.

Database schema:
{schema}

Rules:
- Return ONLY the SQL query, no explanations or markdown formatting
- Use PostgreSQL syntax
- Always use LIMIT when returning raw data rows
- Use aggregate functions with GROUP BY when summarising
- Prefix column names with table names when joining
- Only use SELECT statements
- Never modify or delete data"""


def generate_sql(
    question: str,
    long_term_context: str = "",
    conversation_history: str = "",
) -> str:
    schema = get_table_schema()
    system_prompt = SQL_SYSTEM_PROMPT.format(schema=schema)
    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.append({
            "role": "system",
            "content": (
                "Recent conversation (use to resolve pronouns/follow-ups):\n"
                f"{conversation_history}"
            ),
        })

    if long_term_context:
        messages.append({
            "role": "system",
            "content": long_term_context,
        })

    messages.append({"role": "user", "content": question})
    raw = chat(messages)

    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    sql = sql.strip().rstrip(";")

    return sql


def _format_row(row: dict) -> str:
    """Format a single result row as readable text."""
    parts = []
    for key, val in row.items():
        if val is None:
            continue
        label = key.replace("_", " ").title()
        if isinstance(val, float | int):
            parts.append(f"{label}: {val}")
        else:
            parts.append(f"{label}: {val}")
    return " | ".join(parts)


def _format_results(question: str, results: list[dict], row_count: int) -> str:
    """Format results as clean readable text, no LLM call."""
    if row_count == 0:
        return "No results found."

    cols = list(results[0].keys())
    is_aggregate = len(cols) <= 2 and row_count == 1

    if is_aggregate and len(cols) == 1:
        return str(results[0][cols[0]])

    lines = []
    for row in results[:20]:
        lines.append(_format_row(row))
    if row_count > 20:
        lines.append(f"... and {row_count - 20} more rows")
    return "\n".join(lines)


def process_question(
    question: str,
    long_term_context: str = "",
    conversation_history: str = "",
) -> dict:
    sql = generate_sql(
        question,
        long_term_context=long_term_context,
        conversation_history=conversation_history,
    )

    if not sql:
        return {
            "success": False,
            "sql": "",
            "answer": "Could not generate a valid SQL query.",
        }

    try:
        results = execute_sql(sql)
        row_count = len(results)
        data = _format_results(question, results, row_count)
        answer = f"{data}\n\n---\n*SQL query used:* `{sql}`"
        return {"success": True, "sql": sql, "answer": answer}
    except Exception as e:
        return {
            "success": False,
            "sql": sql,
            "answer": f"Query error: {e}\n\n---\n*SQL attempted:* `{sql}`",
        }
