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

RESPONSE_SYSTEM_PROMPT = """Given results from a database query, answer the user's question in 1-2 clear sentences. Do not mention SQL or queries — just answer naturally. If the result is empty, say so."""


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


def format_response(question: str, sql: str, results: list, row_count: int) -> str:
    if row_count == 0:
        return f"No results found.\n\n---\n*SQL query used:* `{sql}`"

    preview = json.dumps(results[:10], indent=2, default=str)
    if len(results) > 10:
        preview += "\n..."

    answer = chat([
        {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\nResults ({row_count} rows):\n{preview}",
        },
    ])

    return f"{answer}\n\n---\n*SQL query used:* `{sql}`"


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
        answer = format_response(question, sql, results, row_count)
        return {"success": True, "sql": sql, "answer": answer}
    except Exception as e:
        return {
            "success": False,
            "sql": sql,
            "answer": f"Query error: {e}\n\n---\n*SQL attempted:* `{sql}`",
        }
