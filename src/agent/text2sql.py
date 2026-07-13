import re

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
                "This is a conversation. Use the history below to resolve "
                "pronouns and follow-up references in the user's question.\n"
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


def query_database(
    question: str,
    long_term_context: str = "",
    conversation_history: str = "",
) -> dict:
    sql = generate_sql(
        question,
        long_term_context=long_term_context,
        conversation_history=conversation_history,
    )
    try:
        results = execute_sql(sql)
        row_count = len(results)
        return {
            "sql": sql,
            "success": True,
            "results": results[:50],
            "truncated": row_count > 50,
            "row_count": row_count,
        }
    except Exception as e:
        return {
            "sql": sql,
            "success": False,
            "error": str(e),
            "results": [],
        }
