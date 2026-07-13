# Database Agent PoC — Text-to-SQL with Memory Layers

A proof-of-concept agent that answers natural language questions about a PostgreSQL database using a free LLM API, with short-term (conversation buffer) and long-term (summarised memories) memory.

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────┐
│  DatabaseAgent (src/agent/) │
│  ┌───────────┐ ┌─────────┐  │
│  │ text2sql  │ │ llm_cli │  │
│  │           │ │ ent     │  │
│  └─────┬─────┘ └─────────┘  │
│        │                    │
│  ┌─────▼─────┐              │
│  │ short_term│ memory       │
│  │ long_term │              │
│  └───────────┘              │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  PostgreSQL (Docker)        │
│  ┌─────────┐ ┌───────────┐  │
│  │customers│ │ products   │  │
│  │orders   │ │ order_items│  │
│  │agent_   │ │            │  │
│  │memory   │ │            │  │
│  └─────────┘ └───────────┘  │
└─────────────────────────────┘
```

## Project Structure

```
poc-agent-db/
├── .env.example         # Environment variables template
├── requirements.txt     # Python dependencies
├── test_db.py           # Quick DB verification script
└── src/
    ├── __init__.py
    ├── main.py          # CLI entry point
    ├── db/
    │   ├── __init__.py
    │   ├── connection.py  # SQLAlchemy engine & session management
    │   ├── models.py      # ORM models (Customer, Product, Order, OrderItem)
    │   └── seed.py        # Sample data seeding
    ├── agent/
    │   ├── __init__.py
    │   ├── llm_client.py  # OpenAI-compatible LLM wrapper
    │   ├── text2sql.py    # Text-to-SQL generation & execution
    │   └── agent.py       # Agent orchestrator
    └── memory/
        ├── __init__.py
        ├── short_term.py  # Conversation buffer (last N turns)
        └── long_term.py   # Persisted summaries (stored in DB)
```

## Quick Start

### 1. Start PostgreSQL

```bash
docker run --name poc-postgres -e POSTGRES_PASSWORD=poc_password \
  -e POSTGRES_DB=agent_db -p 5432:5432 -d postgres:16-alpine
```

### 2. Configure LLM API

Copy `.env.example` to `.env` and set your API key.

**OpenRouter** (recommended — no credit card):
1. Sign up at https://openrouter.ai/
2. Create an API key at https://openrouter.ai/keys
3. Set `LLM_API_KEY=your_key` in `.env`

**NVIDIA NIM**:
1. Go to https://build.nvidia.com/settings/api-keys
2. Phone verification required
3. Set `LLM_BASE_URL=https://integrate.api.nvidia.com/v1` in `.env`
4. Set `LLM_MODEL=meta/llama-3.1-70b-instruct` in `.env`

### 3. Install & Run

```bash
pip install -r requirements.txt
python src/main.py
```

### 4. Try Some Questions

- "How many customers do we have?"
- "What are the total sales per product category?"
- "Which customers have the highest total spending?"
- "Show me all pending orders"
- "What products are low on stock?"
- "Which country has the most customers?"

## CLI Commands

| Command      | Description                        |
|-------------|------------------------------------|
| `/exit`     | Exit the application               |
| `/history`  | Show short-term conversation log   |
| `/memory`   | Show long-term memory summaries    |

## Memory Architecture

### Short-Term Memory (`src/memory/short_term.py`)
- Stores the last N conversation turns (configurable, default 10)
- Used as context for the LLM to maintain conversation flow
- In-memory buffer (cleared on restart)

### Long-Term Memory (`src/memory/long_term.py`)
- Every 5 turns, the agent summarises the recent interaction
- Summaries are stored in the `agent_memory` table in PostgreSQL
- On each query, recent summaries are injected as context
- Persists across sessions via the session ID

## LLM Providers Supported

The agent uses the OpenAI-compatible API format, so it works with any provider that supports it:

| Provider         | Base URL                              | Free Tier                          |
|-----------------|---------------------------------------|-----------------------------------|
| OpenRouter      | https://openrouter.ai/api/v1          | Free models available             |
| NVIDIA NIM      | https://integrate.api.nvidia.com/v1   | 40 RPM, phone verification        |
| GitHub Models   | https://models.github.ai/inference    | Free with GitHub account          |
| Any OpenAI-compat | Your provider's URL                 | Varies                            |

## Sample Data

- **6 customers** from France, USA, Italy, South Korea, China
- **10 products** across Electronics, Sports, Home, Stationery, Accessories
- **12 orders** with various statuses (completed, pending, shipped, cancelled)
