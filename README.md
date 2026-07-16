# Database Agent PoC — Text-to-SQL with Memory Layers

A proof-of-concept agent that answers natural language questions about a PostgreSQL database using a free LLM API, with short-term (conversation buffer) and long-term (summarised memories) memory.

## Architecture

```
User Input
    │
    ▼
┌──────────────────────────────────────────┐
│  DatabaseAgent (src/agent/agent.py)      │
│                                          │
│  ┌─ ReAct decision ──────────────────┐   │
│  │  Gemma decides: REPLY or TOOL     │   │
│  │  (sees long-term + short-term)    │   │
│  └──────────┬────────────────────────┘   │
│             │                            │
│  ┌──────────▼──────────┐                 │
│  │  REPLY: return      │                 │
│  │  directly           │                 │
│  └─────────────────────┘                 │
│  ┌──────────▼──────────┐                 │
│  │  TOOL:              │                 │
│  │  ➜ generate_sql()  │                 │
│  │  ➜ validate_sql()  │                 │
│  │  ➜ execute_sql()   │                 │
│  │  ➜ reflect (Gemma) │ ◄── reasons     │
│  │    evaluates output │     over results│
│  └─────────────────────┘                 │
│                                          │
│  Memory layers:                          │
│  ┌────────────┐ ┌───────────────┐        │
│  │ Short-term │ │ Long-term     │        │
│  │ (RAM)      │ │ (RAM cache +  │        │
│  │ last 6     │ │  DB for       │        │
│  │ turns raw  │ │  persistence) │        │
│  └────────────┘ └───────────────┘        │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────┐
│  PostgreSQL (Docker)         │
│  ┌─────────┐ ┌──────────┐   │
│  │customers│ │ products  │   │
│  │orders   │ │order_items│   │
│  │agent_   │ │          │   │
│  │memory   │ │          │   │
│  └─────────┘ └──────────┘   │
└──────────────────────────────┘
```

## How Tool Calling Works

The agent uses **native API-level function calling** (not text parsing) for Gemma 4:

```
LLM prompt (text):
  "How many customers?"

API metadata (tools=):
  query_database(sql: string) → JSON rows

LLM response:
  finish_reason = "tool_calls"
  message.tool_calls = [
    { name: "query_database", args: { sql: "SELECT count(*) FROM customers" } }
  ]

  ↓

System parses tool_calls → validates SQL → executes → feeds results back

LLM reflection prompt (text):
  "Results: [{'count': 6}] — enough data or need more?"

API metadata (tools=):
  query_database(sql: string) → JSON rows

LLM response:
  finish_reason = "stop"
  message.content = "There are 6 customers."
```

### Key differences from text-parsing agents

| Aspect | Text parsing (our v0.12) | Native tool calling (v0.13+) |
|--------|--------------------------|-----------------------------|
| How tool is detected | Regex on `TOOL`/`REPLY` prefixes | `message.tool_calls` struct from API |
| Output format | Free text + code blocks | Structured JSON with typed arguments |
| Multiple tools | Ambiguous prefixes | Model selects by `name` |
| Fallback | Regex tries best guess | Falls back to text parsing if no `tool_calls` |
| Compatibility | Any model | Models with native tool support |

### Fallback chain

When `tools` are passed to `chat()`, the function returns the full `Message` object. The `_parse_response_from_msg()` function:

1. **If `tool_calls` present** — extracts the first `query_database` call, parses the `sql` argument from JSON
2. **If no `tool_calls`** — falls back to the text-based `_parse_agent_response()` with `TOOL`/`REPLY` prefix parsing

This means the system works with both native-tool models (Gemma 4) and text-only models (Llama 3.2), handling gracefully in the fallback chain.

## Project Structure

```
poc-agent-db/
├── .env.example         # Environment variables template
├── requirements.txt     # Python dependencies
├── todo.txt             # Local scratchpad (gitignored)
└── src/
    ├── __init__.py
    ├── main.py          # CLI entry point with session picker
    ├── db/
    │   ├── __init__.py
    │   ├── connection.py  # SQLAlchemy engine & session management
    │   ├── models.py      # ORM models (Customer, Product, Order, OrderItem)
    │   └── seed.py        # Sample data seeding
    ├── agent/
    │   ├── __init__.py
    │   ├── llm_client.py  # OpenAI-compatible LLM wrapper, fallback chain
    │   ├── text2sql.py    # ReAct agent + SQL pipeline + guardrails
    │   └── agent.py       # Memory orchestration, summarization, rollup
    └── memory/
        ├── __init__.py
        ├── short_term.py  # In-memory conversation buffer (last 10 turns)
        └── long_term.py   # Hierarchical memory: leafs → blocks → broads
```

**Two-model split:**
- **Gemma (31B):** Agent decision (tool call vs direct reply), SQL generation, result reflection — uses native function calling via API `tools` parameter
- **Llama 3.2 (3B):** Hierarchical memory summarization only (leafs, blocks, broads) — uses text prompt, no tool calling

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
- Stores the last 10 conversation turns (5 user + 5 assistant) in RAM
- The last 6 turns are injected verbatim into the agent prompt every turn
- Cleared on restart (ephemeral)

### Long-Term Memory (`src/memory/long_term.py`)

Hierarchical summarization stored in the `agent_memory` PostgreSQL table:

```
Level 1 — Leaf:     Every 5 turns, Llama summarizes the last 5 turns
Level 2 — Block:    When 4 leafs exist, they roll into 1 block summary
Level 3 — Broad:    When 2 blocks exist, they roll into 1 broad summary
```

**Lifecycle example (20 turns):**

```
Turns 1-5:   Leaf1 created (active)
Turns 6-10:  Leaf2 created (active)
Turns 11-15: Leaf3 created (active)
Turns 16-20: Leaf4 created → rollup → Block1 replaces Leaf1-4 (inactive)
```

| After turn | Active in DB | Injected into prompt |
|-----------|-------------|---------------------|
| 1-5 | Leaf1 | Leaf1 + raw turns 1-5 |
| 6-10 | Leaf1, Leaf2 | Leaf1-2 + raw turns 6-10 |
| 11-15 | Leaf1, Leaf2, Leaf3 | Leaf1-3 + raw turns 11-15 |
| 20 | Block1 | Block1 + raw turns 16-20 |
| 25 | Block1, Leaf5 | Block1 + Leaf5 + raw turns 21-25 |

**Context injection behaviour:**
- **Cold start:** Active entries loaded from PostgreSQL into a RAM cache once on the first turn
- **During session:** New leafs appended to RAM cache (zero DB reads). Rollups reload the cache from DB
- **Result:** Zero database queries during normal turns. The last 6 raw turns (from short-term) fill in the detailed recent window

**Summarization model:** Llama 3.2 3B — used for all summarization (leafs, blocks, broads). Not used in the SQL pipeline. Gemma handles agent decisions, SQL generation, and result reflection.

**Persistence:** Leafs, blocks, and broads are all stored permanently in PostgreSQL. Inactive entries remain in the DB (is_active=False) for future retrieval via semantic search.

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
