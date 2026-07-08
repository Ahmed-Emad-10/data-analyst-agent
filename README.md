# CSV Agent

Ask natural language questions about any CSV dataset. A CrewAI-powered agent inspects the dataset's schema, writes pandas code to answer your question, and responds in plain English — served via FastAPI with persistent, session-based chat history and a built-in web UI.

## Features

- **Natural language → pandas**: ask questions in plain English; the agent writes and executes the pandas code needed to answer them.
- **Dataset-agnostic**: works with any CSV — the agent reads the schema and sample rows at query time rather than assuming a fixed domain.
- **Any LLM provider**: built on CrewAI's `LLM` class, so you can point it at any provider CrewAI supports (OpenAI, Anthropic, a local Ollama model, or any OpenAI-compatible endpoint) by editing `config.py`.
- **Persistent sessions**: each conversation is tracked by a server-generated `session_id` and stored in SQLite, so history survives restarts.
- **Recency-windowed context**: the last N full turns (configurable) are injected into each prompt, so follow-up questions work without unbounded context growth.
- **Built-in web UI**: a single-file chat frontend served directly by FastAPI — no separate frontend build/deploy step.

## Architecture

```
main.py     — FastAPI app: routes, startup checks, serves the frontend
agent.py    — CrewAI agent + task definition; builds the crew per request
tools.py    — pandas execution tool + schema/sample-row inspection
db.py       — SQLite persistence: sessions, turns, recency-windowed history
config.py   — environment-driven configuration (CSV path, LLM, DB, history size)
static/
  index.html — single-file chat UI (HTML/CSS/JS, no build step)
```

## How it works

1. A CSV is loaded once and cached in memory.
2. On each question, the agent is given the dataset's schema and a few sample rows (never the full raw data) plus the last few turns of conversation history.
3. The agent writes pandas code, executes it against the loaded `DataFrame`, and turns the result into a natural language answer.
4. The question and answer are persisted to SQLite under the session's ID.

## Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn crewai pandas
```

### 2. Configure environment variables

| Variable | Description | Default |
|---|---|---|
| `CSV_PATH` | Path to the CSV file to query | `data.csv` |
| `SCHEMA_SAMPLE_ROWS` | Number of sample rows shown to the agent | `5` |
| `LLM_MODEL` | Model identifier passed to CrewAI's `LLM` | — |
| `LLM_API_KEY` | API key for your chosen LLM provider | — |
| `LLM_BASE_URL` | Base URL of the LLM endpoint | — |
| `DB_PATH` | Path to the SQLite database file | `sessions.db` |
| `TOP_K_TURNS` | Number of most recent full turns injected into each prompt | `5` |

Example:

```bash
export CSV_PATH="path/to/your.csv"
export LLM_MODEL="your-model-id"
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://your-provider-endpoint/v1"
```

> The LLM client is built once at startup in `agent.py` and reused across all requests. Swap in any provider CrewAI supports by adjusting the `LLM(...)` call and the corresponding environment variables in `config.py`.

### 3. Run

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000/` for the chat UI, or use the API directly (see below).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/sessions` | Create a new session, returns `{session_id}` |
| `POST` | `/chat` | Body: `{session_id, question}` → `{answer}` |
| `GET` | `/sessions/{session_id}/history` | Full stored history for a session |
| `DELETE` | `/sessions/{session_id}` | Delete a session and its turns |

`session_id` must be created via `/sessions` before calling `/chat` — it is not auto-created.

## Frontend

A single static HTML file (`static/index.html`) served directly by FastAPI at `/`. It creates a session on first load, persists the `session_id` in the browser's `localStorage` so refreshing the page keeps your conversation, and includes a "New session" button to start fresh.

## Notes

- The dataset is loaded into memory once and cached; large CSVs should fit comfortably in memory before use.
- Chat history is windowed to the most recent `TOP_K_TURNS` full turns per prompt to keep context size bounded; full history remains queryable via `/sessions/{session_id}/history`.
