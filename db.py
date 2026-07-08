"""
SQLite persistence layer for sessions and chat turns.

Schema:
    sessions(session_id TEXT PK, created_at TEXT)
    turns(id INTEGER PK, session_id TEXT FK, role TEXT, content TEXT, created_at TEXT)

All access is synchronous (stdlib sqlite3), matching the sync FastAPI
endpoints, which run in FastAPI's threadpool.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

from config import DB_PATH, TOP_K_TURNS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns (session_id)"
        )


def create_session() -> str:
    session_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
            (session_id, _now()),
        )
    return session_id


def session_exists(session_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return row is not None


def add_turn(session_id: str, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO turns (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )


def get_recent_turns(session_id: str, k: int = TOP_K_TURNS) -> list[dict]:
    """
    Return the last `k` FULL TURNS (k user + k assistant rows = 2k rows),
    in chronological order (oldest first) so they read naturally in a prompt.
    """
    limit_rows = k * 2
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at FROM (
                SELECT role, content, created_at
                FROM turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY created_at ASC
            """,
            (session_id, limit_rows),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_history(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM turns "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
