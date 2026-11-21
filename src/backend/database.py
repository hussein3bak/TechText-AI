"""
TechText AI - SQLite Database Handler
Persists user prompts, generated texts, timestamps, and model sampling hyperparameters.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "history.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the generations history table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            continuation TEXT NOT NULL,
            full_text TEXT NOT NULL,
            model_name TEXT NOT NULL,
            temperature REAL NOT NULL,
            top_k INTEGER NOT NULL,
            tokens_generated INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_generation(
    prompt: str,
    continuation: str,
    full_text: str,
    model_name: str,
    temperature: float,
    top_k: int,
    tokens_generated: int,
) -> int:
    """Inserts a generation record and returns the new ID."""
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        """
        INSERT INTO generations 
        (prompt, continuation, full_text, model_name, temperature, top_k, tokens_generated, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prompt,
            continuation,
            full_text,
            model_name,
            temperature,
            top_k,
            tokens_generated,
            created_at,
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves the most recent generation records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, prompt, continuation, full_text, model_name, temperature, top_k, tokens_generated, created_at
        FROM generations
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_history_item(item_id: int) -> bool:
    """Deletes a generation item by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM generations WHERE id = ?", (item_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_all_history() -> int:
    """Clears all history records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM generations")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


# Auto-initialize database schema on module load
init_db()

