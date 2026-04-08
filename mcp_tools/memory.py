"""
MCP Tool: memory_write / memory_read / memory_list
====================================================
Long-term memory system using SQLite for persistent storage.
Stores key-value pairs with timestamps so the agent can
remember player names, events, and preferences across sessions.
"""

import sqlite3
import time
import logging
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mcp.memory")

# Default database path — stored alongside the project
DB_PATH = Path(__file__).parent.parent / "agent_memory.db"


class MemoryTool:
    """MCP tool for persistent long-term memory.

    Uses SQLite to store key-value pairs with timestamps.
    Supports reading, writing, listing, and searching memory.

    Attributes:
        db_path: Path to the SQLite database file
    """

    def __init__(self, db_path: Optional[str] = None):
        # File path for the SQLite database
        self.db_path = db_path or str(DB_PATH)
        # Initialize the database schema
        self._init_db()

    def _init_db(self) -> None:
        """Create the memory table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Memory database initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def memory_write(self, key: str, value: str, category: str = "general") -> dict:
        """Store a key-value pair in long-term memory.

        If the key already exists, its value is updated.

        Args:
            key: Memory key (e.g., 'player_alice_favorite_color')
            value: Memory value (e.g., 'blue')
            category: Optional category for organization

        Returns:
            Status dict with written key details
        """
        key = key.strip()
        if not key:
            return {"success": False, "error": "Empty key"}

        now = time.time()
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO memory (key, value, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = excluded.updated_at
                """,
                (key, value, category, now, now),
            )
            conn.commit()
            logger.info(f"Memory written: {key} = {value[:50]}")
            return {"success": True, "key": key, "value": value}
        except Exception as e:
            logger.error(f"Memory write failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def memory_read(self, key: str) -> dict:
        """Retrieve a value from long-term memory.

        Args:
            key: Memory key to look up

        Returns:
            Dict with value if found, or error if not found
        """
        key = key.strip()
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT value, category, updated_at FROM memory WHERE key = ?",
                (key,),
            ).fetchone()

            if row is None:
                return {"success": True, "found": False, "key": key}

            return {
                "success": True,
                "found": True,
                "key": key,
                "value": row["value"],
                "category": row["category"],
            }
        except Exception as e:
            logger.error(f"Memory read failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def memory_list(self, category: Optional[str] = None, limit: int = 20) -> dict:
        """List all memory keys, optionally filtered by category.

        Args:
            category: Optional filter by category
            limit: Max entries to return (default 20)

        Returns:
            Dict with list of memory entries
        """
        conn = self._get_conn()
        try:
            if category:
                rows = conn.execute(
                    "SELECT key, value, category FROM memory "
                    "WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key, value, category FROM memory "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            entries = [
                {"key": r["key"], "value": r["value"], "category": r["category"]}
                for r in rows
            ]
            return {"success": True, "count": len(entries), "entries": entries}
        except Exception as e:
            logger.error(f"Memory list failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    @property
    def tool_schema_write(self) -> dict:
        """MCP tool definition for memory_write."""
        return {
            "name": "memory_write",
            "description": (
                "Store information in long-term memory. Use this to remember "
                "player names, preferences, events, and important details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key (e.g., 'player_alice_name')",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to remember",
                    },
                },
                "required": ["key", "value"],
            },
        }

    @property
    def tool_schema_read(self) -> dict:
        """MCP tool definition for memory_read."""
        return {
            "name": "memory_read",
            "description": (
                "Recall information from long-term memory. "
                "Use this to remember things about players or past events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key to look up",
                    },
                },
                "required": ["key"],
            },
        }
