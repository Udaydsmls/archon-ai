import json
import sqlite3
from pathlib import Path

from backend.config import settings
from backend.storage.base import StorageBackend


class SQLiteBackend(StorageBackend):
    """SQLite-backed storage for local development and single-instance deployments."""

    def __init__(self) -> None:
        db_path = settings.database_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._bootstrap()

    def _bootstrap(self) -> None:
        """Create tables if they do not already exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                tenant_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tenant_id, run_id)
            );
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._conn.commit()

    def save_run(self, tenant_id: str, run_id: str, data: dict) -> None:
        """Insert or replace a run record."""
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (tenant_id, run_id, data) VALUES (?, ?, ?)",
            (tenant_id, run_id, json.dumps(data)),
        )
        self._conn.commit()

    def get_run(self, tenant_id: str, run_id: str) -> dict | None:
        """Fetch a run record by composite key."""
        row = self._conn.execute(
            "SELECT data FROM runs WHERE tenant_id=? AND run_id=?", (tenant_id, run_id)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def list_runs(self, tenant_id: str) -> list[dict]:
        """Return all runs for a tenant ordered by creation time descending."""
        rows = self._conn.execute(
            "SELECT data FROM runs WHERE tenant_id=? ORDER BY created_at DESC",
            (tenant_id,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def save_tenant(self, tenant_id: str, data: dict) -> None:
        """Insert or replace a tenant record."""
        self._conn.execute(
            "INSERT OR REPLACE INTO tenants (tenant_id, data) VALUES (?, ?)",
            (tenant_id, json.dumps(data)),
        )
        self._conn.commit()

    def get_tenant(self, tenant_id: str) -> dict | None:
        """Fetch a tenant record by ID."""
        row = self._conn.execute(
            "SELECT data FROM tenants WHERE tenant_id=?", (tenant_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None
