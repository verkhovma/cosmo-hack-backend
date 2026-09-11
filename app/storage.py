from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
import os
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "projects.db"))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            scenario TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )"""
    )
    return conn


def save_project(title: str, scenario: dict) -> str:
    pid = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO projects (id, title, scenario) VALUES (?, ?, ?)",
            (pid, title, json.dumps(scenario, ensure_ascii=False)),
        )
    return pid


def load_project(pid: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT scenario FROM projects WHERE id = ?", (pid,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def list_projects() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM projects ORDER BY created_at DESC"
        ).fetchall()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def save_result(project_id: str, result: dict) -> str:
    rid = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO results (id, project_id, result) VALUES (?, ?, ?)",
            (rid, project_id, json.dumps(result, ensure_ascii=False)),
        )
    return rid


def load_result(rid: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT result FROM results WHERE id = ?", (rid,)
        ).fetchone()
    return json.loads(row[0]) if row else None
