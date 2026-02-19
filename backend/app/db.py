from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .config import Settings


def _db_path(settings: Settings) -> Path:
    url = settings.database_url
    if url.startswith("sqlite:///"):
        # NOTE: keep Windows paths simple; we store relative db under backend/
        rel = url.removeprefix("sqlite:///")
        return Path(rel).resolve()
    if url.startswith("sqlite://"):
        rel = url.removeprefix("sqlite://")
        return Path(rel).resolve()
    raise ValueError("Only sqlite database_url is supported in this starter")


def connect(settings: Settings) -> sqlite3.Connection:
    path = _db_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(settings: Settings) -> None:
    conn = connect(settings)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              subject TEXT NOT NULL,
              unit TEXT,
              topic TEXT,
              source_type TEXT NOT NULL,
              source_name TEXT NOT NULL,
              source_page INTEGER,
              content TEXT NOT NULL,
              embedding BLOB NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_document_chunks_subject ON document_chunks(subject);
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_chunk(
    *,
    settings: Settings,
    subject: str,
    unit: Optional[str],
    topic: Optional[str],
    source_type: str,
    source_name: str,
    source_page: Optional[int],
    content: str,
    embedding: bytes,
) -> None:
    conn = connect(settings)
    try:
        conn.execute(
            """
            INSERT INTO document_chunks
              (subject, unit, topic, source_type, source_name, source_page, content, embedding, created_at)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject,
                unit,
                topic,
                source_type,
                source_name,
                source_page,
                content,
                sqlite3.Binary(embedding),
                dt.datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_subjects(settings: Settings) -> List[str]:
    conn = connect(settings)
    try:
        rows = conn.execute("SELECT DISTINCT subject FROM document_chunks ORDER BY subject").fetchall()
        return [str(r[0]) for r in rows if r[0]]
    finally:
        conn.close()


def iter_embeddings(settings: Settings, subject: str) -> Iterable[tuple[int, bytes]]:
    conn = connect(settings)
    try:
        rows = conn.execute(
            "SELECT id, embedding FROM document_chunks WHERE subject = ?",
            (subject,),
        ).fetchall()
        for r in rows:
            yield int(r[0]), bytes(r[1])
    finally:
        conn.close()


def fetch_chunks_by_ids(settings: Settings, ids: List[int]) -> List[dict[str, Any]]:
    if not ids:
        return []

    conn = connect(settings)
    try:
        placeholders = ",".join(["?"] * len(ids))
        rows = conn.execute(
            f"""
            SELECT id, subject, unit, topic, source_type, source_name, source_page, content
            FROM document_chunks
            WHERE id IN ({placeholders})
            """,
            tuple(ids),
        ).fetchall()

        result: list[dict[str, Any]] = []
        for r in rows:
            result.append(
                {
                    "id": int(r[0]),
                    "subject": r[1],
                    "unit": r[2],
                    "topic": r[3],
                    "source_type": r[4],
                    "source_name": r[5],
                    "source_page": r[6],
                    "content": r[7],
                }
            )
        return result
    finally:
        conn.close()
