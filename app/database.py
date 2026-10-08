import sqlite3
from enum import StrEnum
from pathlib import Path

from app.paths import ProjectPaths, ensure_data_directories, get_project_paths


class DocumentStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ExtractionMethod(StrEnum):
    NATIVE = "native"
    OCR = "ocr"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    page_count INTEGER CHECK (page_count IS NULL OR page_count >= 0),
    extraction_method TEXT CHECK (
        extraction_method IS NULL OR extraction_method IN ('native', 'ocr')
    ),
    status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETE', 'FAILED')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    UNIQUE (file_path, file_hash)
);

CREATE INDEX IF NOT EXISTS idx_documents_hash_status
    ON documents (file_hash, status);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect_database(database_path: Path | None = None) -> sqlite3.Connection:
    path = database_path or get_project_paths().database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(paths: ProjectPaths | None = None) -> None:
    project_paths = paths or get_project_paths()
    ensure_data_directories(project_paths)
    with connect_database(project_paths.database_path) as connection:
        connection.executescript(_SCHEMA)


def register_document(
    connection: sqlite3.Connection,
    *,
    filename: str,
    file_path: str,
    file_hash: str,
) -> int:
    connection.execute(
        """
        INSERT INTO documents (filename, file_path, file_hash)
        VALUES (?, ?, ?)
        ON CONFLICT (file_path, file_hash) DO NOTHING
        """,
        (filename, file_path, file_hash),
    )
    row = connection.execute(
        "SELECT id FROM documents WHERE file_path = ? AND file_hash = ?",
        (file_path, file_hash),
    ).fetchone()
    if row is None:
        raise RuntimeError("Document registration did not return a document ID")
    return int(row["id"])


def set_document_status(
    connection: sqlite3.Connection,
    document_id: int,
    status: DocumentStatus,
) -> None:
    if status is DocumentStatus.COMPLETE:
        raise ValueError("Use complete_document to mark a document complete")

    cursor = connection.execute(
        """
        UPDATE documents
        SET status = ?, completed_at = NULL
        WHERE id = ?
        """,
        (status.value, document_id),
    )
    if cursor.rowcount != 1:
        raise LookupError(f"Document ID {document_id} does not exist")


def update_page_count(
    connection: sqlite3.Connection,
    document_id: int,
    page_count: int,
) -> None:
    if page_count < 0:
        raise ValueError("page_count must be non-negative")
    cursor = connection.execute(
        "UPDATE documents SET page_count = ? WHERE id = ?",
        (page_count, document_id),
    )
    if cursor.rowcount != 1:
        raise LookupError(f"Document ID {document_id} does not exist")


def complete_document(
    connection: sqlite3.Connection,
    document_id: int,
    *,
    page_count: int,
    extraction_method: ExtractionMethod,
) -> None:
    if page_count < 0:
        raise ValueError("page_count must be non-negative")
    current = connection.execute(
        "SELECT status FROM documents WHERE id = ?",
        (document_id,),
    ).fetchone()
    if current is None:
        raise LookupError(f"Document ID {document_id} does not exist")
    if current["status"] != DocumentStatus.PROCESSING.value:
        raise ValueError("Only a PROCESSING document can be marked complete")

    cursor = connection.execute(
        """
        UPDATE documents
        SET page_count = ?,
            extraction_method = ?,
            status = 'COMPLETE',
            completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (page_count, extraction_method.value, document_id),
    )
    if cursor.rowcount != 1:
        raise LookupError(f"Document ID {document_id} does not exist")


def record_error(
    connection: sqlite3.Connection,
    document_id: int,
    error: str,
) -> int:
    if not error.strip():
        raise ValueError("error must not be empty")
    cursor = connection.execute(
        "INSERT INTO errors (document_id, error) VALUES (?, ?)",
        (document_id, error),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("Error recording did not return an error ID")
    return cursor.lastrowid


def has_completed_hash(connection: sqlite3.Connection, file_hash: str) -> bool:
    row = connection.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM documents
            WHERE file_hash = ? AND status = 'COMPLETE'
        )
        """,
        (file_hash,),
    ).fetchone()
    return bool(row[0])
