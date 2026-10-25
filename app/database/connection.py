"""SQLite connection handling and schema creation.

A new connection is opened for each operation. That is simple, safe with
FastAPI's thread pool, and fast enough for a local SQLite file.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS id_records (
    id                TEXT PRIMARY KEY,
    created_at        TEXT    NOT NULL,
    status            TEXT    NOT NULL,
    document_detected INTEGER NOT NULL,
    face_detected     INTEGER NOT NULL,
    fields_found      INTEGER NOT NULL,
    result_json       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_id_records_created_at ON id_records (created_at);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        """Create the database file and table if they do not exist yet."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success, rolls back on error and always closes."""
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def is_healthy(self) -> bool:
        try:
            with self.connect() as connection:
                connection.execute("SELECT 1 FROM id_records LIMIT 1")
            return True
        except sqlite3.Error:
            return False
