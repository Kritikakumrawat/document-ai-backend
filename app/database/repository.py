"""Read and write processed records.

Each record is stored as one row: a few searchable columns plus the full API
response as JSON. Every query uses ``?`` placeholders, never string formatting,
so values can never change the meaning of the SQL.
"""

from typing import List, Optional, Tuple

from app.database.connection import Database
from app.schemas.record import IdRecord, RecordSummary


class RecordRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, record: IdRecord) -> None:
        fields_found = sum(value is not None for value in record.extracted_fields.model_dump().values())
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO id_records
                    (id, created_at, status, document_detected, face_detected, fields_found, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.created_at.isoformat(),
                    record.status,
                    int(record.document_detection.detected),
                    int(record.face.status == "detected"),
                    fields_found,
                    record.model_dump_json(),
                ),
            )

    def get(self, record_id: str) -> Optional[IdRecord]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM id_records WHERE id = ?", (record_id,)
            ).fetchone()
        return IdRecord.model_validate_json(row["result_json"]) if row else None

    def list_records(self, limit: int, offset: int) -> Tuple[List[RecordSummary], int]:
        with self.database.connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM id_records").fetchone()[0]
            rows = connection.execute(
                """
                SELECT id, created_at, status, document_detected, face_detected, fields_found
                FROM id_records
                ORDER BY created_at DESC, id
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        items = [
            RecordSummary(
                record_id=row["id"],
                created_at=row["created_at"],
                status=row["status"],
                document_detected=bool(row["document_detected"]),
                face_detected=bool(row["face_detected"]),
                fields_found=row["fields_found"],
            )
            for row in rows
        ]
        return items, int(total)

    def delete(self, record_id: str) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute("DELETE FROM id_records WHERE id = ?", (record_id,))
        return cursor.rowcount > 0
