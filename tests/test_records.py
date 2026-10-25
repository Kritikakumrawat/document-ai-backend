"""GET/DELETE /api/v1/records and thumbnail serving, plus SQLite persistence."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.database.connection import Database
from app.database.repository import RecordRepository
from app.main import create_app
from app.schemas.record import (
    DocumentDetection,
    ExtractedFields,
    FaceThumbnail,
    FieldEvidenceSet,
    IdRecord,
    ImageMetadata,
    OcrOutput,
)
from tests.helpers import FakeOcrEngine, UploadFn, make_settings


def test_record_can_be_retrieved_after_processing(client: TestClient, upload: UploadFn, license_png: bytes) -> None:
    created = upload(license_png).json()
    response = client.get(f"/api/v1/records/{created['record_id']}")
    assert response.status_code == 200
    assert response.json() == created


def test_unknown_record_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/records/" + "0" * 32)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "record_not_found"


def test_malformed_record_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/records/not-a-record-id")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "record_not_found"


def test_records_survive_an_application_restart(tmp_path: Path, license_png: bytes) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings, ocr_engine=FakeOcrEngine())) as first_run:
        created = first_run.post("/api/v1/process-id", files={"file": ("c.png", license_png, "image/png")}).json()

    # A brand-new app instance reading the same SQLite file.
    with TestClient(create_app(settings, ocr_engine=FakeOcrEngine())) as second_run:
        response = second_run.get(f"/api/v1/records/{created['record_id']}")
    assert response.status_code == 200
    assert response.json()["extracted_fields"] == created["extracted_fields"]
    assert settings.database_file.is_file()


def test_list_records_newest_first_with_pagination(client: TestClient, upload: UploadFn, license_png: bytes) -> None:
    first = upload(license_png).json()["record_id"]
    second = upload(license_png).json()["record_id"]

    page = client.get("/api/v1/records").json()
    assert page["total"] == 2
    assert [item["record_id"] for item in page["items"]] == [second, first]
    assert page["items"][0]["fields_found"] == 4
    assert "extracted_fields" not in page["items"][0]  # summaries only

    limited = client.get("/api/v1/records", params={"limit": 1, "offset": 1}).json()
    assert [item["record_id"] for item in limited["items"]] == [first]


def test_list_records_rejects_invalid_pagination(client: TestClient) -> None:
    response = client.get("/api/v1/records", params={"limit": 0})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["location"] == ["query", "limit"]


def test_delete_record_removes_row_and_thumbnail(
    client: TestClient, upload: UploadFn, license_png: bytes, tmp_path: Path
) -> None:
    record_id = upload(license_png).json()["record_id"]
    thumbnail = tmp_path / "thumbnails" / f"{record_id}_face.jpg"
    assert thumbnail.is_file()

    assert client.delete(f"/api/v1/records/{record_id}").status_code == 204
    assert client.get(f"/api/v1/records/{record_id}").status_code == 404
    assert not thumbnail.exists()
    assert client.delete(f"/api/v1/records/{record_id}").status_code == 404


def test_thumbnail_endpoint_rejects_unsafe_names(client: TestClient) -> None:
    for name in ("..%2Frecords.db", "..%5C..%5Crecords.db", "evil.txt", "0" * 32 + "_face.jpg"):
        response = client.get(f"/api/v1/thumbnails/{name}")
        assert response.status_code == 404, name


def _sample_record(record_id: str) -> IdRecord:
    return IdRecord(
        record_id=record_id,
        status="partial",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        processing_time_ms=12,
        image=ImageMetadata(detected_format="png", color_mode="RGB", file_size_bytes=10, width=200, height=100),
        document_detection=DocumentDetection(
            detected=False, method="none", fallback_used=True, aligned_width=200, aligned_height=100
        ),
        extracted_fields=ExtractedFields(full_name="ALEX SAMPLE"),
        field_details=FieldEvidenceSet(),
        ocr=OcrOutput(engine="fake"),
        face=FaceThumbnail(status="not_detected"),
    )


def test_repository_round_trip_uses_sqlite_file(tmp_path: Path) -> None:
    database = Database(tmp_path / "unit.db")
    database.initialize()
    repository = RecordRepository(database)
    record = _sample_record("a" * 32)

    repository.add(record)
    assert RecordRepository(Database(tmp_path / "unit.db")).get("a" * 32) == record
    items, total = repository.list_records(limit=10, offset=0)
    assert total == 1 and items[0].fields_found == 1
    assert repository.delete("a" * 32) is True
    assert repository.get("a" * 32) is None


def test_repository_treats_sql_like_input_as_plain_data(tmp_path: Path) -> None:
    database = Database(tmp_path / "unit.db")
    database.initialize()
    repository = RecordRepository(database)
    repository.add(_sample_record("b" * 32))
    assert repository.get("x' OR '1'='1") is None
    assert repository.list_records(10, 0)[1] == 1
