"""POST /api/v1/process-id - happy path, response shape and failure handling.

These tests use FakeOcrEngine for predictable text; document detection, image
preprocessing and face extraction run for real on synthetic images.
"""

from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.helpers import FakeOcrEngine, UploadFn, make_settings

RECORD_KEYS = {
    "record_id", "status", "created_at", "processing_time_ms", "image", "document_detection",
    "extracted_fields", "field_details", "ocr", "face", "warnings", "disclaimer",
}


def test_valid_synthetic_image_is_accepted(upload: UploadFn, license_png: bytes) -> None:
    response = upload(license_png)
    assert response.status_code == 201
    body = response.json()
    assert response.headers["location"] == f"/api/v1/records/{body['record_id']}"


def test_response_has_expected_structure(upload: UploadFn, license_png: bytes) -> None:
    body = upload(license_png, filename="synthetic_card.png").json()

    assert set(body) == RECORD_KEYS
    assert len(body["record_id"]) == 32
    assert body["status"] == "completed"
    datetime.fromisoformat(body["created_at"].replace("Z", "+00:00"))
    assert body["disclaimer"].startswith("OCR extraction is best-effort")

    assert body["extracted_fields"] == {
        "full_name": "ALEX SAMPLE",
        "document_number": "SYN-4821-7730",
        "date_of_birth": "1991-03-14",
        "expiry_date": "2031-03-14",
    }
    details = body["field_details"]["date_of_birth"]
    assert details["method"] == "label"
    assert details["raw_text"] == "DOB: 14/03/1991"
    assert details["ocr_confidence"] == pytest.approx(0.95)

    assert body["image"] == {
        "original_filename": "synthetic_card.png",
        "declared_content_type": "image/png",
        "detected_format": "png",
        "color_mode": "RGB",
        "file_size_bytes": len(license_png),
        "width": 1012,
        "height": 638,
    }
    assert body["ocr"]["engine"] == "fake"
    assert body["ocr"]["mean_confidence"] == pytest.approx(0.95)
    assert "NAME: ALEX SAMPLE" in body["ocr"]["raw_text"]
    assert body["ocr"]["lines"][1] == {"text": "NAME: ALEX SAMPLE", "confidence": 0.95}


def test_perspective_photo_is_detected_and_aligned(upload: UploadFn, license_photo_jpg: bytes) -> None:
    detection = upload(license_photo_jpg, "photo.jpg", "image/jpeg").json()["document_detection"]
    assert detection["detected"] is True
    assert detection["fallback_used"] is False
    assert len(detection["corners"]) == 4
    aspect = detection["aligned_width"] / detection["aligned_height"]
    assert 1.4 < aspect < 1.8  # a standard ID card is 1.59:1


def test_flat_scan_falls_back_to_whole_image(upload: UploadFn, license_png: bytes) -> None:
    body = upload(license_png).json()
    assert body["document_detection"]["detected"] is False
    assert body["document_detection"]["fallback_used"] is True
    assert any("whole image" in warning for warning in body["warnings"])


def test_face_thumbnail_is_saved_and_served(
    client: TestClient, upload: UploadFn, license_png: bytes, settings: Settings
) -> None:
    body = upload(license_png).json()
    face = body["face"]
    assert face["status"] == "detected"
    assert face["thumbnail_url"] == f"/api/v1/thumbnails/{body['record_id']}_face.jpg"
    assert (settings.thumbnails_path / f"{body['record_id']}_face.jpg").is_file()

    thumbnail = client.get(face["thumbnail_url"])
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/jpeg"
    assert thumbnail.content[:3] == b"\xff\xd8\xff"


def test_response_does_not_expose_filesystem_paths(upload: UploadFn, license_png: bytes, settings: Settings) -> None:
    text = upload(license_png).text
    assert str(settings.storage_dir) not in text
    assert str(settings.storage_dir).replace("\\", "\\\\") not in text


def test_card_without_face_reports_not_detected(upload: UploadFn, no_face_png: bytes, settings: Settings) -> None:
    body = upload(no_face_png).json()
    assert body["face"]["status"] == "not_detected"
    assert body["face"]["thumbnail_url"] is None
    assert list(settings.thumbnails_path.glob("*.jpg")) == []


def test_face_extraction_crash_does_not_fail_the_request(
    app: FastAPI, upload: UploadFn, license_png: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_extract(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated face detector crash")

    monkeypatch.setattr(app.state.processor.face_extractor, "extract", broken_extract)
    response = upload(license_png)
    assert response.status_code == 201
    body = response.json()
    assert body["face"]["status"] == "error"
    assert body["face"]["thumbnail_url"] is None
    assert body["extracted_fields"]["full_name"] == "ALEX SAMPLE"  # the rest still worked


def test_missing_fields_are_null_not_invented(tmp_path: Path, license_png: bytes) -> None:
    engine = FakeOcrEngine(lines=["SYNTHETIC DRIVER LICENSE", "CLASS: C", "4821 7730"])
    with TestClient(create_app(make_settings(tmp_path), ocr_engine=engine)) as client:
        body = client.post("/api/v1/process-id", files={"file": ("c.png", license_png, "image/png")}).json()
    assert body["status"] == "partial"
    assert body["extracted_fields"] == {
        "full_name": None, "document_number": None, "date_of_birth": None, "expiry_date": None,
    }
    assert all(value is None for value in body["field_details"].values())


def test_ocr_failure_is_stored_with_ocr_failed_status(tmp_path: Path, license_png: bytes) -> None:
    with TestClient(create_app(make_settings(tmp_path), ocr_engine=FakeOcrEngine(fail=True))) as client:
        response = client.post("/api/v1/process-id", files={"file": ("c.png", license_png, "image/png")})
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "ocr_failed"
        assert set(body["extracted_fields"].values()) == {None}
        assert any("OCR failed" in warning for warning in body["warnings"])
        assert client.get(f"/api/v1/records/{body['record_id']}").status_code == 200


def test_unavailable_ocr_engine_returns_503_and_stores_nothing(tmp_path: Path, license_png: bytes) -> None:
    with TestClient(create_app(make_settings(tmp_path), ocr_engine=FakeOcrEngine(available=False))) as client:
        response = client.post("/api/v1/process-id", files={"file": ("c.png", license_png, "image/png")})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "ocr_unavailable"
        assert client.get("/api/v1/records").json()["total"] == 0


def test_ocr_text_can_be_left_out_of_responses(tmp_path: Path, license_png: bytes) -> None:
    app = create_app(make_settings(tmp_path, include_ocr_text=False), ocr_engine=FakeOcrEngine())
    with TestClient(app) as client:
        body = client.post("/api/v1/process-id", files={"file": ("c.png", license_png, "image/png")}).json()
    assert body["ocr"]["raw_text"] is None
    assert body["ocr"]["lines"] == []
    assert body["extracted_fields"]["full_name"] == "ALEX SAMPLE"


def test_debug_images_are_saved_only_when_enabled(tmp_path: Path, license_png: bytes) -> None:
    settings = make_settings(tmp_path, save_debug_images=True)
    with TestClient(create_app(settings, ocr_engine=FakeOcrEngine())) as client:
        response = client.post("/api/v1/process-id", files={"file": ("c.png", license_png, "image/png")})
    record_id = response.json()["record_id"]
    saved = sorted(path.name for path in settings.debug_images_path.iterdir())
    assert saved == [f"{record_id}_aligned.png", f"{record_id}_binary.png", f"{record_id}_enhanced.png"]


def test_unexpected_error_returns_500_without_internal_details(
    app: FastAPI, license_png: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret detail C:\\internal\\path")

    monkeypatch.setattr(app.state.processor.detector, "detect", explode)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/process-id", files={"file": ("c.png", license_png, "image/png")})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text
    assert "Traceback" not in response.text
