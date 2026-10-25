"""Upload validation: empty, wrong type, disguised, corrupted, oversized and odd-sized files."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.upload_validator import (
    detect_image_format,
    read_upload,
    sanitize_filename,
    validate_image_upload,
)
from app.utils.errors import FileTooLargeError, ImageDimensionError, UnreadableImageError
from tests.helpers import FakeOcrEngine, UploadFn, make_settings

ALL_FORMATS = frozenset({"jpeg", "png", "webp", "bmp", "tiff"})


def _error(response) -> dict:  # noqa: ANN001
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["message"]
    return body["error"]


def test_empty_upload_is_rejected(upload: UploadFn) -> None:
    response = upload(b"")
    assert response.status_code == 400
    assert _error(response)["code"] == "empty_file"


def test_text_file_is_rejected(upload: UploadFn) -> None:
    response = upload(b"hello world", "notes.txt", "text/plain")
    assert response.status_code == 415
    error = _error(response)
    assert error["code"] == "unsupported_file_type"
    assert "jpeg" in error["details"]["allowed_formats"]


def test_pdf_is_rejected(upload: UploadFn) -> None:
    response = upload(b"%PDF-1.7\n%synthetic", "scan.pdf", "application/pdf")
    assert response.status_code == 415


def test_non_image_content_with_image_name_is_rejected(upload: UploadFn) -> None:
    # Filename and MIME type claim PNG, but the bytes are text: content wins.
    response = upload(b"definitely not an image", "card.png", "image/png")
    assert response.status_code == 415
    assert _error(response)["code"] == "unsupported_file_type"


def test_truncated_image_is_rejected(upload: UploadFn, license_png: bytes) -> None:
    response = upload(license_png[:64])
    assert response.status_code == 422
    assert _error(response)["code"] == "unreadable_image"


def test_corrupted_image_body_is_rejected(upload: UploadFn) -> None:
    corrupted = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8
    response = upload(corrupted)
    assert response.status_code == 422
    assert _error(response)["code"] == "unreadable_image"


def test_oversized_upload_is_rejected(upload: UploadFn) -> None:
    too_big = b"\xff\xd8\xff" + b"0" * (3 * 1024 * 1024)  # limit in tests is 2 MB
    response = upload(too_big, "big.jpg", "image/jpeg")
    assert response.status_code == 413
    assert _error(response)["code"] == "file_too_large"


def test_streamed_upload_stops_reading_past_the_limit() -> None:
    with pytest.raises(FileTooLargeError):
        read_upload(io.BytesIO(b"x" * 5000), max_bytes=4096)
    assert read_upload(io.BytesIO(b"x" * 4096), max_bytes=4096) == b"x" * 4096


def test_too_small_image_is_rejected(upload: UploadFn) -> None:
    ok, tiny = cv2.imencode(".png", np.zeros((40, 60, 3), np.uint8))
    response = upload(tiny.tobytes())
    assert response.status_code == 422
    assert _error(response)["code"] == "invalid_image_dimensions"


def test_too_many_pixels_is_rejected() -> None:
    ok, image = cv2.imencode(".png", np.zeros((400, 500, 3), np.uint8))
    with pytest.raises(ImageDimensionError):
        validate_image_upload(
            image.tobytes(), "x.png", "image/png",
            allowed_formats=ALL_FORMATS, max_bytes=10_000_000, max_pixels=100_000, min_side=16,
        )


def test_missing_file_field_returns_validation_error(client: TestClient) -> None:
    response = client.post("/api/v1/process-id", data={"not_file": "x"})
    assert response.status_code == 422
    assert _error(response)["code"] == "validation_error"


def test_format_disabled_in_settings_is_rejected(tmp_path: Path) -> None:
    ok, bmp = cv2.imencode(".bmp", np.full((200, 300, 3), 255, np.uint8))
    app = create_app(make_settings(tmp_path, allowed_image_formats="jpeg,png"), ocr_engine=FakeOcrEngine())
    with TestClient(app) as client:
        response = client.post("/api/v1/process-id", files={"file": ("card.bmp", bmp.tobytes(), "image/bmp")})
    assert response.status_code == 415


FORMAT_CASES = [(".png", "png"), (".jpg", "jpeg"), (".webp", "webp"), (".bmp", "bmp"), (".tiff", "tiff")]


@pytest.mark.parametrize("extension,expected", FORMAT_CASES)
def test_every_supported_format_is_decoded(extension: str, expected: str) -> None:
    ok, encoded = cv2.imencode(extension, np.full((120, 180, 3), 200, np.uint8))
    assert ok
    data = encoded.tobytes()
    assert detect_image_format(data) == expected
    result = validate_image_upload(
        data, f"card{extension}", None,
        allowed_formats=ALL_FORMATS, max_bytes=10_000_000, max_pixels=40_000_000, min_side=100,
    )
    assert (result.width, result.height) == (180, 120)
    assert result.image.shape == (120, 180, 3)


def test_unreadable_header_raises_unreadable_image() -> None:
    with pytest.raises(UnreadableImageError):
        validate_image_upload(
            b"\xff\xd8\xff" + b"\x00" * 100, "x.jpg", "image/jpeg",
            allowed_formats=ALL_FORMATS, max_bytes=10_000_000, max_pixels=40_000_000, min_side=16,
        )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\someone\\card.png", "card.png"),
        ("<script>alert(1)</script>.png", "script_.png"),
        ("normal name (1).jpg", "normal name (1).jpg"),
        ("..", None),
        ("", None),
        (None, None),
    ],
)
def test_filenames_are_sanitised(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected
