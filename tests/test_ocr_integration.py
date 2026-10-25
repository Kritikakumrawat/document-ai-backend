"""End-to-end tests with the REAL OCR engine (no mocks).

They use the engine configured in ``.env`` / environment variables (Tesseract
by default) and are skipped automatically when it is not installed, so the
rest of the suite still runs anywhere. Run only these with::

    python -m pytest -m ocr
"""

from pathlib import Path
from typing import Tuple

import cv2
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services.ocr_service import OcrEngine, create_ocr_engine
from scripts.generate_sample_id import SyntheticIdSpec, badge_spec, encode_image, generate_synthetic_id
from tests.helpers import make_settings

pytestmark = pytest.mark.ocr

LICENSE = ("ALEX SAMPLE", "SYN-4821-7730", "1991-03-14", "2031-03-14")
BADGE = ("JORDAN TESTER", "EMP-004521", "1988-08-02", "2027-12-31")


@pytest.fixture(scope="module")
def real_engine() -> OcrEngine:
    engine = create_ocr_engine(get_settings())
    if not engine.is_available():
        pytest.skip(f"OCR engine '{engine.name}' is not installed or misconfigured; real-OCR tests skipped")
    return engine


def _process(tmp_path: Path, engine: OcrEngine, image_bytes: bytes) -> dict:
    with TestClient(create_app(make_settings(tmp_path), ocr_engine=engine)) as client:
        response = client.post("/api/v1/process-id", files={"file": ("card.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize(
    "spec,scene,expected",
    [
        (SyntheticIdSpec(), False, LICENSE),
        (SyntheticIdSpec(), True, LICENSE),
        (badge_spec(), False, BADGE),
        (badge_spec(), True, BADGE),
    ],
    ids=["license-flat", "license-photo", "badge-flat", "badge-photo"],
)
def test_real_ocr_extracts_fields_from_synthetic_cards(
    tmp_path: Path, real_engine: OcrEngine, spec: SyntheticIdSpec, scene: bool, expected: Tuple[str, ...]
) -> None:
    body = _process(tmp_path, real_engine, encode_image(generate_synthetic_id(spec, scene=scene, seed=4), ".jpg"))
    fields = body["extracted_fields"]
    assert (fields["full_name"], fields["document_number"], fields["date_of_birth"], fields["expiry_date"]) == expected
    assert body["status"] == "completed"
    assert body["document_detection"]["detected"] is scene
    assert body["face"]["status"] == "detected"
    assert body["ocr"]["mean_confidence"] > 0.5


def test_real_ocr_reads_an_upside_down_card(tmp_path: Path, real_engine: OcrEngine) -> None:
    card = cv2.rotate(generate_synthetic_id(SyntheticIdSpec()), cv2.ROTATE_180)
    body = _process(tmp_path, real_engine, encode_image(card, ".jpg"))
    assert body["document_detection"]["rotation_applied"] == 180
    assert body["extracted_fields"]["document_number"] == LICENSE[1]
