"""OpenCV stages: card detection, preprocessing, face thumbnails, OCR line grouping."""

from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from app.services.document_detector import DocumentDetector, order_corners
from app.services.face_extractor import FaceExtractor
from app.services.image_preprocessor import ImagePreprocessor
from app.services.ocr_service import OcrEngine, OcrService, OcrWord, group_words_into_lines
from scripts.generate_sample_id import SyntheticIdSpec, badge_spec, generate_synthetic_id
from tests.helpers import words_from_lines


@pytest.mark.parametrize("spec", [SyntheticIdSpec(), badge_spec()], ids=["license", "badge"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_card_is_found_and_straightened_in_photos(spec: SyntheticIdSpec, seed: int) -> None:
    photo = generate_synthetic_id(spec, scene=True, seed=seed)
    result = DocumentDetector().detect(photo)
    assert result.detected
    height, width = result.aligned_image.shape[:2]
    assert 1.4 < max(width, height) / min(width, height) < 1.8
    assert (width > height) == (spec.style == "license")  # orientation is preserved
    assert 0.12 < result.area_ratio < 0.6


def test_flat_scan_and_blank_image_fall_back_to_full_image() -> None:
    for image in (generate_synthetic_id(SyntheticIdSpec()), np.full((600, 900, 3), 128, np.uint8)):
        result = DocumentDetector().detect(image)
        assert result.detected is False
        assert result.method == "none"
        assert result.aligned_image.shape == image.shape


def test_corners_are_ordered_clockwise_from_top_left() -> None:
    shuffled = np.array([[300, 220], [10, 15], [20, 200], [310, 5]], dtype=np.float32)
    assert order_corners(shuffled).tolist() == [[10, 15], [310, 5], [300, 220], [20, 200]]


def test_preprocessor_produces_ocr_ready_images() -> None:
    card = generate_synthetic_id(SyntheticIdSpec())
    prepared = ImagePreprocessor(target_long_side=1400).process(card)
    assert max(prepared.color.shape[:2]) == 1400
    assert prepared.gray.ndim == 2 and prepared.enhanced.ndim == 2
    assert set(np.unique(prepared.binary)) <= {0, 255}
    assert set(prepared.variants()) == {"enhanced", "binary"}


def test_face_thumbnail_is_cropped_from_the_portrait(tmp_path: Path) -> None:
    card = generate_synthetic_id(SyntheticIdSpec())
    result = FaceExtractor(tmp_path).extract(card, "a" * 32, on_aligned_card=True)
    assert result.status == "detected"
    assert result.thumbnail_filename == "a" * 32 + "_face.jpg"
    x, y, w, h = result.bounding_box
    assert 40 <= x and x + w <= 300 and 120 <= y and y + h <= 450  # inside the photo box
    thumbnail = cv2.imread(str(tmp_path / result.thumbnail_filename))
    assert thumbnail is not None and max(thumbnail.shape[:2]) <= 320


def test_no_face_means_no_thumbnail(tmp_path: Path) -> None:
    card = generate_synthetic_id(SyntheticIdSpec(include_face=False))
    result = FaceExtractor(tmp_path).extract(card, "b" * 32, on_aligned_card=True)
    assert result.status == "not_detected"
    assert result.thumbnail_filename is None
    assert list(tmp_path.iterdir()) == []


def test_missing_face_model_is_reported_not_raised(tmp_path: Path) -> None:
    extractor = FaceExtractor(tmp_path / "thumbs", cascade_path=tmp_path / "missing.xml")
    result = extractor.extract(generate_synthetic_id(SyntheticIdSpec()), "c" * 32, on_aligned_card=True)
    assert result.status == "error"


def test_words_are_grouped_into_lines_and_columns() -> None:
    words = [
        OcrWord("DOB:", 0.9, 10, 100, 60, 30),
        OcrWord("14/03/1991", 0.9, 80, 102, 150, 30),  # small gap -> same line
        OcrWord("EXP:", 0.9, 600, 101, 60, 30),  # large gap -> new column
        OcrWord("NAME:", 0.9, 10, 40, 80, 30),
    ]
    lines = group_words_into_lines(words)
    assert [line.text for line in lines] == ["NAME:", "DOB: 14/03/1991", "EXP:"]
    assert lines[1].confidence == pytest.approx(0.9)


class UpsideDownEngine(OcrEngine):
    """Reads text only when the image has been turned the right way up."""

    name = "upside-down-test"

    def is_available(self) -> bool:
        return True

    def read_words(self, image: np.ndarray) -> List[OcrWord]:
        upright = image[-1, -1] == 255  # the marker starts in the top-left corner
        return words_from_lines(["NAME: ALEX SAMPLE", "DOB: 14/03/1991", "EXP: 14/03/2031"]) if upright else []


def test_ocr_service_tries_other_rotations_for_weak_results() -> None:
    image = np.zeros((100, 200), np.uint8)
    image[0, 0] = 255
    result = OcrService(UpsideDownEngine(), try_rotations=True).recognize({"enhanced": image})
    assert result.rotation == 180
    assert result.words

    no_retry = OcrService(UpsideDownEngine(), try_rotations=False).recognize({"enhanced": image})
    assert no_retry.rotation == 0 and not no_retry.words
