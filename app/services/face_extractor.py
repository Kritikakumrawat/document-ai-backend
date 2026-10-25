"""Find the portrait photo printed on the card and save a thumbnail of it.

OpenCV's bundled Haar cascade (``haarcascade_frontalface_default.xml``) is used,
so no model download is needed. The search runs on the aligned card image, not
the raw photo, so a face in the background of the photo is not picked up when
the card boundary was found. When several faces are found on the card, the
largest is used (cards often also carry a small "ghost" copy of the portrait).

This module only crops a thumbnail. It does not recognise or verify who the
person is. When no face is found, nothing is saved and the status says so.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceExtractionResult:
    status: str  # "detected", "not_detected" or "error"
    thumbnail_filename: Optional[str] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height on the searched image
    method: Optional[str] = None
    message: Optional[str] = None


def thumbnail_filename(record_id: str) -> str:
    """Deterministic thumbnail name: one file per record, derived from its ID only."""
    return f"{record_id}_face.jpg"


class FaceExtractor:
    def __init__(
        self,
        thumbnails_dir: Path,
        margin: float = 0.35,
        max_thumbnail_side: int = 320,
        cascade_path: Optional[Path] = None,
    ) -> None:
        self.thumbnails_dir = thumbnails_dir
        self.margin = margin
        self.max_thumbnail_side = max_thumbnail_side
        path = cascade_path or Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(path))
        self._cascade: Optional[cv2.CascadeClassifier] = None if cascade.empty() else cascade
        if self._cascade is None:
            logger.warning("Face cascade could not be loaded from %s", path)

    def extract(self, image: np.ndarray, record_id: str, on_aligned_card: bool) -> FaceExtractionResult:
        method = "haar_cascade_on_aligned_card" if on_aligned_card else "haar_cascade_on_full_image"
        if self._cascade is None:
            return FaceExtractionResult(status="error", method=method, message="Face detector model is unavailable.")

        box = self._find_face(image, on_aligned_card)
        if box is None:
            return FaceExtractionResult(
                status="not_detected", method=method, message="No face photo was detected on the card."
            )

        crop = self._crop_with_margin(image, box)
        filename = thumbnail_filename(record_id)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.thumbnails_dir / filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            return FaceExtractionResult(status="error", method=method, message="The thumbnail could not be saved.")

        message = None if on_aligned_card else (
            "Card boundary not found; the face was searched in the whole photo and may not be the card portrait."
        )
        return FaceExtractionResult(
            status="detected", thumbnail_filename=filename, bounding_box=box, method=method, message=message
        )

    def _find_face(self, image: np.ndarray, on_aligned_card: bool) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        height, width = gray.shape[:2]
        min_side = max(24, int(min(height, width) * (0.12 if on_aligned_card else 0.05)))
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
        )
        boxes = [tuple(int(v) for v in face) for face in faces]
        if on_aligned_card:
            # A portrait on an ID card is never wider than ~70% of the card.
            boxes = [b for b in boxes if b[2] <= 0.7 * width and b[3] <= 0.9 * height]
        if not boxes:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])

    def _crop_with_margin(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        pad_x, pad_y = int(w * self.margin), int(h * self.margin)
        top, bottom = max(0, y - pad_y), min(image.shape[0], y + h + pad_y)
        left, right = max(0, x - pad_x), min(image.shape[1], x + w + pad_x)
        crop = image[top:bottom, left:right]
        scale = self.max_thumbnail_side / float(max(crop.shape[:2]))
        if scale < 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return crop
