"""Test doubles and helpers shared by several test modules."""

from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
from httpx import Response

from app.core.config import Settings
from app.services.ocr_service import OcrEngine, OcrEngineError, OcrWord

#: Signature of the ``upload`` fixture: upload(data, filename=..., content_type=...).
UploadFn = Callable[..., Response]

LICENSE_LINES = [
    "SYNTHETIC DRIVER LICENSE",
    "NAME: ALEX SAMPLE",
    "ID NO: SYN-4821-7730",
    "DOB: 14/03/1991",
    "ISS: 15/03/2021",
    "EXP: 14/03/2031",
    "CLASS: C",
    "SYNTHETIC TEST DATA - NOT A REAL ID",
]


def words_from_lines(lines: Sequence[str], confidence: float = 0.95) -> List[OcrWord]:
    """Lay text out like an OCR engine would: one row per line, words left to right."""
    words = []
    for row, line in enumerate(lines):
        left = 40
        for text in line.split():
            width = 14 * len(text)
            words.append(OcrWord(text, confidence, left, 40 + row * 50, width, 30))
            left += width + 12
    return words


class FakeOcrEngine(OcrEngine):
    """Test double: returns predefined words, or fails on demand."""

    name = "fake"
    preferred_variants = ("enhanced",)

    def __init__(self, lines: Sequence[str] = LICENSE_LINES, available: bool = True, fail: bool = False) -> None:
        self.lines = list(lines)
        self.available = available
        self.fail = fail
        self.calls = 0

    def is_available(self) -> bool:
        return self.available

    def read_words(self, image: np.ndarray) -> List[OcrWord]:
        self.calls += 1
        if self.fail:
            raise OcrEngineError("simulated OCR failure")
        return words_from_lines(self.lines)


def make_settings(storage: Path, **overrides: object) -> Settings:
    """Settings that keep everything inside a temporary folder and ignore any local .env."""
    values = dict(
        storage_dir=storage,
        database_path=storage / "records.db",
        thumbnails_dir=storage / "thumbnails",
        debug_images_dir=storage / "debug",
        max_upload_size_mb=2,
        cors_allow_origins="",
        include_ocr_text=True,
        save_debug_images=False,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)
