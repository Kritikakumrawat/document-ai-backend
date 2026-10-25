"""OCR engines and the service that chooses the best reading of a card.

Two open-source engines are supported behind one small interface:

* ``TesseractEngine`` - Tesseract via ``pytesseract`` (default). Needs the
  Tesseract program installed on the machine (see README).
* ``EasyOcrEngine`` - EasyOCR, a pure-pip alternative (PyTorch based). It
  downloads its model weights on first use.

Engines return words with bounding boxes. ``group_words_into_lines`` turns those
words into reading-order lines, splitting a row where there is a wide gap so
that two columns of text are not glued together.
"""

import importlib.util
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.config import Settings

logger = logging.getLogger(__name__)

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class OcrWord:
    text: str
    confidence: Optional[float]  # 0..1, None when the engine gives none
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass
class OcrLine:
    """A run of words on one row, without a large horizontal gap."""

    text: str
    confidence: Optional[float]
    left: int
    top: int
    right: int
    bottom: int

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)


@dataclass
class OcrResult:
    engine: str
    variant: str
    rotation: int
    words: List[OcrWord] = field(default_factory=list)
    lines: List[OcrLine] = field(default_factory=list)

    @property
    def mean_confidence(self) -> Optional[float]:
        values = [w.confidence for w in self.words if w.confidence is not None]
        return round(sum(values) / len(values), 4) if values else None

    @property
    def raw_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def score(self) -> float:
        """Higher is better: total confidence of words that look like real text."""
        return sum(
            (w.confidence if w.confidence is not None else 0.5)
            for w in self.words
            if sum(ch.isalnum() for ch in w.text) >= 2
        )

    def strong_word_count(self, min_confidence: float = 0.7) -> int:
        return sum(
            1
            for w in self.words
            if sum(ch.isalnum() for ch in w.text) >= 2 and (w.confidence or 0.0) >= min_confidence
        )


class OcrEngineError(Exception):
    """Raised when an engine fails while reading an image."""


class OcrEngine(ABC):
    name: str = "base"
    #: Preprocessed variants to try, best first (see ImagePreprocessor.variants).
    preferred_variants: Tuple[str, ...] = ("enhanced",)

    @abstractmethod
    def is_available(self) -> bool:
        """True when the engine can run on this machine."""

    @abstractmethod
    def read_words(self, image: np.ndarray) -> List[OcrWord]:
        """Recognise words in ``image``; raise ``OcrEngineError`` on failure."""


class TesseractEngine(OcrEngine):
    name = "tesseract"
    preferred_variants = ("enhanced", "binary")

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        language: str = "eng",
        page_segmentation_mode: int = 11,
        timeout_seconds: int = 30,
        data_dir: Optional[str] = None,
    ) -> None:
        import pytesseract  # small pure-Python wrapper; the binary is checked in is_available()

        self._pytesseract = pytesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        if data_dir:
            # Tesseract reads this variable to find its language files; the
            # subprocess started by pytesseract inherits it.
            os.environ["TESSDATA_PREFIX"] = data_dir
        self.language = language
        self.page_segmentation_mode = page_segmentation_mode
        self.timeout_seconds = timeout_seconds
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """True when the Tesseract program runs and the configured language data is installed."""
        if self._available is None:
            try:
                version = self._pytesseract.get_tesseract_version()
                installed = set(self._pytesseract.get_languages())
                missing = [lang for lang in self.language.split("+") if lang not in installed]
                if missing:
                    logger.warning("Tesseract %s found but language data is missing: %s", version, missing)
                self._available = not missing
            except Exception:  # binary missing or not executable
                self._available = False
        return self._available

    def read_words(self, image: np.ndarray) -> List[OcrWord]:
        config = f"--oem 1 --psm {self.page_segmentation_mode}"
        try:
            data = self._pytesseract.image_to_data(
                image,
                lang=self.language,
                config=config,
                output_type=self._pytesseract.Output.DICT,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:  # TesseractError, TesseractNotFoundError, timeout RuntimeError
            raise OcrEngineError(f"Tesseract failed: {type(exc).__name__}") from exc

        words: List[OcrWord] = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            confidence = float(data["conf"][i])
            if not text or confidence < 0:  # -1 marks layout boxes, not words
                continue
            words.append(
                OcrWord(
                    text=text,
                    confidence=round(confidence / 100.0, 4),
                    left=int(data["left"][i]),
                    top=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                )
            )
        return words


class EasyOcrEngine(OcrEngine):
    name = "easyocr"
    preferred_variants = ("enhanced",)

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        model_dir: Optional[Path] = None,
        gpu: bool = False,
    ) -> None:
        self.languages = list(languages)
        self.model_dir = model_dir
        self.gpu = gpu
        self._reader = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return importlib.util.find_spec("easyocr") is not None

    def _get_reader(self):  # noqa: ANN202 - easyocr.Reader is an optional dependency
        with self._lock:
            if self._reader is None:
                import easyocr

                logger.info("Loading EasyOCR models (first run downloads them)")
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    model_storage_directory=str(self.model_dir) if self.model_dir else None,
                    verbose=False,
                )
            return self._reader

    def read_words(self, image: np.ndarray) -> List[OcrWord]:
        try:
            results = self._get_reader().readtext(image, detail=1, paragraph=False)
        except Exception as exc:
            raise OcrEngineError(f"EasyOCR failed: {type(exc).__name__}") from exc

        words: List[OcrWord] = []
        for box, text, confidence in results:
            text = (text or "").strip()
            if not text:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            words.append(
                OcrWord(
                    text=text,
                    confidence=round(float(confidence), 4),
                    left=int(min(xs)),
                    top=int(min(ys)),
                    width=max(1, int(max(xs) - min(xs))),
                    height=max(1, int(max(ys) - min(ys))),
                )
            )
        return words


def create_ocr_engine(settings: Settings) -> OcrEngine:
    """Build the engine selected by the ``OCR_ENGINE`` setting."""
    if settings.ocr_engine == "easyocr":
        languages = [lang.strip() for lang in settings.easyocr_languages.split(",") if lang.strip()]
        model_dir = settings.resolve(settings.easyocr_model_dir) if settings.easyocr_model_dir else None
        return EasyOcrEngine(languages=languages, model_dir=model_dir, gpu=settings.easyocr_gpu)
    return TesseractEngine(
        tesseract_cmd=settings.tesseract_cmd,
        language=settings.tesseract_lang,
        page_segmentation_mode=settings.tesseract_psm,
        data_dir=settings.tesseract_data_dir,
    )


def rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate clockwise by 0, 90, 180 or 270 degrees."""
    if rotation == 0:
        return image
    return cv2.rotate(image, _ROTATIONS[rotation])


def group_words_into_lines(words: Sequence[OcrWord], gap_factor: float = 2.0) -> List[OcrLine]:
    """Group words into rows by vertical position, then split rows at wide gaps."""
    rows: List[List[OcrWord]] = []
    for word in sorted(words, key=lambda w: w.top + w.height / 2):
        center = word.top + word.height / 2
        if rows:
            row = rows[-1]
            row_center = sum(w.top + w.height / 2 for w in row) / len(row)
            row_height = median(w.height for w in row)
            if abs(center - row_center) <= 0.5 * max(row_height, word.height):
                row.append(word)
                continue
        rows.append([word])

    lines: List[OcrLine] = []
    for row in rows:
        row.sort(key=lambda w: w.left)
        row_height = median(w.height for w in row)
        segment = [row[0]]
        for word in row[1:]:
            if word.left - segment[-1].right > gap_factor * row_height:
                lines.append(_make_line(segment))
                segment = [word]
            else:
                segment.append(word)
        lines.append(_make_line(segment))
    return lines


def _make_line(words: Sequence[OcrWord]) -> OcrLine:
    confidences = [w.confidence for w in words if w.confidence is not None]
    return OcrLine(
        text=" ".join(w.text for w in words),
        confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
        left=min(w.left for w in words),
        top=min(w.top for w in words),
        right=max(w.right for w in words),
        bottom=max(w.bottom for w in words),
    )


class OcrService:
    """Run an engine on the preprocessed variants (and rotations) and keep the best result."""

    def __init__(self, engine: OcrEngine, try_rotations: bool = True, good_word_count: int = 5) -> None:
        self.engine = engine
        self.try_rotations = try_rotations
        self.good_word_count = good_word_count

    def recognize(self, variants: Dict[str, np.ndarray]) -> OcrResult:
        names = [name for name in self.engine.preferred_variants if name in variants] or list(variants)
        best: Optional[OcrResult] = None
        for name in names:
            best = self._better(best, self._read(variants[name], name, 0))
            if self._good_enough(best):
                return best

        assert best is not None
        if self.try_rotations:
            # Weak result: the card may be upside down or sideways.
            for rotation in (180, 90, 270):
                image = rotate_image(variants[best.variant], rotation)
                best = self._better(best, self._read(image, best.variant, rotation))
                if self._good_enough(best):
                    break
        return best

    def _read(self, image: np.ndarray, variant: str, rotation: int) -> OcrResult:
        words = self.engine.read_words(image)
        return OcrResult(
            engine=self.engine.name,
            variant=variant,
            rotation=rotation,
            words=words,
            lines=group_words_into_lines(words),
        )

    def _good_enough(self, result: OcrResult) -> bool:
        return result.strong_word_count() >= self.good_word_count

    @staticmethod
    def _better(current: Optional[OcrResult], candidate: OcrResult) -> OcrResult:
        if current is None or candidate.score() > current.score():
            return candidate
        return current
