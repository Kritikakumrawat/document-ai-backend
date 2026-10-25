"""Prepare the aligned card image for OCR.

Steps (each one is a standard OpenCV call):

1. Resize so the card's long side is about ``target_long_side`` pixels. OCR
   engines read text best when letters are roughly 20-40 pixels tall.
2. Convert to grayscale - colour does not help text recognition.
3. Remove noise with non-local means denoising.
4. Boost local contrast with CLAHE, which evens out shadows and glare.
5. Binarise with adaptive thresholding, which copes with uneven lighting better
   than one global threshold.

Both the contrast-enhanced grayscale image and the binary image are returned,
because different OCR engines (and different photos) prefer different inputs.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessedImage:
    color: np.ndarray  # resized BGR card, used for face extraction
    gray: np.ndarray
    enhanced: np.ndarray  # denoised + CLAHE grayscale
    binary: np.ndarray  # adaptive threshold (black text on white)
    scale: float  # resize factor applied to the aligned card

    def variants(self) -> Dict[str, np.ndarray]:
        """Images the OCR service may try, keyed by name."""
        return {"enhanced": self.enhanced, "binary": self.binary}


class ImagePreprocessor:
    def __init__(self, target_long_side: int = 1400) -> None:
        self.target_long_side = target_long_side

    def resize(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        scale = self.target_long_side / float(max(height, width))
        if abs(scale - 1.0) < 0.02:
            return image.copy(), 1.0
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation), scale

    def process(self, card: np.ndarray) -> PreprocessedImage:
        color, scale = self.resize(card)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        block_size = self._odd(max(15, int(min(enhanced.shape[:2]) / 25)))
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 15
        )
        binary = cv2.medianBlur(binary, 3)  # remove isolated speckles left by thresholding
        return PreprocessedImage(color=color, gray=gray, enhanced=enhanced, binary=binary, scale=scale)

    @staticmethod
    def _odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1
