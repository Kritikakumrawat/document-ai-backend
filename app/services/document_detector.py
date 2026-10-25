"""Find the ID card in a photo and straighten it with a perspective transform.

How it works:

1. Shrink the photo to a working size so detection is fast and predictable.
2. Build binary "maps" of where the card might be:
   * edge map - Canny edges, slightly thickened so the card outline is closed;
   * colour map - how different each pixel's colour is from the photo's border
     (usually the table or background), split with Otsu's threshold. This finds
     cards whose coloured bands are as dark as the background;
   * threshold maps - Otsu thresholding of brightness, which separates a light
     card from a darker background (and the inverse).
3. In each map, test the biggest outlines. An outline is a candidate when it:
   * covers a large part of the photo (small boxes such as the portrait photo
     printed on the card are ignored),
   * has four corners, or is almost a perfect rectangle (rounded card corners),
   * has an ID-card-like width/height ratio (a standard card is 1.59:1),
   * shows a clear brightness change between just inside and just outside its
     border (a real card edge, not a cluster of text).
   The largest candidate from all maps wins, so a coloured band or the white
   middle of the card is not mistaken for the whole card.
4. Warp the four corners onto a flat rectangle.

If nothing card-like is found, the whole image is used instead (safe fallback).
That is the right behaviour for scans where the card already fills the frame.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    aligned_image: np.ndarray
    detected: bool
    method: str  # "edge_contour", "threshold_contour" or "none"
    corners: Optional[np.ndarray]  # 4x2 float32: tl, tr, br, bl in original pixels
    area_ratio: Optional[float]


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return the four points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = points.reshape(4, 2).astype(np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]  # clockwise in image coordinates (y points down)
    start = int(np.argmin(pts.sum(axis=1)))  # top-left has the smallest x + y
    return np.roll(pts, -start, axis=0)


def warp_to_rectangle(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Map the quadrilateral ``corners`` onto a flat, upright rectangle."""
    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width, height = max(width, 1), max(height, 1)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


class DocumentDetector:
    """Locate the card boundary using contour analysis."""

    def __init__(
        self,
        working_size: int = 900,
        min_area_ratio: float = 0.12,
        max_area_ratio: float = 0.98,
        min_aspect_ratio: float = 1.35,
        max_aspect_ratio: float = 2.0,
        min_edge_contrast: float = 12.0,
        min_color_distance: float = 22.0,
    ) -> None:
        self.working_size = working_size
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        # A standard ID card is 1.59:1. Portrait photo boxes (about 1.25-1.33:1)
        # fall below the minimum, so they are never mistaken for the card.
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_edge_contrast = min_edge_contrast
        self.min_color_distance = min_color_distance

    def detect(self, image: np.ndarray) -> DetectionResult:
        height, width = image.shape[:2]
        scale = min(1.0, self.working_size / float(max(height, width)))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        small = cv2.GaussianBlur(small, (5, 5), 0)
        blurred = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        candidate_maps = [
            ("edge_contour", self._edge_map(blurred)),
            ("color_contour", self._color_map(small)),
            *self._threshold_maps(blurred),
        ]
        best: Optional[Tuple[str, np.ndarray]] = None
        for method, mask in candidate_maps:
            quad = self._find_card_quad(mask, blurred)
            if quad is not None and (best is None or cv2.contourArea(quad) > cv2.contourArea(best[1])):
                best = (method, quad)

        if best is None:
            logger.debug("No card boundary found; using the full image")
            return DetectionResult(image.copy(), False, "none", None, None)

        method, quad = best
        corners = order_corners(quad / scale)
        area_ratio = float(cv2.contourArea(corners)) / float(width * height)
        logger.debug("Card found with %s (area ratio %.2f)", method, area_ratio)
        return DetectionResult(warp_to_rectangle(image, corners), True, method, corners, round(area_ratio, 4))

    # --- candidate maps ----------------------------------------------------
    @staticmethod
    def _edge_map(blurred: np.ndarray) -> np.ndarray:
        median = float(np.median(blurred))
        lower = int(max(10, 0.66 * median))
        upper = int(min(255, max(lower + 40, 1.33 * median)))
        edges = cv2.Canny(blurred, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    def _color_map(self, small_bgr: np.ndarray) -> np.ndarray:
        # Float input gives true CIELAB values, where a distance of ~20 is a clearly visible colour change.
        lab = cv2.cvtColor(small_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        band = max(2, int(0.03 * min(lab.shape[:2])))
        edges = (lab[:band], lab[-band:], lab[:, :band], lab[:, -band:])
        border = np.concatenate([edge.reshape(-1, 3) for edge in edges])
        distance = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
        mask = (distance > self.min_color_distance).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _threshold_maps(blurred: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        _, light = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, kernel, iterations=2)
        dark = cv2.bitwise_not(light)
        return [("threshold_contour", light), ("threshold_contour", dark)]

    # --- shape tests -------------------------------------------------------
    def _find_card_quad(self, mask: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        image_area = float(mask.shape[0] * mask.shape[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(hull)
            if not self.min_area_ratio * image_area <= area <= self.max_area_ratio * image_area:
                continue
            quad = self._as_quadrilateral(hull)
            if quad is None or not self._has_card_shape(quad):
                continue
            if self._sides_on_image_border(quad, mask.shape[:2]) >= 2:
                continue  # a tight scan or crop: the whole image already is the card
            if self._edge_contrast(gray, quad) < self.min_edge_contrast:
                continue
            return quad.astype(np.float32)
        return None

    @staticmethod
    def _as_quadrilateral(hull: np.ndarray) -> Optional[np.ndarray]:
        perimeter = cv2.arcLength(hull, True)
        for epsilon in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2)
        # Rounded corners can defeat polygon fitting; accept the bounding
        # rotated rectangle when the shape fills almost all of it.
        rect = cv2.minAreaRect(hull)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area > 0 and cv2.contourArea(hull) / rect_area >= 0.9:
            return cv2.boxPoints(rect)
        return None

    def _has_card_shape(self, quad: np.ndarray) -> bool:
        tl, tr, br, bl = order_corners(quad)
        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        if min(width, height) < 1:
            return False
        aspect = max(width, height) / min(width, height)
        return self.min_aspect_ratio <= aspect <= self.max_aspect_ratio

    @staticmethod
    def _sides_on_image_border(quad: np.ndarray, shape: Tuple[int, int]) -> int:
        """How many image edges the quad's bounding box runs along (within 1%)."""
        height, width = shape
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        margin_x, margin_y = 0.01 * width, 0.01 * height
        return sum(
            (
                x <= margin_x,
                y <= margin_y,
                x + w >= width - 1 - margin_x,
                y + h >= height - 1 - margin_y,
            )
        )

    @staticmethod
    def _edge_contrast(gray: np.ndarray, quad: np.ndarray) -> float:
        """Mean brightness difference between thin bands inside and outside the quad."""
        region = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(region, order_corners(quad).astype(np.int32), 255)
        band = max(3, int(0.015 * min(gray.shape[:2])))
        kernel = np.ones((band, band), dtype=np.uint8)
        inner = cv2.subtract(region, cv2.erode(region, kernel))
        outer = cv2.subtract(cv2.dilate(region, kernel), region)
        if not inner.any() or not outer.any():
            return 0.0
        return float(abs(gray[inner > 0].mean() - gray[outer > 0].mean()))
