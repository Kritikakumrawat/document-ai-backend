"""Run the full ID-card pipeline for one upload and store the result.

    upload -> validate -> detect card -> preprocess -> OCR -> extract fields
           -> face thumbnail -> save record -> response

Each step lives in its own module; this class only wires them together and
turns their results into the API's ``IdRecord`` model.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import cv2

from app.core.config import Settings
from app.database.repository import RecordRepository
from app.schemas.record import (
    BoundingBox,
    DocumentDetection,
    ExtractedFields,
    FaceThumbnail,
    FieldEvidence,
    FieldEvidenceSet,
    IdRecord,
    ImageMetadata,
    OcrLine,
    OcrOutput,
    Point,
)
from app.services.document_detector import DetectionResult, DocumentDetector
from app.services.face_extractor import FaceExtractionResult, FaceExtractor
from app.services.field_extractor import ExtractionResult, FieldExtractor, FieldValue, normalize_text
from app.services.image_preprocessor import ImagePreprocessor, PreprocessedImage
from app.services.ocr_service import OcrEngine, OcrEngineError, OcrResult, OcrService, rotate_image
from app.services.upload_validator import ValidatedImage, validate_image_upload
from app.utils.errors import OcrUnavailableError

logger = logging.getLogger(__name__)

THUMBNAIL_URL_PREFIX = "/api/v1/thumbnails"


class IdCardProcessor:
    def __init__(
        self,
        settings: Settings,
        ocr_engine: OcrEngine,
        repository: RecordRepository,
        detector: Optional[DocumentDetector] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        extractor: Optional[FieldExtractor] = None,
        face_extractor: Optional[FaceExtractor] = None,
    ) -> None:
        self.settings = settings
        self.ocr_engine = ocr_engine
        self.ocr_service = OcrService(ocr_engine, try_rotations=settings.ocr_try_rotations)
        self.repository = repository
        self.detector = detector or DocumentDetector()
        self.preprocessor = preprocessor or ImagePreprocessor(settings.ocr_target_long_side)
        self.extractor = extractor or FieldExtractor(date_order=settings.date_order)
        self.face_extractor = face_extractor or FaceExtractor(settings.thumbnails_path)

    def process(self, data: bytes, filename: Optional[str], content_type: Optional[str]) -> IdRecord:
        started = time.perf_counter()
        upload = validate_image_upload(
            data,
            filename,
            content_type,
            allowed_formats=self.settings.allowed_formats,
            max_bytes=self.settings.max_upload_bytes,
            max_pixels=self.settings.max_image_pixels,
            min_side=self.settings.min_image_side_px,
        )
        if not self.ocr_engine.is_available():
            raise OcrUnavailableError(
                f"The OCR engine '{self.ocr_engine.name}' is not available on this server. "
                "See the README section on installing Tesseract or EasyOCR."
            )

        record_id = uuid.uuid4().hex
        warnings: List[str] = []

        detection = self.detector.detect(upload.image)
        if not detection.detected:
            warnings.append("Card boundary not detected; the whole image was processed instead.")
        prepared = self.preprocessor.process(detection.aligned_image)

        ocr_result: Optional[OcrResult] = None
        try:
            ocr_result = self.ocr_service.recognize(prepared.variants())
        except OcrEngineError:
            logger.warning("OCR failed for record %s", record_id, exc_info=True)
            warnings.append("OCR failed for this image, so no fields could be extracted.")

        extraction = self.extractor.extract(ocr_result.lines) if ocr_result else ExtractionResult()
        if ocr_result is not None and not ocr_result.words:
            warnings.append("No text was recognised on the card.")
        warnings.extend(extraction.warnings)

        rotation = ocr_result.rotation if ocr_result else 0
        face = self._extract_face(rotate_image(prepared.color, rotation), record_id, detection.detected)

        if self.settings.save_debug_images:
            self._save_debug_images(record_id, detection, prepared)

        record = IdRecord(
            record_id=record_id,
            status=self._status(ocr_result, extraction),
            created_at=datetime.now(timezone.utc),
            processing_time_ms=int((time.perf_counter() - started) * 1000),
            image=self._image_metadata(upload),
            document_detection=self._detection_model(detection, rotation),
            extracted_fields=ExtractedFields(
                full_name=_value(extraction.full_name),
                document_number=_value(extraction.document_number),
                date_of_birth=_value(extraction.date_of_birth),
                expiry_date=_value(extraction.expiry_date),
            ),
            field_details=FieldEvidenceSet(
                full_name=_evidence(extraction.full_name),
                document_number=_evidence(extraction.document_number),
                date_of_birth=_evidence(extraction.date_of_birth),
                expiry_date=_evidence(extraction.expiry_date),
            ),
            ocr=self._ocr_model(ocr_result),
            face=self._face_model(face),
            warnings=warnings,
        )

        try:
            self.repository.add(record)
        except Exception:
            self._remove_thumbnail(face)  # do not leave an orphan file behind
            raise

        # Log only technical facts - never names, numbers or dates from the card.
        logger.info(
            "Processed record %s: status=%s card_detected=%s face=%s fields=%d time_ms=%d",
            record_id,
            record.status,
            detection.detected,
            face.status,
            extraction.found_count(),
            record.processing_time_ms,
        )
        return record

    # --- helpers -----------------------------------------------------------
    def _extract_face(self, card, record_id: str, on_aligned_card: bool) -> FaceExtractionResult:  # noqa: ANN001
        try:
            return self.face_extractor.extract(card, record_id, on_aligned_card)
        except Exception:
            logger.warning("Face extraction failed for record %s", record_id, exc_info=True)
            return FaceExtractionResult(status="error", message="Face extraction failed for this image.")

    def _remove_thumbnail(self, face: FaceExtractionResult) -> None:
        if face.thumbnail_filename:
            (self.settings.thumbnails_path / face.thumbnail_filename).unlink(missing_ok=True)

    def _save_debug_images(self, record_id: str, detection: DetectionResult, prepared: PreprocessedImage) -> None:
        folder = self.settings.debug_images_path
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{record_id}_aligned.png"), detection.aligned_image)
        cv2.imwrite(str(folder / f"{record_id}_enhanced.png"), prepared.enhanced)
        cv2.imwrite(str(folder / f"{record_id}_binary.png"), prepared.binary)

    @staticmethod
    def _status(ocr_result: Optional[OcrResult], extraction: ExtractionResult) -> str:
        if ocr_result is None:
            return "ocr_failed"
        return "completed" if extraction.found_count() == 4 else "partial"

    @staticmethod
    def _image_metadata(upload: ValidatedImage) -> ImageMetadata:
        return ImageMetadata(
            original_filename=upload.original_filename,
            declared_content_type=upload.declared_content_type,
            detected_format=upload.detected_format,
            color_mode=upload.color_mode,
            file_size_bytes=upload.file_size_bytes,
            width=upload.width,
            height=upload.height,
        )

    @staticmethod
    def _detection_model(detection: DetectionResult, rotation: int) -> DocumentDetection:
        corners = None
        if detection.corners is not None:
            corners = [Point(x=int(round(x)), y=int(round(y))) for x, y in detection.corners]
        height, width = detection.aligned_image.shape[:2]
        return DocumentDetection(
            detected=detection.detected,
            method=detection.method,
            corners=corners,
            area_ratio=detection.area_ratio,
            fallback_used=not detection.detected,
            rotation_applied=rotation,
            aligned_width=width,
            aligned_height=height,
        )

    def _ocr_model(self, ocr_result: Optional[OcrResult]) -> OcrOutput:
        if ocr_result is None:
            return OcrOutput(engine=self.ocr_engine.name)
        output = OcrOutput(
            engine=ocr_result.engine,
            image_variant=ocr_result.variant,
            mean_confidence=ocr_result.mean_confidence,
            word_count=len(ocr_result.words),
        )
        if self.settings.include_ocr_text:
            output.raw_text = ocr_result.raw_text
            output.normalized_text = "\n".join(normalize_text(line.text) for line in ocr_result.lines)
            output.lines = [OcrLine(text=line.text, confidence=line.confidence) for line in ocr_result.lines]
        return output

    @staticmethod
    def _face_model(face: FaceExtractionResult) -> FaceThumbnail:
        box = None
        if face.bounding_box is not None:
            x, y, w, h = face.bounding_box
            box = BoundingBox(x=x, y=y, width=w, height=h)
        url = f"{THUMBNAIL_URL_PREFIX}/{face.thumbnail_filename}" if face.thumbnail_filename else None
        return FaceThumbnail(
            status=face.status, thumbnail_url=url, bounding_box=box, method=face.method, message=face.message
        )


def _value(field: Optional[FieldValue]) -> Optional[str]:
    return field.value if field else None


def _evidence(field: Optional[FieldValue]) -> Optional[FieldEvidence]:
    if field is None:
        return None
    return FieldEvidence(
        value=field.value, raw_text=field.raw_text, ocr_confidence=field.ocr_confidence, method=field.method
    )
