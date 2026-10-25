"""Response models for processed ID records.

``POST /api/v1/process-id`` and ``GET /api/v1/records/{id}`` return the same
``IdRecord`` shape, so a client can treat both responses identically.
"""

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

ProcessingStatus = Literal["completed", "partial", "ocr_failed"]
FaceStatus = Literal["detected", "not_detected", "error"]

DISCLAIMER = (
    "OCR extraction is best-effort. Values can be missing or misread, so verify them "
    "manually before relying on them in a real-world process."
)


class ImageMetadata(BaseModel):
    original_filename: Optional[str] = Field(
        None, description="Sanitised client filename. Stored as metadata only, never used as a path."
    )
    declared_content_type: Optional[str] = Field(None, description="MIME type sent by the client.")
    detected_format: str = Field(..., description="Format detected from the file's magic bytes.")
    color_mode: str = Field(..., description="Colour mode reported by the image header, e.g. RGB.")
    file_size_bytes: int
    width: int
    height: int


class Point(BaseModel):
    x: int
    y: int


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DocumentDetection(BaseModel):
    detected: bool = Field(..., description="True when a card boundary was found.")
    method: str = Field(..., description="edge_contour, threshold_contour or none.")
    corners: Optional[List[Point]] = Field(
        None, description="Card corners (top-left, top-right, bottom-right, bottom-left) in original pixels."
    )
    area_ratio: Optional[float] = Field(None, description="Card area divided by image area.")
    fallback_used: bool = Field(..., description="True when the whole image was processed instead.")
    rotation_applied: int = Field(0, description="Clockwise rotation (degrees) chosen for OCR.")
    aligned_width: int
    aligned_height: int


class OcrLine(BaseModel):
    text: str
    confidence: Optional[float] = Field(None, description="Mean engine confidence, 0 to 1.")


class OcrOutput(BaseModel):
    engine: str
    image_variant: Optional[str] = Field(None, description="Preprocessed image variant used for OCR.")
    mean_confidence: Optional[float] = None
    word_count: int = 0
    raw_text: Optional[str] = Field(None, description="Engine output before normalisation.")
    normalized_text: Optional[str] = None
    lines: List[OcrLine] = Field(default_factory=list)


class ExtractedFields(BaseModel):
    full_name: Optional[str] = None
    document_number: Optional[str] = None
    date_of_birth: Optional[str] = Field(None, description="ISO 8601 date (YYYY-MM-DD).")
    expiry_date: Optional[str] = Field(None, description="ISO 8601 date (YYYY-MM-DD).")


class FieldEvidence(BaseModel):
    value: str
    raw_text: str = Field(..., description="OCR text the value was taken from.")
    ocr_confidence: Optional[float] = Field(None, description="Engine confidence of that text, 0 to 1.")
    method: str = Field(..., description="How the value was found, e.g. label or label_next_line.")


class FieldEvidenceSet(BaseModel):
    full_name: Optional[FieldEvidence] = None
    document_number: Optional[FieldEvidence] = None
    date_of_birth: Optional[FieldEvidence] = None
    expiry_date: Optional[FieldEvidence] = None


class FaceThumbnail(BaseModel):
    status: FaceStatus
    thumbnail_url: Optional[str] = Field(None, description="API URL of the saved thumbnail.")
    bounding_box: Optional[BoundingBox] = Field(
        None, description="Face box on the normalised card image (resized and rotated as used for OCR)."
    )
    method: Optional[str] = None
    message: Optional[str] = None


class IdRecord(BaseModel):
    record_id: str
    status: ProcessingStatus = Field(
        ..., description="completed (all fields found), partial (some missing) or ocr_failed."
    )
    created_at: datetime
    processing_time_ms: int
    image: ImageMetadata
    document_detection: DocumentDetection
    extracted_fields: ExtractedFields
    field_details: FieldEvidenceSet
    ocr: OcrOutput
    face: FaceThumbnail
    warnings: List[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class RecordSummary(BaseModel):
    record_id: str
    created_at: datetime
    status: ProcessingStatus
    document_detected: bool
    face_detected: bool
    fields_found: int = Field(..., description="How many of the four fields were extracted.")


class RecordList(BaseModel):
    items: List[RecordSummary]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    ocr_engine: str
    ocr_available: bool
    database: Literal["ok", "error"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
