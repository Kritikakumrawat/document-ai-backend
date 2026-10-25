"""HTTP endpoints.

Routes stay thin: they read the request, call a service and return a model.
Shared objects (settings, repository, processor) are created once in
``app.main.create_app`` and reached through small dependency functions, which
tests can replace.
"""

import re

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse

from app import __version__
from app.core.config import Settings
from app.database.repository import RecordRepository
from app.schemas.record import ErrorResponse, HealthResponse, IdRecord, RecordList
from app.services.face_extractor import thumbnail_filename
from app.services.id_processor import IdCardProcessor
from app.services.upload_validator import read_upload
from app.utils.errors import RecordNotFoundError, ThumbnailNotFoundError

# Record IDs are uuid4 hex strings; anything else cannot exist.
_RECORD_ID = re.compile(r"^[0-9a-f]{32}$")
_THUMBNAIL_NAME = re.compile(r"^[0-9a-f]{32}_face\.jpg$")

_ERRORS = {
    400: {"model": ErrorResponse, "description": "Empty or invalid upload"},
    404: {"model": ErrorResponse, "description": "Not found"},
    413: {"model": ErrorResponse, "description": "File too large"},
    415: {"model": ErrorResponse, "description": "Unsupported file type"},
    422: {"model": ErrorResponse, "description": "Unreadable image or invalid request"},
    503: {"model": ErrorResponse, "description": "OCR engine unavailable"},
}


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repository(request: Request) -> RecordRepository:
    return request.app.state.repository


def get_processor(request: Request) -> IdCardProcessor:
    return request.app.state.processor


system_router = APIRouter(tags=["system"])
api_router = APIRouter(prefix="/api/v1", tags=["id-processing"])


@system_router.get("/health", response_model=HealthResponse, summary="Service health")
def health(
    processor: IdCardProcessor = Depends(get_processor),
    repository: RecordRepository = Depends(get_repository),
) -> HealthResponse:
    ocr_available = processor.ocr_engine.is_available()
    database_ok = repository.database.is_healthy()
    return HealthResponse(
        status="ok" if ocr_available and database_ok else "degraded",
        version=__version__,
        ocr_engine=processor.ocr_engine.name,
        ocr_available=ocr_available,
        database="ok" if database_ok else "error",
    )


@api_router.post(
    "/process-id",
    response_model=IdRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Process an ID-card image",
    responses={code: _ERRORS[code] for code in (400, 413, 415, 422, 503)},
)
def process_id(
    response: Response,
    file: UploadFile = File(..., description="ID-card image (JPEG, PNG, WebP, BMP or TIFF)."),
    processor: IdCardProcessor = Depends(get_processor),
    settings: Settings = Depends(get_settings),
) -> IdRecord:
    """Detect the card, run OCR, extract fields and the face thumbnail, then store the record."""
    data = read_upload(file.file, settings.max_upload_bytes)
    record = processor.process(data, file.filename, file.content_type)
    response.headers["Location"] = f"/api/v1/records/{record.record_id}"
    return record


@api_router.get("/records", response_model=RecordList, summary="List processed records")
def list_records(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    repository: RecordRepository = Depends(get_repository),
) -> RecordList:
    """Newest first. Summaries only - fetch a record by ID for its extracted fields."""
    items, total = repository.list_records(limit, offset)
    return RecordList(items=items, total=total, limit=limit, offset=offset)


@api_router.get(
    "/records/{record_id}",
    response_model=IdRecord,
    summary="Get a processed record",
    responses={404: _ERRORS[404]},
)
def get_record(record_id: str, repository: RecordRepository = Depends(get_repository)) -> IdRecord:
    record = repository.get(record_id) if _RECORD_ID.fullmatch(record_id) else None
    if record is None:
        raise RecordNotFoundError(f"No record found with ID '{record_id[:64]}'.")
    return record


@api_router.delete(
    "/records/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a record and its thumbnail",
    responses={404: _ERRORS[404]},
)
def delete_record(
    record_id: str,
    repository: RecordRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> Response:
    if not (_RECORD_ID.fullmatch(record_id) and repository.delete(record_id)):
        raise RecordNotFoundError(f"No record found with ID '{record_id[:64]}'.")
    (settings.thumbnails_path / thumbnail_filename(record_id)).unlink(missing_ok=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@api_router.get(
    "/thumbnails/{filename}",
    response_class=FileResponse,
    summary="Download a face thumbnail",
    responses={200: {"content": {"image/jpeg": {}}}, 404: _ERRORS[404]},
)
def get_thumbnail(filename: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    # Only names this service generates are accepted, so "../" and similar tricks cannot work.
    path = settings.thumbnails_path / filename
    if not _THUMBNAIL_NAME.fullmatch(filename) or not path.is_file():
        raise ThumbnailNotFoundError("Thumbnail not found.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
