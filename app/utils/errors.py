"""Application errors and the handlers that turn them into JSON responses.

Every error response has the same shape::

    {"error": {"code": "unsupported_file_type", "message": "...", "details": null}}

Stack traces are written to the server log only, never to the response.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for errors that map to a specific HTTP status and error code."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidUploadError(AppError):
    status_code = 400
    code = "invalid_upload"


class EmptyFileError(AppError):
    status_code = 400
    code = "empty_file"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class UnreadableImageError(AppError):
    status_code = 422
    code = "unreadable_image"


class ImageDimensionError(AppError):
    status_code = 422
    code = "invalid_image_dimensions"


class RecordNotFoundError(AppError):
    status_code = 404
    code = "record_not_found"


class ThumbnailNotFoundError(AppError):
    status_code = 404
    code = "thumbnail_not_found"


class OcrUnavailableError(AppError):
    status_code = 503
    code = "ocr_unavailable"


_HTTP_STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "file_too_large",
    415: "unsupported_file_type",
    422: "validation_error",
    503: "service_unavailable",
}


def error_body(code: str, message: str, details: Optional[Any] = None) -> Dict[str, Any]:
    """Build the standard JSON error payload."""
    return {"error": {"code": code, "message": message, "details": details}}


def _clean_validation_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Drop the "input" and "ctx" keys: they can echo uploaded data back to the client.
    return [
        {"location": [str(part) for part in err.get("loc", ())], "message": err.get("msg", "")}
        for err in errors
    ]


async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message, exc.details))


async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=error_body(
            "validation_error",
            "The request is missing required data or contains invalid values.",
            _clean_validation_errors(list(exc.errors())),
        ),
    )


async def _handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_STATUS_CODES.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code, message),
        headers=getattr(exc, "headers", None),
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=error_body("internal_error", "An unexpected error occurred. Please try again later."),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all handlers so every error uses the same JSON shape."""
    app.add_exception_handler(AppError, _handle_app_error)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _handle_http_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unexpected_error)
