"""FastAPI application factory.

Start the server from the project root with::

    uvicorn app.main:app --host 127.0.0.1 --port 8000

Interactive API docs are then at http://127.0.0.1:8000/docs.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app import __version__
from app.api.routes import api_router, system_router
from app.core.config import Settings, get_settings
from app.database.connection import Database
from app.database.repository import RecordRepository
from app.schemas.record import DISCLAIMER
from app.services.id_processor import IdCardProcessor
from app.services.ocr_service import OcrEngine, create_ocr_engine
from app.utils.errors import error_body, register_exception_handlers

logger = logging.getLogger("app")

# Room for multipart boundaries and headers on top of the file itself.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024

DESCRIPTION = f"""
Upload an image of an ID card (driver's licence, employee badge, ...). The service
finds the card, straightens it, runs OCR, extracts the **full name**, **document
number**, **date of birth** and **expiry date**, saves a **face thumbnail** and
stores the result in SQLite.

**{DISCLAIMER}**

Use synthetic test images only. Do not upload real identity documents to a
shared or public deployment.
"""


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("app").setLevel(level)


def create_app(settings: Optional[Settings] = None, ocr_engine: Optional[OcrEngine] = None) -> FastAPI:
    """Build the app. Tests pass their own settings and a fake OCR engine."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    database = Database(settings.database_file)
    repository = RecordRepository(database)
    engine = ocr_engine or create_ocr_engine(settings)
    processor = IdCardProcessor(settings=settings, ocr_engine=engine, repository=repository)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.thumbnails_path.mkdir(parents=True, exist_ok=True)
        database.initialize()
        if engine.is_available():
            logger.info("OCR engine '%s' is ready", engine.name)
        else:
            logger.warning("OCR engine '%s' is NOT available; /api/v1/process-id will return 503", engine.name)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.repository = repository
    app.state.processor = processor

    register_exception_handlers(app)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type"],
            allow_credentials=False,
        )

    @app.middleware("http")
    async def limit_body_and_add_headers(request: Request, call_next):  # noqa: ANN001, ANN202
        # Reject oversized uploads early when the client declares the size.
        content_length = request.headers.get("content-length", "")
        limit = settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES
        if request.method == "POST" and content_length.isdigit() and int(content_length) > limit:
            return JSONResponse(
                status_code=413,
                content=error_body(
                    "file_too_large",
                    f"The request is larger than the {settings.max_upload_size_mb:g} MB upload limit.",
                    {"max_bytes": settings.max_upload_bytes},
                ),
            )
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    app.include_router(system_router)
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
