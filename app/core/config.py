"""Application settings.

Values come from environment variables or a ``.env`` file in the project root
(see ``.env.example``). Relative paths are resolved against the project root,
so the app behaves the same no matter which directory it is started from.
"""

from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Every image format the code knows how to recognise from its magic bytes.
SUPPORTED_IMAGE_FORMATS = ("jpeg", "png", "webp", "bmp", "tiff")


class Settings(BaseSettings):
    """All tunable values in one place, kept out of the business logic."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Document AI Backend"
    log_level: str = "INFO"

    # --- Storage -----------------------------------------------------------
    storage_dir: Path = Path("storage")
    database_path: Path = Path("storage/document_ai.db")
    thumbnails_dir: Path = Path("storage/thumbnails")
    debug_images_dir: Path = Path("storage/debug")
    save_debug_images: bool = False

    # --- Upload validation -------------------------------------------------
    max_upload_size_mb: float = Field(10.0, gt=0, le=100)
    max_image_pixels: int = Field(40_000_000, gt=0)
    min_image_side_px: int = Field(100, ge=16)
    allowed_image_formats: str = "jpeg,png,webp,bmp,tiff"

    # --- OCR ---------------------------------------------------------------
    ocr_engine: Literal["tesseract", "easyocr"] = "tesseract"
    tesseract_cmd: Optional[str] = None
    # Folder holding eng.traineddata; only needed when Tesseract cannot find it itself.
    tesseract_data_dir: Optional[str] = None
    tesseract_lang: str = "eng"
    tesseract_psm: int = Field(11, ge=0, le=13)
    easyocr_languages: str = "en"
    easyocr_model_dir: Optional[Path] = None
    easyocr_gpu: bool = False
    ocr_target_long_side: int = Field(1400, ge=400, le=4000)
    ocr_try_rotations: bool = True

    # --- Field extraction --------------------------------------------------
    # How to read ambiguous numeric dates such as 04/05/1990.
    date_order: Literal["DMY", "MDY"] = "DMY"

    # --- API ---------------------------------------------------------------
    include_ocr_text: bool = True
    cors_allow_origins: str = ""

    @field_validator("tesseract_cmd", "tesseract_data_dir", "easyocr_model_dir", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("allowed_image_formats")
    @classmethod
    def _check_formats(cls, value: str) -> str:
        formats = {item.strip().lower() for item in value.split(",") if item.strip()}
        unknown = formats - set(SUPPORTED_IMAGE_FORMATS)
        if not formats or unknown:
            raise ValueError(
                f"allowed_image_formats must be a subset of {SUPPORTED_IMAGE_FORMATS}"
            )
        return ",".join(sorted(formats))

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("log_level must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
        return level

    # --- Derived values ----------------------------------------------------
    @staticmethod
    def resolve(path: Path) -> Path:
        """Turn a configured path into an absolute path under the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def database_file(self) -> Path:
        return self.resolve(self.database_path)

    @property
    def thumbnails_path(self) -> Path:
        return self.resolve(self.thumbnails_dir)

    @property
    def debug_images_path(self) -> Path:
        return self.resolve(self.debug_images_dir)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    @property
    def allowed_formats(self) -> FrozenSet[str]:
        return frozenset(self.allowed_image_formats.split(","))

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them for the life of the process."""
    return Settings()
