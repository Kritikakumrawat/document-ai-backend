"""Validate an uploaded file and decode it into an OpenCV image.

Checks run from cheapest to most expensive:

1. size (empty / too large),
2. filename extension and declared MIME type,
3. real content type from the file's magic bytes,
4. pixel dimensions from the image header (blocks "decompression bombs"),
5. the actual decode with OpenCV.

The filename and MIME type sent by the client are never trusted on their own.
"""

import io
import re
import unicodedata
from dataclasses import dataclass
from typing import BinaryIO, FrozenSet, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.utils.errors import (
    EmptyFileError,
    FileTooLargeError,
    ImageDimensionError,
    UnreadableImageError,
    UnsupportedFileTypeError,
)

_EXTENSION_FORMATS = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".jpe": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
}
_ACCEPTED_GENERIC_MIME_TYPES = {"application/octet-stream", "binary/octet-stream"}
_CHUNK_SIZE = 1024 * 1024
_MAX_FILENAME_LENGTH = 120


@dataclass
class ValidatedImage:
    """A decoded image plus the metadata gathered while validating it."""

    image: np.ndarray  # BGR, 8-bit, 3 channels
    detected_format: str
    color_mode: str
    width: int
    height: int
    file_size_bytes: int
    original_filename: Optional[str]
    declared_content_type: Optional[str]


def read_upload(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read an upload in chunks and stop as soon as it exceeds ``max_bytes``."""
    chunks = []
    total = 0
    while True:
        chunk = stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise FileTooLargeError(
                f"The file is larger than the {max_bytes // (1024 * 1024)} MB upload limit.",
                details={"max_bytes": max_bytes},
            )
        chunks.append(chunk)
    return b"".join(chunks)


def detect_image_format(data: bytes) -> Optional[str]:
    """Identify the image format from the first bytes of the file."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"BM"):
        return "bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return None


def sanitize_filename(filename: Optional[str]) -> Optional[str]:
    """Keep a harmless version of the client's filename for display only.

    Directory parts, control characters and unusual symbols are removed. The
    result is stored as metadata; files on disk always get generated names.
    """
    if not filename:
        return None
    name = unicodedata.normalize("NFKC", filename)
    name = re.split(r"[\\/]", name)[-1]  # drop any directory part ("../../x.png" -> "x.png")
    name = "".join(ch for ch in name if ch.isprintable())
    name = re.sub(r"[^\w.\- ()]", "_", name).strip(" .")
    if not name:
        return None
    return name[:_MAX_FILENAME_LENGTH]


def _extension(filename: Optional[str]) -> Optional[str]:
    if not filename or "." not in filename:
        return None
    return "." + filename.rsplit(".", 1)[-1].lower()


def validate_image_upload(
    data: bytes,
    filename: Optional[str],
    content_type: Optional[str],
    *,
    allowed_formats: FrozenSet[str],
    max_bytes: int,
    max_pixels: int,
    min_side: int,
) -> ValidatedImage:
    """Run every upload check and return the decoded image or raise an ``AppError``."""
    if not data:
        raise EmptyFileError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"The file is larger than the {max_bytes // (1024 * 1024)} MB upload limit.",
            details={"max_bytes": max_bytes},
        )

    allowed_list = sorted(allowed_formats)
    extension = _extension(filename)
    if extension is not None and _EXTENSION_FORMATS.get(extension) not in allowed_formats:
        raise UnsupportedFileTypeError(
            f"File extension '{extension}' is not allowed.",
            details={"allowed_formats": allowed_list},
        )

    declared = (content_type or "").split(";")[0].strip().lower() or None
    if declared and not (declared.startswith("image/") or declared in _ACCEPTED_GENERIC_MIME_TYPES):
        raise UnsupportedFileTypeError(
            f"Content type '{declared}' is not an image type.",
            details={"allowed_formats": allowed_list},
        )

    detected_format = detect_image_format(data)
    if detected_format is None or detected_format not in allowed_formats:
        raise UnsupportedFileTypeError(
            "The file content is not a supported image format.",
            details={"allowed_formats": allowed_list},
        )

    # Read only the header to get the size before decoding any pixels.
    try:
        with Image.open(io.BytesIO(data)) as header:
            width, height = header.size
            color_mode = header.mode
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableImageError("The image header could not be read; the file may be corrupted.") from exc

    if width * height > max_pixels:
        raise ImageDimensionError(
            f"The image is too large ({width}x{height}). Maximum is {max_pixels:,} pixels.",
            details={"width": width, "height": height, "max_pixels": max_pixels},
        )
    if min(width, height) < min_side:
        raise ImageDimensionError(
            f"The image is too small ({width}x{height}). Each side must be at least {min_side}px.",
            details={"width": width, "height": height, "min_side": min_side},
        )

    # IMREAD_COLOR returns 8-bit BGR and applies the EXIF orientation of phone photos.
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise UnreadableImageError("The image could not be decoded; the file may be corrupted.")

    decoded_height, decoded_width = image.shape[:2]
    return ValidatedImage(
        image=image,
        detected_format=detected_format,
        color_mode=color_mode,
        width=decoded_width,
        height=decoded_height,
        file_size_bytes=len(data),
        original_filename=sanitize_filename(filename),
        declared_content_type=declared,
    )
