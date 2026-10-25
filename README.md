# Document AI Backend: ID Card OCR Microservice

A RESTful backend that accepts a photo or scan of an identity card (driver's
licence, employee badge, ...), finds and straightens the card with **OpenCV**,
reads it with an open-source **OCR** engine, extracts the **full name**,
**document number**, **date of birth** and **expiry date**, saves a **face
thumbnail**, and stores the structured result in **SQLite**.

Built for the *Computer Vision & Microservice Backend (Document AI)* internship
assignment with **Python 3.9+** and **FastAPI**.

> **OCR extraction is best-effort.** Values can be missing or misread, so verify
> them manually before relying on them in a real-world process. Every API
> response repeats this in its `disclaimer` field.

---

## Contents

1. [Assignment requirements and how they are met](#1-assignment-requirements-and-how-they-are-met)
2. [Technology stack](#2-technology-stack)
3. [Architecture](#3-architecture)
4. [Project structure](#4-project-structure)
5. [Prerequisites](#5-prerequisites)
6. [Installation](#6-installation)
7. [Configuration (.env)](#7-configuration-env)
8. [Database initialisation](#8-database-initialisation)
9. [Running the application](#9-running-the-application)
10. [Using Swagger UI to upload an ID image](#10-using-swagger-ui-to-upload-an-id-image)
11. [API reference and curl examples](#11-api-reference-and-curl-examples)
12. [Example request and response](#12-example-request-and-response)
13. [How the processing works](#13-how-the-processing-works)
14. [Error handling](#14-error-handling)
15. [Running the tests](#15-running-the-tests)
16. [Generating synthetic test images](#16-generating-synthetic-test-images)
17. [Running with Docker](#17-running-with-docker)
18. [Known limitations](#18-known-limitations)
19. [Privacy and data retention](#19-privacy-and-data-retention)
20. [Uploading the project to GitHub](#20-uploading-the-project-to-github)
21. [Verification record](#21-verification-record)

---

## 1. Assignment requirements and how they are met

| Requirement | Implementation |
|---|---|
| Accept image uploads via POST | `POST /api/v1/process-id` (multipart field `file`) - `app/api/routes.py` |
| Validate the upload | Size limit, extension, MIME type, **magic bytes**, header dimensions, real decode - `app/services/upload_validator.py` |
| Detect card boundary, find 4 corners | Edge, colour and brightness contour maps, quadrilateral + shape + contrast checks - `app/services/document_detector.py` |
| Correct perspective / crop ROI | `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to a flat card |
| Fallback when no card is found | The whole image is processed and `fallback_used: true` is reported |
| Preprocessing for OCR | Resize, grayscale, non-local-means denoising, CLAHE contrast, adaptive threshold - `app/services/image_preprocessor.py` |
| Open-source OCR | **Tesseract** via `pytesseract` (default) or **EasyOCR** (optional) - `app/services/ocr_service.py` |
| Extract name, ID number, DOB, expiry | Label-based rules, regular expressions and plausibility checks - `app/services/field_extractor.py`, `app/services/date_parser.py` |
| Common date formats, DOB vs expiry | `14/03/1991`, `1991-03-14`, `14 MAR 1991`, `March 14, 1991`, two-digit years; labels decide the field |
| Null instead of guesses | Ambiguous or invalid candidates leave the field `null` (with a warning) |
| OCR confidence | Engine confidence per line and per extracted field; mean confidence per record |
| Raw + normalised OCR text | `ocr.raw_text` and `ocr.normalized_text` in every record |
| Modular rules | `ExtractionProfile` holds all labels and patterns; add a profile for a new layout |
| Face thumbnail saved to disk | OpenCV Haar cascade on the aligned card, saved as `storage/thumbnails/<record_id>_face.jpg` - `app/services/face_extractor.py` |
| Thumbnail URL | `face.thumbnail_url` -> `GET /api/v1/thumbnails/{filename}` |
| Unique record ID + persistence | UUID4 hex ID, SQLite with parameterised queries - `app/database/` |
| `GET /api/v1/records/{id}` | Returns the stored record, `404` if unknown, survives restarts |
| HTTP error handling | One JSON error shape, meaningful status codes, no stack traces - `app/utils/errors.py` |
| Python 3.9+, FastAPI, OpenCV, Tesseract/EasyOCR, SQLite | All used; tests verified on Python 3.9.23 and 3.11.4 |

Extra endpoints (documented in [section 11](#11-api-reference-and-curl-examples)):
`GET /health`, `GET /api/v1/records`, `DELETE /api/v1/records/{id}`,
`GET /api/v1/thumbnails/{filename}`.

## 2. Technology stack

| Area | Choice | Why |
|---|---|---|
| Language | Python 3.9+ (verified on 3.9.23 and 3.11.4) | Assignment requirement |
| Web framework | FastAPI + Uvicorn | Automatic validation and OpenAPI/Swagger docs |
| Validation / settings | Pydantic v2, pydantic-settings | Typed request/response models, `.env` support |
| Computer vision | OpenCV 4.x (`opencv-python-headless`) | Contours, perspective transform, preprocessing, Haar face cascade |
| OCR | Tesseract 5 via `pytesseract` (default); EasyOCR (optional) | Open source; Tesseract is light and easy to package in Docker |
| Storage | SQLite (Python standard library) | Zero setup, persistent single file |
| Tests | pytest + FastAPI `TestClient` | Fast API-level tests with a fake OCR engine plus real-OCR tests |

OpenCV is limited to `<5.0` on purpose: OpenCV 5 no longer ships the Haar face
cascade that the thumbnail extraction relies on, and the 4.x line needs no model
download.

## 3. Architecture

```
            POST /api/v1/process-id (multipart image)
                              |
                              v
  +-----------------  app/api/routes.py  (thin HTTP layer) ------------------+
  |                           |                                              |
  |                           v                                              |
  |          app/services/id_processor.py  (orchestration)                   |
  |                                                                          |
  |  upload_validator  ->  document_detector  ->  image_preprocessor         |
  |  (size, magic        (find card, warp to     (resize, denoise, CLAHE,    |
  |   bytes, decode)      flat rectangle)         adaptive threshold)        |
  |                                                     |                    |
  |                                                     v                    |
  |  face_extractor  <-  rotation chosen by OCR  <-  ocr_service             |
  |  (Haar cascade,                                  (Tesseract / EasyOCR,   |
  |   thumbnail file)                                 best variant+rotation) |
  |                                                     |                    |
  |                                                     v                    |
  |                                  field_extractor + date_parser           |
  |                                  (labels, regex, plausibility checks)    |
  +-----------------------------------|--------------------------------------+
                                      v
             app/database/repository.py  ->  SQLite (storage/document_ai.db)
```

Each box is a separate module with one job, so each can be tested on its own.
Shared objects (settings, repository, processor) are created once in
`app/main.py:create_app()` and injected into routes with FastAPI `Depends`;
tests swap in a fake OCR engine and temporary folders the same way.

## 4. Project structure

```
Document_AI_Backend_Intern_Assignment/
├── app/
│   ├── main.py                  # FastAPI app factory, middleware, startup
│   ├── api/routes.py            # HTTP endpoints
│   ├── core/config.py           # Settings from environment / .env
│   ├── schemas/record.py        # Pydantic response models
│   ├── services/
│   │   ├── upload_validator.py  # Upload checks and safe decoding
│   │   ├── document_detector.py # Card boundary + perspective correction
│   │   ├── image_preprocessor.py# OCR preprocessing
│   │   ├── ocr_service.py       # Tesseract / EasyOCR engines, line grouping
│   │   ├── date_parser.py       # Date formats and day/month handling
│   │   ├── field_extractor.py   # Name, number, DOB, expiry rules
│   │   ├── face_extractor.py    # Face thumbnail
│   │   └── id_processor.py      # Runs the whole pipeline, saves the record
│   ├── database/
│   │   ├── connection.py        # SQLite connection + schema
│   │   └── repository.py        # Parameterised queries
│   └── utils/errors.py          # Error classes + JSON error handlers
├── tests/                       # pytest suite (see section 15)
├── scripts/generate_sample_id.py# Synthetic ID-card generator
├── sample_data/                 # One synthetic example + README
├── storage/                     # Runtime data (database, thumbnails) - git-ignored
├── requirements.txt             # Runtime dependencies (version ranges, Python 3.9+)
├── requirements-dev.txt         # + pytest, httpx
├── requirements-easyocr.txt     # Optional EasyOCR engine
├── requirements-lock.txt        # Exact versions verified on Python 3.11
├── .env.example                 # All settings with defaults
├── Dockerfile, docker-compose.yml, .dockerignore
├── Makefile                     # Linux/macOS shortcuts
├── pytest.ini, .gitignore, LICENSE
└── README.md
```

## 5. Prerequisites

* **Python 3.9 or newer** (`python --version`)
* **Tesseract OCR 5.x** installed on the machine (see 6.3), *or* the optional EasyOCR engine
* Git (to publish the project), Docker (optional)

## 6. Installation

All commands run from the project root (the folder that contains `README.md`).

### 6.1 Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
> `.venv\Scripts\activate.bat` from Command Prompt.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 6.2 Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the runtime packages plus pytest. For a server
only, `requirements.txt` is enough. For the exact versions this project was
verified with (Python 3.11), add the lock file as constraints:

```bash
python -m pip install -r requirements-dev.txt -c requirements-lock.txt
```

### 6.3 Install Tesseract OCR

| System | Command |
|---|---|
| Windows | Download the installer from <https://github.com/UB-Mannheim/tesseract/wiki> (or `winget install UB-Mannheim.TesseractOCR`). It installs to `C:\Program Files\Tesseract-OCR\`. |
| Windows without admin rights | `conda create -p .tools\tesseract -c conda-forge tesseract`, then set `TESSERACT_CMD` and `TESSERACT_DATA_DIR` as shown below |
| Ubuntu / Debian | `sudo apt-get install tesseract-ocr tesseract-ocr-eng` |
| macOS | `brew install tesseract` |

Check it: `tesseract --version`. If `tesseract` is not on your `PATH` (common on
Windows), tell the app where it is in `.env`:

```ini
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

Only if Tesseract reports *"Error opening data file eng.traineddata"* (for
example with the conda-forge build), also point it at its language folder:

```ini
TESSERACT_DATA_DIR=D:/path/to/project/.tools/tesseract/share/tessdata
```

`GET /health` shows `"ocr_available": true` once Tesseract and its English data are found.

### 6.4 Optional: EasyOCR instead of Tesseract

```bash
python -m pip install -r requirements-easyocr.txt
```

Then set `OCR_ENGINE=easyocr` in `.env`. EasyOCR installs PyTorch (a large
download) and **downloads its model weights (~100 MB) from the internet the
first time it runs**; set `EASYOCR_MODEL_DIR` to keep them in a known folder.
It runs on CPU by default (`EASYOCR_GPU=false`) and is noticeably slower than
Tesseract on CPU.

> **Windows + Anaconda:** if `import torch` fails with
> `OSError: [WinError 1114] ... c10.dll`, the virtual environment was probably
> created from an Anaconda `python.exe`. Some Anaconda installs ship an old
> Visual C++ runtime DLL (14.2x) in their own folder that Windows loads before
> the system's newer copy, and current PyTorch cannot start with it. Create the
> environment from a python.org installer or a conda-forge Python instead. If
> EasyOCR fails at runtime anyway, the API still answers: records get status
> `ocr_failed` with a warning, and nothing crashes.

## 7. Configuration (.env)

Copy the template and edit what you need:

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Every setting is optional. The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `OCR_ENGINE` | `tesseract` | `tesseract` or `easyocr` |
| `TESSERACT_CMD` | *(PATH)* | Full path to the Tesseract executable |
| `TESSERACT_DATA_DIR` | *(auto)* | Folder with `eng.traineddata`, only if Tesseract cannot find it |
| `TESSERACT_LANG` | `eng` | Tesseract language(s), e.g. `eng+deu` |
| `TESSERACT_PSM` | `11` | Page segmentation mode; 11 ("sparse text") suits cards |
| `DATE_ORDER` | `DMY` | How to read ambiguous dates like `04/05/1990`; use `MDY` for US cards |
| `MAX_UPLOAD_SIZE_MB` | `10` | Upload size limit |
| `MAX_IMAGE_PIXELS` | `40000000` | Rejects huge images before decoding them |
| `ALLOWED_IMAGE_FORMATS` | `jpeg,png,webp,bmp,tiff` | Accepted formats (checked by content) |
| `DATABASE_PATH` | `storage/document_ai.db` | SQLite file |
| `THUMBNAILS_DIR` | `storage/thumbnails` | Face thumbnail folder |
| `INCLUDE_OCR_TEXT` | `true` | Include raw/normalised OCR text in records |
| `SAVE_DEBUG_IMAGES` | `false` | Save aligned and preprocessed card images to `storage/debug` |
| `OCR_TRY_ROTATIONS` | `true` | Retry at 90/180/270 degrees when the first reading is weak |
| `CORS_ALLOW_ORIGINS` | *(empty)* | Comma-separated browser origins; empty keeps CORS disabled |
| `LOG_LEVEL` | `INFO` | Logging level |

Relative paths are resolved from the project root, so the server behaves the
same whichever folder you start it from. Environment variables override `.env`.

## 8. Database initialisation

Nothing to do by hand. On startup the app creates `storage/`, the thumbnails
folder and the SQLite database with its `id_records` table if they do not exist
(`CREATE TABLE IF NOT EXISTS`). Records stay in that file across restarts. To
start fresh, stop the server and delete `storage/document_ai.db` and the files
in `storage/thumbnails/`.

## 9. Running the application

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Add `--reload` while developing (on Linux/macOS, `make run` starts it that way).
The log line `OCR engine 'tesseract' is ready` confirms OCR was found; otherwise
the server still starts, `/health` reports `"degraded"` and uploads return `503`.

## 10. Using Swagger UI to upload an ID image

1. Start the server and open <http://127.0.0.1:8000/docs> (the root URL redirects there).
2. Expand **POST /api/v1/process-id** and click **Try it out**.
3. Click **Choose File** and select `sample_data/synthetic_license_example.jpg`.
4. Click **Execute**. The response body shows the extracted fields; note the `record_id`.
5. Expand **GET /api/v1/records/{record_id}**, click **Try it out**, paste the ID and **Execute**.

ReDoc documentation is at <http://127.0.0.1:8000/redoc>.

## 11. API reference and curl examples

| Method | Path | Purpose | Success |
|---|---|---|---|
| `POST` | `/api/v1/process-id` | Process an ID image and store the record | `201` + `Location` header |
| `GET` | `/api/v1/records/{id}` | Get one stored record | `200` |
| `GET` | `/api/v1/records?limit=20&offset=0` | List record summaries, newest first (no personal fields) | `200` |
| `DELETE` | `/api/v1/records/{id}` | Delete a record and its thumbnail | `204` |
| `GET` | `/api/v1/thumbnails/{record_id}_face.jpg` | Download a face thumbnail | `200` `image/jpeg` |
| `GET` | `/health` | OCR engine and database status | `200` |

Linux/macOS (bash), from the project root:

```bash
# Process an image
curl -X POST http://127.0.0.1:8000/api/v1/process-id \
  -F "file=@sample_data/synthetic_license_example.jpg;type=image/jpeg"

# Retrieve it (use the record_id from the previous response)
curl http://127.0.0.1:8000/api/v1/records/bbaf249c09c1455898f878421fcda73f

# List, download the thumbnail, delete, health
curl "http://127.0.0.1:8000/api/v1/records?limit=10"
curl -o face.jpg http://127.0.0.1:8000/api/v1/thumbnails/bbaf249c09c1455898f878421fcda73f_face.jpg
curl -X DELETE http://127.0.0.1:8000/api/v1/records/bbaf249c09c1455898f878421fcda73f
curl http://127.0.0.1:8000/health
```

Windows PowerShell: use `curl.exe` (plain `curl` is an alias for `Invoke-WebRequest`)
and put the command on one line:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/process-id -F "file=@sample_data/synthetic_license_example.jpg;type=image/jpeg"
curl.exe http://127.0.0.1:8000/api/v1/records/<record_id>
```

## 12. Example request and response

Request: `POST /api/v1/process-id` with `sample_data/synthetic_license_example.jpg`
(a synthetic card photographed at an angle). The response below was produced by
this server with Tesseract 5.5.3 (`ocr.lines` shortened); the data is fictional.

```json
{
  "record_id": "bbaf249c09c1455898f878421fcda73f",
  "status": "completed",
  "created_at": "2026-10-22T06:35:19.592385Z",
  "processing_time_ms": 1944,
  "image": {
    "original_filename": "synthetic_license_example.jpg",
    "declared_content_type": "image/jpeg",
    "detected_format": "jpeg",
    "color_mode": "RGB",
    "file_size_bytes": 280700,
    "width": 1600,
    "height": 1200
  },
  "document_detection": {
    "detected": true,
    "method": "color_contour",
    "corners": [{"x": 340, "y": 304}, {"x": 1333, "y": 292}, {"x": 1349, "y": 880}, {"x": 356, "y": 873}],
    "area_ratio": 0.2995,
    "fallback_used": false,
    "rotation_applied": 0,
    "aligned_width": 994,
    "aligned_height": 589
  },
  "extracted_fields": {
    "full_name": "ALEX SAMPLE",
    "document_number": "SYN-4821-7730",
    "date_of_birth": "1991-03-14",
    "expiry_date": "2031-03-14"
  },
  "field_details": {
    "full_name": {"value": "ALEX SAMPLE", "raw_text": "NAME: ALEX SAMPLE", "ocr_confidence": 0.9533, "method": "label"},
    "document_number": {"value": "SYN-4821-7730", "raw_text": "ID NO: SYN-4821-7730", "ocr_confidence": 0.92, "method": "label"},
    "date_of_birth": {"value": "1991-03-14", "raw_text": "DOB: 14/03/1991", "ocr_confidence": 0.96, "method": "label"},
    "expiry_date": {"value": "2031-03-14", "raw_text": "EXP: 14/03/2031", "ocr_confidence": 0.955, "method": "label"}
  },
  "ocr": {
    "engine": "tesseract",
    "image_variant": "enhanced",
    "mean_confidence": 0.9089,
    "word_count": 28,
    "raw_text": "SYNTHETIC DRIVER LICENSE\nTEST ISSUING AUTHORITY\nNAME: ALEX SAMPLE\nID NO: SYN-4821-7730\nao\nDOB: 14/03/1991\nISS: 15/03/2021\nEXP: 14/03/2031\nCLASS: C\nSYNTHETIC TEST DATA- NOT A REAL ID",
    "normalized_text": "SYNTHETIC DRIVER LICENSE\nTEST ISSUING AUTHORITY\nNAME: ALEX SAMPLE\n...",
    "lines": [
      {"text": "NAME: ALEX SAMPLE", "confidence": 0.9533},
      {"text": "ao", "confidence": 0.03},
      {"text": "DOB: 14/03/1991", "confidence": 0.96}
    ]
  },
  "face": {
    "status": "detected",
    "thumbnail_url": "/api/v1/thumbnails/bbaf249c09c1455898f878421fcda73f_face.jpg",
    "bounding_box": {"x": 121, "y": 232, "width": 235, "height": 235},
    "method": "haar_cascade_on_aligned_card",
    "message": null
  },
  "warnings": [],
  "disclaimer": "OCR extraction is best-effort. Values can be missing or misread, so verify them manually before relying on them in a real-world process."
}
```

Note the real OCR noise (`"ao"` with confidence 0.03, read from the portrait):
the extractor ignores it because it carries no label and no field-shaped value.

`status` is `completed` when all four fields were found, `partial` when some
are `null`, and `ocr_failed` when the OCR engine crashed on the image (the record
is still stored, with a warning). `GET /api/v1/records/{id}` returns exactly the
same JSON.

## 13. How the processing works

**Card detection** (`document_detector.py`). The photo is shrunk to ~900 px and
three binary maps are built: Canny edges, colour distance from the photo's
border (the background), and Otsu brightness thresholds. In each map the largest
outlines are tested. A candidate must cover at least 12% of the photo, have four
corners (or nearly fill its rotated bounding rectangle, for rounded corners),
have a card-like aspect ratio of 1.35-2.0 (ID-1 cards are 1.59:1), show a clear
brightness step across its border, and not run along two image edges (that
would be a tight scan). The largest valid candidate is warped flat. If none
qualifies, the whole image is used and `fallback_used` is `true`.

**Preprocessing** (`image_preprocessor.py`). The card is resized so its long
side is 1400 px, converted to grayscale, denoised (non-local means), contrast
boosted with CLAHE, and binarised with an adaptive Gaussian threshold.

**OCR** (`ocr_service.py`). The engine reads the enhanced grayscale image first,
then the binary image if the result is weak. If it is still weak (fewer than five
confident words) the card is tried at 180, 90 and 270 degrees, so upside-down or
sideways photos work. Words are grouped into lines by position, and a row is
split where there is a wide gap, so two columns are not merged.

**Field extraction** (`field_extractor.py`, `date_parser.py`):

* **Labels first.** `NAME`, `SURNAME`/`GIVEN NAMES`, `ID NO`, `DLN`,
  `EMPLOYEE ID`, `DOB`, `DATE OF BIRTH`, `EXP`, `EXPIRES`, `VALID UNTIL`,
  `ISS` and more. A value is taken from the same line after the label, or from
  the line directly below it.
* **Issue dates are recognised** and never used as birth or expiry dates.
* **Validation.** Names contain only letters and name punctuation and are not
  card vocabulary ("DRIVER LICENSE"). Document numbers need digits and must not
  be dates. A birth date must be in the past and at most 120 years ago. The
  expiry date must be after the birth date.
* **OCR fixes.** `O`->`0`, `I`/`l`->`1` and similar, only when the letter sits
  between digits (`14/O3/1991` -> `1991-03-14`, `48O1` -> `4801`).
* **Cautious fallbacks without labels**, each named in `field_details.*.method`
  and usually with a warning: a single future date becomes the expiry date; a
  date at least ten years older than all others becomes the birth date; a single
  distinctive ID-shaped token (`SYN-4821-7730`, `D1234567`) becomes the document
  number; a single line printed noticeably larger than the rest with 2-4
  alphabetic words becomes the name (typical of badges). **When two candidates
  compete, the field stays `null`.**
* **Extending it.** All labels and patterns live in `ExtractionProfile`
  (`DEFAULT_PROFILE`). A new layout or language means a new profile, e.g. adding
  `nombre` and `fecha de nacimiento` (see `tests/test_field_extractor.py`).

**Face thumbnail** (`face_extractor.py`). OpenCV's bundled Haar frontal-face
cascade runs on the **aligned card**, not the raw photo, so a face in the
background is not picked up when the card was found. Faces wider than 70% of the
card are ignored; the largest remaining face (the main portrait, not a small
"ghost" copy) is cropped with a margin and saved as
`storage/thumbnails/<record_id>_face.jpg`. No face found means status
`not_detected` and no file. A crash inside face extraction is caught and
reported as `error`; the rest of the record is still saved. This is thumbnail
cropping only: there is no face recognition or identity verification.

## 14. Error handling

Every error uses the same JSON shape, and stack traces never reach the client:

```json
{"error": {"code": "unsupported_file_type", "message": "The file content is not a supported image format.", "details": {"allowed_formats": ["bmp", "jpeg", "png", "tiff", "webp"]}}}
```

| Status | `code` | When |
|---|---|---|
| 400 | `empty_file` | The uploaded file has no bytes |
| 404 | `record_not_found` | Unknown or malformed record ID |
| 404 | `thumbnail_not_found` / `not_found` | Unknown thumbnail, unsafe filename, unknown route |
| 413 | `file_too_large` | Upload above `MAX_UPLOAD_SIZE_MB` (checked from `Content-Length` and while reading) |
| 415 | `unsupported_file_type` | Disallowed extension, non-image MIME type, or content that is not JPEG/PNG/WebP/BMP/TIFF |
| 422 | `unreadable_image` | Corrupted or truncated image |
| 422 | `invalid_image_dimensions` | Smaller than `MIN_IMAGE_SIDE_PX` or more than `MAX_IMAGE_PIXELS` |
| 422 | `validation_error` | Missing `file` field or invalid query parameters |
| 500 | `internal_error` | Unexpected failure (details only in the server log) |
| 503 | `ocr_unavailable` | The configured OCR engine is not installed |

## 15. Running the tests

```bash
python -m pytest            # whole suite
python -m pytest -m ocr -v  # only the end-to-end tests with the real OCR engine
python -m pytest -m "not ocr"
```

| File | Covers |
|---|---|
| `test_health.py` | Health (ok / degraded), docs redirect, OpenAPI paths, 404 shape, security headers |
| `test_process_id.py` | Valid upload (201 + Location), full JSON structure, perspective photo alignment, flat-scan fallback, thumbnail saved and served, no-face card, face-extractor crash, missing fields are `null`, OCR failure (`ocr_failed`), OCR unavailable (503), hidden OCR text, debug images, 500 without internals |
| `test_records.py` | Retrieval, 404s, persistence across an app restart, listing + pagination, delete, path-traversal attempts, repository round trip, SQL-injection-shaped input |
| `test_validation.py` | Empty, text, PDF, disguised text, truncated, corrupted, oversized, too small, too many pixels, missing field, disabled format, all five formats, filename sanitising |
| `test_field_extractor.py` | All fields from known OCR text, label-above-value layouts, DOB vs expiry by label, issue dates, nulls instead of guesses, OCR look-alikes, document-number variants, unlabelled heuristics, custom profiles |
| `test_date_parser.py` | 14 date formats, DMY/MDY ambiguity, two-digit years, impossible dates, non-dates |
| `test_image_pipeline.py` | Card detection on 6 synthetic photos, fallback, corner order, preprocessing, face crop, no face, missing model, line grouping, rotation retry |
| `test_ocr_integration.py` | **Real OCR, no mocks**: four synthetic cards (flat and photographed) and an upside-down card |

Most API tests use a fake OCR engine (`tests/helpers.py`) so they are fast and
run anywhere. The `ocr`-marked tests use the real engine from your `.env`; when
it is not installed they are **skipped** with a clear reason, not failed. The
results recorded for this submission are in [section 21](#21-verification-record).

## 16. Generating synthetic test images

```bash
python scripts/generate_sample_id.py --all --output-dir sample_data/generated
python scripts/generate_sample_id.py --style badge --scene --output my_badge.jpg
```

Every generated card is stamped **"SYNTHETIC TEST DATA - NOT A REAL ID"** and
uses invented data and a cartoon portrait. See `sample_data/README.md` for all
options. Never use real identity documents as test data.

## 17. Running with Docker

The image installs Tesseract and its English data, so no model download is
needed at runtime and no internet access is required after the build.

```bash
docker compose up --build -d
curl http://127.0.0.1:8000/health
docker compose logs -f api
docker compose down          # add -v to also delete the stored data volume
```

The SQLite database and thumbnails live in the named volume `docai-storage`
(mounted at `/app/storage`), so they survive container restarts and rebuilds.
The container runs as a non-root user and has a health check. Plain Docker:

```bash
docker build -t document-ai-backend .
docker run -p 8000:8000 -v docai-storage:/app/storage document-ai-backend
```

> **Not verified:** Docker is not installed on the machine this project was
> built on, so the Dockerfile and Compose file were written carefully but have
> **not been built or run**. Please report any build issue.

## 18. Known limitations

* **OCR is best-effort.** Accuracy was measured only on the synthetic cards in
  this repository (clean fonts). Real cards with guilloche backgrounds,
  holograms, glare, tiny print or unusual fonts will produce more errors.
* **The rules target Latin-script, English-labelled cards.** Other layouts need
  a new `ExtractionProfile` (and `TESSERACT_LANG` data for other scripts).
  Machine-readable zones (MRZ) and barcodes are not decoded.
* **Ambiguous dates** such as `04/05/1990` follow `DATE_ORDER` and produce a warning.
* **Card detection** needs the card to stand out from the background and cover
  at least ~12% of the photo. Cards on a same-coloured surface, heavily occluded
  cards or cards cut off by the frame fall back to whole-image processing.
* **Face detection** uses a classic Haar cascade: it can miss tilted, small,
  low-contrast or partly covered faces, and it may fire on face-like patterns.
* **Rotation** is handled in 90-degree steps; small tilts are fixed only when
  the card boundary is detected.
* **Speed:** about 1-2 s per image with Tesseract on a laptop CPU; images with no
  readable text take longer (up to ~8 s) because every rotation is tried.
  Processing is synchronous; a production deployment would use a job queue.
* **Upload limit:** a request without a `Content-Length` header is still
  received in full before its size is checked; put a reverse proxy with a body
  limit (e.g. nginx `client_max_body_size`) in front of a public deployment.
* **No authentication** is included; add it before exposing the API beyond localhost.

## 19. Privacy and data retention

Identity documents are sensitive personal data. This service is a demo and
keeps the footprint small:

* **What is stored:** for each upload, one SQLite row with the extracted
  fields, OCR text (unless `INCLUDE_OCR_TEXT=false`), image metadata, and a face
  thumbnail file. **The uploaded image itself is never written to disk**, unless
  you turn on `SAVE_DEBUG_IMAGES`.
* **Where:** only on the local disk, under `storage/` (or the Docker volume). Nothing
  is sent to any external service; OCR runs locally.
* **How long:** until you delete it. Use `DELETE /api/v1/records/{id}` (removes
  the row and the thumbnail) or delete the `storage/` contents. There is no
  automatic expiry, so add a scheduled clean-up for any real deployment.
* **Logs** contain record IDs, statuses and timings only, never names, numbers,
  dates or OCR text.
* **Filenames from clients** are sanitised and stored as metadata only; files on
  disk always get generated names (`<record_id>_face.jpg`).
* **Not encrypted at rest.** Use disk encryption and access control if real data is ever processed.
* `storage/`, `.env`, databases and generated images are excluded by
  `.gitignore`; only the clearly marked synthetic example is tracked.
* The face thumbnail is a crop for display. **No facial recognition, matching
  or identity verification** is performed.

## 20. Uploading the project to GitHub

1. Create an **empty** repository on GitHub (no README, no .gitignore), e.g. `document-ai-backend`.
2. From the project root:

   ```bash
   git init
   git add .
   git status          # check: no .venv, .env, storage/*.db, thumbnails or images except the synthetic example
   git commit -m "Document AI backend: ID card OCR microservice"
   git branch -M main
   git remote add origin https://github.com/<your-username>/document-ai-backend.git
   git push -u origin main
   ```

3. Open the repository page and confirm the README renders and `storage/`
   contains only `.gitkeep` files.

Submission checklist: README renders; `requirements*.txt` present; tests pass
locally (`python -m pytest`); no secrets, databases, real ID images or private
thumbnails in the history.

## 21. Verification record

Checks run on 22 October 2026 on Windows 11, before packaging:

| Check | Result |
|---|---|
| `python -m pytest` on Python 3.11.4 with Tesseract 5.5.3 | **132 passed**, 0 failed, 0 skipped (includes the 5 real-OCR tests) |
| Same suite on Python 3.9.23 (pip-resolved older FastAPI 0.128, NumPy 2.0) | **132 passed** |
| Same suite with Tesseract made unavailable | 127 passed, **5 skipped** (real-OCR tests), 0 failed |
| Real OCR on 8 synthetic cards (flat + photographed licence and badge) | 32/32 fields correct |
| Real OCR on harder inputs | Upside-down and 90-degree-rotated cards, an 800x600 JPEG at quality 45 and a darkened photo: all fields correct. Random noise and a plain gradient: no fields invented |
| Card detection on 30 synthetic photos (seeds 1-15, licence and badge) | 30/30 detected; 29/30 within 0.12 of the 1.59 card ratio (one at 1.71) |
| Live server (`uvicorn`) | `/health`, upload, retrieval, list, thumbnail, 400/404/413/415/422 cases checked with curl; record retrieved after a server restart; Swagger UI loads |
| Server log | Contains no names, numbers, dates or OCR text |
| `ruff` lint (pyflakes, pycodestyle, bugbear, isort rules) | Clean, apart from FastAPI's standard `Depends()` default-argument pattern (B008) |
| EasyOCR engine (`OCR_ENGINE=easyocr`) | Real-OCR tests **5 passed** with EasyOCR 1.7.2 + PyTorch 2.8 (CPU) on Python 3.9.23, about 3 min including the first model download. On the build machine's Anaconda-based Python 3.11 venv PyTorch could not load (`WinError 1114`, see 6.4); the API then stored records as `ocr_failed` without crashing |
| Docker build / run | **Not run** (Docker not installed on the build machine) |

## License

MIT - see [LICENSE](LICENSE).
