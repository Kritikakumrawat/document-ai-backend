# Shortcuts for Linux/macOS (Windows users: run the commands from the README directly).
# Activate the virtual environment first, or pass PYTHON=.venv/bin/python.
PYTHON ?= python

.PHONY: install install-dev run test test-ocr check samples docker-build docker-up docker-down clean

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

run:
	$(PYTHON) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

test:
	$(PYTHON) -m pytest

test-ocr:
	$(PYTHON) -m pytest -m ocr -v

check:
	$(PYTHON) -m compileall -q app scripts tests

samples:
	$(PYTHON) scripts/generate_sample_id.py --all --output-dir sample_data/generated

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
