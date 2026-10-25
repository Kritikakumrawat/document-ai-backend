from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from tests.helpers import FakeOcrEngine, make_settings


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "1.0.0",
        "ocr_engine": "fake",
        "ocr_available": True,
        "database": "ok",
    }


def test_health_reports_degraded_when_ocr_engine_is_missing(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path), ocr_engine=FakeOcrEngine(available=False))
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["ocr_available"] is False


def test_root_redirects_to_swagger_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_openapi_lists_required_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/api/v1/process-id"]
    assert "get" in paths["/api/v1/records/{record_id}"]


def test_unknown_route_uses_json_error_shape(client: TestClient) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_security_headers_are_set(client: TestClient) -> None:
    response = client.get("/api/v1/records")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
