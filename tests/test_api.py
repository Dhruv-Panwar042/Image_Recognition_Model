import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api.v1.endpoints import _parse_filter_classes
from backend.app.schemas.analysis import BoundingBox, PerformanceMetrics, DetectionResponse

client = TestClient(app)

def test_health_check():
    """Verify that the system health check endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_supported_classes():
    """Verify that the classes endpoint returns the 80 standard COCO classes."""
    response = client.get("/api/v1/classes")
    assert response.status_code == 200
    classes = response.json()
    assert isinstance(classes, list)
    assert len(classes) == 80
    assert "person" in classes
    assert "car" in classes

def test_parse_filter_classes():
    """Verify comma-separated class filter parsing."""
    assert _parse_filter_classes(None) is None
    assert _parse_filter_classes("") is None
    assert _parse_filter_classes("person, car, dog ") == ["person", "car", "dog"]

def test_bounding_box_schema():
    """Verify BoundingBox schema validation."""
    box = BoundingBox(
        class_id=0,
        class_name="person",
        confidence=0.92,
        box=[10, 20, 100, 200],
        normalized_centroid=[0.1, 0.2]
    )
    assert box.class_name == "person"
    assert box.confidence == 0.92
    assert len(box.box) == 4

def test_invalid_image_upload():
    """Verify that submitting non-image bytes returns 400 Bad Request."""
    response = client.post(
        "/api/v1/detect",
        files={"file": ("corrupt.txt", b"not-a-valid-image", "text/plain")}
    )
    assert response.status_code == 400
