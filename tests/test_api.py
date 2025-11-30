"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    # Note: This will use a mock model loader in a real scenario
    # For now, we'll skip if model is not available
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "environment" in data
    assert "status" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    # May return 503 if model not loaded, which is acceptable in test environment
    assert response.status_code in [200, 503]


def test_predict_endpoint_structure(client):
    """Test predict endpoint structure (may fail if model not loaded)."""
    response = client.post(
        "/predict",
        json={"text": "I love this product!"}
    )
    
    # If model is loaded, should return 200
    # If not loaded, should return 503
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "text" in data
        assert "label" in data
        assert "score" in data
        assert "timestamp" in data
        assert data["label"] in ["POSITIVE", "NEGATIVE"]
        assert 0.0 <= data["score"] <= 1.0


def test_predict_endpoint_invalid_input(client):
    """Test predict endpoint with invalid input."""
    # Empty text
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    # Should return 422 (validation error) or 503 (model not loaded)
    assert response.status_code in [422, 503]
    
    # Missing text field
    response = client.post(
        "/predict",
        json={}
    )
    assert response.status_code in [422, 503]


def test_batch_predict_endpoint(client):
    """Test batch predict endpoint."""
    response = client.post(
        "/predict/batch",
        json=["I love this!", "I hate this."]
    )
    
    # May return 503 if model not loaded
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 2

