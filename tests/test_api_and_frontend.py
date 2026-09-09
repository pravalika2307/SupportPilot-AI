"""
Tests for SupportPilot AI Backend API and Frontend Serving.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """TestClient fixture for FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify health check returns healthy status with agent configuration."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["agent_loaded"] is True
    assert data["classifier"] == "TF-IDF Logistic Regression"
    assert "MiniLM" in data["retriever"]
    assert data["confidence_threshold"] == 0.12
    assert data["similarity_threshold"] == 0.45


def test_analyze_empty_message(client):
    """Verify empty query returns 400 Bad Request."""
    response = client.post("/api/analyze", json={"message": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_battery_drain_query(client):
    """Verify live analysis of a standard self-service query."""
    query = "How do I turn on low power mode on iPhone 7? My battery drains fast."
    response = client.post("/api/analyze", json={"message": query})
    assert response.status_code == 200
    data = response.json()

    assert data["customer_query"] == query
    assert data["predicted_intent"] == "battery_power"
    assert data["confidence"] > 0.12
    assert data["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert isinstance(data["draft_reply"], str)
    assert len(data["draft_reply"]) > 0
    assert data["grounding_status"] in ["grounded", "abstained"]
    assert len(data["retrieved_evidence"]) <= 3
    assert "battery_power" in data["top_probabilities"]


def test_analyze_hardware_escalation_query(client):
    """Verify live analysis triggers escalation for hardware damage."""
    query = "My iPhone screen is completely cracked and shattered after dropping it on the street."
    response = client.post("/api/analyze", json={"message": query})
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_intent"] == "hardware_repair_service"
    assert data["decision"] == "ESCALATE"
    assert "hardware" in data["escalation_reason"].lower() or "repair" in data["escalation_reason"].lower()
    assert isinstance(data["draft_reply"], str)
    assert len(data["draft_reply"]) > 0


def test_metrics_endpoint(client):
    """Verify benchmark metrics endpoint returns human-verified results."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "PASSED" in data["zero_leakage_status"]
    audit = data["audit"]
    reviewer_status = audit["reviewer_status"]
    assert reviewer_status["actually_human_reviewed"] is True
    assert reviewer_status["reviewer_id"] == "Pravalika"
    assert audit["total_golden_records"] == 200
    assert audit["number_reviewed"] == 200

    human_metrics = data["human_verified_benchmark"]
    clf = human_metrics["intent_classification"]
    assert round(clf["overall_accuracy"] * 100, 2) == 65.50

    esc = human_metrics["escalation_policy"]
    assert round(esc["escalation_recall"] * 100, 2) == 88.33

    reply = human_metrics["retrieval_and_reply_quality"]
    assert round(reply["grounded_response_rate"] * 100, 2) == 70.50


def test_failures_endpoint(client):
    """Verify failures endpoint returns safety philosophy and top 5 cases."""
    response = client.get("/api/failures")
    assert response.status_code == 200
    data = response.json()

    assert "Fail-Closed" in data["safety_philosophy"]
    failures = data["top_failures"]
    assert len(failures) == 5
    for f in failures:
        assert "golden_id" in f
        assert "customer_query" in f
        assert "predicted_intent" in f
        assert "expected_intent" in f
        assert "why_it_failed" in f
        assert "improvement_hypothesis" in f


def test_frontend_serving(client):
    """Verify root URL serves the frontend index.html dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "SupportPilot AI" in html
    assert "Intent Accuracy" in html
    assert "65.50%" in html
    assert "Escalation Recall" in html
    assert "88.33%" in html
