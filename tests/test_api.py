"""Integration tests for MedAI FastAPI endpoints with mocked runtime components."""

import pytest
from fastapi.testclient import TestClient

from src import api as api_module


class DummyModel:
    """Predictable model stub for API endpoint tests."""

    def predict(self, text: str, tokenizer: object, device: str, id2ner: dict = None) -> dict:
        """Return a stable MedAI prediction payload."""
        return {
            "entities": [
                {"type": "TEST", "text": "HbA1c", "confidence": 0.96},
                {"type": "VALUE", "text": "9.1%", "confidence": 0.92},
            ],
            "risk_level": "HIGH",
            "risk_probs": {"LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.80},
            "tokens": ["HbA1c", "9.1%"],
        }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Create a TestClient with model, agent, PDF extraction, and startup mocked."""
    api_module.RATE_LIMIT_STORE.clear()
    api_module.model_manager.model = DummyModel()
    api_module.model_manager.tokenizer = object()
    api_module.model_manager.agent = object()
    api_module.model_manager.rag = object()
    api_module.model_manager.device = "cpu"

    monkeypatch.setattr(api_module.model_manager, "load_model", lambda: None)
    monkeypatch.setattr(api_module.model_manager, "load_rag", lambda: None)
    monkeypatch.setattr(api_module.model_manager, "load_agent", lambda: None)
    monkeypatch.setattr(
        api_module,
        "run_agent_query",
        lambda agent, query, session_id=None: {
            "response": "This is an educational explanation.",
            "tools_used": ["run_ner", "get_risk_level"],
            "session_id": session_id or "test-session",
        },
    )
    monkeypatch.setattr(
        api_module,
        "extract_text_from_pdf",
        lambda file_path: "Patient has HbA1c 9.1% and WBC 14000/uL.",
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """GET /health should report a healthy mocked service."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_text_endpoint(client: TestClient) -> None:
    """POST /analyze should return entities and a risk level for report text."""
    response = client.post("/analyze", json={"text": "Patient has HbA1c 9.1%."})
    payload = response.json()
    assert response.status_code == 200
    assert "entities" in payload
    assert payload["risk_level"] == "HIGH"


def test_analyze_pdf_endpoint(client: TestClient) -> None:
    """POST /analyze-pdf should accept a PDF upload and analyze extracted text."""
    response = client.post(
        "/analyze-pdf",
        files={"file": ("sample.pdf", b"%PDF-1.4\n% test pdf bytes", "application/pdf")},
    )
    payload = response.json()
    assert response.status_code == 200
    assert "entities" in payload
    assert payload["risk_level"] == "HIGH"


def test_chat_endpoint(client: TestClient) -> None:
    """POST /chat should call the mocked agent runner and return a response."""
    response = client.post("/chat", json={"message": "Explain HbA1c", "session_id": "abc"})
    payload = response.json()
    assert response.status_code == 200
    assert "response" in payload
    assert payload["session_id"] == "abc"


def test_invalid_input(client: TestClient) -> None:
    """POST /analyze with empty text should fail validation."""
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422
