from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert "model_loaded" in payload
    assert "video_dim" in payload
    assert "audio_dim" in payload
    assert "meta_dim" in payload
