from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.detector import PredictionResult


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
    assert "decision_threshold" in payload
    assert "video_dim" in payload
    assert "audio_dim" in payload
    assert "meta_dim" in payload


def test_predict_url_endpoint(monkeypatch, tmp_path) -> None:
    import backend.app.main as main_module

    temp_video = tmp_path / "video.mp4"
    temp_video.write_bytes(b"fake-bytes")

    class DummyDownloaded:
        def __init__(self, path):
            self.path = path
            self.cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True
            self.path.unlink(missing_ok=True)

    downloaded = DummyDownloaded(temp_video)

    def fake_fetch_video_from_url(url: str, max_upload_mb: int, allowed_extensions: tuple[str, ...]):
        assert url.startswith("https://")
        assert max_upload_mb > 0
        assert ".mp4" in allowed_extensions
        return downloaded

    def fake_predict(path):
        assert path == temp_video
        return PredictionResult(
            score=0.91,
            label="fake",
            confidence=0.91,
            modality_weights={"video": 0.6, "audio": 0.25, "metadata": 0.15},
            extracted_modalities={"video": True, "audio": True, "metadata": True},
            model_source="unit_test_model",
            report_id="RPT-TEST",
            risk_level="high",
            summary="Unit test summary",
            recommendation="Unit test recommendation",
            media_info={
                "duration_seconds": 1.0,
                "fps": 30.0,
                "width_px": 1280.0,
                "height_px": 720.0,
                "frame_count": 30.0,
                "file_size_mb": 1.0,
                "resolution_px": 921600.0,
                "bitrate_proxy_mb_per_sec": 1.0,
                "model_source_code": 1.0,
            },
            forensic_signals={
                "visual_artifact_score": 0.7,
                "temporal_inconsistency_score": 0.6,
                "audio_anomaly_score": 0.5,
                "compression_score": 0.4,
                "sharpness_mean": 12.0,
                "motion_std": 1.0,
                "saturation_std": 2.0,
                "rms_std": 0.1,
                "zcr_std": 0.1,
                "spectral_bandwidth_mean": 3.0,
            },
        )

    monkeypatch.setattr(main_module, "fetch_video_from_url", fake_fetch_video_from_url)
    monkeypatch.setattr(main_module.detector, "predict", fake_predict)

    response = client.post("/predict-url", json={"url": "https://example.com/video.mp4"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "fake"
    assert payload["report"]["report_id"] == "RPT-TEST"
    assert downloaded.cleaned is True
