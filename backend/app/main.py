"""FastAPI app entrypoint for multimodal deepfake detection."""

from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.schemas import (
    AnalysisReport,
    ExtractedModalities,
    ForensicSignals,
    HealthResponse,
    MediaInfo,
    ModelInfoResponse,
    ModalityWeights,
    PredictionResponse,
    URLPredictionRequest,
)
from backend.app.services.detector import DeepfakeDetector, PredictionResult
from backend.app.services.url_fetcher import DownloadedVideo, fetch_video_from_url


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = DeepfakeDetector(
    model_path=settings.model_path,
    decision_threshold=settings.decision_threshold,
)


def _to_prediction_response(result: PredictionResult) -> PredictionResponse:
    return PredictionResponse(
        label=result.label,
        score=result.score,
        confidence=result.confidence,
        model_source=result.model_source,
        modality_weights=ModalityWeights(**result.modality_weights),
        extracted_modalities=ExtractedModalities(**result.extracted_modalities),
        report=AnalysisReport(
            report_id=result.report_id,
            risk_level=result.risk_level,
            summary=result.summary,
            recommendation=result.recommendation,
            media_info=MediaInfo(**result.media_info),
            forensic_signals=ForensicSignals(**result.forensic_signals),
        ),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    dims = detector.extractor.dims
    return ModelInfoResponse(
        model_loaded=detector.loaded is not None,
        model_source=detector.loaded.source if detector.loaded else "heuristic_baseline",
        model_path=str(settings.model_path),
        decision_threshold=settings.decision_threshold,
        video_dim=dims.video,
        audio_dim=dims.audio,
        meta_dim=dims.metadata,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(video: UploadFile = File(...)) -> PredictionResponse:
    extension = Path(video.filename or "upload.mp4").suffix.lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {extension}")

    contents = await video.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > settings.max_upload_mb:
        raise HTTPException(status_code=413, detail="File too large")

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(contents)
        temp_path = Path(tmp.name)

    try:
        result = detector.predict(temp_path)
        return _to_prediction_response(result)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/predict-url", response_model=PredictionResponse)
async def predict_url(payload: URLPredictionRequest) -> PredictionResponse:
    downloaded: DownloadedVideo | None = None
    try:
        downloaded = fetch_video_from_url(
            url=payload.url,
            max_upload_mb=settings.max_upload_mb,
            allowed_extensions=settings.allowed_extensions,
        )
        result = detector.predict(downloaded.path)
        return _to_prediction_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if downloaded is not None:
            downloaded.cleanup()
