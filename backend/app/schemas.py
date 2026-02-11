"""Pydantic schemas for API I/O."""

from pydantic import BaseModel, Field


class ModalityWeights(BaseModel):
    video: float = Field(..., ge=0.0, le=1.0)
    audio: float = Field(..., ge=0.0, le=1.0)
    metadata: float = Field(..., ge=0.0, le=1.0)


class ExtractedModalities(BaseModel):
    video: bool
    audio: bool
    metadata: bool


class MediaInfo(BaseModel):
    duration_seconds: float
    fps: float
    width_px: float
    height_px: float
    frame_count: float
    file_size_mb: float
    resolution_px: float
    bitrate_proxy_mb_per_sec: float
    model_source_code: float


class ForensicSignals(BaseModel):
    visual_artifact_score: float
    temporal_inconsistency_score: float
    audio_anomaly_score: float
    compression_score: float
    sharpness_mean: float
    motion_std: float
    saturation_std: float
    rms_std: float
    zcr_std: float
    spectral_bandwidth_mean: float


class AnalysisReport(BaseModel):
    report_id: str
    risk_level: str
    summary: str
    recommendation: str
    media_info: MediaInfo
    forensic_signals: ForensicSignals


class PredictionResponse(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_source: str
    modality_weights: ModalityWeights
    extracted_modalities: ExtractedModalities
    report: AnalysisReport


class URLPredictionRequest(BaseModel):
    url: str = Field(..., min_length=5)


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_source: str
    model_path: str
    decision_threshold: float
    video_dim: int
    audio_dim: int
    meta_dim: int
