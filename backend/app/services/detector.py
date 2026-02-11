"""Inference orchestrator for multimodal deepfake detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import numpy as np

from backend.app.services.feature_extractor import MultimodalFeatureExtractor

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-specific fallback
    torch = None  # type: ignore[assignment]

if torch is not None:
    from backend.app.services.model import load_checkpoint
else:  # pragma: no cover - torch-missing path
    load_checkpoint = None  # type: ignore[assignment]


@dataclass
class PredictionResult:
    score: float
    label: str
    confidence: float
    modality_weights: dict[str, float]
    extracted_modalities: dict[str, bool]
    model_source: str
    report_id: str
    risk_level: str
    summary: str
    recommendation: str
    media_info: dict[str, float]
    forensic_signals: dict[str, float]


class DeepfakeDetector:
    def __init__(self, model_path: Path, decision_threshold: float = 0.5) -> None:
        self.extractor = MultimodalFeatureExtractor()
        self.model_path = model_path
        self.decision_threshold = float(max(0.0, min(1.0, decision_threshold)))

        if torch is not None and load_checkpoint is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.loaded = load_checkpoint(model_path, self.device)
        else:
            self.device = None
            self.loaded = None

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + float(np.exp(-x)))

    @staticmethod
    def _clamp(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.85:
            return "critical"
        if score >= 0.65:
            return "high"
        if score >= 0.40:
            return "medium"
        return "low"

    def _build_report(
        self,
        features: dict[str, object],
        score: float,
        label: str,
        confidence: float,
        weights: dict[str, float],
        source: str,
    ) -> tuple[str, str, str, dict[str, float], dict[str, float]]:
        video = np.asarray(features["video"], dtype=np.float32)
        audio = np.asarray(features["audio"], dtype=np.float32)
        meta = np.asarray(features["metadata"], dtype=np.float32)
        available = dict(features["available"])

        duration = float(meta[0])
        fps = float(meta[1])
        width = int(meta[2])
        height = int(meta[3])
        size_mb = float(meta[4])
        frame_count = int(meta[5])

        resolution = float(width * height)
        bitrate_proxy = float(size_mb / max(duration, 1.0))

        sharpness_mean = float(video[4])
        motion_std = float(video[9])
        saturation_std = float(video[7])

        rms_std = float(audio[1])
        zcr_std = float(audio[7])
        spectral_bandwidth_mean = float(audio[4])

        visual_artifact_score = self._clamp((100.0 - min(sharpness_mean, 100.0)) / 100.0)
        temporal_inconsistency_score = self._clamp(motion_std / 20.0)
        compression_score = self._clamp((0.6 - min(bitrate_proxy, 0.6)) / 0.6)
        audio_anomaly_score = self._clamp((self._clamp((0.05 - min(rms_std, 0.05)) / 0.05) + self._clamp(zcr_std / 0.15)) / 2.0)

        risk_level = self._risk_level(score)
        top_modality = max(weights, key=weights.get) if weights else "unknown"
        missing_modalities = [name for name, ok in available.items() if not ok]

        rationale = []
        if visual_artifact_score >= 0.55:
            rationale.append("visual artifacts are elevated")
        if temporal_inconsistency_score >= 0.55:
            rationale.append("frame-to-frame motion is inconsistent")
        if audio_anomaly_score >= 0.55 and available.get("audio", False):
            rationale.append("audio dynamics look atypical")
        if compression_score >= 0.55:
            rationale.append("bitrate/compression pattern is suspicious")
        if not rationale:
            rationale.append("forensic signals are generally consistent")

        summary = (
            f"Predicted {label.upper()} with score {score:.3f} and confidence {confidence:.3f}. "
            f"Primary modality contribution: {top_modality}. Key observation: {rationale[0]}."
        )
        if missing_modalities:
            summary += f" Missing modalities: {', '.join(missing_modalities)}."

        if label == "fake":
            recommendation = (
                "Treat this media as potentially manipulated. Request source verification and "
                "run a secondary detector before publishing or acting on it."
            )
        else:
            recommendation = (
                "No strong deepfake signal detected. Keep provenance checks and metadata validation "
                "in the review workflow."
            )

        media_info = {
            "duration_seconds": duration,
            "fps": fps,
            "width_px": float(width),
            "height_px": float(height),
            "frame_count": float(frame_count),
            "file_size_mb": size_mb,
            "resolution_px": resolution,
            "bitrate_proxy_mb_per_sec": bitrate_proxy,
            "model_source_code": 1.0 if source != "heuristic_baseline" else 0.0,
        }
        forensic_signals = {
            "visual_artifact_score": visual_artifact_score,
            "temporal_inconsistency_score": temporal_inconsistency_score,
            "audio_anomaly_score": audio_anomaly_score if available.get("audio", False) else 0.0,
            "compression_score": compression_score,
            "sharpness_mean": sharpness_mean,
            "motion_std": motion_std,
            "saturation_std": saturation_std,
            "rms_std": rms_std if available.get("audio", False) else 0.0,
            "zcr_std": zcr_std if available.get("audio", False) else 0.0,
            "spectral_bandwidth_mean": spectral_bandwidth_mean if available.get("audio", False) else 0.0,
        }
        return risk_level, summary, recommendation, media_info, forensic_signals

    def _heuristic_prediction(self, features: dict[str, object]) -> tuple[float, dict[str, float]]:
        video = features["video"]
        audio = features["audio"]
        meta = features["metadata"]
        available = features["available"]

        v_sharpness = float(video[4])
        v_motion_std = float(video[9])
        a_rms_std = float(audio[1])
        a_zcr_std = float(audio[7])

        duration = float(meta[0])
        size_mb = float(meta[4])
        bitrate_proxy = size_mb / max(duration, 1.0)

        visual_artifact = self._clamp((100.0 - min(v_sharpness, 100.0)) / 100.0)
        temporal_inconsistency = self._clamp(v_motion_std / 20.0)
        audio_flatness = self._clamp((0.05 - min(a_rms_std, 0.05)) / 0.05)
        audio_instability = self._clamp(a_zcr_std / 0.15)
        compression_signal = self._clamp((0.6 - min(bitrate_proxy, 0.6)) / 0.6)

        score = self._sigmoid(
            (1.6 * visual_artifact)
            + (1.0 * temporal_inconsistency)
            + (0.8 * audio_flatness)
            + (0.7 * audio_instability)
            + (0.6 * compression_signal)
            - 1.9
        )

        video_weight = 0.55 if available["video"] else 0.0
        audio_weight = 0.30 if available["audio"] else 0.0
        metadata_weight = 0.15 if available["metadata"] else 0.0
        weight_sum = max(video_weight + audio_weight + metadata_weight, 1e-6)

        weights = {
            "video": video_weight / weight_sum,
            "audio": audio_weight / weight_sum,
            "metadata": metadata_weight / weight_sum,
        }
        return score, weights

    def _normalize(self, x: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
        return (x - mean.cpu().numpy()) / (std.cpu().numpy() + 1e-6)

    def predict(self, video_path: Path) -> PredictionResult:
        features = self.extractor.extract(video_path)

        if self.loaded is not None:
            video = np.asarray(features["video"], dtype=np.float32)
            audio = np.asarray(features["audio"], dtype=np.float32)
            meta = np.asarray(features["metadata"], dtype=np.float32)
            mask = np.asarray(features["mask"], dtype=np.float32)

            if self.loaded.norm is not None:
                video = self._normalize(video, self.loaded.norm.video_mean, self.loaded.norm.video_std)
                audio = self._normalize(audio, self.loaded.norm.audio_mean, self.loaded.norm.audio_std)
                meta = self._normalize(meta, self.loaded.norm.meta_mean, self.loaded.norm.meta_std)

            video_t = torch.tensor(video, dtype=torch.float32, device=self.device).unsqueeze(0)
            audio_t = torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)
            meta_t = torch.tensor(meta, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits, modality_weights = self.loaded.model(video_t, audio_t, meta_t, mask_t)
                score = float(torch.sigmoid(logits).cpu().item())
                weights_np = modality_weights.squeeze(0).cpu().numpy()

            weights = {
                "video": float(weights_np[0]),
                "audio": float(weights_np[1]),
                "metadata": float(weights_np[2]),
            }
            source = self.loaded.source
        else:
            score, weights = self._heuristic_prediction(features)
            source = "heuristic_baseline"

        label = "fake" if score >= self.decision_threshold else "real"
        confidence = score if label == "fake" else 1.0 - score
        risk_level, summary, recommendation, media_info, forensic_signals = self._build_report(
            features=features,
            score=score,
            label=label,
            confidence=float(confidence),
            weights=weights,
            source=source,
        )
        report_id = f"RPT-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

        return PredictionResult(
            score=score,
            label=label,
            confidence=float(confidence),
            modality_weights=weights,
            extracted_modalities=dict(features["available"]),
            model_source=source,
            report_id=report_id,
            risk_level=risk_level,
            summary=summary,
            recommendation=recommendation,
            media_info=media_info,
            forensic_signals=forensic_signals,
        )
