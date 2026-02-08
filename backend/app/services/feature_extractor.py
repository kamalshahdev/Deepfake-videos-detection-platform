"""Feature extraction for video, audio, and metadata modalities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - environment-specific fallback
    cv2 = None  # type: ignore[assignment]

try:
    import librosa
except ModuleNotFoundError:  # pragma: no cover - environment-specific fallback
    librosa = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FeatureDims:
    video: int = 10
    audio: int = 8
    metadata: int = 6


class MultimodalFeatureExtractor:
    def __init__(self, frame_stride: int = 12, max_frames: int = 64, sample_rate: int = 16000) -> None:
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.dims = FeatureDims()

    @staticmethod
    def _aggregate(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    def extract_video_features(self, video_path: Path) -> tuple[np.ndarray, bool]:
        if cv2 is None:
            return np.zeros(self.dims.video, dtype=np.float32), False

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return np.zeros(self.dims.video, dtype=np.float32), False

        brightness = []
        contrast = []
        sharpness = []
        saturation = []
        motion = []

        frame_index = 0
        sampled = 0
        prev_gray = None

        while sampled < self.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % self.frame_stride != 0:
                frame_index += 1
                continue

            frame_index += 1
            sampled += 1

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            brightness.append(float(np.mean(gray)))
            contrast.append(float(np.std(gray)))
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            saturation.append(float(np.mean(hsv[:, :, 1])))

            if prev_gray is not None:
                motion.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
            prev_gray = gray

        capture.release()

        if sampled == 0:
            return np.zeros(self.dims.video, dtype=np.float32), False

        b_mean, b_std = self._aggregate(np.asarray(brightness, dtype=np.float32))
        c_mean, c_std = self._aggregate(np.asarray(contrast, dtype=np.float32))
        s_mean, s_std = self._aggregate(np.asarray(sharpness, dtype=np.float32))
        sat_mean, sat_std = self._aggregate(np.asarray(saturation, dtype=np.float32))
        m_mean, m_std = self._aggregate(np.asarray(motion, dtype=np.float32))

        features = np.asarray(
            [
                b_mean,
                b_std,
                c_mean,
                c_std,
                s_mean,
                s_std,
                sat_mean,
                sat_std,
                m_mean,
                m_std,
            ],
            dtype=np.float32,
        )
        return features, True

    def _extract_wav_with_ffmpeg(self, video_path: Path, wav_path: Path) -> bool:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            return False

        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-loglevel",
            "error",
            str(wav_path),
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return result.returncode == 0 and wav_path.exists()

    def extract_audio_features(self, video_path: Path) -> tuple[np.ndarray, bool]:
        if librosa is None:
            return np.zeros(self.dims.audio, dtype=np.float32), False

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav = Path(tmp.name)

        try:
            if not self._extract_wav_with_ffmpeg(video_path, temp_wav):
                return np.zeros(self.dims.audio, dtype=np.float32), False

            waveform, _ = librosa.load(str(temp_wav), sr=self.sample_rate, mono=True)
            if waveform.size == 0:
                return np.zeros(self.dims.audio, dtype=np.float32), False

            rms = librosa.feature.rms(y=waveform)[0]
            centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.sample_rate)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=self.sample_rate)[0]
            zcr = librosa.feature.zero_crossing_rate(y=waveform)[0]

            rms_mean, rms_std = self._aggregate(rms)
            c_mean, c_std = self._aggregate(centroid)
            bw_mean, bw_std = self._aggregate(bandwidth)
            z_mean, z_std = self._aggregate(zcr)

            features = np.asarray(
                [
                    rms_mean,
                    rms_std,
                    c_mean,
                    c_std,
                    bw_mean,
                    bw_std,
                    z_mean,
                    z_std,
                ],
                dtype=np.float32,
            )
            return features, True
        finally:
            if temp_wav.exists():
                os.remove(temp_wav)

    def extract_metadata_features(self, video_path: Path) -> tuple[np.ndarray, bool]:
        if cv2 is None:
            size_mb = float(video_path.stat().st_size / (1024 * 1024)) if video_path.exists() else 0.0
            features = np.asarray([0.0, 0.0, 0.0, 0.0, size_mb, 0.0], dtype=np.float32)
            return features, True

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            size_mb = float(video_path.stat().st_size / (1024 * 1024)) if video_path.exists() else 0.0
            features = np.asarray([0.0, 0.0, 0.0, 0.0, size_mb, 0.0], dtype=np.float32)
            return features, True

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        capture.release()

        duration = float(frame_count / fps) if fps > 0 else 0.0
        size_mb = float(video_path.stat().st_size / (1024 * 1024)) if video_path.exists() else 0.0

        features = np.asarray([duration, fps, width, height, size_mb, frame_count], dtype=np.float32)
        return features, True

    def extract(self, video_path: Path) -> dict[str, object]:
        video, has_video = self.extract_video_features(video_path)
        audio, has_audio = self.extract_audio_features(video_path)
        metadata, has_metadata = self.extract_metadata_features(video_path)

        return {
            "video": video,
            "audio": audio,
            "metadata": metadata,
            "mask": np.asarray(
                [
                    1.0 if has_video else 0.0,
                    1.0 if has_audio else 0.0,
                    1.0 if has_metadata else 0.0,
                ],
                dtype=np.float32,
            ),
            "available": {
                "video": has_video,
                "audio": has_audio,
                "metadata": has_metadata,
            },
        }
