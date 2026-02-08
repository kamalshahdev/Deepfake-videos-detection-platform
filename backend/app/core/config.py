"""Application-wide settings and constants."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    app_name: str
    model_path: Path
    max_upload_mb: int
    allowed_extensions: tuple[str, ...]


settings = Settings(
    app_name="ByteGuard Multimodal API",
    model_path=Path(os.getenv("MODEL_PATH", "models/deepfake_multimodal.pt")),
    max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "200")),
    allowed_extensions=(".mp4", ".mov", ".avi", ".mkv", ".webm"),
)
