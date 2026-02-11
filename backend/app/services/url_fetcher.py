"""Download helper for URL-based deepfake analysis."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import shutil
import socket
import tempfile
from urllib.parse import urlparse

import requests


VIDEO_CONTENT_TYPE_EXT = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}


@dataclass
class DownloadedVideo:
    path: Path
    temp_dir: Path | None = None

    def cleanup(self) -> None:
        self.path.unlink(missing_ok=True)
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def _is_private_or_local_host(hostname: str) -> bool:
    if hostname.lower() in {"localhost"}:
        return True

    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True

    for address_info in addresses:
        ip_raw = address_info[4][0]
        ip_obj = ipaddress.ip_address(ip_raw)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            return True

    return False


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported")
    if not parsed.hostname:
        raise ValueError("Invalid URL host")
    if _is_private_or_local_host(parsed.hostname):
        raise ValueError("Private/local hosts are not allowed")


def _safe_extension(url: str, content_type: str, allowed_extensions: tuple[str, ...]) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in allowed_extensions:
        return ext

    ext_from_type = VIDEO_CONTENT_TYPE_EXT.get(content_type.lower())
    if ext_from_type in allowed_extensions:
        return ext_from_type

    return ".mp4"


def _download_direct_video(
    url: str,
    max_upload_mb: int,
    allowed_extensions: tuple[str, ...],
) -> DownloadedVideo:
    max_bytes = int(max_upload_mb * 1024 * 1024)
    user_agent = "ByteGuard-Deepfake-Detector/1.0"

    with requests.get(url, stream=True, timeout=(10, 120), allow_redirects=True, headers={"User-Agent": user_agent}) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        content_length_raw = resp.headers.get("Content-Length")
        if content_length_raw and content_length_raw.isdigit():
            if int(content_length_raw) > max_bytes:
                raise ValueError("Remote video exceeds max upload size")

        # For direct file URLs, many servers use application/octet-stream.
        is_videoish = content_type.startswith("video/") or content_type in {"application/octet-stream", ""}
        if not is_videoish:
            raise ValueError(f"URL content type is not video: {content_type or 'unknown'}")

        suffix = _safe_extension(url, content_type, allowed_extensions)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            total = 0
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Downloaded file exceeds max upload size")
                tmp_file.write(chunk)
            temp_path = Path(tmp_file.name)

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Downloaded file is empty")

    return DownloadedVideo(path=temp_path)


def _download_with_yt_dlp(
    url: str,
    max_upload_mb: int,
    allowed_extensions: tuple[str, ...],
) -> DownloadedVideo:
    try:
        import yt_dlp
    except ModuleNotFoundError as exc:
        raise ValueError(
            "URL appears to require social-media extraction. Install yt-dlp to support this source."
        ) from exc

    max_bytes = int(max_upload_mb * 1024 * 1024)
    temp_dir = Path(tempfile.mkdtemp(prefix="bg-url-"))
    outtmpl = str(temp_dir / "video.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        files = [path for path in temp_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions]
        if not files:
            files = [path for path in temp_dir.iterdir() if path.is_file()]
        if not files:
            raise ValueError("Could not extract downloadable video from URL")

        video_path = max(files, key=lambda item: item.stat().st_size)
        if video_path.stat().st_size > max_bytes:
            raise ValueError("Downloaded video exceeds max upload size")
        return DownloadedVideo(path=video_path, temp_dir=temp_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def fetch_video_from_url(
    url: str,
    max_upload_mb: int,
    allowed_extensions: tuple[str, ...],
) -> DownloadedVideo:
    clean_url = url.strip()
    if not clean_url:
        raise ValueError("URL is required")

    _validate_url(clean_url)

    try:
        return _download_direct_video(clean_url, max_upload_mb=max_upload_mb, allowed_extensions=allowed_extensions)
    except Exception:
        # Fallback for social URLs (X/Twitter/etc.) that require extraction.
        return _download_with_yt_dlp(clean_url, max_upload_mb=max_upload_mb, allowed_extensions=allowed_extensions)
