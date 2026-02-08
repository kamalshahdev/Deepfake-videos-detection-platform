"""Build a training manifest from public deepfake dataset directory layouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Iterable


DEFAULT_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train manifest from public deepfake datasets")
    parser.add_argument(
        "--dataset-type",
        choices=["dfdc", "celebdfv2", "faceforensicspp", "dfd", "generic"],
        required=True,
        help="Dataset layout parser to use",
    )
    parser.add_argument("--root", type=Path, required=True, help="Path to dataset root directory")
    parser.add_argument("--output", type=Path, default=Path("data/train_manifest.csv"))
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="Video extensions to include (example: .mp4 .avi)",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute video paths in manifest (default writes workspace-relative paths)",
    )
    return parser.parse_args()


def is_video_file(path: Path, extensions: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in extensions


def normalize_path(path: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve().as_posix())
    return str(path.as_posix())


def collect_dfdc(root: Path, extensions: set[str]) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    for metadata_file in root.rglob("metadata.json"):
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        for filename, info in payload.items():
            label_raw = str(info.get("label", "")).upper()
            if label_raw not in {"REAL", "FAKE"}:
                continue

            video_path = metadata_file.parent / filename
            if not is_video_file(video_path, extensions):
                continue

            label = 1 if label_raw == "FAKE" else 0
            rows.append((video_path, label))

    return rows


def collect_celebdfv2(root: Path, extensions: set[str]) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    mapping = {
        "Celeb-real": 0,
        "YouTube-real": 0,
        "Celeb-synthesis": 1,
    }

    for directory, label in mapping.items():
        base = root / directory
        if not base.exists():
            continue

        for path in base.rglob("*"):
            if is_video_file(path, extensions):
                rows.append((path, label))

    return rows


def collect_faceforensicspp(root: Path, extensions: set[str]) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []

    for path in root.rglob("*"):
        if not is_video_file(path, extensions):
            continue

        parts = set(path.parts)
        if "original_sequences" in parts:
            rows.append((path, 0))
        elif "manipulated_sequences" in parts:
            rows.append((path, 1))

    return rows


def collect_generic(root: Path, extensions: set[str]) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    mapping = {"real": 0, "fake": 1}

    for directory, label in mapping.items():
        base = root / directory
        if not base.exists():
            continue

        for path in base.rglob("*"):
            if is_video_file(path, extensions):
                rows.append((path, label))

    return rows


def collect_dfd(root: Path, extensions: set[str]) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    mapping = {
        "DFD_original sequences": 0,
        "DFD_manipulated_sequences": 1,
    }

    for directory, label in mapping.items():
        base = root / directory
        if not base.exists():
            continue

        for path in base.rglob("*"):
            if is_video_file(path, extensions):
                rows.append((path, label))

    return rows


def deduplicate(rows: Iterable[tuple[Path, int]]) -> list[tuple[Path, int]]:
    seen: set[tuple[str, int]] = set()
    unique: list[tuple[Path, int]] = []

    for path, label in rows:
        key = (str(path.resolve().as_posix()), label)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, label))

    return unique


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}

    collectors = {
        "dfdc": collect_dfdc,
        "celebdfv2": collect_celebdfv2,
        "faceforensicspp": collect_faceforensicspp,
        "dfd": collect_dfd,
        "generic": collect_generic,
    }

    rows = collectors[args.dataset_type](root, extensions)
    rows = deduplicate(rows)

    if not rows:
        raise ValueError(
            "No samples found. Check dataset type, root path, and extension filters."
        )

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    else:
        rows.sort(key=lambda item: str(item[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["video_path", "label"])
        for path, label in rows:
            writer.writerow([normalize_path(path, args.absolute_paths), label])

    real_count = sum(1 for _, label in rows if label == 0)
    fake_count = sum(1 for _, label in rows if label == 1)

    print(f"Dataset root: {root}")
    print(f"Manifest: {args.output}")
    print(f"Samples: {len(rows)} (real={real_count}, fake={fake_count})")


if __name__ == "__main__":
    main()
