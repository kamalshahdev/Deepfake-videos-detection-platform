"""Generate a starter manifest CSV template for training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a starter train manifest")
    parser.add_argument("--output", type=Path, default=Path("data/train_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "video_path": ["/path/to/real_01.mp4", "/path/to/fake_01.mp4"],
            "label": [0, 1],
        }
    )
    frame.to_csv(args.output, index=False)
    print(f"Wrote template manifest: {args.output}")


if __name__ == "__main__":
    main()
