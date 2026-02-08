"""Merge multiple manifest CSV files into one deduplicated manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge deepfake manifest CSV files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input manifest CSV files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample majority class so real and fake counts match",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames = []
    for csv_path in args.inputs:
        if not csv_path.exists():
            raise FileNotFoundError(f"Input manifest not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        expected = {"video_path", "label"}
        missing = expected.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
        frames.append(frame[["video_path", "label"]].copy())

    merged = pd.concat(frames, ignore_index=True)
    merged["video_path"] = merged["video_path"].astype(str)
    merged["label"] = merged["label"].astype(int)

    before = len(merged)
    merged = merged.drop_duplicates(subset=["video_path"], keep="first")

    if args.balance:
        real = merged[merged["label"] == 0]
        fake = merged[merged["label"] == 1]
        if len(real) == 0 or len(fake) == 0:
            raise ValueError("Cannot balance: one class is empty")

        minority = min(len(real), len(fake))
        rnd = random.Random(args.seed)

        real_idx = list(real.index)
        fake_idx = list(fake.index)
        rnd.shuffle(real_idx)
        rnd.shuffle(fake_idx)

        keep = real_idx[:minority] + fake_idx[:minority]
        merged = merged.loc[keep].copy()

    if args.shuffle:
        merged = merged.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        merged = merged.sort_values("video_path").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    real_count = int((merged["label"] == 0).sum())
    fake_count = int((merged["label"] == 1).sum())

    print(f"Input rows: {before}")
    print(f"Unique rows: {len(merged)}")
    print(f"Output: {args.output}")
    print(f"Class counts -> real: {real_count}, fake: {fake_count}")


if __name__ == "__main__":
    main()
