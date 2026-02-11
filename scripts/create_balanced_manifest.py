"""Create a class-balanced manifest from an existing train CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create balanced deepfake training manifest")
    parser.add_argument("--input", type=Path, required=True, help="Input manifest CSV with video_path,label")
    parser.add_argument("--output", type=Path, required=True, help="Output balanced manifest CSV")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class after balancing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    required = {"video_path", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label"] = df["label"].astype(int)
    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]

    if real_df.empty or fake_df.empty:
        raise ValueError("Both classes are required to build a balanced manifest")

    per_class = min(len(real_df), len(fake_df))
    if args.max_per_class is not None:
        if args.max_per_class <= 0:
            raise ValueError("--max-per-class must be > 0")
        per_class = min(per_class, args.max_per_class)

    real_sample = real_df.sample(n=per_class, random_state=args.seed)
    fake_sample = fake_df.sample(n=per_class, random_state=args.seed)
    balanced = pd.concat([real_sample, fake_sample], axis=0)

    if args.shuffle:
        balanced = balanced.sample(frac=1.0, random_state=args.seed)
    else:
        balanced = balanced.sort_values(by=["label", "video_path"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(args.output, index=False)

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Input rows: {len(df)} (real={len(real_df)}, fake={len(fake_df)})")
    print(f"Balanced rows: {len(balanced)} (real={per_class}, fake={per_class})")


if __name__ == "__main__":
    main()
