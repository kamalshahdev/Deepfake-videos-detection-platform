"""Train and evaluate multiple model variants on cached features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.model import MultimodalClassifier


@dataclass(frozen=True)
class VariantConfig:
    name: str
    epochs: int
    hidden_dim: int
    dropout: float
    lr: float
    seed: int


DEFAULT_VARIANTS: list[VariantConfig] = [
    VariantConfig("v1_baseline", epochs=12, hidden_dim=128, dropout=0.20, lr=1e-3, seed=42),
    VariantConfig("v2_low_lr", epochs=16, hidden_dim=128, dropout=0.20, lr=5e-4, seed=42),
    VariantConfig("v3_bigger_hidden", epochs=14, hidden_dim=192, dropout=0.25, lr=7e-4, seed=42),
    VariantConfig("v4_more_dropout", epochs=14, hidden_dim=128, dropout=0.35, lr=7e-4, seed=42),
    VariantConfig("v5_smaller_hidden", epochs=12, hidden_dim=96, dropout=0.20, lr=1e-3, seed=42),
    VariantConfig("v6_seed7", epochs=14, hidden_dim=192, dropout=0.25, lr=7e-4, seed=7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize model variants on cached features")
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/train_manifest_dfd_2000.csv"))
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("data/train_manifest_dfd_2000.csv.features.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("A:/deepfake-data/models/variants_2000"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--min-fake-recall", type=float, default=0.75)
    parser.add_argument("--summary-json", type=Path, default=Path("A:/deepfake-data/models/variants_2000_summary.json"))
    return parser.parse_args()


def _normalize(values: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    mean_np = np.asarray(mean, dtype=np.float32)
    std_np = np.asarray(std, dtype=np.float32) + 1e-6
    return (values - mean_np) / std_np


def _compute_threshold_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (probs >= threshold).astype(np.float32)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float((specificity + float(recall)) / 2.0)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "specificity_real": specificity,
        "balanced_accuracy": bal_acc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _evaluate_checkpoint(checkpoint_path: Path, cache_data: dict[str, np.ndarray]) -> dict[str, Any]:
    ck = torch.load(checkpoint_path, map_location="cpu")

    video = cache_data["video"]
    audio = cache_data["audio"]
    metadata = cache_data["metadata"]
    mask = cache_data["mask"]
    labels = cache_data["labels"]
    val_indices = np.asarray(ck["val_indices"], dtype=np.int64)

    norm = ck["normalization"]
    video_norm = _normalize(video, norm["video_mean"], norm["video_std"])
    audio_norm = _normalize(audio, norm["audio_mean"], norm["audio_std"])
    meta_norm = _normalize(metadata, norm["meta_mean"], norm["meta_std"])

    x_video = torch.tensor(video_norm[val_indices], dtype=torch.float32)
    x_audio = torch.tensor(audio_norm[val_indices], dtype=torch.float32)
    x_meta = torch.tensor(meta_norm[val_indices], dtype=torch.float32)
    x_mask = torch.tensor(mask[val_indices], dtype=torch.float32)
    y_true = labels[val_indices]

    model = MultimodalClassifier(
        video_dim=int(ck["video_dim"]),
        audio_dim=int(ck["audio_dim"]),
        meta_dim=int(ck["meta_dim"]),
        hidden_dim=int(ck["hidden_dim"]),
        dropout=float(ck["dropout"]),
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits, _ = model(x_video, x_audio, x_meta, x_mask)
        probs = torch.sigmoid(logits).numpy()

    auc = float(roc_auc_score(y_true, probs))
    metrics_05 = _compute_threshold_metrics(y_true, probs, 0.5)

    thresholds = np.arange(0.30, 0.901, 0.05)
    threshold_metrics = [_compute_threshold_metrics(y_true, probs, float(t)) for t in thresholds]
    return {
        "val_samples": int(len(val_indices)),
        "auc": auc,
        "metrics_at_0_5": metrics_05,
        "threshold_metrics": threshold_metrics,
    }


def _select_threshold(
    threshold_metrics: list[dict[str, float]],
    min_fake_recall: float,
) -> dict[str, float]:
    candidates = [m for m in threshold_metrics if m["recall_fake"] >= min_fake_recall]
    if candidates:
        return max(
            candidates,
            key=lambda m: (m["balanced_accuracy"], m["precision_fake"], m["accuracy"]),
        )
    return max(
        threshold_metrics,
        key=lambda m: (m["balanced_accuracy"], m["recall_fake"], m["precision_fake"]),
    )


def _train_variant(args: argparse.Namespace, variant: VariantConfig, checkpoint_path: Path) -> float:
    cmd = [
        str((ROOT_DIR / ".venv" / "Scripts" / "python.exe").resolve()),
        "scripts/train.py",
        "--dataset-csv",
        str(args.dataset_csv),
        "--output",
        str(checkpoint_path),
        "--epochs",
        str(variant.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(variant.lr),
        "--hidden-dim",
        str(variant.hidden_dim),
        "--dropout",
        str(variant.dropout),
        "--seed",
        str(variant.seed),
        "--frame-stride",
        str(args.frame_stride),
        "--max-frames",
        str(args.max_frames),
        "--progress-every",
        "0",
        "--feature-cache",
        str(args.feature_cache),
    ]

    started = time.time()
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)
    return time.time() - started


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    if not args.dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV missing: {args.dataset_csv}")
    if not args.feature_cache.exists():
        raise FileNotFoundError(f"Feature cache missing: {args.feature_cache}")

    payload = np.load(args.feature_cache)
    required = {"video", "audio", "metadata", "mask", "labels"}
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"Feature cache missing arrays: {sorted(missing)}")
    cache_data = {
        "video": np.asarray(payload["video"], dtype=np.float32),
        "audio": np.asarray(payload["audio"], dtype=np.float32),
        "metadata": np.asarray(payload["metadata"], dtype=np.float32),
        "mask": np.asarray(payload["mask"], dtype=np.float32),
        "labels": np.asarray(payload["labels"], dtype=np.float32),
    }

    results: list[dict[str, Any]] = []
    print(f"Running {len(DEFAULT_VARIANTS)} variants...")

    for variant in DEFAULT_VARIANTS:
        checkpoint_path = args.output_dir / f"{variant.name}.pt"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        print(f"\n=== {variant.name} ===")
        train_seconds = _train_variant(args, variant, checkpoint_path)
        eval_payload = _evaluate_checkpoint(checkpoint_path, cache_data)
        best_threshold = _select_threshold(
            eval_payload["threshold_metrics"],
            min_fake_recall=float(args.min_fake_recall),
        )

        auc = float(eval_payload["auc"])
        score = (0.6 * auc) + (0.4 * float(best_threshold["balanced_accuracy"]))

        result = {
            "variant": asdict(variant),
            "checkpoint": str(checkpoint_path),
            "train_seconds": round(train_seconds, 2),
            "auc": auc,
            "metrics_at_0_5": eval_payload["metrics_at_0_5"],
            "best_threshold": best_threshold,
            "ranking_score": float(score),
        }
        results.append(result)
        print(
            f"AUC={auc:.4f} | thr*={best_threshold['threshold']:.2f} | "
            f"bal_acc={best_threshold['balanced_accuracy']:.4f} | "
            f"recall_fake={best_threshold['recall_fake']:.4f}"
        )

    best = max(
        results,
        key=lambda r: (
            r["ranking_score"],
            r["auc"],
            r["best_threshold"]["balanced_accuracy"],
            r["best_threshold"]["precision_fake"],
        ),
    )

    summary = {
        "dataset_csv": str(args.dataset_csv),
        "feature_cache": str(args.feature_cache),
        "generated_at_epoch_time": int(time.time()),
        "min_fake_recall": float(args.min_fake_recall),
        "variants": results,
        "best": best,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBest variant: {best['variant']['name']}")
    print(f"Recommended checkpoint: {best['checkpoint']}")
    print(f"Recommended threshold: {best['best_threshold']['threshold']:.2f}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
