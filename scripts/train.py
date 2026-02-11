"""Train multimodal deepfake model from a manifest CSV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.feature_extractor import MultimodalFeatureExtractor
from backend.app.services.model import MultimodalClassifier


@dataclass
class FeatureBatch:
    video: np.ndarray
    audio: np.ndarray
    metadata: np.ndarray
    mask: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal deepfake detector")
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/deepfake_multimodal.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-class-weighting",
        action="store_true",
        help="Disable automatic BCE pos_weight computed from training label distribution",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --output checkpoint if it exists",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Explicit checkpoint to resume from",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Path for cached extracted features (.npz). Default: <dataset_csv>.features.npz",
    )
    parser.add_argument(
        "--disable-feature-cache",
        action="store_true",
        help="Disable feature cache read/write",
    )
    parser.add_argument(
        "--disable-epoch-checkpoint",
        action="store_true",
        help="Save model only at the very end (not recommended)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=12,
        help="Video frame sampling stride for feature extraction",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=64,
        help="Maximum sampled frames per video for feature extraction",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print extraction progress every N processed rows",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Optional JSON file to persist extraction/training progress",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_feature_cache_path(dataset_csv: Path) -> Path:
    return dataset_csv.with_suffix(dataset_csv.suffix + ".features.npz")


def load_feature_cache(cache_path: Path) -> FeatureBatch | None:
    if not cache_path.exists():
        return None

    payload = np.load(cache_path)
    required = {"video", "audio", "metadata", "mask", "labels"}
    if not required.issubset(set(payload.files)):
        print(f"Feature cache is invalid, rebuilding: {cache_path}")
        return None

    batch = FeatureBatch(
        video=np.asarray(payload["video"], dtype=np.float32),
        audio=np.asarray(payload["audio"], dtype=np.float32),
        metadata=np.asarray(payload["metadata"], dtype=np.float32),
        mask=np.asarray(payload["mask"], dtype=np.float32),
        labels=np.asarray(payload["labels"], dtype=np.float32),
    )
    print(f"Loaded feature cache: {cache_path} (samples={len(batch.labels)})")
    return batch


def save_feature_cache(cache_path: Path, batch: FeatureBatch) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        video=batch.video,
        audio=batch.audio,
        metadata=batch.metadata,
        mask=batch.mask,
        labels=batch.labels,
    )
    print(f"Saved feature cache: {cache_path}")


def write_progress(progress_file: Path | None, payload: dict[str, Any]) -> None:
    if progress_file is None:
        return
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path = progress_file.with_suffix(progress_file.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(progress_file)


def build_features(
    dataset_csv: Path,
    extractor: MultimodalFeatureExtractor,
    progress_every: int = 50,
    progress_file: Path | None = None,
) -> FeatureBatch:
    df = pd.read_csv(dataset_csv)
    expected = {"video_path", "label"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset CSV: {sorted(missing)}")

    total_rows = int(len(df))
    missing_paths = 0
    invalid_labels = 0

    videos: list[np.ndarray] = []
    audios: list[np.ndarray] = []
    metas: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[float] = []

    started_at = time.time()
    total_for_eta = max(total_rows, 1)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        path = Path(row.video_path)
        if not path.exists():
            missing_paths += 1
            if progress_every > 0 and idx % progress_every == 0:
                elapsed = time.time() - started_at
                rate = idx / max(elapsed, 1e-6)
                eta = (total_for_eta - idx) / max(rate, 1e-6)
                write_progress(
                    progress_file,
                    {
                        "stage": "extracting_features",
                        "processed_rows": idx,
                        "total_rows": total_rows,
                        "valid_samples": len(labels),
                        "missing_paths": missing_paths,
                        "invalid_labels": invalid_labels,
                        "elapsed_seconds": round(elapsed, 2),
                        "eta_seconds": round(eta, 2),
                    },
                )
                print(
                    f"[extract] {idx}/{total_rows} rows | valid={len(labels)} | "
                    f"missing={missing_paths} | elapsed={elapsed:.1f}s | eta={eta/60:.1f}m"
                )
            continue

        try:
            label = float(row.label)
        except (TypeError, ValueError):
            invalid_labels += 1
            if progress_every > 0 and idx % progress_every == 0:
                elapsed = time.time() - started_at
                rate = idx / max(elapsed, 1e-6)
                eta = (total_for_eta - idx) / max(rate, 1e-6)
                write_progress(
                    progress_file,
                    {
                        "stage": "extracting_features",
                        "processed_rows": idx,
                        "total_rows": total_rows,
                        "valid_samples": len(labels),
                        "missing_paths": missing_paths,
                        "invalid_labels": invalid_labels,
                        "elapsed_seconds": round(elapsed, 2),
                        "eta_seconds": round(eta, 2),
                    },
                )
                print(
                    f"[extract] {idx}/{total_rows} rows | valid={len(labels)} | "
                    f"invalid_labels={invalid_labels} | elapsed={elapsed:.1f}s | eta={eta/60:.1f}m"
                )
            continue

        extracted = extractor.extract(path)
        videos.append(np.asarray(extracted["video"], dtype=np.float32))
        audios.append(np.asarray(extracted["audio"], dtype=np.float32))
        metas.append(np.asarray(extracted["metadata"], dtype=np.float32))
        masks.append(np.asarray(extracted["mask"], dtype=np.float32))
        labels.append(label)

        if progress_every > 0 and idx % progress_every == 0:
            elapsed = time.time() - started_at
            rate = idx / max(elapsed, 1e-6)
            eta = (total_for_eta - idx) / max(rate, 1e-6)
            write_progress(
                progress_file,
                {
                    "stage": "extracting_features",
                    "processed_rows": idx,
                    "total_rows": total_rows,
                    "valid_samples": len(labels),
                    "missing_paths": missing_paths,
                    "invalid_labels": invalid_labels,
                    "elapsed_seconds": round(elapsed, 2),
                    "eta_seconds": round(eta, 2),
                },
            )
            print(
                f"[extract] {idx}/{total_rows} rows | valid={len(labels)} | "
                f"elapsed={elapsed:.1f}s | eta={eta/60:.1f}m"
            )

    if not labels:
        raise ValueError("No valid samples found. Check video paths and labels in dataset CSV.")

    total_elapsed = time.time() - started_at
    write_progress(
        progress_file,
        {
            "stage": "features_ready",
            "processed_rows": total_rows,
            "total_rows": total_rows,
            "valid_samples": len(labels),
            "missing_paths": missing_paths,
            "invalid_labels": invalid_labels,
            "elapsed_seconds": round(total_elapsed, 2),
            "eta_seconds": 0.0,
        },
    )
    print(
        f"Manifest rows={total_rows} | valid={len(labels)} | missing_paths={missing_paths} | "
        f"invalid_labels={invalid_labels} | extract_time={total_elapsed/60:.1f}m"
    )

    return FeatureBatch(
        video=np.stack(videos),
        audio=np.stack(audios),
        metadata=np.stack(metas),
        mask=np.stack(masks),
        labels=np.asarray(labels, dtype=np.float32),
    )


def compute_norm(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def make_loader(
    video: np.ndarray,
    audio: np.ndarray,
    meta: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(video, dtype=torch.float32),
        torch.tensor(audio, dtype=torch.float32),
        torch.tensor(meta, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: MultimodalClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    y_true = []
    y_prob = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for video, audio, meta, mask, labels in loader:
            video = video.to(device)
            audio = audio.to(device)
            meta = meta.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            logits, _ = model(video, audio, meta, mask)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_hat = [1.0 if p >= 0.5 else 0.0 for p in y_prob]
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_hat)))

    metrics = {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_acc": acc,
    }

    classes = set(y_true)
    if len(classes) > 1:
        metrics["val_auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def resolve_train_val_indices(
    labels: np.ndarray,
    seed: int,
    resume_checkpoint: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if resume_checkpoint is not None:
        raw_train = resume_checkpoint.get("train_indices")
        raw_val = resume_checkpoint.get("val_indices")
        if raw_train is not None and raw_val is not None:
            train_idx = np.asarray(raw_train, dtype=np.int64)
            val_idx = np.asarray(raw_val, dtype=np.int64)
            if (
                train_idx.size > 0
                and val_idx.size > 0
                and train_idx.max(initial=-1) < len(labels)
                and val_idx.max(initial=-1) < len(labels)
            ):
                print("Using train/val split from resume checkpoint")
                return train_idx, val_idx
            print("Checkpoint split is invalid for current dataset; recomputing split")

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def build_checkpoint(
    *,
    epoch: int,
    model: MultimodalClassifier,
    optimizer: torch.optim.Optimizer,
    extractor: MultimodalFeatureExtractor,
    hidden_dim: int,
    dropout: float,
    video_mean: np.ndarray,
    video_std: np.ndarray,
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    meta_mean: np.ndarray,
    meta_std: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "video_dim": extractor.dims.video,
        "audio_dim": extractor.dims.audio,
        "meta_dim": extractor.dims.metadata,
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "normalization": {
            "video_mean": video_mean.tolist(),
            "video_std": video_std.tolist(),
            "audio_mean": audio_mean.tolist(),
            "audio_std": audio_std.tolist(),
            "meta_mean": meta_mean.tolist(),
            "meta_std": meta_std.tolist(),
        },
    }


def save_checkpoint(checkpoint_path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(checkpoint_path)


def load_resume_checkpoint(resume_path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(resume_path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint format: {resume_path}")
    return payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    extractor = MultimodalFeatureExtractor(frame_stride=args.frame_stride, max_frames=args.max_frames)
    print(
        f"Extractor config: frame_stride={args.frame_stride} max_frames={args.max_frames} "
        f"sample_rate={extractor.sample_rate}"
    )
    write_progress(
        args.progress_file,
        {
            "stage": "initializing",
            "dataset_csv": str(args.dataset_csv),
            "output": str(args.output),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "frame_stride": int(args.frame_stride),
            "max_frames": int(args.max_frames),
        },
    )

    cache_path: Path | None = None
    if not args.disable_feature_cache:
        cache_path = args.feature_cache or default_feature_cache_path(args.dataset_csv)

    batch: FeatureBatch
    if cache_path is not None:
        cached = load_feature_cache(cache_path)
        if cached is not None:
            batch = cached
        else:
            batch = build_features(
                args.dataset_csv,
                extractor,
                progress_every=args.progress_every,
                progress_file=args.progress_file,
            )
            save_feature_cache(cache_path, batch)
    else:
        batch = build_features(
            args.dataset_csv,
            extractor,
            progress_every=args.progress_every,
            progress_file=args.progress_file,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_path: Path | None = None
    if args.resume_from is not None:
        resume_path = args.resume_from
    elif args.resume and args.output.exists():
        resume_path = args.output

    resume_checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume_checkpoint = load_resume_checkpoint(resume_path, device)
        print(f"Resuming from checkpoint: {resume_path}")

    hidden_dim = (
        int(resume_checkpoint.get("hidden_dim", args.hidden_dim))
        if resume_checkpoint is not None
        else args.hidden_dim
    )
    dropout = (
        float(resume_checkpoint.get("dropout", args.dropout))
        if resume_checkpoint is not None
        else args.dropout
    )

    model = MultimodalClassifier(
        video_dim=extractor.dims.video,
        audio_dim=extractor.dims.audio,
        meta_dim=extractor.dims.metadata,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        print(f"Resume point: epoch={start_epoch - 1}")

    train_idx, val_idx = resolve_train_val_indices(batch.labels, args.seed, resume_checkpoint)

    norm_dict = resume_checkpoint.get("normalization") if resume_checkpoint is not None else None
    if norm_dict is not None:
        video_mean = np.asarray(norm_dict["video_mean"], dtype=np.float32)
        video_std = np.asarray(norm_dict["video_std"], dtype=np.float32)
        audio_mean = np.asarray(norm_dict["audio_mean"], dtype=np.float32)
        audio_std = np.asarray(norm_dict["audio_std"], dtype=np.float32)
        meta_mean = np.asarray(norm_dict["meta_mean"], dtype=np.float32)
        meta_std = np.asarray(norm_dict["meta_std"], dtype=np.float32)
        print("Using normalization stats from resume checkpoint")
    else:
        video_mean, video_std = compute_norm(batch.video[train_idx])
        audio_mean, audio_std = compute_norm(batch.audio[train_idx])
        meta_mean, meta_std = compute_norm(batch.metadata[train_idx])

    video_norm = apply_norm(batch.video, video_mean, video_std)
    audio_norm = apply_norm(batch.audio, audio_mean, audio_std)
    meta_norm = apply_norm(batch.metadata, meta_mean, meta_std)

    train_loader = make_loader(
        video_norm[train_idx],
        audio_norm[train_idx],
        meta_norm[train_idx],
        batch.mask[train_idx],
        batch.labels[train_idx],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        video_norm[val_idx],
        audio_norm[val_idx],
        meta_norm[val_idx],
        batch.mask[val_idx],
        batch.labels[val_idx],
        batch_size=args.batch_size,
        shuffle=False,
    )

    train_labels = batch.labels[train_idx]
    pos_count = float(np.sum(train_labels == 1.0))
    neg_count = float(np.sum(train_labels == 0.0))
    print(f"Train label counts: real={int(neg_count)} fake={int(pos_count)}")

    if not args.disable_class_weighting and pos_count > 0 and neg_count > 0:
        pos_weight_value = neg_count / pos_count
        if pos_weight_value > 1.0:
            pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using BCE pos_weight={pos_weight_value:.6f}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print(
                "Skipping BCE pos_weight because fake class is not minority "
                f"(computed pos_weight={pos_weight_value:.6f})"
            )
    else:
        criterion = nn.BCEWithLogitsLoss()
        if args.disable_class_weighting:
            print("Class weighting disabled")

    if start_epoch > args.epochs:
        write_progress(
            args.progress_file,
            {
                "stage": "complete",
                "message": "checkpoint already at or above requested epoch",
                "current_epoch": int(start_epoch - 1),
                "target_epochs": int(args.epochs),
            },
        )
        print(
            f"Checkpoint epoch ({start_epoch - 1}) is already >= target epochs ({args.epochs}). Nothing to train."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        write_progress(
            args.progress_file,
            {
                "stage": "training_epoch",
                "epoch": int(epoch),
                "target_epochs": int(args.epochs),
                "status": "started",
            },
        )
        model.train()
        epoch_losses = []

        for video, audio, meta, mask, labels in train_loader:
            video = video.to(device)
            audio = audio.to(device)
            meta = meta.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(video, audio, meta, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        metrics = evaluate(model, val_loader, device)

        message = (
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_acc={metrics['val_acc']:.4f}"
        )
        if "val_auc" in metrics:
            message += f" | val_auc={metrics['val_auc']:.4f}"
        print(message)
        write_progress(
            args.progress_file,
            {
                "stage": "training_epoch",
                "epoch": int(epoch),
                "target_epochs": int(args.epochs),
                "status": "completed",
                "train_loss": round(train_loss, 6),
                "val_loss": round(float(metrics["val_loss"]), 6),
                "val_acc": round(float(metrics["val_acc"]), 6),
                "val_auc": round(float(metrics.get("val_auc", 0.0)), 6),
            },
        )

        if not args.disable_epoch_checkpoint:
            checkpoint = build_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                extractor=extractor,
                hidden_dim=hidden_dim,
                dropout=dropout,
                video_mean=video_mean,
                video_std=video_std,
                audio_mean=audio_mean,
                audio_std=audio_std,
                meta_mean=meta_mean,
                meta_std=meta_std,
                train_idx=train_idx,
                val_idx=val_idx,
            )
            save_checkpoint(args.output, checkpoint)
            print(f"Saved checkpoint: {args.output} (epoch {epoch}/{args.epochs})")

    final_checkpoint = build_checkpoint(
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        extractor=extractor,
        hidden_dim=hidden_dim,
        dropout=dropout,
        video_mean=video_mean,
        video_std=video_std,
        audio_mean=audio_mean,
        audio_std=audio_std,
        meta_mean=meta_mean,
        meta_std=meta_std,
        train_idx=train_idx,
        val_idx=val_idx,
    )
    save_checkpoint(args.output, final_checkpoint)
    write_progress(
        args.progress_file,
        {
            "stage": "complete",
            "message": "training finished",
            "epoch": int(args.epochs),
            "checkpoint": str(args.output),
        },
    )
    print(f"Training complete. Final checkpoint: {args.output}")


if __name__ == "__main__":
    main()
