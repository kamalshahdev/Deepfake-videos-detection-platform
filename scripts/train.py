"""Train multimodal deepfake model from a manifest CSV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_features(dataset_csv: Path, extractor: MultimodalFeatureExtractor) -> FeatureBatch:
    df = pd.read_csv(dataset_csv)
    expected = {"video_path", "label"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset CSV: {sorted(missing)}")

    videos: list[np.ndarray] = []
    audios: list[np.ndarray] = []
    metas: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[float] = []

    for row in df.itertuples(index=False):
        path = Path(row.video_path)
        if not path.exists():
            continue

        extracted = extractor.extract(path)
        videos.append(np.asarray(extracted["video"], dtype=np.float32))
        audios.append(np.asarray(extracted["audio"], dtype=np.float32))
        metas.append(np.asarray(extracted["metadata"], dtype=np.float32))
        masks.append(np.asarray(extracted["mask"], dtype=np.float32))
        labels.append(float(row.label))

    if not labels:
        raise ValueError("No valid samples found. Check video paths in dataset CSV.")

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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    extractor = MultimodalFeatureExtractor()
    batch = build_features(args.dataset_csv, extractor)

    indices = np.arange(len(batch.labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=batch.labels if len(np.unique(batch.labels)) > 1 else None,
    )

    video_train = batch.video[train_idx]
    audio_train = batch.audio[train_idx]
    meta_train = batch.metadata[train_idx]

    video_mean, video_std = compute_norm(video_train)
    audio_mean, audio_std = compute_norm(audio_train)
    meta_mean, meta_std = compute_norm(meta_train)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalClassifier(
        video_dim=extractor.dims.video,
        audio_dim=extractor.dims.audio,
        meta_dim=extractor.dims.metadata,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_labels = batch.labels[train_idx]
    pos_count = float(np.sum(train_labels == 1.0))
    neg_count = float(np.sum(train_labels == 0.0))
    print(f"Train label counts: real={int(neg_count)} fake={int(pos_count)}")

    if not args.disable_class_weighting and pos_count > 0:
        pos_weight_value = neg_count / pos_count
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCE pos_weight={pos_weight_value:.6f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        if args.disable_class_weighting:
            print("Class weighting disabled")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
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

    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "video_dim": extractor.dims.video,
        "audio_dim": extractor.dims.audio,
        "meta_dim": extractor.dims.metadata,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "model_state_dict": model.state_dict(),
        "normalization": {
            "video_mean": video_mean.tolist(),
            "video_std": video_std.tolist(),
            "audio_mean": audio_mean.tolist(),
            "audio_std": audio_std.tolist(),
            "meta_mean": meta_mean.tolist(),
            "meta_std": meta_std.tolist(),
        },
    }
    torch.save(checkpoint, args.output)

    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
