# Architecture Overview

## Core Components

1. Inference API (`backend/app/main.py`)
- Receives videos over HTTP
- Runs multimodal feature extraction and classification
- Returns score, label, and modality contribution

2. Feature Extraction (`backend/app/services/feature_extractor.py`)
- Video modality: frame-level forensic descriptors + temporal motion stats
- Audio modality: RMS, centroid, bandwidth, zero-crossing features
- Metadata modality: fps, duration, file size, resolution, frame count

3. Model (`backend/app/services/model.py`)
- Separate encoder per modality
- Mask-aware fusion using learned modality gates
- Binary classifier head (fake vs real)

4. Training (`scripts/train.py`)
- Reads `(video_path, label)` manifest CSV
- Extracts multimodal features
- Trains PyTorch model
- Saves checkpoint with normalization stats

5. Frontend (`frontend/app.py`)
- Upload and preview video
- Call API `/predict`
- Display score, label, and modality weights

## Data Contract

Input API:
- Multipart file upload (`video`)

Output API:
- `score`: float in [0, 1]
- `label`: `fake` or `real`
- `confidence`: confidence of predicted label
- `modality_weights`: model fusion weights
- `extracted_modalities`: booleans for available features

## Extension Path

- Replace handcrafted visual features with CLIP/ViT + temporal transformer
- Replace statistical audio features with wav2vec2 embeddings
- Add transcript modality with ASR + language model inconsistencies
- Add model monitoring, drift detection, and calibration
