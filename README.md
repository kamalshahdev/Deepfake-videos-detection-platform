# ByteGuard: Deepfake Video Detection Platform (Multimodal)

ByteGuard is a starter platform for detecting manipulated videos using multimodal learning.
The baseline model fuses:
- Visual forensic features from sampled video frames
- Audio consistency features from extracted waveform statistics
- Video metadata features (fps, duration, size, resolution)

This repository includes:
- `backend/`: FastAPI inference service
- `frontend/`: Streamlit web app
- `scripts/`: training utilities
- `models/`: model checkpoints (generated after training)
- `docs/`: architecture notes

## 1. Quick Start (Local)

Run all commands from the repository root (`Deepfake-video-detection-platform`).

### 1.1 Create environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 1.2 Run backend API

```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 1.3 Run frontend

```bash
python -m streamlit run frontend/app.py
```

The frontend calls `http://localhost:8000` by default.
Set `API_URL` to override.

### 1.4 Ready-to-Use Mode (Fastest)

If you are short on time, run with the existing checkpoint and start analyzing uploads immediately:

```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
python -m streamlit run frontend/app.py
```

Open `http://localhost:8501`, upload a video, click **Run detection**, then download:
- report JSON
- report TXT

You can also paste a video URL (direct `.mp4` link or social post URL) in the frontend and run detection from URL.

## 2. Train a Model

Prepare a CSV file with columns:
- `video_path`: absolute or relative path to video
- `label`: `0` for real, `1` for fake

Example (`data/train_manifest.csv`):

```csv
video_path,label
/path/to/real_01.mp4,0
/path/to/fake_01.mp4,1
```

Train:

```bash
python scripts/train.py --dataset-csv data/train_manifest.csv --output models/deepfake_multimodal.pt
```

Resume-safe training (recommended for large datasets):

```bash
python scripts/train.py --dataset-csv data/train_manifest_dfd_full.csv --output A:/deepfake-data/models/deepfake_multimodal_dfd.pt --epochs 12 --batch-size 32 --resume
```

- The trainer now saves checkpoint state every epoch to `--output`
- If interrupted, rerun the same command with `--resume`
- Feature extraction cache is saved to `data/train_manifest_dfd_full.csv.features.npz` by default
- You can speed up extraction on CPU with `--frame-stride` and `--max-frames`

Fast balanced-subset mode (CPU-friendly):

```bash
python scripts/create_balanced_manifest.py --input data/train_manifest_dfd_full.csv --output data/train_manifest_dfd_balanced.csv --shuffle
python scripts/train.py --dataset-csv data/train_manifest_dfd_balanced.csv --output models/deepfake_multimodal_balanced.pt --epochs 10 --batch-size 32 --frame-stride 24 --max-frames 24 --progress-every 25 --resume
```

Post-training evaluation (go/no-go):

```bash
python scripts/evaluate_checkpoint.py --checkpoint A:/deepfake-data/models/deepfake_multimodal_balanced.pt --feature-cache data/train_manifest_dfd_balanced.csv.features.npz --split val-from-checkpoint --output-json A:/deepfake-data/models/eval_balanced_val.json
```

8-hour optimization sweep on cached features:

```bash
python scripts/optimize_variants.py --dataset-csv data/train_manifest_dfd_2000.csv --feature-cache data/train_manifest_dfd_2000.csv.features.npz --output-dir A:/deepfake-data/models/variants_2000 --summary-json A:/deepfake-data/models/variants_2000_summary.json
```

Set decision threshold for API predictions (PowerShell):

```bash
$env:DECISION_THRESHOLD="0.45"
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2.1 Import Public Dataset Layouts

If you already downloaded a public dataset locally, build a manifest with:

```bash
python scripts/import_public_dataset.py --dataset-type dfdc --root D:/datasets/dfdc --output data/train_manifest.csv --absolute-paths
```

Supported `--dataset-type` values:
- `dfdc`: parses `metadata.json` labels from DFDC-style directories
- `celebdfv2`: parses `Celeb-real`, `YouTube-real`, and `Celeb-synthesis`
- `faceforensicspp`: parses `original_sequences` (real) and `manipulated_sequences` (fake)
- `dfd`: parses `DFD_original sequences` (real) and `DFD_manipulated_sequences` (fake)
- `generic`: expects `real/` and `fake/` subfolders

Example (Celeb-DF v2):

```bash
python scripts/import_public_dataset.py --dataset-type celebdfv2 --root D:/datasets/Celeb-DF-v2 --output data/train_manifest.csv --absolute-paths
python scripts/train.py --dataset-csv data/train_manifest.csv --output models/deepfake_multimodal.pt
```

### 2.2 Real Public Dataset Workflow (Recommended)

For stronger results, train on a real benchmark dataset instead of synthetic samples.

Recommended order:
1. Start with DFDC (largest public challenge dataset)
2. Add FaceForensics++ and Celeb-DF v2 for cross-dataset robustness
3. Re-train or fine-tune with merged manifests

#### A) DFDC (Kaggle competition data)

1) Install Kaggle CLI:

```bash
pip install kaggle
```

2) Create Kaggle API token at `https://www.kaggle.com/settings` and configure one method:

- Access token env (current shell): `set KAGGLE_API_TOKEN=<your_token>` (or PowerShell `$env:KAGGLE_API_TOKEN="<your_token>"`)
- Access token file: `~/.kaggle/access_token` containing only the token string
- Legacy file (still supported): `C:\Users\<your-user>\.kaggle\kaggle.json`

3) Accept competition rules on Kaggle (required once), then download:

```bash
kaggle competitions download -c deepfake-detection-challenge -p data/raw/dfdc
```

4) Extract zip parts into a folder such as `data/raw/dfdc_extracted`.

5) Build manifest from DFDC metadata:

```bash
python scripts/import_public_dataset.py --dataset-type dfdc --root data/raw/dfdc_extracted --output data/train_manifest_dfdc.csv --absolute-paths --shuffle
```

6) Train:

```bash
python scripts/train.py --dataset-csv data/train_manifest_dfdc.csv --output models/deepfake_multimodal.pt --epochs 20 --batch-size 32
```

Run inference API with a specific trained checkpoint (PowerShell):

```bash
$env:MODEL_PATH="A:/deepfake-data/models/deepfake_multimodal_dfd.pt"
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Automated alternative (download + extract):

```bash
powershell -ExecutionPolicy Bypass -File scripts/download_dfdc_kaggle.ps1
```

#### B) FaceForensics++ / Celeb-DF v2

Both datasets require access request forms from their official repositories.

After download and extraction, generate manifests:

```bash
python scripts/import_public_dataset.py --dataset-type faceforensicspp --root D:/datasets/FaceForensics++ --output data/train_manifest_ffpp.csv --absolute-paths --shuffle
python scripts/import_public_dataset.py --dataset-type celebdfv2 --root D:/datasets/Celeb-DF-v2 --output data/train_manifest_celebdf.csv --absolute-paths --shuffle
```

You can merge manifests manually (CSV concat) and retrain for broader generalization.

Manifest merge helper:

```bash
python scripts/merge_manifests.py --inputs data/train_manifest_dfdc.csv data/train_manifest_ffpp.csv data/train_manifest_celebdf.csv --output data/train_manifest_all.csv --shuffle
python scripts/train.py --dataset-csv data/train_manifest_all.csv --output models/deepfake_multimodal.pt --epochs 20 --batch-size 32
```

## 3. API Endpoints

- `GET /health`: service health
- `GET /model-info`: model loading and config
- `POST /predict`: upload a video file and get deepfake score + full analysis report
- `POST /predict-url`: send JSON `{ "url": "https://..." }` and analyze a video URL

## 4. Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`

## 5. Notes

- The baseline feature extractor is intentionally lightweight and explainable.
- Accuracy depends on training data quality and diversity.
- For production use, add robust face tracking, temporal transformers, and adversarial hard-negative mining.
- In this DFD manifest, real videos are fewer than fake videos. A balanced subset may reduce majority-class bias.
