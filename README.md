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

### 1.1 Create environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 1.2 Run backend API

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 1.3 Run frontend

```bash
streamlit run frontend/app.py
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
