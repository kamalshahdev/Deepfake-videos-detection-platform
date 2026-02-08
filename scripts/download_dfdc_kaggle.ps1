param(
    [string]$Competition = "deepfake-detection-challenge",
    [string]$RawDir = "data/raw/dfdc",
    [string]$ExtractDir = "data/raw/dfdc_extracted"
)

$ErrorActionPreference = "Stop"

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Get-KaggleCommand {
    $resolved = Get-Command "kaggle" -ErrorAction SilentlyContinue
    if ($resolved) {
        return $resolved.Source
    }

    $local = Join-Path (Get-Location) ".venv/Scripts/kaggle.exe"
    if (Test-Path $local) {
        return $local
    }

    throw "Kaggle CLI not found. Install with '.\\.venv\\Scripts\\python -m pip install kaggle'."
}

$kaggleCmd = Get-KaggleCommand

$hasLegacyEnvCreds = [bool]$env:KAGGLE_USERNAME -and [bool]$env:KAGGLE_KEY
$hasAccessTokenEnv = [bool]$env:KAGGLE_API_TOKEN
$legacyTokenPath = Join-Path $HOME ".kaggle/kaggle.json"
$accessTokenPath = Join-Path $HOME ".kaggle/access_token"
$accessTokenTxtPath = Join-Path $HOME ".kaggle/access_token.txt"

$hasLegacyTokenFile = Test-Path $legacyTokenPath
$hasAccessTokenFile = (Test-Path $accessTokenPath) -or (Test-Path $accessTokenTxtPath)

if (-not $hasLegacyEnvCreds -and -not $hasAccessTokenEnv -and -not $hasLegacyTokenFile -and -not $hasAccessTokenFile) {
    throw "Kaggle auth missing. Configure one of: KAGGLE_API_TOKEN, KAGGLE_USERNAME+KAGGLE_KEY, ~/.kaggle/access_token, or ~/.kaggle/kaggle.json."
}

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null

Write-Host "Downloading competition files to $RawDir ..."
& $kaggleCmd competitions download -c $Competition -p $RawDir

$zips = Get-ChildItem -Path $RawDir -Filter *.zip -File
if (-not $zips) {
    throw "No zip files found in $RawDir after download. Verify Kaggle access and accepted rules."
}

Write-Host "Extracting $($zips.Count) archives to $ExtractDir ..."
foreach ($zip in $zips) {
    Write-Host "Extracting $($zip.Name)"
    Expand-Archive -Path $zip.FullName -DestinationPath $ExtractDir -Force
}

Write-Host "Done. Next commands:"
Write-Host "  .\\.venv\\Scripts\\python scripts\\import_public_dataset.py --dataset-type dfdc --root $ExtractDir --output data\\train_manifest_dfdc.csv --absolute-paths --shuffle"
Write-Host "  .\\.venv\\Scripts\\python scripts\\train.py --dataset-csv data\\train_manifest_dfdc.csv --output models\\deepfake_multimodal.pt --epochs 20 --batch-size 32"
