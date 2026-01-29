param(
  [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvPath)) {
  python -m venv $VenvPath
}

$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
  throw "Activation script not found at $activate"
}

. $activate
$cuda = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($cuda) {
  Write-Host "CUDA detected. Installing CUDA-enabled PyTorch..."
  pip install -r requirements-cuda.txt
} else {
  Write-Host "CUDA not detected. Installing CPU-only deps..."
}
pip install -r requirements.txt
uvicorn app.main:app --reload
