#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA detected. Installing CUDA-enabled PyTorch..."
  python -m pip install -r requirements-cuda.txt
else
  echo "CUDA not detected. Installing CPU-only deps..."
fi
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
