@echo off
setlocal
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo CUDA detected. Installing CUDA-enabled PyTorch...
  pip install -r requirements-cuda.txt
) else (
  echo CUDA not detected. Installing CPU-only deps...
)
pip install -r requirements.txt
uvicorn app.main:app --reload
