# Single-Photo to HDRI Generator

This project turns a single HEIC photo into a stitched HDRI (EXR/HDR) with a web UI, live progress, and a 360 preview viewer.

## Features
- Upload a HEIC photo, infer EXIF, estimate FOV, and project into a cubemap
- AI-based outpaint/inpaint to fill missing regions across multiple EV exposures
- Merge LDR panoramas into an HDR equirectangular map
- Web UI with progress updates and a live 360 viewer

## Requirements
- Python 3.11+
- CUDA GPU strongly recommended (SDXL on CPU is very slow)
- Dependencies listed in `requirements.txt`

## Install

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

Note: FastAPI file uploads require `python-multipart`:

```powershell
pip install python-multipart
```

GPU note: `requirements.txt` does not include PyTorch. For CUDA installs, use:

```powershell
pip install -r requirements-cuda.txt
```

The start scripts auto-detect CUDA (via `nvidia-smi`) and install `requirements-cuda.txt` when available.

Optional: For faster Hugging Face downloads with Xet Storage, install:

```powershell
pip install huggingface_hub[hf_xet]
# or
pip install hf_xet
```

Note: In some network environments Xet can be slower than plain HTTP; if downloads seem worse, skip or uninstall `hf_xet`.

3) (Optional) Set Hugging Face cache location

```powershell
$env:HF_HOME = "D:\hf-cache"
# or
$env:TRANSFORMERS_CACHE = "D:\hf-cache"
```

## Run

```powershell
uvicorn app.main:app --reload
```

### Start Scripts

```powershell
.\start.ps1
```

```cmd
start.bat
```

```bash
./start.sh
```

Open `http://127.0.0.1:8000` in your browser.

## Usage
1) Upload a `.heic` photo
2) Choose scene type (Auto/Outdoor/Indoor/Night)
3) Click **Generate HDRI**
4) Watch progress and preview updates
5) Download the HDR and final preview when done

## Project Structure

```
project_root/
  app/
    main.py
    routes.py
    websocket.py
    models.py
    jobs.py
    pipeline/
      heic_loader.py
      exif_utils.py
      fov_estimation.py
      linearization.py
      masks.py
      projection.py
      ai_engine.py
      exposures.py
      hdr_merge.py
      preview.py
  static/
    index.html
    js/
      main.js
  assets/
    uploads/
    previews/
    hdr/
  config.py
  requirements.txt
```

## API

### `POST /api/upload`
- Multipart fields:
  - `file`: HEIC file
  - `scene_type`: `auto`, `outdoor`, `indoor`, `night`
- Returns: `{ "job_id": "<uuid>" }`

### `GET /api/status/{job_id}`
- Returns job status, progress, stage, preview URL, and error message

### `GET /api/result/{job_id}/hdr`
- Downloads the HDR `.exr`

### `GET /api/result/{job_id}/preview`
- Downloads the current/final LDR preview

### `WebSocket /ws/jobs/{job_id}`
- `status` events
- `preview` events
- `done` events
- `error` events

## SDXL Implementation Details (Mandatory)

### AI Model
- Uses HuggingFace diffusers
- Primary model: `stabilityai/stable-diffusion-xl-base-1.0`
- No refiner model (v1)
- Model is loaded exactly once at application startup

### Model Download & Caching
- Models are downloaded automatically by diffusers
- Uses HuggingFace cache (`HF_HOME` / `TRANSFORMERS_CACHE` if set)
- No model weights are committed to the repo

### Precision & Device
- Default device: CUDA if available, otherwise CPU
- `torch.float16` on CUDA
- `torch.float32` on CPU
- Pipeline explicitly moved to the selected device

### Pipeline Instantiation
- `StableDiffusionXLImg2ImgPipeline`
- Attention slicing enabled to reduce VRAM usage
- Safety checker disabled
- Scheduler replaced with **DPM++ 2M Karras**

### Initialization Flow
At application startup:
1) Detect device
2) Load SDXL base via `from_pretrained()`
3) Replace scheduler
4) Disable safety checker
5) Enable attention slicing
6) Store the pipeline globally (singleton)

### Inpainting / Img2Img Usage
- Only img2img/inpainting style usage (no text-to-image)
- Input image is passed as a PIL image (uint8)
- Mask is a single-channel PIL image:
  - White = AI allowed
  - Black = preserve
- Internal prompt is short and generic
- Negative prompt is minimal: `cartoon, illustration, painting, unrealistic`
- Output is composited with the original using the mask to preserve known regions

### Default Generation Parameters
- `num_inference_steps`: 20?30
- `guidance_scale` (CFG): 3.0?5.0
- `strength`:
  - Faces with original content: ~0.3
  - Faces mostly empty: up to ~0.8
- Fixed random seed per job (derived from `job_id`)

### Error Handling
- If model loading fails: startup raises an exception and the app does not run
- If generation fails for any face: job errors, WebSocket + `/api/status` report it

### No Other Models
- No Midjourney/DALL?E/web APIs
- No ControlNet/Depth/Refiners for v1

## Notes
- EXR output requires a backend supported by `imageio`. If OIIO is missing, output may fall back to another writer or fail; install OpenImageIO if needed.
- The current cubemap projection is a straightforward mapping and can show seams. Higher cube resolution helps.
- CPU-only SDXL is extremely slow; use CUDA for practical runtime.

## Troubleshooting
- If the server fails at startup with model errors, ensure `diffusers`, `torch`, and `transformers` are installed and compatible.
- If EXR writing fails, install OpenImageIO and restart the server.

