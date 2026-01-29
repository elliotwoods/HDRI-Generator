# Specification ? Single?Photo to HDRI Generator with Web UI and 360 Viewer

## Purpose
Create a Python web app that ingests a single HEIC photo, infers camera/scene parameters, projects the image into a partial 360? environment, uses SDXL (diffusers) to outpaint/inpaint missing regions across multiple synthetic exposures, merges those exposures into a true HDR equirectangular map (8k ? 4k), and serves a browser UI with live progress and a 360? preview viewer.

## Inputs and Outputs
### Inputs
- A single HEIC file uploaded by the user.
- Optional scene type override: `auto`, `outdoor`, `indoor`, `night`.

### Outputs
- HDR equirectangular panorama (`.exr` or `.hdr`).
- LDR preview image (`.jpg` or `.png`).
- Live preview updates during generation.

## High?Level Workflow
1. **Load HEIC + EXIF**
   - Decode HEIC to sRGB float32 array (H?W?3, [0,1]).
   - Extract EXIF for FOV inference and scene classification.

2. **Infer Camera/Scene Parameters**
   - Estimate FOV from EXIF (prefer `FocalLengthIn35mmFilm`).
   - Infer scene type (`outdoor`, `indoor`, `night`) from EXIF + luminance stats.
   - Build internal prompt based on scene type.

3. **Preprocess**
   - Convert sRGB to linear space.
   - Compute highlight mask (clipped luminance) and dilate it.

4. **Project to Cubemap**
   - Project the input image into 6 cubemap faces at `CUBE_RES`.
   - Track coverage masks per face.

5. **Generate Multiple Exposures**
   - Use EV offsets (default `[-3, -1, 0, 1, 3]`) **reordered to start with EV 0**.
   - For each EV:
     - Apply exposure scaling in linear space (`img_lin * 2^EV`).
     - Inpaint/outpaint missing regions on each cubemap face using SDXL.
     - Convert the cubemap back to an equirectangular panorama.
     - Emit live preview updates during face generation.

6. **Merge HDR**
   - Merge EV panoramas into an HDR radiance map using OpenCV Debevec.
   - Save HDR as `.exr` (preferred) or `.hdr`.

7. **Final Preview**
   - Tone map HDR for LDR preview.
   - Save final preview and notify UI.

## SDXL Implementation Requirements (Mandatory)
### Model
- Use HuggingFace **diffusers**.
- Primary model: `stabilityai/stable-diffusion-xl-base-1.0`.
- No SDXL Refiner in v1.
- Model loaded **once at startup** (not per job).

### Caching
- Respect HuggingFace cache env vars (`HF_HOME`, `TRANSFORMERS_CACHE`).
- Do not bundle model weights in repo.

### Device + Precision
- Default device: **CUDA if available**, else CPU.
- CUDA ? `torch.float16`, CPU ? `torch.float32`.
- Move pipeline to selected device explicitly.

### Pipeline
- Use `StableDiffusionXLImg2ImgPipeline`.
- Scheduler: **DPM++ 2M Karras**.
- Enable attention slicing.
- Disable safety checker.
- Enable VAE slicing/tiling if available.

### Generation
- Use img2img/inpainting style only.
- Input image: PIL (uint8 or float32 converted to PIL).
- Mask: single?channel PIL (white = allowed, black = preserve).
- Prompt: short and generic, e.g. ?a realistic outdoor environment with natural sky and ground?.
- Negative prompt: minimal, e.g. ?cartoon, illustration, painting, unrealistic?.

### Default Parameters
- `num_inference_steps`: 20?30.
- `guidance_scale`: 3.0?5.0.
- `strength`: ~0.3 when image content exists; up to ~0.8 for empty faces.
- Fixed random seed per job (derived from `job_id`).

### Error Handling
- Model load failure ? **raise at startup** (app should not silently continue).
- Generation failure per face ? job error + WebSocket + `/api/status`.

### VRAM Constraints
- Use `AI_FACE_RES` to downscale faces for SDXL (e.g. 1024 on 12GB GPU).
- Re?upscale outputs to full face resolution for cubemap.
- Clear CUDA cache between faces.

## Backend Architecture
### Tech
- Python 3.11+
- FastAPI + Uvicorn
- pillow?heif + Pillow
- exif/metadata parsing (piexif or exifread)
- numpy, OpenCV
- imageio for HDR saving
- diffusers + torch

### Modules
- `heic_loader.py`: load HEIC + EXIF
- `fov_estimation.py`: estimate horizontal/vertical FOV
- `exif_utils.py`: scene inference + internal prompt
- `linearization.py`: sRGB ? linear
- `masks.py`: highlight mask
- `projection.py`: cubemap projection (OpenCV remap)
- `ai_engine.py`: SDXL pipeline + per?face generation
- `exposures.py`: EV offsets + exposure scaling
- `hdr_merge.py`: Debevec merge
- `preview.py`: tone mapping
- `jobs.py`: orchestration + progress + previews

### Job Control
- Each upload creates a job with a UUID.
- Job progress is broadcast via WebSocket and available via REST.
- Job preview updates emitted during face generation and after EV 0 completes.

## API
### `POST /api/upload`
- Multipart fields: `file` (HEIC), `scene_type` (`auto`/`outdoor`/`indoor`/`night`).
- Returns `{ "job_id": "<uuid>" }`.

### `GET /api/status/{job_id}`
Returns:
```json
{
  "status": "queued"|"processing"|"done"|"error",
  "progress": 0.0-1.0,
  "current_stage": "string",
  "preview_url": "optional",
  "error_message": "optional"
}
```

### `GET /api/result/{job_id}/hdr`
- Downloads HDR `.exr` or `.hdr`.

### `GET /api/result/{job_id}/preview`
- Returns current/final LDR preview.

### `WebSocket /ws/jobs/{job_id}`
Events:
- `status`: `{ "stage": string, "progress": float }`
- `preview`: `{ "preview_url": string }`
- `done`: `{ "hdr_url": string, "preview_url": string }`
- `error`: `{ "message": string }`

## Frontend UI
### Layout
- Left panel: upload + scene type, progress, status log, download buttons.
- Right panel: 360? viewer.

### Behavior
- Drag?drop or file input auto?uploads.
- WebSocket (plus polling fallback) updates progress.
- Preview updates as faces finish; EV 0 preview shown ASAP.

### Viewer
- Uses Photo Sphere Viewer when available.
- Falls back to a simple `<img>` element if the viewer library fails to load.

## Progress Reporting
- Stages include ?Loading image?, ?Reading EXIF + scene inference?, ?Projecting into cubemap?, ?Preparing AI generation?, per?EV and per?face step updates, ?Merging HDR stack?.
- Per?step updates include percent completion within a face (e.g. ?step 10/25 (40%)?).

## Configuration
All key parameters are centralized in `config.py`, including:
- `CUBE_RES`, `PANORAMA_WIDTH`, `PANORAMA_HEIGHT`
- `AI_FACE_RES` (SDXL input size cap)
- `EV_OFFSETS`
- `AI_CFG_SCALE`, `AI_STEPS`, `AI_STRENGTH_ORIGINAL`, `AI_DENOISE_NEW`

## Non?Functional Requirements
- Stable, ?boring? generations preferred (low CFG, moderate steps).
- Reproducible results (fixed seed per job).
- Robust error messages exposed to UI.
- Works without user?supplied text prompts.

## Deployment Notes
- GPU recommended for SDXL (12GB+ suggested).
- CUDA builds are installed via `requirements-cuda.txt`.
- Start scripts detect CUDA and install correct wheels automatically.

