from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import HDR_DIR, PREVIEWS_DIR, UPLOADS_DIR
from .jobs import JOBS, start_job
from .models import JobStatus, UploadResponse

router = APIRouter()


@router.post("/api/upload", response_model=UploadResponse)
async def upload_heic(file: UploadFile = File(...), scene_type: Optional[str] = Form(None)) -> UploadResponse:
    if not file.filename.lower().endswith(".heic"):
        raise HTTPException(status_code=400, detail="Only HEIC files are supported.")
    job_id = str(uuid.uuid4())
    upload_path = UPLOADS_DIR / f"{job_id}.heic"
    contents = await file.read()
    upload_path.write_bytes(contents)
    start_job(str(upload_path), scene_type if scene_type != "auto" else None, job_id=job_id)
    return UploadResponse(job_id=job_id)


@router.get("/api/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str) -> JobStatus:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")
    state = JOBS[job_id]
    return JobStatus(
        status=state.status,
        progress=state.progress,
        current_stage=state.current_stage,
        preview_url=state.preview_url,
        error_message=state.error_message,
    )


@router.get("/api/result/{job_id}/hdr")
async def get_hdr(job_id: str) -> FileResponse:
    hdr_path = HDR_DIR / f"{job_id}.exr"
    if not hdr_path.exists():
        raise HTTPException(status_code=404, detail="HDR file not ready.")
    return FileResponse(hdr_path, filename=hdr_path.name, media_type="application/octet-stream")


@router.get("/api/result/{job_id}/preview")
async def get_preview(job_id: str) -> FileResponse:
    preview_path = PREVIEWS_DIR / f"{job_id}_final.jpg"
    if not preview_path.exists():
        preview_path = PREVIEWS_DIR / f"{job_id}_preview.jpg"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not ready.")
    return FileResponse(preview_path, filename=preview_path.name, media_type="image/jpeg")

