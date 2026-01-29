from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: str
    progress: float
    current_stage: str
    preview_url: Optional[str] = None
    error_message: Optional[str] = None

