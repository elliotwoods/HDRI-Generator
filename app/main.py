from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import ASSETS_DIR
from .jobs import JOBS, WS_MANAGER
from .pipeline.ai_engine import initialize_pipeline
from .routes import router

app = FastAPI(title="Single-Photo to HDRI Generator")


@app.on_event("startup")
async def _startup() -> None:
    initialize_pipeline()

app.include_router(router)

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.websocket("/ws/jobs/{job_id}")
async def job_ws(websocket: WebSocket, job_id: str) -> None:
    if job_id not in JOBS:
        await websocket.accept()
        await websocket.send_json({"event": "error", "message": "Job not found."})
        await websocket.close()
        return
    await WS_MANAGER.connect(job_id, websocket)
    state = JOBS[job_id]
    await websocket.send_json(
        {
            "event": "status",
            "stage": state.current_stage,
            "progress": state.progress,
        }
    )
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await WS_MANAGER.disconnect(job_id, websocket)

