from __future__ import annotations

import asyncio
from typing import Dict, Set

from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.setdefault(job_id, set()).add(websocket)

    async def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            if job_id in self._connections:
                self._connections[job_id].discard(websocket)
                if not self._connections[job_id]:
                    self._connections.pop(job_id, None)

    async def broadcast(self, job_id: str, message: dict) -> None:
        async with self._lock:
            sockets = list(self._connections.get(job_id, set()))
        if not sockets:
            return
        await asyncio.gather(*(ws.send_json(message) for ws in sockets), return_exceptions=True)

