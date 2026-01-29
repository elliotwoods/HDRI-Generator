from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import time

import imageio.v3 as iio
import numpy as np
import logging

from config import (AI_DENOISE_NEW, AI_STRENGTH_ORIGINAL, CUBE_RES,
                    HDR_DIR, PANORAMA_HEIGHT, PANORAMA_WIDTH, PREVIEWS_DIR,
                    UPLOADS_DIR)
from .websocket import WebSocketManager
from .pipeline.ai_engine import AIEngine
from .pipeline.exif_utils import build_internal_prompt, infer_scene_type
from .pipeline.exposures import get_default_ev_offsets
from .pipeline.fov_estimation import estimate_fov
from .pipeline.heic_loader import load_heic_with_exif
from .pipeline.linearization import srgb_to_linear
from .pipeline.masks import compute_highlight_mask
from .pipeline.projection import cubemap_to_equirectangular, project_to_cubemap
from .pipeline.hdr_merge import merge_hdr_from_panos
from .pipeline.preview import tone_map_for_preview


@dataclass
class JobState:
    status: str = "queued"
    progress: float = 0.0
    current_stage: str = "queued"
    preview_url: Optional[str] = None
    error_message: Optional[str] = None
    hdr_url: Optional[str] = None


JOBS: Dict[str, JobState] = {}
WS_MANAGER = WebSocketManager()


def _update_job(job_id: str, **kwargs) -> None:
    state = JOBS[job_id]
    for key, value in kwargs.items():
        setattr(state, key, value)


def _job_seed(job_id: str) -> int:
    return int(uuid.UUID(job_id)) % (2**32 - 1)


def start_job(heic_path: str, scene_type_override: Optional[str], job_id: Optional[str] = None) -> str:
    job_id = job_id or str(uuid.uuid4())
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    HDR_DIR.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = JobState()
    asyncio.create_task(run_pipeline(job_id, heic_path, scene_type_override))
    return job_id


def _project_mask_to_cubemap(mask: np.ndarray, fov_h_deg: float, fov_v_deg: float, cube_res: int) -> Dict[str, np.ndarray]:
    faces = {name: np.zeros((cube_res, cube_res), dtype=bool) for name in ["px", "nx", "py", "ny", "pz", "nz"]}
    mask_faces, _coverage = project_to_cubemap(mask[..., None].astype(np.float32), fov_h_deg, fov_v_deg, cube_res)
    for name in faces:
        faces[name] = mask_faces[name][..., 0] > 0.5
    return faces


async def run_pipeline(job_id: str, heic_path: str, scene_type_override: Optional[str]) -> None:
    try:
        logging.info("Job %s starting pipeline", job_id)
        loop = asyncio.get_running_loop()

        def _schedule_status(stage: str, progress: float) -> None:
            def _do() -> None:
                _update_job(job_id, current_stage=stage, progress=progress)
                asyncio.create_task(
                    WS_MANAGER.broadcast(job_id, {"event": "status", "stage": stage, "progress": progress})
                )

            loop.call_soon_threadsafe(_do)

        def _schedule_preview_snapshot(snapshot: Dict[str, np.ndarray]) -> None:
            async def _emit_preview() -> None:
                pano_lin = await asyncio.to_thread(
                    cubemap_to_equirectangular, snapshot, PANORAMA_WIDTH, PANORAMA_HEIGHT
                )
                preview = await asyncio.to_thread(tone_map_for_preview, pano_lin)
                preview_path = PREVIEWS_DIR / f"{job_id}_preview.jpg"
                await asyncio.to_thread(iio.imwrite, preview_path, (preview * 255.0).astype(np.uint8))
                preview_url = f"/api/result/{job_id}/preview"
                _update_job(job_id, preview_url=preview_url)
                await WS_MANAGER.broadcast(job_id, {"event": "preview", "preview_url": preview_url})

            loop.call_soon_threadsafe(lambda: asyncio.create_task(_emit_preview()))

        _update_job(job_id, status="processing", current_stage="Loading image")
        await WS_MANAGER.broadcast(job_id, {"event": "status", "stage": "Loading image", "progress": 0.05})

        img_srgb, exif = await asyncio.to_thread(load_heic_with_exif, heic_path)
        _update_job(job_id, current_stage="Reading EXIF + scene inference", progress=0.08)
        await WS_MANAGER.broadcast(job_id, {"event": "status", "stage": "Reading EXIF + scene inference", "progress": 0.08})
        fov_h, fov_v = await asyncio.to_thread(estimate_fov, exif)
        scene_type = scene_type_override or await asyncio.to_thread(infer_scene_type, exif, img_srgb)
        internal_prompt = build_internal_prompt(scene_type)

        img_lin = await asyncio.to_thread(srgb_to_linear, img_srgb)
        highlight_mask = await asyncio.to_thread(compute_highlight_mask, img_lin)
        _update_job(job_id, current_stage="Projecting into cubemap", progress=0.12)
        await WS_MANAGER.broadcast(job_id, {"event": "status", "stage": "Projecting into cubemap", "progress": 0.12})

        faces_lin, coverage = await asyncio.to_thread(project_to_cubemap, img_lin, fov_h, fov_v, CUBE_RES)
        highlight_faces = await asyncio.to_thread(_project_mask_to_cubemap, highlight_mask, fov_h, fov_v, CUBE_RES)
        for name in coverage:
            coverage[name] = coverage[name] & (~highlight_faces[name])
        _update_job(job_id, current_stage="Preparing AI generation", progress=0.16)
        await WS_MANAGER.broadcast(job_id, {"event": "status", "stage": "Preparing AI generation", "progress": 0.16})

        ai = AIEngine()
        ev_offsets = get_default_ev_offsets()
        pano_stack = []
        seed = _job_seed(job_id)

        for idx, ev in enumerate(ev_offsets):
            stage = f"EV {ev:+.0f} face generation"
            progress = 0.18 + 0.65 * (idx / max(len(ev_offsets), 1))
            _update_job(job_id, current_stage=stage, progress=progress)
            await WS_MANAGER.broadcast(job_id, {"event": "status", "stage": stage, "progress": progress})

            faces_ev = {k: v * (2.0 ** ev) for k, v in faces_lin.items()}
            faces_out_current = {k: v.copy() for k, v in faces_ev.items()}
            last_preview = {"t": 0.0}

            face_names = list(faces_ev.keys())
            face_total = max(len(face_names), 1)
            face_index = {"i": 0}
            ev_span = 0.65 / max(len(ev_offsets), 1)

            def _face_progress(face_name: str) -> None:
                if face_name in face_names:
                    idx_face = face_names.index(face_name)
                else:
                    idx_face = face_index["i"]
                    face_index["i"] += 1
                face_stage = f"{stage} ({face_name})"
                face_progress = progress + ev_span * ((idx_face + 1) / face_total)
                _schedule_status(face_stage, face_progress)

            def _step_progress(face_name: str, step: int, total: int) -> None:
                if step % 5 != 0 and step != total:
                    return
                if face_name in face_names:
                    idx_face = face_names.index(face_name)
                else:
                    idx_face = face_index["i"]
                step_frac = (step / max(total, 1))
                face_frac = (idx_face + step_frac) / face_total
                pct = int(round((step / max(total, 1)) * 100))
                face_stage = f"{stage} ({face_name}) step {step}/{total} ({pct}%)"
                face_progress = progress + ev_span * face_frac
                _schedule_status(face_stage, face_progress)

            def _face_done(face_name: str, face_img: np.ndarray) -> None:
                faces_out_current[face_name] = face_img
                now = time.monotonic()
                if now - last_preview["t"] < 2.0:
                    return
                last_preview["t"] = now
                snapshot = {k: v.copy() for k, v in faces_out_current.items()}
                _schedule_preview_snapshot(snapshot)
                def _emit_face_preview() -> None:
                    try:
                        face_preview = tone_map_for_preview(face_img)
                        face_path = PREVIEWS_DIR / f"{job_id}_ev{ev:+.0f}_{face_name}.jpg"
                        iio.imwrite(face_path, (face_preview * 255.0).astype(np.uint8))
                        face_url = f"/api/result/{job_id}/face/{ev:+.0f}/{face_name}"
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(
                                WS_MANAGER.broadcast(
                                    job_id,
                                    {
                                        "event": "face_preview",
                                        "ev": f"{ev:+.0f}",
                                        "face": face_name,
                                        "url": face_url,
                                    },
                                )
                            )
                        )
                    except Exception:
                        pass

                loop.call_soon_threadsafe(lambda: loop.run_in_executor(None, _emit_face_preview))

            faces_out = await asyncio.to_thread(
                ai.enhance_cubemap_for_ev,
                faces_ev,
                coverage,
                ev,
                internal_prompt,
                AI_STRENGTH_ORIGINAL,
                AI_DENOISE_NEW,
                seed,
                _face_progress,
                _step_progress,
                _face_done,
            )
            logging.info(
                "Job %s EV %s face shapes: %s",
                job_id,
                ev,
                {k: v.shape for k, v in faces_out.items()},
            )
            pano_lin = await asyncio.to_thread(
                cubemap_to_equirectangular, faces_out, PANORAMA_WIDTH, PANORAMA_HEIGHT
            )
            logging.info("Job %s EV %s pano shape: %s", job_id, ev, pano_lin.shape)
            if pano_lin.shape[0] != PANORAMA_HEIGHT or pano_lin.shape[1] != PANORAMA_WIDTH:
                try:
                    import cv2  # type: ignore

                    pano_lin = cv2.resize(
                        pano_lin,
                        (PANORAMA_WIDTH, PANORAMA_HEIGHT),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32)
                except Exception:
                    from PIL import Image

                    pano_img = Image.fromarray(np.clip(pano_lin * 255.0, 0, 255).astype(np.uint8))
                    pano_img = pano_img.resize((PANORAMA_WIDTH, PANORAMA_HEIGHT), Image.BILINEAR)
                    pano_lin = np.asarray(pano_img).astype(np.float32) / 255.0
                logging.info("Job %s EV %s pano resized to: %s", job_id, ev, pano_lin.shape)
            pano_stack.append(pano_lin)

            if ev == 0.0:
                preview = await asyncio.to_thread(tone_map_for_preview, pano_lin)
                preview_path = PREVIEWS_DIR / f"{job_id}_preview.jpg"
                await asyncio.to_thread(iio.imwrite, preview_path, (preview * 255.0).astype(np.uint8))
                preview_url = f"/api/result/{job_id}/preview"
                _update_job(job_id, preview_url=preview_url)
                await WS_MANAGER.broadcast(job_id, {"event": "preview", "preview_url": preview_url})

        _update_job(job_id, current_stage="Merging HDR stack", progress=0.9)
        await WS_MANAGER.broadcast(job_id, {"event": "status", "stage": "Merging HDR stack", "progress": 0.9})
        hdr = await asyncio.to_thread(merge_hdr_from_panos, pano_stack, ev_offsets)
        hdr_path = HDR_DIR / f"{job_id}.exr"
        try:
            await asyncio.to_thread(iio.imwrite, hdr_path, hdr, plugin="OIIO")
        except Exception:
            await asyncio.to_thread(iio.imwrite, hdr_path, hdr)

        final_preview = await asyncio.to_thread(tone_map_for_preview, hdr)
        final_preview_path = PREVIEWS_DIR / f"{job_id}_final.jpg"
        await asyncio.to_thread(iio.imwrite, final_preview_path, (final_preview * 255.0).astype(np.uint8))

        hdr_url = f"/api/result/{job_id}/hdr"
        preview_url = f"/api/result/{job_id}/preview"
        _update_job(job_id, status="done", progress=1.0, current_stage="done", preview_url=preview_url, hdr_url=hdr_url)
        await WS_MANAGER.broadcast(job_id, {"event": "done", "hdr_url": hdr_url, "preview_url": preview_url})
    except Exception as exc:
        logging.exception("Job %s failed: %s", job_id, exc)
        msg = str(exc)
        _update_job(job_id, status="error", current_stage="error", error_message=msg)
        await WS_MANAGER.broadcast(job_id, {"event": "error", "message": msg})

