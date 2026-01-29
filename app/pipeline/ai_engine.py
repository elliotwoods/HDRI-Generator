from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from config import AI_CFG_SCALE, AI_DENOISE_NEW, AI_STEPS, AI_STRENGTH_ORIGINAL, AI_FACE_RES
from .linearization import linear_to_srgb, srgb_to_linear

try:
    import torch  # type: ignore
    from diffusers import StableDiffusionXLImg2ImgPipeline  # type: ignore
    from diffusers.schedulers import DPMSolverMultistepScheduler  # type: ignore
except Exception:
    torch = None
    StableDiffusionXLImg2ImgPipeline = None
    DPMSolverMultistepScheduler = None

_PIPELINE: Optional["StableDiffusionXLImg2ImgPipeline"] = None
_DEVICE: Optional[str] = None
_DTYPE: Optional["torch.dtype"] = None


def _select_device() -> Tuple[str, "torch.dtype"]:
    if torch is None:
        raise RuntimeError("PyTorch is not available. Install torch to use SDXL.")
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def initialize_pipeline(device: Optional[str] = None) -> None:
    global _PIPELINE, _DEVICE, _DTYPE
    if _PIPELINE is not None:
        return
    if StableDiffusionXLImg2ImgPipeline is None or torch is None or DPMSolverMultistepScheduler is None:
        raise RuntimeError("Diffusers or required scheduler is not available. Install diffusers and torch.")
    if device is None:
        device, dtype = _select_device()
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    pipe.to(device)
    _PIPELINE = pipe
    _DEVICE = device
    _DTYPE = dtype


class AIEngine:
    def __init__(self, device: str = "cuda") -> None:
        self.device = _DEVICE or device
        self.pipe = self._get_pipeline()

    def _prepare_inpaint_inputs(self, face_lin: np.ndarray) -> Image.Image:
        face_srgb = linear_to_srgb(np.clip(face_lin, 0.0, 1.0))
        face_u8 = (face_srgb * 255.0).astype(np.uint8)
        return Image.fromarray(face_u8)

    def _mask_to_image(self, mask: np.ndarray) -> Image.Image:
        mask_u8 = (mask.astype(np.uint8) * 255)
        return Image.fromarray(mask_u8)

    def _get_pipeline(self) -> "StableDiffusionXLImg2ImgPipeline":
        if _PIPELINE is None:
            raise RuntimeError("SDXL pipeline not initialized. Call initialize_pipeline() at startup.")
        return _PIPELINE

    def enhance_cubemap_for_ev(
        self,
        faces_lin: Dict[str, np.ndarray],
        coverage_masks: Dict[str, np.ndarray],
        ev: float,
        internal_prompt: str,
        strength_original: float = AI_STRENGTH_ORIGINAL,
        denoise_new: float = AI_DENOISE_NEW,
        seed: int | None = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        step_callback: Optional[Callable[[str, int, int], None]] = None,
        face_done_callback: Optional[Callable[[str, np.ndarray], None]] = None,
    ) -> Dict[str, np.ndarray]:
        updated = {}
        rng = random.Random(seed)
        for face_name, face_lin in faces_lin.items():
            if progress_callback is not None:
                progress_callback(face_name)
            coverage = coverage_masks[face_name]
            generate_mask = ~coverage
            if not np.any(generate_mask):
                updated[face_name] = face_lin
                continue
            prompt = internal_prompt
            face_for_ai = face_lin
            mask_for_ai = generate_mask
            if face_lin.shape[0] > AI_FACE_RES or face_lin.shape[1] > AI_FACE_RES:
                scale = AI_FACE_RES / max(face_lin.shape[0], face_lin.shape[1])
                new_h = max(1, int(round(face_lin.shape[0] * scale)))
                new_w = max(1, int(round(face_lin.shape[1] * scale)))
                face_for_ai = np.asarray(
                    Image.fromarray(np.clip(face_lin * 255.0, 0, 255).astype(np.uint8)).resize(
                        (new_w, new_h), Image.BILINEAR
                    )
                ).astype(np.float32) / 255.0
                mask_for_ai = np.asarray(
                    Image.fromarray((generate_mask.astype(np.uint8) * 255)).resize(
                        (new_w, new_h), Image.NEAREST
                    )
                ) > 0
            image = self._prepare_inpaint_inputs(face_for_ai)
            mask_img = self._mask_to_image(mask_for_ai)
            gen = torch.Generator(device=self.device)
            if seed is None:
                seed = rng.randint(0, 2**32 - 1)
            gen.manual_seed(seed)
            coverage_ratio = float(np.mean(coverage))
            strength = denoise_new if coverage_ratio < 0.4 else strength_original
            def _step_cb(step: int, timestep: int, _kwargs=None) -> None:
                if step_callback is not None:
                    step_callback(face_name, step + 1, AI_STEPS)

            call_kwargs = dict(
                prompt=prompt,
                negative_prompt="cartoon, illustration, painting, unrealistic",
                image=image,
                strength=strength,
                guidance_scale=AI_CFG_SCALE,
                num_inference_steps=AI_STEPS,
                generator=gen,
            )
            try:
                import inspect

                sig = inspect.signature(self.pipe.__call__)
                if "callback_on_step_end" in sig.parameters:
                    call_kwargs["callback_on_step_end"] = (
                        lambda pipe, step, timestep, kwargs: (_step_cb(step, timestep), kwargs)[1]
                    )
                elif "callback" in sig.parameters:
                    call_kwargs["callback"] = _step_cb
                    call_kwargs["callback_steps"] = 1
                result = self.pipe(**call_kwargs).images[0]
            except Exception as exc:
                msg = str(exc)
                if "out of memory" in msg.lower():
                    raise RuntimeError(
                        "SDXL CUDA OOM. Reduce CUBE_RES or AI_FACE_RES, or close other GPU apps."
                    ) from exc
                raise RuntimeError(f"SDXL generation failed for face {face_name}: {exc}") from exc
            out_srgb = np.asarray(result).astype(np.float32) / 255.0
            mask_f = (np.asarray(mask_img).astype(np.float32) / 255.0)[..., None]
            if out_srgb.shape[:2] != face_lin.shape[:2]:
                out_srgb = np.asarray(
                    Image.fromarray(np.clip(out_srgb * 255.0, 0, 255).astype(np.uint8)).resize(
                        (face_lin.shape[1], face_lin.shape[0]), Image.BILINEAR
                    )
                ).astype(np.float32) / 255.0
                mask_f = np.asarray(
                    Image.fromarray((mask_f[..., 0] * 255).astype(np.uint8)).resize(
                        (face_lin.shape[1], face_lin.shape[0]), Image.NEAREST
                    )
                ).astype(np.float32) / 255.0
                mask_f = mask_f[..., None]
            out = srgb_to_linear(out_srgb)
            out = out * mask_f + face_lin * (1.0 - mask_f)
            updated[face_name] = out
            if face_done_callback is not None:
                face_done_callback(face_name, out)
            if torch is not None and self.device == "cuda":
                torch.cuda.empty_cache()
        return updated
