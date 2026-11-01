"""Module 5 Unified Kaggle Server: kaggle_server.py

Single-process unified GPU backend for Claustroscope:
  - Stage 1: Feature Extraction (extractors.py: Depth Anything V2, SegFormer, YOLO11-Pose, OpenCV)
  - Stage 2: Claustrophobia Scoring Engine (scorer.py: T -> R -> C mathematical scoring)
  - Stage 3: Spatial Modulation (modulators.py: Radial vignette, inward depth warp, canny jitter)
  - Stage 4: Generative Multi-ControlNet Diffusion (diffusers: Lykon/dreamshaper-8 + Depth & Canny ControlNet)
  - Post-Processing: Non-linear claustrophobic barrel lens distortion with peripheral chromatic aberration
  - Automated Cloudflare Tunnel: Exposes public trycloudflare.com URL for Next.js / Vercel frontend.

Usage:
  python kaggle_server.py
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import gc
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import torch
import uvicorn

# Import Claustroscope Pipeline Modules
from extractors import FeatureExtractor
from modulators import (
    apply_claustrophobic_lens_distortion,
    apply_radial_vignette,
    distort_final_canvas,
    modulate_canny_edges,
    warp_depth_inward,
)
from scorer import build_control_payload

load_dotenv()

# Global Model Singletons
EXTRACTOR: FeatureExtractor | None = None
DIFFUSION_PIPE: Any | None = None
MODELS_LOADED: bool = False


def smart_resize(img: Image.Image, max_side: int = 768) -> Image.Image:
    """Clamps longest edge to max_side while rounding both width and height to multiples of 8."""
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(int(round((w * scale) / 8.0)) * 8, 64)
    new_h = max(int(round((h * scale) / 8.0)) * 8, 64)
    if (new_w, new_h) != (w, h):
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def b64_encode_img(img_arr: np.ndarray, is_bgr: bool = True) -> str:
    """Encodes a numpy image (BGR, RGB, or grayscale) to a data:image/png;base64 string in memory."""
    if img_arr.ndim == 3 and not is_bgr:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", img_arr)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context manager replacing deprecated on_event handlers."""
    global EXTRACTOR, DIFFUSION_PIPE, MODELS_LOADED

    print("=" * 72)
    print("      CLAUSTROSCOPE UNIFIED KAGGLE GPU SERVER INITIALIZING")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using compute device: {device.upper()}")
    if device == "cuda":
        print(f"[Device] GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"[Device] Initial VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    # Stage 1: Pre-load Perception Models
    print("[Stage 1] Pre-loading perception models (Depth Anything V2, SegFormer, YOLO11-Pose, OpenCV)...")
    EXTRACTOR = FeatureExtractor(device=device)

    # Stage 4: Pre-load Multi-ControlNet Stable Diffusion Pipeline
    print("[Stage 4] Pre-loading Multi-ControlNet adapters and DreamShaper-8...")
    try:
        from diffusers import (
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionControlNetImg2ImgPipeline,
        )

        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        controlnets = [
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=torch_dtype,
            ),
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny",
                torch_dtype=torch_dtype,
            ),
        ]

        base_model_id = "Lykon/dreamshaper-8"
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnets,
            torch_dtype=torch_dtype,
            safety_checker=None,
        ).to(device)

        # Critical Scheduler Fix: Override DreamShaper's default 'deis' configuration tag
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[Diffusers] Enabled xFormers memory efficient attention.")
        except Exception:
            pipe.enable_attention_slicing()
            print("[Diffusers] Enabled attention slicing.")

        DIFFUSION_PIPE = pipe
        MODELS_LOADED = True
        print("[Pipeline] All 4 stages successfully initialized and resident in VRAM.")
    except Exception as e:
        print(f"[Warning] Failed to load Stage 4 diffusion models: {e}")
        print("[Warning] Server will operate in Stage 1-3 perception and preview mode.")
        DIFFUSION_PIPE = None
        MODELS_LOADED = False

    yield

    print("[Shutdown] Releasing GPU resources and flushing memory caches...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


app = FastAPI(
    title="Claustroscope Unified Kaggle GPU Server",
    description="Unified 4-stage psychological claustrophobia visualization and diffusion engine",
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend (Vercel, localhost, and custom domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Returns server health, GPU status, and loaded models."""
    gpu_active = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_active else "CPU"
    vram_mb = round(torch.cuda.memory_allocated(0) / 1024**2, 1) if gpu_active else 0.0

    return {
        "status": "ok",
        "gpu_active": gpu_active,
        "device": device_name,
        "vram_allocated_mb": vram_mb,
        "models_loaded": [
            "DepthAnythingV2",
            "SegFormer-B2",
            "YOLO11-Pose",
            "DreamShaper-8-MultiControlNet",
        ] if MODELS_LOADED else ["DepthAnythingV2", "SegFormer-B2", "YOLO11-Pose"],
    }


@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    apply_lens_distortion: bool = Form(True),
    distortion_mode: str = Form("barrel"),
    subject: str = Form("a photograph of the scene"),
) -> dict[str, Any]:
    """Execute end-to-end processing: Stages 1-4 + optical lens distortion in memory."""
    t0 = time.perf_counter()

    if not EXTRACTOR:
        raise HTTPException(status_code=503, detail="Perception models are not loaded.")

    try:
        # 1. Read and smart-resize input image
        contents = await image.read()
        pil_raw = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_resized = smart_resize(pil_raw, max_side=768)

        rgb = np.array(pil_resized)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 2. Stage 1: Feature Extraction
        feature_payload, artifacts = EXTRACTOR.extract_with_artifacts(rgb)

        # 3. Stage 2: Claustrophobia Scoring Engine
        control_payload = build_control_payload(
            feature_payload, base_subject_description=subject
        )

        # 4. Stage 3: Spatial Modulation
        fp = control_payload["focal_point"]
        mod_params = control_payload["modulation_parameters"]
        depth_u8 = (artifacts.depth_closeness * 255).astype(np.uint8)
        canny_u8 = artifacts.edges

        vignette_bgr = apply_radial_vignette(bgr, fp, mod_params["vignette_intensity"])
        warped_depth = warp_depth_inward(depth_u8, fp, mod_params["depth_warp_kappa"])

        primary_bbox = None
        if artifacts.pose.person_detected and artifacts.pose.primary_index is not None:
            primary_bbox = artifacts.pose.boxes[artifacts.pose.primary_index]

        modulated_canny = modulate_canny_edges(
            canny_u8, primary_bbox, mod_params["edge_jitter_amplitude"]
        )

        # 5. Optical Lens Preview (Stage 3 + Barrel Distortion)
        if apply_lens_distortion:
            preview_bgr = distort_final_canvas(
                vignette_bgr, control_payload, mode=distortion_mode
            )
        else:
            preview_bgr = vignette_bgr.copy()

        # 6. Stage 4: Generative Multi-ControlNet Diffusion
        final_canvas_bgr: np.ndarray
        if DIFFUSION_PIPE is not None:
            vignette_pil = Image.fromarray(cv2.cvtColor(vignette_bgr, cv2.COLOR_BGR2RGB))
            depth_pil = Image.fromarray(warped_depth).convert("RGB")
            canny_pil = Image.fromarray(modulated_canny).convert("RGB")

            prompts = control_payload["prompts"]
            hyper = control_payload["diffusion_hyperparameters"]

            gen_result = DIFFUSION_PIPE(
                prompt=prompts["positive_prompt"],
                negative_prompt=prompts["negative_prompt"],
                image=vignette_pil,
                control_image=[depth_pil, canny_pil],
                strength=float(hyper["denoising_strength"]),
                guidance_scale=float(hyper["cfg_scale"]),
                controlnet_conditioning_scale=[
                    float(hyper["controlnet_depth_scale"]),
                    float(hyper["controlnet_edge_scale"]),
                ],
                num_inference_steps=int(hyper["num_inference_steps"]),
            ).images[0]

            gen_rgb = np.array(gen_result)
            gen_bgr = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR)

            if apply_lens_distortion:
                final_canvas_bgr = distort_final_canvas(
                    gen_bgr, control_payload, mode=distortion_mode
                )
            else:
                final_canvas_bgr = gen_bgr
        else:
            # Fallback to optical modulated preview if diffusion is not active
            final_canvas_bgr = preview_bgr.copy()

        # 7. Convert all maps to Base64 in memory
        b64_original = b64_encode_img(bgr, is_bgr=True)
        b64_vignette = b64_encode_img(vignette_bgr, is_bgr=True)
        b64_depth = b64_encode_img(warped_depth, is_bgr=False)
        b64_canny = b64_encode_img(modulated_canny, is_bgr=False)
        b64_preview = b64_encode_img(preview_bgr, is_bgr=True)
        b64_final = b64_encode_img(final_canvas_bgr, is_bgr=True)

        # 8. Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        elapsed_time = round(time.perf_counter() - t0, 2)

        return {
            "elapsed_time": elapsed_time,
            "scores": control_payload["scores"],
            "classification": control_payload["classification"],
            "focal_point": control_payload["focal_point"],
            "features": feature_payload,
            "control": control_payload,
            "images": {
                "original": b64_original,
                "vignette": b64_vignette,
                "depth": b64_depth,
                "canny": b64_canny,
                "preview": b64_preview,
                "final_canvas": b64_final,
            },
        }

    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=str(exc))


# ==============================================================================
# Cloudflare Automated Tunnel Launcher & Server Runner
# ==============================================================================

def ensure_cloudflared() -> Path:
    """Checks for or downloads the official cloudflared Linux binary."""
    bin_path = Path("./cloudflared").resolve()
    if bin_path.exists() and os.access(bin_path, os.X_OK):
        return bin_path

    print("[Cloudflare] Downloading cloudflared-linux-amd64 binary...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    urllib.request.urlretrieve(url, bin_path)
    bin_path.chmod(0o755)
    print(f"[Cloudflare] Binary installed at {bin_path}")
    return bin_path


def run_server_and_tunnel(host: str = "127.0.0.1", port: int = 7860):
    """Starts Uvicorn in a daemon thread and launches the Cloudflare quick tunnel."""
    cloudflared = ensure_cloudflared()

    print(f"[Server] Starting FastAPI backend on {host}:{port}...")
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    time.sleep(2.0)

    print(f"[Cloudflare] Starting tunnel forwarding to {host}:{port}...")
    cmd = [str(cloudflared), "tunnel", "--url", f"http://{host}:{port}"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Read stderr until tunnel URL is discovered
    assert proc.stderr is not None
    for line in proc.stderr:
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if tunnel_url:
        print("\n" + "=" * 72)
        print("                     TUNNEL ONLINE & READY")
        print("=" * 72)
        print(f"PUBLIC URL: {tunnel_url}")
        print("\nHOW TO CONNECT YOUR FRONTEND:")
        print("----------------------------")
        print("1. If using Vercel:")
        print("   In your Vercel Project Settings -> Environment Variables, set:")
        print(f"   NEXT_PUBLIC_KAGGLE_API_URL = {tunnel_url}")
        print("   Then click Redeploy (or update).")
        print("\n2. If running locally:")
        print("   In claustroscope/frontend/.env.local, set:")
        print(f"   NEXT_PUBLIC_KAGGLE_API_URL={tunnel_url}")
        print("\n3. Runtime Dynamic Override:")
        print("   Open the frontend, click the 'Kaggle GPU' status badge in the header,")
        print(f"   and paste: {tunnel_url}")
        print("=" * 72 + "\n")
    else:
        print("[Cloudflare] Warning: Cloudflare tunnel URL could not be auto-detected.")

    # Background thread to drain stderr and avoid pipe buffer lockups
    def drain_pipe():
        assert proc.stderr is not None
        for _ in proc.stderr:
            pass

    threading.Thread(target=drain_pipe, daemon=True).start()

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping tunnel and shutting down server...")
        proc.terminate()
        server.should_exit = True


if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", "7860"))
    run_server_and_tunnel(host="127.0.0.1", port=server_port)
