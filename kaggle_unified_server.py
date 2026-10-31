"""ClaustroScope Unified Server for Kaggle GPU (kaggle_unified_server.py)

Unifies the entire 4-stage Claustrophobia Visualizer pipeline into a single
high-performance service running on a Kaggle GPU (T4 / P100):
  1. FastAPI web server exposing /api/analyze and REST endpoints
  2. Stage 1: Feature Extraction (Depth Anything V2, SegFormer ADE20K, YOLO11-Pose, OpenCV)
  3. Stage 2: Claustrophobia Scoring (T -> R -> C Non-linear Formula)
  4. Stage 3: Spatial Conditioning (Radial Vignette, Depth Warping, Canny Jitter)
  5. Stage 4: Multi-ControlNet Diffusion (Lykon/dreamshaper-8 + Depth & Canny ControlNets)
  6. Stage 4 Post-Processing: Non-Linear Claustrophobic Barrel Lens Distortion
  7. Automated Cloudflare Tunnel forwarding port 8000 to a public HTTPS URL

How to run on Kaggle:
--------------------
Cell 1 (Clone & Dependencies):
  !git clone https://github.com/nixnord/claustroscope.git /kaggle/working/claustroscope
  %cd /kaggle/working/claustroscope
  !pip install -q -r requirements.txt

Cell 2 (Start Unified Server):
  !python kaggle_unified_server.py
  # OR paste this entire file into Cell 2 and click Run.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any
import urllib.request

# Ensure repo root is on sys.path
CWD = Path.cwd()
REPO_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else CWD
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))
KAGGLE_PATH = Path("/kaggle/working/claustroscope")
if KAGGLE_PATH.exists() and str(KAGGLE_PATH) not in sys.path:
  sys.path.insert(0, str(KAGGLE_PATH))

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import numpy as np
from pipeline import ClaustrophobiaPipeline
import torch
import uvicorn

load_dotenv()

# Global state
PIPELINE: ClaustrophobiaPipeline | None = None
TUNNEL_URL: str = ""
TUNNEL_PROC: subprocess.Popen | None = None
TEMP_ROOT = Path(tempfile.gettempdir()) / "claustrophobia_unified_runs"
SAMPLE_DIR = REPO_ROOT / "test"
INDEX_FILE = REPO_ROOT / "index.html"


def ensure_cloudflared() -> str:
  """Finds or downloads the cloudflared binary on Linux x86_64."""
  existing = shutil.which("cloudflared")
  if existing:
    return existing

  candidates = [
      Path("/tmp/cloudflared"),
      REPO_ROOT / "cloudflared",
      Path("/kaggle/working/cloudflared"),
  ]
  for c in candidates:
    if c.exists() and os.access(str(c), os.X_OK):
      return str(c)

  target = Path("/tmp/cloudflared")
  print("[Cloudflare] Downloading cloudflared binary for Linux amd64...")
  url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
  urllib.request.urlretrieve(url, str(target))
  os.chmod(str(target), 0o755)
  print(f"[Cloudflare] cloudflared ready at {target}")
  return str(target)


def start_cloudflare_tunnel(port: int = 8000, timeout: float = 30.0) -> str:
  """Starts a Cloudflare quick tunnel and captures the public trycloudflare.com URL."""
  global TUNNEL_PROC, TUNNEL_URL
  cf_bin = ensure_cloudflared()

  print(f"[Cloudflare] Starting tunnel forwarding 127.0.0.1:{port}...")
  cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"]
  TUNNEL_PROC = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      bufsize=1,
  )

  atexit.register(cleanup_tunnel)

  tunnel_url = ""
  url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
  start_time = time.time()

  # Cloudflare logs the URL in stderr
  assert TUNNEL_PROC.stderr is not None
  while time.time() - start_time < timeout:
    line = TUNNEL_PROC.stderr.readline()
    if not line and TUNNEL_PROC.poll() is not None:
      break
    match = url_pattern.search(line)
    if match:
      tunnel_url = match.group(0)
      break

  if not tunnel_url:
    print(
        "[Cloudflare] Warning: Could not detect tunnel URL within timeout."
        " Server will continue on localhost.",
        file=sys.stderr,
    )
    return ""

  TUNNEL_URL = tunnel_url

  # Background log drainer so stderr buffer never blocks
  def _drain():
    try:
      while TUNNEL_PROC and TUNNEL_PROC.poll() is None:
        _ = TUNNEL_PROC.stderr.readline()
    except Exception:
      pass

  drain_thread = threading.Thread(target=_drain, daemon=True)
  drain_thread.start()

  return tunnel_url


def cleanup_tunnel():
  """Ensures cloudflared tunnel is cleanly terminated upon exit."""
  global TUNNEL_PROC
  if TUNNEL_PROC and TUNNEL_PROC.poll() is None:
    try:
      TUNNEL_PROC.terminate()
      TUNNEL_PROC.wait(timeout=2)
    except Exception:
      try:
        TUNNEL_PROC.kill()
      except Exception:
        pass


def encode_b64(filepath: str | Path | None) -> str | None:
  if not filepath:
    return None
  p = Path(filepath)
  if not p.exists():
    return None
  data = base64.b64encode(p.read_bytes()).decode("utf-8")
  ext = p.suffix.lower()
  mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
  return f"data:{mime};base64,{data}"


# ==============================================================================
# FastAPI Application Definition
# ==============================================================================
app = FastAPI(
    title="ClaustroScope Unified Engine",
    description="Unified 4-Stage Claustrophobia Visualizer on Kaggle GPU",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
  global PIPELINE
  TEMP_ROOT.mkdir(parents=True, exist_ok=True)
  print("\n" + "=" * 70)
  print("[ClaustroScope Unified Engine] Pre-loading all pipeline models...")
  print("  - Stage 1: Depth Anything V2 Small, SegFormer B2, YOLO11n-Pose, OpenCV")
  print("  - Stage 2: Claustrophobia Scoring Engine (T -> R -> C)")
  print("  - Stage 3: Spatial Modulation (Vignette, Depth Warp, Canny Jitter)")
  print(
      "  - Stage 4: Multi-ControlNet Stable Diffusion (DreamShaper 8, Depth,"
      " Canny)"
  )
  print("  - Stage 4: Non-Linear Claustrophobic Barrel Lens Distortion")
  print("=" * 70)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"[Device] Using compute device: {device.upper()}")
  if device == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    print(
        "[Device] Initial VRAM Allocated:"
        f" {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB"
    )

  # Initialize unified pipeline with local in-process diffusion enabled
  PIPELINE = ClaustrophobiaPipeline(device=device, enable_local_diffusion=True)

  if device == "cuda":
    print(
        "[Device] Models loaded into VRAM:"
        f" {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated."
    )
  print("[ClaustroScope Unified Engine] Server is ready to process requests.\n")


@app.on_event("shutdown")
async def shutdown_event():
  shutil.rmtree(TEMP_ROOT, ignore_errors=True)
  cleanup_tunnel()


@app.get("/api/health")
def health_check() -> dict[str, Any]:
  """Returns health status, active compute device, and unified mode flag."""
  device_str = "cuda" if torch.cuda.is_available() else "cpu"
  gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
  return {
      "status": "ok",
      "models_loaded": PIPELINE is not None,
      "unified_mode": True,
      "device": device_str,
      "gpu": gpu_name,
      "tunnel_url": TUNNEL_URL,
      "primary_engine_url": TUNNEL_URL,
  }


@app.get("/api/config")
def get_config() -> dict[str, Any]:
  """Returns public tunnel URL so frontend automatically connects."""
  return {
      "primary_engine_url": TUNNEL_URL,
      "kaggle_tunnel_url": TUNNEL_URL,
  }


@app.get("/api/samples")
def list_samples() -> list[dict[str, str]]:
  """Returns available sample test images."""
  if not SAMPLE_DIR.exists():
    return []
  descriptions = {
      "testimg1.jpeg": "Crawlspace with low ducts",
      "testimg2.jpeg": "Cave squeeze with caver",
      "testimg3.jpeg": "Narrow brick alleyway",
      "testimg4.jpeg": "Open sunlit grass park",
      "testimg5.jpg": "Park grove with trees",
      "testimg6.jpeg": "Packed stadium crowd",
  }
  samples = []
  for f in sorted(SAMPLE_DIR.iterdir()):
    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
      samples.append({
          "filename": f.name,
          "label": descriptions.get(f.name, f.stem),
          "url": f"/api/sample/{f.name}",
      })
  return samples


@app.get("/api/sample/{filename}")
def get_sample_image(filename: str):
  """Serves sample image files."""
  target = SAMPLE_DIR / filename
  if not target.exists() or not target.is_file():
    raise HTTPException(status_code=404, detail="Sample image not found")
  ext = target.suffix.lower()
  media_type = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
  return FileResponse(str(target), media_type=media_type)


@app.post("/api/analyze")
def analyze_image(
    image: UploadFile = File(...),
    subject: str = Form("a photograph of the scene"),
    apply_distortion: bool = Form(True),
    distortion_mode: str = Form("barrel"),
    kaggle_url: str = Form(""),  # Handled locally, ignored
) -> dict[str, Any]:
  """Unified inference endpoint: executes Stages 1-4 end-to-end on Kaggle GPU."""
  if PIPELINE is None:
    raise HTTPException(status_code=503, detail="Models are still initializing.")

  t0 = time.perf_counter()
  run_id = f"kgl_{int(time.time())}_{os.urandom(3).hex()}"
  run_dir = TEMP_ROOT / run_id
  run_dir.mkdir(parents=True, exist_ok=True)

  try:
    orig_filename = image.filename or "input.jpg"
    ext = Path(orig_filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
      ext = ".jpg"
    input_path = run_dir / f"sample{ext}"
    input_path.write_bytes(image.file.read())

    # Execute full 4-stage pipeline directly in GPU memory
    res = PIPELINE.run(
        image_path=input_path,
        output_dir=run_dir,
        subject_description=subject.strip() or "a photograph of the scene",
        apply_lens_distortion=apply_distortion,
        distortion_mode=distortion_mode,
    )

    stem = input_path.stem
    preview_candidate = res.preview_path or (run_dir / f"{stem}_mod_preview.png")

    if res.final_canvas_path and Path(res.final_canvas_path).exists():
      remote_status = "success"
      remote_msg = (
          "Unified Kaggle GPU Multi-ControlNet diffusion canvas synthesized"
          " successfully."
      )
    else:
      remote_status = "failed_fallback_to_preview"
      remote_msg = "Diffusion generation failed. Returning Stage 3 preview."

    return {
        "run_id": run_id,
        "elapsed_time": round(time.perf_counter() - t0, 2),
        "remote_status": remote_status,
        "remote_message": remote_msg,
        "features": res.feature_payload,
        "control": res.control_payload,
        "images": {
            "original": encode_b64(input_path),
            "vignette": encode_b64(res.vignetted_path),
            "depth": encode_b64(res.warped_depth_path),
            "canny": encode_b64(res.modulated_canny_path),
            "preview": encode_b64(preview_candidate),
            "final_canvas": encode_b64(res.final_canvas_path),
        },
    }
  except Exception as exc:
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=str(exc)) from exc
  finally:
    shutil.rmtree(run_dir, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
  """Serves index.html when opening the Cloudflare tunnel directly."""
  if INDEX_FILE.exists():
    return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
  return HTMLResponse(
      "<h1>ClaustroScope Unified Engine Online</h1>"
      "<p>Connected to Kaggle GPU.</p>"
  )


# ==============================================================================
# Runner (Compatible with both terminal execution and Kaggle notebook cells)
# ==============================================================================
def run_server(port: int = 8000):
  """Starts tunnel, configures Uvicorn, and serves requests safely in any environment."""
  print("\n" + "=" * 80)
  print("        CLAUSTROSCOPE UNIFIED KAGGLE GPU SERVER INITIALIZING")
  print("=" * 80)

  public_url = start_cloudflare_tunnel(port=port)

  print("\n" + "=" * 80)
  print("                     TUNNEL ONLINE & READY")
  print("=" * 80)
  if public_url:
    print(f"PUBLIC URL: {public_url}\n")
    print("HOW TO CONNECT YOUR FRONTEND:")
    print("----------------------------")
    print("1. If using Vercel:")
    print(
        "   In your Vercel Project Settings -> Environment Variables, set:"
    )
    print(f"   PRIMARY_ENGINE_URL = {public_url}")
    print("   Then click Redeploy (or update).\n")
    print("2. If using the Web UI directly:")
    print(f"   Simply visit {public_url} in your browser, or open the UI")
    print(f"   and paste {public_url} into the settings gear icon.")
  else:
    print("Warning: Cloudflare tunnel did not start. Running on local port only.")
  print("=" * 80 + "\n")

  config = uvicorn.Config(
      app=app,
      host="0.0.0.0",
      port=port,
      log_level="info",
  )
  server = uvicorn.Server(config)

  # Check if running inside an existing asyncio event loop (e.g. Jupyter kernel)
  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = None

  if loop and loop.is_running():
    print("[Environment] Active Jupyter / Kaggle event loop detected.")
    print("[Environment] Serving FastAPI via background daemon thread...")
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    try:
      while True:
        time.sleep(1)
    except KeyboardInterrupt:
      print("\n[Shutting down Kaggle Unified Server...]")
      server.should_exit = True
      cleanup_tunnel()
      server_thread.join(timeout=3)
  else:
    try:
      server.run()
    finally:
      cleanup_tunnel()


if __name__ == "__main__":
  run_server(port=8000)
