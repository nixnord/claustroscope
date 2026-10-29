"""Module 5 Backend Server: app.py

FastAPI web application exposing the 4-stage Claustrophobia Visualizer pipeline.
Pre-loads Stage 1-3 perception models in memory as a singleton lifecycle service.
Connects optionally to remote Kaggle GPU worker for Multi-ControlNet diffusion.
"""

from __future__ import annotations

import base64
import os
import shutil
import tempfile
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from pipeline import ClaustrophobiaPipeline

# Load .env variables (e.g. KAGGLE_TUNNEL_URL)
load_dotenv()

PIPELINE: ClaustrophobiaPipeline | None = None
TEMP_ROOT = Path(tempfile.gettempdir()) / "claustrophobia_web_runs"
SAMPLE_DIR = Path(__file__).resolve().parent / "test"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    print("Pre-loading Stage 1-3 perception models into memory...")
    PIPELINE = ClaustrophobiaPipeline()
    print("Claustrophobia perception pipeline ready.")
    yield
    shutil.rmtree(TEMP_ROOT, ignore_errors=True)


app = FastAPI(
    title="Code the Canvas: Claustrophobia Visualizer",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode_b64(filepath: str | Path | None) -> str | None:
    if not filepath:
        return None
    p = Path(filepath)
    if not p.exists():
        return None
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    ext = p.suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{data}"


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Returns server status, model readiness, and default tunnel URL."""
    return {
        "status": "ok",
        "models_loaded": PIPELINE is not None,
        "default_kaggle_url": os.environ.get("KAGGLE_TUNNEL_URL", ""),
        "primary_engine_url": os.environ.get("PRIMARY_ENGINE_URL", ""),
    }


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Returns runtime configuration and tunnel URLs for frontend initialization."""
    return {
        "primary_engine_url": os.environ.get("PRIMARY_ENGINE_URL", "").strip(),
        "kaggle_tunnel_url": os.environ.get("KAGGLE_TUNNEL_URL", "").strip(),
    }


@app.get("/api/samples")
def list_samples() -> list[dict[str, str]]:
    """Returns available sample test images for quick one-click testing."""
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
    """Serves a sample image file from the test folder."""
    target = SAMPLE_DIR / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    ext = target.suffix.lower()
    media_type = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(str(target), media_type=media_type)


@app.post("/api/analyze")
def analyze_image(
    image: UploadFile = File(...),
    kaggle_url: str = Form(""),
    subject: str = Form("a photograph of the scene"),
    apply_distortion: bool = Form(True),
    distortion_mode: str = Form("barrel"),
) -> dict[str, Any]:
    """Executes the full Claustrophobia Visualizer pipeline.

    - Runs Stages 1-3 locally (Perception, Scoring, OpenCV conditioning).
    - Applies non-linear claustrophobic lens distortion to final canvas and preview.
    - If kaggle_url is provided, calls remote Stage 4 diffusion on Kaggle GPU.
    """
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Models are still initializing.")

    run_id = uuid.uuid4().hex[:8]
    run_dir = TEMP_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        orig_filename = image.filename or "input.jpg"
        ext = Path(orig_filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        input_path = run_dir / f"sample{ext}"
        input_path.write_bytes(image.file.read())

        clean_url = kaggle_url.strip() or os.environ.get("KAGGLE_TUNNEL_URL", "").strip() or None
        res = PIPELINE.run(
            image_path=input_path,
            output_dir=run_dir,
            subject_description=subject.strip() or "a photograph of the scene",
            kaggle_url=clean_url,
            apply_lens_distortion=apply_distortion,
            distortion_mode=distortion_mode,
        )

        stem = input_path.stem
        preview_candidate = res.preview_path or (run_dir / f"{stem}_mod_preview.png")

        if res.final_canvas_path and Path(res.final_canvas_path).exists():
            remote_status = "success"
            remote_msg = "Remote Kaggle GPU diffusion canvas synthesized successfully."
        elif clean_url:
            remote_status = "failed_fallback_to_preview"
            remote_msg = f"Remote tunnel '{clean_url}' is unreachable (expired or offline). Falling back to Stage 3 optical preview."
        else:
            remote_status = "skipped_offline_mode"
            remote_msg = "Offline preview mode."

        return {
            "run_id": run_id,
            "elapsed_time": round(res.elapsed_time, 2),
            "remote_status": remote_status,
            "remote_message": remote_msg,
            "features": res.feature_payload,
            "control": res.control_payload,
            "images": {
                "original": _encode_b64(input_path),
                "vignette": _encode_b64(res.vignetted_path),
                "depth": _encode_b64(res.warped_depth_path),
                "canny": _encode_b64(res.modulated_canny_path),
                "preview": _encode_b64(preview_candidate),
                "final_canvas": _encode_b64(res.final_canvas_path),
            },
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    """Serves the interactive single-page application dashboard."""
    html_file = Path(__file__).resolve().parent / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))
