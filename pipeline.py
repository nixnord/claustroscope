"""Module 4 Master Pipeline: pipeline.py

End-to-End Orchestrator connecting all four stages:
  Stage 1: Feature Extraction (extractors.py)
  Stage 2: Claustrophobia Scoring Engine (scorer.py)
  Stage 3: Conditioning & Latent Modulation (modulators.py)
  Stage 4: Generative Multi-ControlNet Diffusion (client_pipeline.py / Kaggle Gradio)

Usage:
  # Run Stages 1-3 (Feature extraction, scoring, OpenCV conditioning):
  python pipeline.py test/testimg3.jpeg -o outputs/testresult

  # Run Stages 1-4 with active Kaggle GPU worker:
  python pipeline.py test/testimg3.jpeg -o outputs/testresult --url https://xxxx.trycloudflare.com
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

import cv2
from dotenv import load_dotenv
from extractors import FeatureExtractor, load_image

load_dotenv()
from modulators import (
    apply_radial_vignette,
    distort_final_canvas,
    modulate_canny_edges,
    warp_depth_inward,
)
import numpy as np
from scorer import build_control_payload


@dataclass
class PipelineResult:
  """End-to-end result container for a single image run."""

  image_path: str
  feature_payload: dict[str, Any]
  control_payload: dict[str, Any]
  vignetted_path: str
  warped_depth_path: str
  modulated_canny_path: str
  preview_path: str | None = None
  final_canvas_path: str | None = None
  elapsed_time: float = 0.0


class ClaustrophobiaPipeline:
  """Full four-stage Claustrophobia Visualizer Pipeline."""

  def __init__(
      self,
      device: str | None = None,
      kaggle_url: str | None = None,
      enable_local_diffusion: bool = False,
  ):
    import torch

    self.device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    self.extractor = FeatureExtractor(device=str(self.device))
    self.kaggle_url = kaggle_url or os.environ.get("KAGGLE_TUNNEL_URL")
    self.enable_local_diffusion = enable_local_diffusion
    self.diffusion_pipe = None

    if enable_local_diffusion:
      self._load_diffusion_models()

  def _load_diffusion_models(self) -> None:
    """Loads Multi-ControlNet Stable Diffusion (DreamShaper 8) into GPU memory."""
    import torch
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetImg2ImgPipeline,
    )

    dtype = torch.float16 if self.device.type == "cuda" else torch.float32
    print(f"[{self.device}] Loading Multi-ControlNet adapters (Depth & Canny)...")
    controlnets = [
        ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=dtype,
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=dtype,
        ),
    ]

    base_model_id = "Lykon/dreamshaper-8"
    print(f"[{self.device}] Loading Base Checkpoint: {base_model_id}...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnets,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(self.device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    try:
      pipe.enable_xformers_memory_efficient_attention()
    except Exception:
      pipe.enable_attention_slicing()

    self.diffusion_pipe = pipe
    print("Multi-ControlNet Diffusion Pipeline loaded and ready in memory.")

  def run(
      self,
      image_path: str | Path,
      output_dir: str | Path = "outputs",
      subject_description: str = "a photograph of the scene",
      kaggle_url: str | None = None,
      apply_lens_distortion: bool = True,
      distortion_mode: str = "barrel",
  ) -> PipelineResult:
    """Execute end-to-end processing on an input photograph.

    Args:
        image_path: Path to input photo.
        output_dir: Folder to store JSON payloads and modulated images.
        subject_description: Base description used in dynamic prompts.
        kaggle_url: Optional override for Kaggle tunnel URL.
        apply_lens_distortion: Whether to apply claustrophobic lens distortion.
        distortion_mode: "barrel" or "tunnel" lens distortion model.

    Returns:
        PipelineResult containing payloads, map paths, and final canvas path.
    """
    t0 = time.perf_counter()
    img_p = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = img_p.stem
    active_url = kaggle_url or self.kaggle_url

    print(
        f"\n[{stem}] Stage 1: Running perception models (Depth Anything V2,"
        " SegFormer, YOLO11-Pose, CV)..."
    )
    feature_payload, artifacts = self.extractor.extract_with_artifacts(img_p)

    print(
        f"[{stem}] Stage 2: Scoring claustrophobia tension & computing"
        " hyperparameters..."
    )
    control_payload = build_control_payload(
        feature_payload, base_subject_description=subject_description
    )

    scores = control_payload["scores"]
    classification = control_payload["classification"]
    print(
        f"[{stem}] Scores: T={scores['trapping_base_T']:.3f},"
        f" R={scores['raw_score_R']:.3f},"
        f" C={scores['claustrophobia_score_C']:.3f} -> Classification:"
        f" {classification}"
    )

    # Save payloads
    feat_file = out_dir / f"{stem}_features.json"
    ctrl_file = out_dir / f"{stem}_control.json"
    feat_file.write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")
    ctrl_file.write_text(json.dumps(control_payload, indent=2), encoding="utf-8")

    print(f"[{stem}] Stage 3: Applying OpenCV conditioning modulations...")
    rgb = artifacts.image_rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_u8 = (artifacts.depth_closeness * 255).astype(np.uint8)
    canny_u8 = artifacts.edges

    fp = control_payload["focal_point"]
    mod_params = control_payload["modulation_parameters"]

    vignette_img = apply_radial_vignette(
        bgr, fp, mod_params["vignette_intensity"]
    )
    warped_depth = warp_depth_inward(
        depth_u8, fp, mod_params["depth_warp_kappa"]
    )

    primary_bbox = None
    if (
        artifacts.pose.person_detected
        and artifacts.pose.primary_index is not None
    ):
      primary_bbox = artifacts.pose.boxes[artifacts.pose.primary_index]

    modulated_canny = modulate_canny_edges(
        canny_u8,
        primary_bbox,
        mod_params["edge_jitter_amplitude"],
    )

    vignette_path = str(out_dir / f"{stem}_mod_vignette.png")
    depth_path = str(out_dir / f"{stem}_mod_depth.png")
    canny_path = str(out_dir / f"{stem}_mod_canny.png")

    cv2.imwrite(vignette_path, vignette_img)
    cv2.imwrite(depth_path, warped_depth)
    cv2.imwrite(canny_path, modulated_canny)

    preview_path = str(out_dir / f"{stem}_mod_preview.png")
    if apply_lens_distortion:
      distort_final_canvas(
          vignette_img,
          control_payload,
          output_path=preview_path,
          mode=distortion_mode,
      )
      print(f"[{stem}] Modulated preview with {distortion_mode} lens distortion saved to {preview_path}")
    else:
      cv2.imwrite(preview_path, vignette_img)

    final_canvas_path = None
    if self.diffusion_pipe is not None:
      print(
          f"[{stem}] Stage 4: Executing in-process Multi-ControlNet Diffusion (DreamShaper 8)..."
      )
      try:
        from PIL import Image

        diff_params = control_payload["diffusion_hyperparameters"]
        prompts = control_payload["prompts"]

        base_rgb = cv2.cvtColor(vignette_img, cv2.COLOR_BGR2RGB)
        base_pil = Image.fromarray(base_rgb)
        depth_pil = Image.fromarray(warped_depth).convert("RGB")
        canny_pil = Image.fromarray(modulated_canny).convert("RGB")

        w, h = base_pil.size
        max_dim = 768
        if max(w, h) > max_dim:
          s = max_dim / max(w, h)
          w_t, h_t = int(round(w * s)), int(round(h * s))
        else:
          w_t, h_t = w, h
        w_adj, h_adj = (w_t // 8) * 8, (h_t // 8) * 8

        base_input = base_pil.resize((w_adj, h_adj), Image.Resampling.LANCZOS)
        depth_input = depth_pil.resize((w_adj, h_adj), Image.Resampling.NEAREST)
        canny_input = canny_pil.resize((w_adj, h_adj), Image.Resampling.NEAREST)

        output = self.diffusion_pipe(
            prompt=prompts["positive"],
            negative_prompt=prompts["negative"],
            image=base_input,
            control_image=[depth_input, canny_input],
            strength=float(diff_params["denoising_strength"]),
            guidance_scale=float(diff_params["cfg_scale"]),
            controlnet_conditioning_scale=[
                float(diff_params["controlnet_depth_scale"]),
                float(diff_params["controlnet_edge_scale"]),
            ],
            num_inference_steps=int(diff_params["num_inference_steps"]),
        ).images[0]

        output_bgr = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        if (w, h) != (w_adj, h_adj):
          output_bgr = cv2.resize(output_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

        canvas_file = str(out_dir / f"{stem}_final_canvas.png")
        if apply_lens_distortion:
          distort_final_canvas(
              output_bgr,
              control_payload,
              output_path=canvas_file,
              mode=distortion_mode,
          )
          print(f"[{stem}] Lens-distorted generative canvas saved to {canvas_file}")
        else:
          cv2.imwrite(canvas_file, output_bgr)
          print(f"[{stem}] Final generative canvas saved to {canvas_file}")
        final_canvas_path = canvas_file
      except Exception as e:
        print(
            f"[{stem}] Warning: In-process diffusion failed ({type(e).__name__}: {e}). "
            "Falling back to local Stage 3 optical preview.",
            file=sys.stderr,
        )
        if os.environ.get("DEBUG_DIFFUSION"):
          traceback.print_exc()

    elif active_url:
      print(
          f"[{stem}] Stage 4: Triggering remote diffusion inference on Kaggle"
          " worker..."
      )
      try:
        from client_pipeline import DiffusionClient

        client = DiffusionClient(active_url)
        canvas_file = str(out_dir / f"{stem}_final_canvas.png")
        final_canvas_path = client.run(
            vignetted_path=vignette_path,
            warped_depth_path=depth_path,
            modulated_canny_path=canny_path,
            control_payload=control_payload,
            output_filepath=canvas_file,
            apply_lens_distortion=apply_lens_distortion,
            distortion_mode=distortion_mode,
        )
      except Exception as e:
        err_msg = str(e)
        if "Name or service not known" in err_msg or "ConnectError" in type(e).__name__:
          print(
              f"[{stem}] Warning: Remote worker at '{active_url}' is unreachable "
              "(DNS lookup failed / tunnel expired). "
              "Check your Kaggle cell or update KAGGLE_TUNNEL_URL in .env.\n"
              f"[{stem}] Gracefully falling back to local Stage 3 optical preview.",
              file=sys.stderr,
          )
        else:
          print(
              f"[{stem}] Warning: Remote diffusion failed ({type(e).__name__}: {e}). "
              "Falling back to local Stage 3 optical preview.",
              file=sys.stderr,
          )
        if os.environ.get("DEBUG_DIFFUSION"):
          traceback.print_exc()
    else:
      print(
          f"[{stem}] Stage 4: Skipped remote diffusion (no --url provided)."
          f" Modulated conditioning maps and preview are ready in {out_dir}/"
      )

    dt = time.perf_counter() - t0
    return PipelineResult(
        image_path=str(img_p),
        feature_payload=feature_payload,
        control_payload=control_payload,
        vignetted_path=vignette_path,
        warped_depth_path=depth_path,
        modulated_canny_path=canny_path,
        preview_path=preview_path,
        final_canvas_path=final_canvas_path,
        elapsed_time=dt,
    )


def main():
  ap = argparse.ArgumentParser(
      description="Claustrophobia Visualizer End-to-End Pipeline"
  )
  ap.add_argument("image", help="Input photograph path")
  ap.add_argument(
      "-o", "--output-dir", default="outputs", help="Output directory"
  )
  ap.add_argument(
      "--subject",
      default="a photograph of the scene",
      help="Base subject description for prompt",
  )
  ap.add_argument(
      "--url", default=None, help="Public Kaggle / Cloudflare tunnel URL"
  )
  ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
  ap.add_argument(
      "--no-lens-distortion",
      action="store_true",
      help="Disable lens distortion on preview and final canvas",
  )
  ap.add_argument(
      "--mode",
      choices=["barrel", "tunnel"],
      default="barrel",
      help="Lens distortion mode (default: barrel)",
  )
  ap.add_argument(
      "--local-diffusion",
      action="store_true",
      help="Run local Multi-ControlNet diffusion with DreamShaper 8 on GPU",
  )
  args = ap.parse_args()

  pipeline = ClaustrophobiaPipeline(
      device=args.device,
      kaggle_url=args.url,
      enable_local_diffusion=args.local_diffusion,
  )
  res = pipeline.run(
      image_path=args.image,
      output_dir=args.output_dir,
      subject_description=args.subject,
      apply_lens_distortion=not args.no_lens_distortion,
      distortion_mode=args.mode,
  )

  print("\n" + "═" * 60)
  print(f"Pipeline finished in {res.elapsed_time:.2f}s")
  print(
      f"Scores: C={res.control_payload['scores']['claustrophobia_score_C']:.3f}"
      f" [{res.control_payload['classification']}]"
  )
  print(f"Vignette map : {res.vignetted_path}")
  print(f"Depth map    : {res.warped_depth_path}")
  print(f"Canny map    : {res.modulated_canny_path}")
  if res.preview_path:
    print(f"Preview map  : {res.preview_path}")
  if res.final_canvas_path:
    print(f"Final canvas : {res.final_canvas_path}")
  print("═" * 60)


if __name__ == "__main__":
  main()
