"""Module 4 Client: client_pipeline.py

Invokes the remote Kaggle Gradio worker via gradio_client.
Consumes modulated maps from Module 3 and diffusion parameters from Module 2 (control payload).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from gradio_client import Client, handle_file
from PIL import Image


class DiffusionClient:
    """Client bridge connecting local pipeline to remote Kaggle Gradio worker."""

    def __init__(self, gradio_live_url: str):
        """Initializes client with the active Kaggle public tunnel."""
        self.url = gradio_live_url.strip()
        print(f"Connecting to remote diffusion worker at {self.url}...")
        self.client = Client(self.url)

    def run(
        self,
        vignetted_path: str | Path,
        warped_depth_path: str | Path,
        modulated_canny_path: str | Path,
        control_payload: str | Path | Mapping[str, Any],
        output_filepath: str | Path,
        apply_lens_distortion: bool = True,
        distortion_mode: str = "barrel",
    ) -> str:
        """Call remote Kaggle Gradio API with modulated maps and control hyperparameters.

        Args:
            vignetted_path: Path to vignetted base image.
            warped_depth_path: Path to warped depth map.
            modulated_canny_path: Path to modulated canny edges.
            control_payload: Path to Stage 2 control JSON file, or parsed dict.
            output_filepath: Destination file path for generated canvas.
            apply_lens_distortion: Whether to apply post-processing lens distortion.
            distortion_mode: "barrel" (convex inward bowing) or "tunnel" mode.

        Returns:
            The saved output filepath string.
        """
        if isinstance(control_payload, (str, Path)):
            with open(control_payload, "r", encoding="utf-8") as f:
                control = json.load(f)
        else:
            control = dict(control_payload)

        diff_params = control["diffusion_hyperparameters"]
        prompts = control["prompts"]

        print(
            f"Invoking remote generation: steps={diff_params['num_inference_steps']}, "
            f"strength={diff_params['denoising_strength']}, cfg={diff_params['cfg_scale']}..."
        )

        # Arguments in the exact positional order expected by gr.Interface:
        args = [
            handle_file(str(vignetted_path)),
            handle_file(str(warped_depth_path)),
            handle_file(str(modulated_canny_path)),
            prompts["positive"],
            prompts["negative"],
            float(diff_params["denoising_strength"]),
            float(diff_params["cfg_scale"]),
            float(diff_params["controlnet_depth_scale"]),
            float(diff_params["controlnet_edge_scale"]),
            int(diff_params["num_inference_steps"]),
        ]

        # Call the primary endpoint using fn_index=0 to avoid endpoint name mismatch
        try:
            result = self.client.predict(*args, fn_index=0)
        except Exception:
            # Fallback to explicit function name if fn_index is not supported by client version
            result = self.client.predict(
                *args, api_name="/generate_visual_canvas"
            )

        out_p = Path(output_filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # Result is a temporary local file path downloaded by gradio_client
        if isinstance(result, str) and Path(result).exists():
            shutil.copy(result, str(out_p))
        else:
            saved_img = Image.open(result)
            saved_img.save(str(out_p))

        if apply_lens_distortion:
            from modulators import distort_final_canvas

            distort_final_canvas(
                str(out_p),
                control,
                output_path=str(out_p),
                mode=distortion_mode,
            )
            print(f"Applied claustrophobic lens distortion ({distortion_mode}) to {out_p}")

        print(f"Canvas saved to {out_p}")
        return str(out_p)


def main():
    ap = argparse.ArgumentParser(description="Module 4 Remote Diffusion Client")
    ap.add_argument(
        "--url", required=True, help="Active public Cloudflare/Gradio URL"
    )
    ap.add_argument(
        "--vignette", required=True, help="Vignetted base image path"
    )
    ap.add_argument("--depth", required=True, help="Warped depth map path")
    ap.add_argument("--canny", required=True, help="Modulated canny edge path")
    ap.add_argument(
        "--control", required=True, help="Control payload JSON path"
    )
    ap.add_argument(
        "-o", "--output", default="final_canvas.png", help="Output filepath"
    )
    ap.add_argument(
        "--no-lens-distortion",
        action="store_true",
        help="Disable final lens distortion post-processing",
    )
    ap.add_argument(
        "--mode",
        choices=["barrel", "tunnel"],
        default="barrel",
        help="Lens distortion mode (default: barrel)",
    )
    args = ap.parse_args()

    client = DiffusionClient(args.url)
    output = client.run(
        vignetted_path=args.vignette,
        warped_depth_path=args.depth,
        modulated_canny_path=args.canny,
        control_payload=args.control,
        output_filepath=args.output,
        apply_lens_distortion=not args.no_lens_distortion,
        distortion_mode=args.mode,
    )
    print(f"Final canvas successfully generated and saved to: {output}")


if __name__ == "__main__":
    main()
