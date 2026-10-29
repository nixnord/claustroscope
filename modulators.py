"""Module 3: modulators.py

Applies OpenCV-based transformations to conditioning maps based on the
Claustrophobia control payload (spec §5, §8):

1. apply_radial_vignette: Darkens outer margins centered at the focal point.
2. warp_depth_inward: Inward polynomial distortion bending walls inward.
3. modulate_canny_edges: Jitters background edge lines while preserving the
   human subject boundary.
4. apply_claustrophobic_lens_distortion: Non-linear radial lens distortion with
   subtle edge chromatic aberration applied to the final produced canvas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def apply_radial_vignette(
    image: np.ndarray,
    focal_point: Mapping[str, float],
    intensity: float,
    r_min: float = 0.18,
    r_max: float = 0.68,
) -> np.ndarray:
    """Darken perimeter of the original photo based on vignette_intensity and focal_point.

    Args:
        image: BGR or RGB uint8 image array (H, W, 3).
        focal_point: dict with keys "x" and "y" normalized to [0, 1].
        intensity: float in [0, 1] from modulation_parameters.vignette_intensity.
        r_min: inner clear radius (normalized to frame).
        r_max: outer full-vignette radius (normalized to frame).

    Returns:
        Vignetted image array with same shape and dtype as input.
    """
    if intensity <= 1e-4:
        return image.copy()

    h, w = image.shape[:2]
    cx = focal_point.get("x", 0.5) * w
    cy = focal_point.get("y", 0.5) * h

    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    dx = (x_indices - cx) / float(w)
    dy = (y_indices - cy) / float(h)
    radius = np.sqrt(dx**2 + dy**2)

    alpha = np.clip((radius - r_min) / max(r_max - r_min, 1e-6), 0.0, 1.0) * float(intensity)
    alpha = np.expand_dims(alpha, axis=-1)

    vignetted = image.astype(np.float32) * (1.0 - alpha)
    return np.clip(vignetted, 0, 255).astype(image.dtype)


def warp_depth_inward(
    depth_map: np.ndarray,
    focal_point: Mapping[str, float],
    kappa: float,
) -> np.ndarray:
    """Physically bend outer depth pixels toward the focal point using polynomial remap.

    Forces the depth conditioning model to see leaning or collapsing architecture.

    Args:
        depth_map: Grayscale or 3-channel depth map (H, W) or (H, W, 3).
        focal_point: dict with keys "x" and "y" normalized to [0, 1].
        kappa: float displacement coefficient from modulation_parameters.depth_warp_kappa.

    Returns:
        Warped depth map with the same shape and dtype as input.
    """
    if kappa <= 1e-4:
        return depth_map.copy()

    h, w = depth_map.shape[:2]
    cx = focal_point.get("x", 0.5) * w
    cy = focal_point.get("y", 0.5) * h

    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    dx = x_indices - cx
    dy = y_indices - cy

    u = dx / float(w)
    v = dy / float(h)
    r2 = u**2 + v**2

    # Map source coordinates using polynomial radial expansion
    map_x = cx + dx * (1.0 + float(kappa) * r2)
    map_y = cy + dy * (1.0 + float(kappa) * r2)

    warped = cv2.remap(
        depth_map,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def modulate_canny_edges(
    canny_map: np.ndarray,
    bbox: list[float] | tuple[float, float, float, float] | None,
    jitter_amp: float,
) -> np.ndarray:
    """Protect subject from distortion while introducing jitter and jagged noise to background edges.

    Args:
        canny_map: Grayscale uint8 edge map (H, W).
        bbox: Subject bounding box [x1, y1, x2, y2] in pixels, or None.
        jitter_amp: float jitter amplitude in pixels from modulation_parameters.edge_jitter_amplitude.

    Returns:
        Modulated canny edge map (H, W) uint8.
    """
    if jitter_amp < 0.5:
        return canny_map.copy()

    h, w = canny_map.shape[:2]
    # Random displacement field smoothed with Gaussian blur
    noise_x = np.random.uniform(-jitter_amp, jitter_amp, (h, w)).astype(np.float32) * 2.0
    noise_y = np.random.uniform(-jitter_amp, jitter_amp, (h, w)).astype(np.float32) * 2.0

    noise_x = cv2.GaussianBlur(noise_x, (5, 5), 0)
    noise_y = cv2.GaussianBlur(noise_y, (5, 5), 0)

    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    map_x = np.clip(x_indices + noise_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(y_indices + noise_y, 0, h - 1).astype(np.float32)

    jittered = cv2.remap(
        canny_map,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Preserve subject boundary if bounding box exists
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            jittered[y1:y2, x1:x2] = canny_map[y1:y2, x1:x2]

    return jittered


def apply_claustrophobic_lens_distortion(
    image: np.ndarray,
    focal_point: Mapping[str, float],
    k: float,
    chromatic_aberration: float = 0.025,
    mode: str = "barrel",
) -> np.ndarray:
    """Applies non-linear radial lens distortion to enhance the claustrophobic feeling.

    Barrel distortion causes architectural walls and ceilings to bow inward around
    the focal point, creating an oppressive fishbowl / collapsing perimeter effect.
    Peripheral chromatic aberration adds subtle optical tension at high scores.

    Args:
        image: BGR or RGB uint8 image array (H, W, 3) or grayscale (H, W).
        focal_point: dict with keys "x" and "y" normalized to [0, 1].
        k: distortion coefficient (typically 0.0 to 0.45, scaled with C).
        chromatic_aberration: relative chromatic divergence between color channels at edges.
        mode: "barrel" (convex inward bowing) or "tunnel" (funnel inward pull).

    Returns:
        Lens-distorted image with the same shape and dtype as input.
    """
    if abs(k) <= 1e-4:
        return image.copy()

    h, w = image.shape[:2]
    cx = focal_point.get("x", 0.5) * w
    cy = focal_point.get("y", 0.5) * h

    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    dx = (x_indices - cx) / float(w)
    dy = (y_indices - cy) / float(h)
    r2 = dx**2 + dy**2

    # Optical chromatic splitting: Red channel expands slightly more than Blue at edges
    if chromatic_aberration > 0 and image.ndim == 3 and image.shape[2] == 3:
        channels = []
        shifts = [-chromatic_aberration, 0.0, chromatic_aberration]  # B, G, R
        for ch_idx, shift in enumerate(shifts):
            k_eff = k * (1.0 + shift)
            scale = 1.0 / (1.0 + k_eff * r2) if mode == "barrel" else (1.0 + k_eff * r2)
            map_x = cx + (x_indices - cx) * scale
            map_y = cy + (y_indices - cy) * scale
            warped_ch = cv2.remap(
                image[..., ch_idx],
                map_x.astype(np.float32),
                map_y.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            channels.append(warped_ch)
        return np.stack(channels, axis=-1)
    else:
        scale = 1.0 / (1.0 + k * r2) if mode == "barrel" else (1.0 + k * r2)
        map_x = cx + (x_indices - cx) * scale
        map_y = cy + (y_indices - cy) * scale
        return cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )


def distort_final_canvas(
    canvas_image: np.ndarray | str | Path,
    control_payload: Mapping[str, Any] | str | Path,
    output_path: str | Path | None = None,
    mode: str = "barrel",
) -> np.ndarray:
    """Enhances the final produced canvas image with claustrophobic lens distortion.

    Args:
        canvas_image: NumPy array or file path to the final canvas image.
        control_payload: Parsed control payload dict or path to JSON file.
        output_path: Optional path to save the distorted canvas.
        mode: "barrel" (convex inward bowing) or "tunnel" (funnel inward pull).

    Returns:
        The distorted image as an ndarray (and writes to output_path if provided).
    """
    if isinstance(canvas_image, (str, Path)):
        img = cv2.imread(str(canvas_image))
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {canvas_image}")
    else:
        img = canvas_image.copy()

    if isinstance(control_payload, (str, Path)):
        with open(control_payload, "r", encoding="utf-8") as f:
            control = json.load(f)
    else:
        control = dict(control_payload)

    fp = control.get("focal_point", {"x": 0.5, "y": 0.5})
    mod_params = control.get("modulation_parameters", {})
    c_score = float(control.get("scores", {}).get("claustrophobia_score_C", 0.0))
    k = float(mod_params.get("lens_distortion_k", 0.35 * c_score))
    chroma = 0.025 * c_score if c_score >= 0.35 else 0.0

    distorted = apply_claustrophobic_lens_distortion(
        img, focal_point=fp, k=k, chromatic_aberration=chroma, mode=mode
    )

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), distorted)

    return distorted


def process_all_modulations(
    image_path: str | Path,
    depth_path: str | Path,
    canny_path: str | Path,
    control_json_path: str | Path,
    output_dir: str | Path,
    bbox: list[float] | None = None,
) -> dict[str, str]:
    """Execute all three modulations and persist the resulting conditioning images.

    Args:
        image_path: Path to original photograph.
        depth_path: Path to depth map (grayscale or RGB).
        canny_path: Path to canny edges (grayscale).
        control_json_path: Path to Stage 2 control payload JSON.
        output_dir: Destination directory for modulated maps.
        bbox: Optional primary subject bounding box [x1, y1, x2, y2].

    Returns:
        dict mapping file keys to output file paths.
    """
    with open(control_json_path, "r", encoding="utf-8") as f:
        control = json.load(f)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth map at {depth_path}")

    canny = cv2.imread(str(canny_path), cv2.IMREAD_GRAYSCALE)
    if canny is None:
        raise FileNotFoundError(f"Failed to read canny map at {canny_path}")

    fp = control.get("focal_point", {"x": 0.5, "y": 0.5})
    mod_params = control.get("modulation_parameters", {})

    vignette_intensity = float(mod_params.get("vignette_intensity", 0.0))
    depth_kappa = float(mod_params.get("depth_warp_kappa", 0.0))
    jitter_amp = float(mod_params.get("edge_jitter_amplitude", 0.0))

    # Perform modulations
    vignetted_img = apply_radial_vignette(img, fp, vignette_intensity)
    warped_depth = warp_depth_inward(depth, fp, depth_kappa)
    jittered_canny = modulate_canny_edges(canny, bbox, jitter_amp)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    vignette_file = str(out_path / "mod_vignette.png")
    depth_file = str(out_path / "mod_depth.png")
    canny_file = str(out_path / "mod_canny.png")

    cv2.imwrite(vignette_file, vignetted_img)
    cv2.imwrite(depth_file, warped_depth)
    cv2.imwrite(canny_file, jittered_canny)

    return {
        "vignetted_path": vignette_file,
        "warped_depth_path": depth_file,
        "modulated_canny_path": canny_file,
    }


def main():
    ap = argparse.ArgumentParser(description="Stage 3: OpenCV Conditioning Map Modulator & Final Canvas Distortion")
    ap.add_argument("--image", help="Input photograph path")
    ap.add_argument("--depth", help="Input depth map path")
    ap.add_argument("--canny", help="Input canny edges path")
    ap.add_argument("--control", required=True, help="Stage 2 control payload JSON path")
    ap.add_argument("-o", "--output-dir", default="outputs", help="Output directory")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    help="Subject bounding box x1 y1 x2 y2 in pixels")
    ap.add_argument("--distort-canvas", help="Path to produced canvas image to apply final lens distortion")
    ap.add_argument("--mode", choices=["barrel", "tunnel"], default="barrel",
                    help="Lens distortion mode (default: barrel)")
    args = ap.parse_args()

    if args.distort_canvas:
        out_p = Path(args.output_dir) / f"{Path(args.distort_canvas).stem}_distorted.png"
        distort_final_canvas(args.distort_canvas, args.control, output_path=out_p, mode=args.mode)
        print(json.dumps({"distorted_canvas_path": str(out_p)}, indent=2))
        return

    if not (args.image and args.depth and args.canny):
        ap.error("--image, --depth, and --canny are required when not using --distort-canvas")

    results = process_all_modulations(
        image_path=args.image,
        depth_path=args.depth,
        canny_path=args.canny,
        control_json_path=args.control,
        output_dir=args.output_dir,
        bbox=args.bbox,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
