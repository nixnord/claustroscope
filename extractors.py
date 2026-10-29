"""
Stage 1 — Feature Extraction.

Runs four perception branches on an input photograph and aggregates them into
the Stage 1 *Feature Payload* (spec §3, §8):

    Depth Anything V2 (Small)   → border_closeness, depth_flatness
    SegFormer-B2 (ADE20K)       → wall_ceiling_ratio, sky_open_ratio
    YOLO11n-Pose                → person_detected, crowd_density,
                                  body_compression, focal_point
    OpenCV (classical)          → edge_clutter, darkness_level, chromatic_warmth

Every scalar feature is normalised to [0, 1]. Intermediate maps (depth,
segmentation, edges, keypoints) are also returned via :class:`Artifacts` so the
Stage 3 modulation layer can warp/filter them without re-running inference.

Usage:
    python extractors.py photo.jpg                   # print payload JSON
    python extractors.py photo.jpg -o payload.json
    python extractors.py photo.jpg | python scorer.py -
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ImageInput = Union[str, Path, Image.Image, np.ndarray]


# ════════════════════════════════════════════════════════════════════════════
# Model identifiers & constants
# ════════════════════════════════════════════════════════════════════════════

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
SEG_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
POSE_MODEL_ID = "yolo11n-pose.pt"   # auto-downloaded by ultralytics

# ADE20K class names (spec §3.2). Water/ground synonyms are included so that
# beaches, lakes and rivers register as open space.
ENCLOSURE_LABELS = {"wall", "ceiling", "building", "fence", "door"}
OPEN_LABELS = {"sky", "earth", "field", "water", "sea", "river", "lake", "sand"}
LIGHT_SOURCE_LABELS = {"sky", "lamp", "light", "fireplace", "streetlight", "traffic light", "windowpane", "window", "candle"}
PERSON_LABELS = {"person"}

# Depth border strips (spec §3.1): top 15 %, left/right 10 %.
BORDER_TOP_FRAC = 0.15
BORDER_SIDE_FRAC = 0.10
# Side strips stop above the bottom of the frame: the floor is always near the
# camera and does not mean "walls are closing in".
BORDER_SIDE_VERTICAL_EXTENT = 0.75

# Pose (COCO-17 keypoint indices)
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_CONF_MIN = 0.40
PERSON_CONF_MIN = 0.35
# Limb-to-torso distance / torso length at which compression is 1 ("tight")
# and 0 ("open"). Calibrated for orientation-invariant torso-length scaling.
WRIST_RANGE = (0.35, 1.80)
ELBOW_RANGE = (0.25, 1.20)
# Crowd: how quickly additional people saturate the count factor.
CROWD_COUNT_SCALE = 3.0

# Classical CV
CV_WORK_MAX_SIDE = 768          # resize before Canny so density is res-invariant
CLUTTER_RADIUS_FRAC = 0.35      # focal-region radius as fraction of min(h, w)
CLUTTER_SATURATION = 0.20       # Canny density that maps to clutter = 1.0
DARK_V_THRESHOLD = 0.25         # spec §3.4
SAT_MIN_CHROMATIC = 0.15        # below this saturation a pixel counts as grey


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def load_image(img: ImageInput) -> Image.Image:
    """Accept a path, PIL image, or RGB ndarray and return an RGB PIL image."""
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)!r}")


def _resize_max_side(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return arr
    return cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _ramp_down(value: float, tight: float, open_: float) -> float:
    """1.0 at/below ``tight``, 0.0 at/above ``open_``, linear in between."""
    return _clip01((open_ - value) / (open_ - tight))


# ════════════════════════════════════════════════════════════════════════════
# Outputs
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PoseResult:
    person_detected: bool = False
    person_count: int = 0
    crowd_density: float = 0.0
    body_compression: float = 0.0
    focal_point: tuple[float, float] = (0.5, 0.5)       # normalised (x, y)
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 17, 2)))
    keypoint_conf: np.ndarray = field(default_factory=lambda: np.zeros((0, 17)))
    primary_index: int | None = None


@dataclass
class Artifacts:
    """Intermediate maps handed to Stage 3 (all at original image resolution)."""

    image_rgb: np.ndarray                 # H×W×3 uint8
    depth_closeness: np.ndarray           # H×W float32, 1 = nearest, 0 = farthest
    segmentation: np.ndarray              # H×W int32 ADE20K class ids
    edges: np.ndarray                     # H×W uint8 Canny map
    pose: PoseResult


# ════════════════════════════════════════════════════════════════════════════
# Extractor
# ════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Lazy-loading wrapper around all Stage 1 models.

    Models are loaded on first use and kept resident, so a single instance can
    process many images efficiently.
    """

    def __init__(self, device: str | None = None, half: bool | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = (self.device.type == "cuda") if half is None else half
        self.dtype = torch.float16 if self.half else torch.float32
        self._depth = None
        self._seg = None
        self._pose = None
        self._enc_ids: np.ndarray | None = None
        self._open_ids: np.ndarray | None = None
        self._light_ids: np.ndarray | None = None
        self._person_ids: np.ndarray | None = None

    # ── model loading ──────────────────────────────────────────────────────
    def _load_depth(self):
        if self._depth is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
            model = AutoModelForDepthEstimation.from_pretrained(
                DEPTH_MODEL_ID, dtype=self.dtype).to(self.device).eval()
            self._depth = (proc, model)
        return self._depth

    def _load_seg(self):
        if self._seg is None:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            proc = AutoImageProcessor.from_pretrained(SEG_MODEL_ID)
            model = AutoModelForSemanticSegmentation.from_pretrained(
                SEG_MODEL_ID, dtype=self.dtype).to(self.device).eval()
            # Resolve class ids by name (labels may carry stray whitespace).
            id2label = {int(i): str(n).strip().lower() for i, n in model.config.id2label.items()}
            self._enc_ids = np.array([i for i, n in id2label.items() if n in ENCLOSURE_LABELS])
            self._open_ids = np.array([i for i, n in id2label.items() if n in OPEN_LABELS])
            self._light_ids = np.array([i for i, n in id2label.items()
                                        if any(w in n for w in ['lamp', 'fireplace', 'streetlight', 'traffic light', 'sky', 'windowpane', 'window', 'candle']) or n == 'light'])
            self._person_ids = np.array([i for i, n in id2label.items() if 'person' in n])
            self._seg = (proc, model)
        return self._seg

    def _load_pose(self):
        if self._pose is None:
            from ultralytics import YOLO
            self._pose = YOLO(POSE_MODEL_ID)
        return self._pose

    # ── 3.1 depth ──────────────────────────────────────────────────────────
    @torch.inference_mode()
    def depth_map(self, image: Image.Image) -> np.ndarray:
        """Relative closeness map in [0, 1] at image resolution (1 = nearest).

        Depth Anything V2 (relative) predicts inverse depth / disparity, so
        larger raw values already mean closer surfaces. Foreground floor at the
        extreme bottom of the frame is excluded from the high-percentile reference
        so tight crawlspaces and rooms are not artificially compressed.
        """
        proc, model = self._load_depth()
        inputs = proc(images=image, return_tensors="pt").to(self.device, self.dtype)
        pred = model(**inputs).predicted_depth                    # (1, h, w)
        pred = F.interpolate(pred.unsqueeze(1).float(), size=image.size[::-1],
                             mode="bicubic", align_corners=False)[0, 0]
        d = pred.float().cpu().numpy()
        h = d.shape[0]
        lo = float(np.percentile(d, 1))
        # Upper percentile reference excludes extreme foreground floor (bottom 15%):
        hi = float(np.percentile(d[:int(0.85 * h), :], 99))
        return np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def depth_features(closeness: np.ndarray) -> dict[str, float]:
        h, w = closeness.shape
        top = max(1, int(round(h * BORDER_TOP_FRAC)))
        side = max(1, int(round(w * BORDER_SIDE_FRAC)))
        side_bottom = max(top, int(round(h * BORDER_SIDE_VERTICAL_EXTENT)))

        mask = np.zeros_like(closeness, dtype=bool)
        mask[:top, :] = True
        mask[:side_bottom, :side] = True
        mask[:side_bottom, w - side:] = True
        border_closeness = float(closeness[mask].mean())

        # Std of a [0,1] variable is ≤ 0.5, so 2·std is a normalised spread.
        depth_flatness = 1.0 - _clip01(2.0 * float(closeness.std()))
        return {"border_closeness": _r(_clip01(border_closeness)),
                "depth_flatness": _r(depth_flatness)}

    # ── 3.2 semantics ──────────────────────────────────────────────────────
    @torch.inference_mode()
    def segmentation_map(self, image: Image.Image) -> np.ndarray:
        proc, model = self._load_seg()
        inputs = proc(images=image, return_tensors="pt").to(self.device, self.dtype)
        logits = model(**inputs).logits                           # (1, C, h/4, w/4)
        logits = F.interpolate(logits.float(), size=image.size[::-1],
                               mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    def semantic_features(self, seg: np.ndarray) -> dict[str, float]:
        self._load_seg()
        total = seg.size
        enc = np.isin(seg, self._enc_ids).sum() / total
        opn = np.isin(seg, self._open_ids).sum() / total
        return {"wall_ceiling_ratio": _r(enc), "sky_open_ratio": _r(opn)}

    # ── 3.3 pose ───────────────────────────────────────────────────────────
    def pose(self, image_rgb: np.ndarray, seg: np.ndarray | None = None) -> PoseResult:
        model = self._load_pose()
        h, w = image_rgb.shape[:2]
        # ultralytics expects BGR ndarrays (OpenCV convention).
        res = model.predict(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                            conf=PERSON_CONF_MIN, device=str(self.device),
                            half=self.half, verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0:
            # Check if dense crowd where pose keypoints failed (Fix 1 fallback)
            crowd = 0.0
            if seg is not None and self._person_ids is not None and len(self._person_ids) > 0:
                seg_person_ratio = float(np.isin(seg, self._person_ids).mean())
                if seg_person_ratio > 0.25:
                    crowd = _clip01(np.sqrt(seg_person_ratio) * 0.8)
            return PoseResult(crowd_density=crowd)   # spec §7: centre focal point

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else np.zeros((len(boxes), 17, 2))
        kcf = (res.keypoints.conf.cpu().numpy()
               if res.keypoints is not None and res.keypoints.conf is not None
               else np.zeros((len(boxes), 17)))

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        n = len(boxes)

        # Primary subject = most salient person (area × confidence).
        primary = int(np.argmax(areas * confs))
        x1, y1, x2, y2 = boxes[primary]
        focal = (_clip01((x1 + x2) / 2 / w), _clip01((y1 + y2) / 2 / h))

        # Fix 1 — Crowd density: other people pressing in.
        # Combines YOLO detections with SegFormer person semantic segmentation ratio.
        # A single detected person is the trapped viewer, not a crowd (count factor = 0 for n = 1).
        area_ratio = _clip01(areas.sum() / (w * h))
        seg_person_ratio = 0.0
        if seg is not None and self._person_ids is not None and len(self._person_ids) > 0:
            seg_person_ratio = float(np.isin(seg, self._person_ids).mean())

        if n <= 1:
            crowd = 0.0
        else:
            yolo_crowd = np.sqrt(area_ratio) * (1.0 - np.exp(-(n - 1) / CROWD_COUNT_SCALE))
            # In dense crowds (e.g. testimg6), YOLO only detects a handful due to heavy occlusion;
            # SegFormer's person class accurately measures the full crowd footprint.
            seg_crowd = np.sqrt(seg_person_ratio) * min(1.0, (n - 1) / 2.0)
            crowd = _clip01(max(yolo_crowd, seg_crowd))

        return PoseResult(
            person_detected=True,
            person_count=n,
            crowd_density=crowd,
            body_compression=self._body_compression(kxy[primary], kcf[primary]),
            focal_point=focal,
            boxes=boxes, keypoints=kxy, keypoint_conf=kcf, primary_index=primary,
        )

    @staticmethod
    def _body_compression(kp: np.ndarray, conf: np.ndarray) -> float:
        """Limb-to-torso distance relative to torso length (Fix 2).

        Uses torso length (shoulder-to-hip) as the reference factor instead of
        shoulder width, providing orientation-invariant posture compression.
        Arms pulled in / tight posture → small ratio → compression → 1.
        """
        ok = conf >= KP_CONF_MIN
        sh_pts = [kp[i] for i in (KP_L_SHOULDER, KP_R_SHOULDER) if ok[i]]
        hip_pts = [kp[i] for i in (KP_L_HIP, KP_R_HIP) if ok[i]]

        if not sh_pts:
            return 0.0

        sh_mid = np.mean(sh_pts, axis=0)
        hip_mid = np.mean(hip_pts, axis=0) if hip_pts else None

        if hip_mid is not None:
            torso_len = float(np.linalg.norm(sh_mid - hip_mid))
            torso_c = (sh_mid + hip_mid) / 2.0
        else:
            torso_len = 0.0
            torso_c = sh_mid

        # Fallback to shoulder width if hips are not detected (e.g. bust framing)
        if torso_len < 4.0:
            if ok[KP_L_SHOULDER] and ok[KP_R_SHOULDER]:
                torso_len = float(np.linalg.norm(kp[KP_L_SHOULDER] - kp[KP_R_SHOULDER]))
            if torso_len < 4.0:
                return 0.0

        scores = []
        for idx, (tight, open_) in ((KP_L_WRIST, WRIST_RANGE), (KP_R_WRIST, WRIST_RANGE),
                                    (KP_L_ELBOW, ELBOW_RANGE), (KP_R_ELBOW, ELBOW_RANGE)):
            if ok[idx]:
                ratio = float(np.linalg.norm(kp[idx] - torso_c)) / torso_len
                scores.append(_ramp_down(ratio, tight, open_))
        return _clip01(float(np.mean(scores))) if scores else 0.0

    # ── 3.4 classical CV ───────────────────────────────────────────────────
    @staticmethod
    def edge_map(image_rgb: np.ndarray) -> np.ndarray:
        """Canny edges with median-based auto thresholds."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        med = float(np.median(gray))
        lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
        if hi <= lo:
            lo, hi = 50, 150
        return cv2.Canny(gray, lo, hi)

    @classmethod
    def cv_features(cls, image_rgb: np.ndarray,
                    focal: tuple[float, float],
                    seg: np.ndarray | None = None,
                    light_ids: np.ndarray | None = None,
                    enc_ids: np.ndarray | None = None) -> dict[str, float]:
        small = _resize_max_side(image_rgb, CV_WORK_MAX_SIDE)
        h, w = small.shape[:2]

        # Edge clutter in a disc around the focal point.
        edges = cls.edge_map(small)
        yy, xx = np.mgrid[:h, :w]
        cx, cy = focal[0] * w, focal[1] * h
        radius = CLUTTER_RADIUS_FRAC * min(h, w)
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        density = float((edges[disc] > 0).mean()) if disc.any() else 0.0
        edge_clutter = _clip01(density / CLUTTER_SATURATION)

        # HSV in [0, 1] (OpenCV uint8 hue is 0–179).
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue, sat, val = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0

        darkness = float((val < DARK_V_THRESHOLD).mean())

        # Raw color warmth: intensity-weighted red/orange/yellow mass vs blue + grey mass.
        chromatic = sat >= SAT_MIN_CHROMATIC
        warm = chromatic & ((hue <= 70) | (hue >= 330))
        cool = chromatic & (hue >= 180) & (hue <= 270)
        grey = ~chromatic
        warm_mass = float((val * sat)[warm].sum())
        cool_mass = float((val * sat)[cool].sum()) + float((val * SAT_MIN_CHROMATIC)[grey].sum())
        raw_warmth = warm_mass / (warm_mass + cool_mass + 1e-6)

        # Fix 3: Consider warmth from light sources and not inherent colour.
        # In a cosy bedroom, warmth comes from sunlight, windows, or lamps.
        # In a cave or narrow brick alley, warmth is purely inherent mineral/rock/brick albedo.
        has_light_source = False
        enc_ratio = 0.0
        if seg is not None:
            if light_ids is not None and len(light_ids) > 0:
                has_light_source = bool((np.isin(seg, light_ids)).sum() > 0.001 * seg.size)
            if enc_ids is not None and len(enc_ids) > 0:
                enc_ratio = float((np.isin(seg, enc_ids)).mean())

        if has_light_source:
            # Confirmed light source (sky, lamp, window, light, fireplace)
            light_source_warmth = raw_warmth
        else:
            # No explicit light source: check for diffuse warm lighting (bright scene, low clutter)
            # vs rough, textured, enclosing surfaces (cave rock, brick walls).
            dark_ok = _clip01((0.20 - darkness) / 0.15)
            clutter_ok = _clip01((0.40 - edge_clutter) / 0.20)
            diffuse_factor = dark_ok * clutter_ok * (1.0 - 0.7 * enc_ratio)
            light_source_warmth = raw_warmth * diffuse_factor

        return {"edge_clutter": _r(edge_clutter),
                "darkness_level": _r(darkness),
                "chromatic_warmth": _r(_clip01(light_source_warmth))}

    # ── aggregation ────────────────────────────────────────────────────────
    def extract_with_artifacts(self, img: ImageInput) -> tuple[dict[str, Any], Artifacts]:
        """Run all branches; return (Stage 1 payload, intermediate maps)."""
        pil = load_image(img)
        rgb = np.asarray(pil)
        h, w = rgb.shape[:2]

        closeness = self.depth_map(pil)
        seg = self.segmentation_map(pil)
        pose = self.pose(rgb, seg=seg)
        cv_feats = self.cv_features(rgb, pose.focal_point, seg=seg,
                                    light_ids=self._light_ids, enc_ids=self._enc_ids)

        payload = {
            "image_dimensions": {"width": w, "height": h},
            "focal_point": {"x": _r(pose.focal_point[0]), "y": _r(pose.focal_point[1])},
            "depth_features": self.depth_features(closeness),
            "semantic_features": self.semantic_features(seg),
            "pose_features": {
                "person_detected": pose.person_detected,
                "person_count": pose.person_count,
                "crowd_density": _r(pose.crowd_density),
                "body_compression": _r(pose.body_compression),
            },
            "cv_features": cv_feats,
        }
        artifacts = Artifacts(image_rgb=rgb, depth_closeness=closeness,
                              segmentation=seg, edges=self.edge_map(rgb), pose=pose)
        return payload, artifacts

        payload = {
            "image_dimensions": {"width": w, "height": h},
            "focal_point": {"x": _r(pose.focal_point[0]), "y": _r(pose.focal_point[1])},
            "depth_features": self.depth_features(closeness),
            "semantic_features": self.semantic_features(seg),
            "pose_features": {
                "person_detected": pose.person_detected,
                "person_count": pose.person_count,
                "crowd_density": _r(pose.crowd_density),
                "body_compression": _r(pose.body_compression),
            },
            "cv_features": cv_feats,
        }
        artifacts = Artifacts(image_rgb=rgb, depth_closeness=closeness,
                              segmentation=seg, edges=self.edge_map(rgb), pose=pose)
        return payload, artifacts

    def extract(self, img: ImageInput) -> dict[str, Any]:
        """Run all branches and return only the Stage 1 Feature Payload."""
        return self.extract_with_artifacts(img)[0]


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Stage 1 feature extractor")
    ap.add_argument("image", help="input photograph")
    ap.add_argument("-o", "--output", help="write payload JSON here")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--save-maps", metavar="DIR",
                    help="also dump depth/segmentation/edge maps as PNGs")
    args = ap.parse_args(argv)

    fx = FeatureExtractor(device=args.device)
    payload, art = fx.extract_with_artifacts(args.image)

    if args.save_maps:
        out = Path(args.save_maps)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image).stem
        cv2.imwrite(str(out / f"{stem}_depth.png"),
                    cv2.applyColorMap((art.depth_closeness * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
        seg_vis = ((art.segmentation.astype(np.uint32) * 2654435761) & 0xFFFFFF)
        seg_rgb = np.stack([(seg_vis >> s) & 0xFF for s in (16, 8, 0)], -1).astype(np.uint8)
        cv2.imwrite(str(out / f"{stem}_seg.png"), seg_rgb)
        cv2.imwrite(str(out / f"{stem}_edges.png"), art.edges)
        print(f"maps written to {out}/", file=sys.stderr)

    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
