"""Batch test: run extractors + scorer over test/ and build a diagnostic contact sheet."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from extractors import FeatureExtractor  # noqa: E402
from modulators import apply_radial_vignette, distort_final_canvas  # noqa: E402
from scorer import build_control_payload  # noqa: E402

IN_DIR = ROOT / "test"
OUT = ROOT / "test_results"
OUT.mkdir(exist_ok=True)
TILE = 320

ENC_COLOR = (60, 60, 230)    # BGR red  = enclosure classes
OPEN_COLOR = (230, 160, 40)  # BGR blue = open classes


def tile(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = TILE / max(h, w)
    img = cv2.resize(img_bgr, (int(w * s), int(h * s)))
    canvas = np.zeros((TILE, TILE, 3), np.uint8)
    y, x = (TILE - img.shape[0]) // 2, (TILE - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def label(img: np.ndarray, lines: list[str], color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    for i, t in enumerate(lines):
        cv2.putText(img, t, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
        cv2.putText(img, t, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
    return img


def main():
    fx = FeatureExtractor()
    rows, summary = [], []
    imgs = sorted(p for p in IN_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in imgs:
        t0 = time.perf_counter()
        payload, art = fx.extract_with_artifacts(p)
        dt = time.perf_counter() - t0
        ctrl = build_control_payload(payload)
        (OUT / f"{p.stem}_features.json").write_text(json.dumps(payload, indent=2))
        (OUT / f"{p.stem}_control.json").write_text(json.dumps(ctrl, indent=2))

        bgr = cv2.cvtColor(art.image_rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        # pose overlay
        pose_vis = bgr.copy()
        for i, b in enumerate(art.pose.boxes):
            c = (0, 255, 0) if i == art.pose.primary_index else (0, 200, 255)
            cv2.rectangle(pose_vis, tuple(map(int, b[:2])), tuple(map(int, b[2:])), c, 2)
        for kp, kc in zip(art.pose.keypoints, art.pose.keypoint_conf):
            for (x, y), cf in zip(kp, kc):
                if cf > 0.4:
                    cv2.circle(pose_vis, (int(x), int(y)), 3, (255, 0, 255), -1)
        fx_, fy_ = payload["focal_point"]["x"] * w, payload["focal_point"]["y"] * h
        cv2.drawMarker(pose_vis, (int(fx_), int(fy_)), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
        # depth
        depth_vis = cv2.applyColorMap((art.depth_closeness * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        # semantic overlay
        seg_vis = bgr.copy() // 3
        seg_vis[np.isin(art.segmentation, fx._enc_ids)] = ENC_COLOR
        seg_vis[np.isin(art.segmentation, fx._open_ids)] = OPEN_COLOR
        edges_vis = cv2.cvtColor(art.edges, cv2.COLOR_GRAY2BGR)

        # Module 3 Vignette + Lens Distortion preview
        fp = ctrl["focal_point"]
        mod_p = ctrl["modulation_parameters"]
        vignetted = apply_radial_vignette(bgr, fp, mod_p["vignette_intensity"])
        preview_distorted = distort_final_canvas(vignetted, ctrl)

        d, s, po, cv = (payload[k] for k in ("depth_features", "semantic_features",
                                             "pose_features", "cv_features"))
        sc = ctrl["scores"]
        cls = ctrl["classification"]
        col = {"OPEN": (120, 255, 120), "ENCLOSED": (0, 220, 255), "CLAUSTROPHOBIC": (80, 80, 255)}[cls]
        row = np.hstack([
            label(tile(pose_vis), [p.name, f"C={sc['claustrophobia_score_C']:.3f} {cls}",
                                    f"T={sc['trapping_base_T']:.3f} R={sc['raw_score_R']:.3f}"], col),
            label(tile(depth_vis), ["depth (bright=near)", f"border={d['border_closeness']:.2f}",
                                    f"flat={d['depth_flatness']:.2f}"]),
            label(tile(seg_vis), ["seg red=enc blue=open", f"enc={s['wall_ceiling_ratio']:.2f}",
                                  f"open={s['sky_open_ratio']:.2f}"]),
            label(tile(edges_vis), ["canny", f"clutter={cv['edge_clutter']:.2f}",
                                    f"dark={cv['darkness_level']:.2f} warm={cv['chromatic_warmth']:.2f}",
                                    f"people={po['person_count']} crowd={po['crowd_density']:.2f}",
                                    f"comp={po['body_compression']:.2f}"]),
            label(tile(preview_distorted), ["lens distortion preview", f"k={mod_p['lens_distortion_k']:.3f}",
                                            f"vignette={mod_p['vignette_intensity']:.2f}"]),
        ])
        rows.append(row)
        summary.append((p.name, payload, ctrl, dt))

    cv2.imwrite(str(OUT / "contact_sheet.png"), np.vstack(rows))

    print(f"{'image':<15}{'Dbord':>6}{'Senc':>6}{'Ssky':>6}{'Pcrw':>6}{'Pcmp':>6}{'Eclt':>6}"
          f"{'Ldrk':>6}{'Lwrm':>6} |{'T':>6}{'R':>6}{'C':>7}  class           time")
    for name, pl, ct, dt in summary:
        d, s, po, cv, sc = (pl["depth_features"], pl["semantic_features"], pl["pose_features"],
                            pl["cv_features"], ct["scores"])
        print(f"{name:<15}{d['border_closeness']:>6.2f}{s['wall_ceiling_ratio']:>6.2f}"
              f"{s['sky_open_ratio']:>6.2f}{po['crowd_density']:>6.2f}{po['body_compression']:>6.2f}"
              f"{cv['edge_clutter']:>6.2f}{cv['darkness_level']:>6.2f}{cv['chromatic_warmth']:>6.2f} |"
              f"{sc['trapping_base_T']:>6.3f}{sc['raw_score_R']:>6.3f}{sc['claustrophobia_score_C']:>7.3f}"
              f"  {ct['classification']:<15}{dt:5.2f}s  (n={po['person_count']})")


if __name__ == "__main__":
    main()
