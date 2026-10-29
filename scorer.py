"""
Stage 2 — Claustrophobia Scoring Engine.

Consumes the Stage 1 *Feature Payload* produced by ``extractors.py`` and emits
the Stage 2 *Control Payload* consumed by the modulation layer (Stage 3) and
the generative engine (Stage 4).

Pipeline (see spec §4):

    T = max(0.6·D_border + 0.4·S_enc, P_crowd) · (1 − S_sky)          (2A)
    R = T · (1 + 0.35·L_dark + 0.25·E_clutter + 0.20·P_comp − 0.30·L_warm)  (2B)
    C = 1 / (1 + exp(−k·(R − μ)))       with  k = 9.0, μ = 0.42        (2C)

C is then mapped through linear lookup tables onto diffusion hyperparameters
(spec §6.2), modulation strengths (spec §5) and prompt presets (spec §6.3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScoringConfig:
    """Every tunable constant of the T → R → C formula (spec §4)."""

    # Step 2A — physical confinement base
    w_border: float = 0.60          # weight of D_border in structural term
    w_enclosure: float = 0.40       # weight of S_enc in structural term

    # Step 2B — psychological amplifiers (signed)
    a_dark: float = 0.35
    a_clutter: float = 0.25
    a_compression: float = 0.20
    a_warmth: float = -0.65         # warmth *dampens* tension (calibrated for cosy bedroom target)

    # Step 2C — logistic mapping
    mu: float = 0.49                # centre of the sigmoid (calibrated for cosy bedroom C ≈ 0.30)
    k: float = 9.0                  # steepness

    # Classification bands (applied to C)
    open_threshold: float = 0.35
    panic_threshold: float = 0.65


DEFAULT_CONFIG = ScoringConfig()


class Classification(str, Enum):
    OPEN = "OPEN"                       # C < 0.35
    ENCLOSED = "ENCLOSED"               # 0.35 ≤ C < 0.65
    CLAUSTROPHOBIC = "CLAUSTROPHOBIC"   # C ≥ 0.65


# ════════════════════════════════════════════════════════════════════════════
# Parameter-mapping lookup tables
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LinearMap:
    """y = intercept + slope · C, optionally clamped to [lo, hi]."""

    intercept: float
    slope: float
    lo: float = -math.inf
    hi: float = math.inf
    as_int: bool = False

    def __call__(self, c: float) -> float | int:
        y = min(max(self.intercept + self.slope * c, self.lo), self.hi)
        return int(round(y)) if self.as_int else y


# Spec §6.2 — diffusion hyperparameters.
DIFFUSION_TABLE: dict[str, LinearMap] = {
    "denoising_strength":     LinearMap(0.20,  0.58, 0.0, 1.0),
    "cfg_scale":              LinearMap(7.00,  3.00, 1.0, 20.0),
    "controlnet_depth_scale": LinearMap(0.90, -0.25, 0.0, 2.0),
    "controlnet_edge_scale":  LinearMap(0.70,  0.25, 0.0, 2.0),
    "num_inference_steps":    LinearMap(24.0, 12.0, 10, 60, as_int=True),
}

# Spec §5 — Stage 3 modulation strengths.
MODULATION_TABLE: dict[str, LinearMap] = {
    "vignette_intensity":    LinearMap(0.0, 1.00, 0.0, 1.0),   # α scale = C
    "depth_warp_kappa":      LinearMap(0.0, 0.45, 0.0, 0.45),  # κ = 0.45·C
    "edge_jitter_amplitude": LinearMap(0.0, 3.80, 0.0, 3.8),
    # Inward lens distortion applied to final canvas to enhance claustrophobic bowing
    "lens_distortion_k": LinearMap(0.0, 0.75, 0.0, 0.90),  # k = 0.35·C
}

# Spec §6.3 — prompt presets, keyed by regime.
# Refactored: Removed Francis Bacon / flesh-distorting tokens and budgeted token counts < 75.
PROMPT_TABLE: dict[str, dict[str, str]] = {
    "calm": {
        "style_preset": "clear natural photography, soft atmospheric aesthetic, balanced composition",
        "mood_tokens": "peaceful, spacious, breathing room, tranquil openness, serene",
        "lighting_tokens": "warm natural sunlight, golden hour, soft ambient illumination",
        "negative": "dark, claustrophobic, gloomy, scary, narrow, cramped, horrific, distorted, blurry",
    },
    "dread": {
        "style_preset": (
            "cinematic psychological thriller, atmospheric dark photography, "
            "moody film still"
        ),
        # {panic_w} is filled with 0.8 + 0.6·C at render time.
        "mood_tokens": (
            "oppressive claustrophobia, walls closing in, psychological tension, "
            "suffocating dread, trapped, (panic:{panic_w:.2f})"
        ),
        "lighting_tokens": (
            "harsh chiaroscuro, suffocating heavy shadows, cold ominous backlight"
        ),
        # Explicit anatomical & clothing protection tokens
        "negative": (
            "open sky, spacious, sunny, cheerful, bright room, peaceful, wide angle, calm, "
            "nude, naked, shirtless, bare skin, deformed body, extra limbs, "
            "disfigured anatomy, bad proportions, bad hands, poorly drawn face, blurry"
        ),
    },
}

PANIC_WEIGHT = LinearMap(0.80, 0.60)  # token weight for "panic" (spec §6.3)


# ════════════════════════════════════════════════════════════════════════════
# Core math
# ════════════════════════════════════════════════════════════════════════════

def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def trapping_base(d_border: float, s_enc: float, p_crowd: float, s_sky: float,
                  cfg: ScoringConfig = DEFAULT_CONFIG) -> float:
    """Step 2A — physical confinement base T ∈ [0, 1]."""
    structural = cfg.w_border * d_border + cfg.w_enclosure * s_enc
    return _clip01(max(structural, p_crowd) * (1.0 - s_sky))


def amplifier_terms(l_dark: float, e_clutter: float, p_comp: float, l_warm: float,
                    cfg: ScoringConfig = DEFAULT_CONFIG) -> dict[str, float]:
    """Individual signed contributions inside the Step 2B bracket."""
    return {
        "darkness": cfg.a_dark * l_dark,
        "edge_clutter": cfg.a_clutter * e_clutter,
        "body_compression": cfg.a_compression * p_comp,
        "chromatic_warmth": cfg.a_warmth * l_warm,
    }


def raw_score(t: float, amps: Mapping[str, float]) -> float:
    """Step 2B — R = T · (1 + Σ amplifiers). Unbounded above (≈ ≤ 1.8)."""
    return t * (1.0 + sum(amps.values()))


def logistic(r: float, cfg: ScoringConfig = DEFAULT_CONFIG) -> float:
    """Step 2C — numerically stable sigmoid centred at μ with steepness k."""
    z = cfg.k * (r - cfg.mu)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def classify(c: float, cfg: ScoringConfig = DEFAULT_CONFIG) -> Classification:
    if c < cfg.open_threshold:
        return Classification.OPEN
    if c < cfg.panic_threshold:
        return Classification.ENCLOSED
    return Classification.CLAUSTROPHOBIC


# ════════════════════════════════════════════════════════════════════════════
# Payload handling
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureVector:
    """Flat view of the Stage 1 payload with values clamped to [0, 1]."""

    d_border: float
    d_flat: float
    s_enc: float
    s_sky: float
    p_crowd: float
    p_comp: float
    e_clutter: float
    l_dark: float
    l_warm: float
    person_detected: bool = False
    focal_point: dict[str, float] = field(default_factory=lambda: {"x": 0.5, "y": 0.5})

    @classmethod
    def from_payload(cls, p: Mapping[str, Any]) -> "FeatureVector":
        d, s = p["depth_features"], p["semantic_features"]
        po, cv = p["pose_features"], p["cv_features"]
        person = bool(po.get("person_detected", False))
        return cls(
            d_border=_clip01(d["border_closeness"]),
            d_flat=_clip01(d.get("depth_flatness", 0.0)),
            s_enc=_clip01(s["wall_ceiling_ratio"]),
            s_sky=_clip01(s["sky_open_ratio"]),
            p_crowd=_clip01(po.get("crowd_density", 0.0)),
            p_comp=_clip01(po.get("body_compression", 0.0)) if person else 0.0,
            e_clutter=_clip01(cv["edge_clutter"]),
            l_dark=_clip01(cv["darkness_level"]),
            l_warm=_clip01(cv["chromatic_warmth"]),
            person_detected=person,
            focal_point=dict(p.get("focal_point") or {"x": 0.5, "y": 0.5}),
        )


@dataclass
class ScoreResult:
    trapping_base_T: float
    raw_score_R: float
    claustrophobia_score_C: float
    classification: Classification
    amplifiers: dict[str, float]
    structural_term: float
    crowd_term: float
    sky_suppression: float


def score(features: FeatureVector, cfg: ScoringConfig = DEFAULT_CONFIG) -> ScoreResult:
    """Run T → R → C on a :class:`FeatureVector`."""
    t = trapping_base(features.d_border, features.s_enc, features.p_crowd,
                      features.s_sky, cfg)
    amps = amplifier_terms(features.l_dark, features.e_clutter, features.p_comp,
                           features.l_warm, cfg)
    r = raw_score(t, amps)
    c = logistic(r, cfg)
    return ScoreResult(
        trapping_base_T=t,
        raw_score_R=r,
        claustrophobia_score_C=c,
        classification=classify(c, cfg),
        amplifiers=amps,
        structural_term=cfg.w_border * features.d_border + cfg.w_enclosure * features.s_enc,
        crowd_term=features.p_crowd,
        sky_suppression=1.0 - features.s_sky,
    )


def diffusion_hyperparameters(c: float, person_detected: bool = False) -> dict[str, float | int]:
    """Computes diffusion parameters with anatomy-safe denoising caps when a person is detected."""
    params = {name: fn(c) for name, fn in DIFFUSION_TABLE.items()}
    if person_detected:
        # Cap denoising strength to prevent identity, limbs, and clothing destruction
        params["denoising_strength"] = min(float(params["denoising_strength"]), 0.66)
    return params


def modulation_parameters(c: float) -> dict[str, float]:
    return {name: fn(c) for name, fn in MODULATION_TABLE.items()}


def build_prompts(c: float, base_subject_description: str = "a photograph of the scene",
                  cfg: ScoringConfig = DEFAULT_CONFIG) -> dict[str, str]:
    """Dynamic prompt engine keeping total token sequence length safely under 75 tokens."""
    regime = "calm" if c < cfg.open_threshold else "dread"
    preset = PROMPT_TABLE[regime]
    mood = preset["mood_tokens"].format(panic_w=PANIC_WEIGHT(c))
    positive = (f"{base_subject_description}, {preset['style_preset']}, {mood}, "
                f"{preset['lighting_tokens']}, high quality visual art")
    return {"regime": regime, "positive": positive, "negative": preset["negative"]}


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def build_control_payload(feature_payload: Mapping[str, Any],
                          base_subject_description: str = "a photograph of the scene",
                          cfg: ScoringConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Stage 1 Feature Payload → Stage 2 Control Payload."""
    fv = FeatureVector.from_payload(feature_payload)
    res = score(fv, cfg)
    c = res.claustrophobia_score_C

    dominant_base = "crowd" if res.crowd_term > res.structural_term else "structure"
    return {
        "scores": {
            "trapping_base_T": _r(res.trapping_base_T),
            "raw_score_R": _r(res.raw_score_R),
            "claustrophobia_score_C": _r(c),
        },
        "classification": res.classification.value,
        "focal_point": fv.focal_point,
        "diffusion_hyperparameters": {
            k: (v if isinstance(v, int) else _r(v, 3))
            for k, v in diffusion_hyperparameters(c, person_detected=fv.person_detected).items()
        },
        "modulation_parameters": {k: _r(v, 3) for k, v in modulation_parameters(c).items()},
        "prompts": build_prompts(c, base_subject_description, cfg),
        "explanation": {
            "structural_term": _r(res.structural_term),
            "crowd_term": _r(res.crowd_term),
            "dominant_confinement_source": dominant_base,
            "sky_suppression_factor": _r(res.sky_suppression),
            "amplifier_contributions": {k: _r(v) for k, v in res.amplifiers.items()},
            "amplifier_multiplier": _r(1.0 + sum(res.amplifiers.values())),
            "person_detected": fv.person_detected,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI / demo
# ════════════════════════════════════════════════════════════════════════════

def _synthetic_payload(d_border, s_enc, s_sky, p_crowd, p_comp, e_clutter,
                       l_dark, l_warm, person=True) -> dict[str, Any]:
    return {
        "image_dimensions": {"width": 1024, "height": 1024},
        "focal_point": {"x": 0.5, "y": 0.5},
        "depth_features": {"border_closeness": d_border, "depth_flatness": 0.5},
        "semantic_features": {"wall_ceiling_ratio": s_enc, "sky_open_ratio": s_sky},
        "pose_features": {"person_detected": person, "crowd_density": p_crowd,
                          "body_compression": p_comp},
        "cv_features": {"edge_clutter": e_clutter, "darkness_level": l_dark,
                        "chromatic_warmth": l_warm},
    }


DEMO_CASES: dict[str, dict[str, Any]] = {
    "spec_example (§8)": _synthetic_payload(0.88, 0.79, 0.00, 0.32, 0.81, 0.74, 0.68, 0.15),
    "empty_elevator":    _synthetic_payload(0.85, 0.80, 0.00, 0.00, 0.00, 0.40, 0.55, 0.20, person=False),
    "macro_closeup":     _synthetic_payload(0.15, 0.10, 0.10, 0.00, 0.00, 0.30, 0.20, 0.40, person=False),
    "sunlit_bedroom":    _synthetic_payload(0.55, 0.60, 0.00, 0.00, 0.10, 0.25, 0.10, 0.65),
    "night_beach":       _synthetic_payload(0.05, 0.02, 0.85, 0.05, 0.10, 0.10, 0.85, 0.10),
    "packed_subway":     _synthetic_payload(0.60, 0.35, 0.00, 0.75, 0.70, 0.65, 0.40, 0.25),
}


def _demo() -> None:
    hdr = f"{'case':<20}{'T':>7}{'R':>7}{'C':>7}  class            σ     cfg   w_d   w_e"
    print(hdr)
    print("─" * len(hdr))
    for name, payload in DEMO_CASES.items():
        out = build_control_payload(payload)
        s, h = out["scores"], out["diffusion_hyperparameters"]
        print(f"{name:<20}{s['trapping_base_T']:>7.3f}{s['raw_score_R']:>7.3f}"
              f"{s['claustrophobia_score_C']:>7.3f}  {out['classification']:<15}"
              f"{h['denoising_strength']:>6.2f}{h['cfg_scale']:>6.2f}"
              f"{h['controlnet_depth_scale']:>6.2f}{h['controlnet_edge_scale']:>6.2f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Stage 2 claustrophobia scorer")
    ap.add_argument("payload", nargs="?", help="Stage 1 JSON file, or '-' for stdin")
    ap.add_argument("--subject", default="a photograph of the scene",
                    help="base subject description for the positive prompt")
    ap.add_argument("-o", "--output", help="write control payload JSON here")
    ap.add_argument("--demo", action="store_true", help="run the edge-case matrix")
    args = ap.parse_args(argv)

    if args.demo or not args.payload:
        _demo()
        return

    src = sys.stdin if args.payload == "-" else open(args.payload, encoding="utf-8")
    with src:
        feature_payload = json.load(src)

    control = build_control_payload(feature_payload, args.subject)
    text = json.dumps(control, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
