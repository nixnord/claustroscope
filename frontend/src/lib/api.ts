/**
 * Claustroscope API Client: lib/api.ts
 *
 * Handles HTTP communication with the unified Kaggle FastAPI server.
 * Supports environment configuration via NEXT_PUBLIC_KAGGLE_API_URL,
 * with runtime localStorage override for seamless Cloudflare tunnel swapping.
 */

export interface AnalysisScores {
  trapping_base_T: number;
  raw_score_R: number;
  claustrophobia_score_C: number;
}

export interface FocalPoint {
  x: number;
  y: number;
}

export interface DepthFeatures {
  border_closeness: number;
  center_depth_gradient: number;
  vertical_depth_gradient: number;
  depth_variance: number;
  mean_closeness: number;
}

export interface SemanticFeatures {
  wall_ceiling_ratio: number;
  sky_open_ratio: number;
  floor_ratio: number;
  structural_enclosure: number;
}

export interface PoseFeatures {
  person_detected: boolean;
  person_count: number;
  crowd_density: number;
  body_compression: number;
}

export interface CvFeatures {
  edge_clutter: number;
  darkness_level: number;
  chromatic_warmth: number;
}

export interface FeaturePayload {
  image_dimensions: { width: number; height: number };
  focal_point: FocalPoint;
  depth_features: DepthFeatures;
  semantic_features: SemanticFeatures;
  pose_features: PoseFeatures;
  cv_features: CvFeatures;
}

export interface ModulationParameters {
  vignette_intensity: number;
  depth_warp_kappa: number;
  edge_jitter_amplitude: number;
  lens_distortion_k: number;
}

export interface DiffusionHyperparameters {
  denoising_strength: number;
  cfg_scale: number;
  num_inference_steps: number;
  controlnet_depth_scale: number;
  controlnet_edge_scale: number;
}

export interface Prompts {
  positive_prompt: string;
  negative_prompt: string;
}

export interface ControlPayload {
  scores: AnalysisScores;
  classification: "OPEN" | "MODERATE" | "CONFINED" | "CLAUSTROPHOBIC";
  focal_point: FocalPoint;
  dominant_confinement_factor: string;
  modulation_parameters: ModulationParameters;
  diffusion_hyperparameters: DiffusionHyperparameters;
  prompts: Prompts;
  explanation?: string;
}

export interface AnalysisImages {
  original: string;
  vignette: string;
  depth: string;
  canny: string;
  preview: string;
  final_canvas: string;
}

export interface AnalysisResponse {
  elapsed_time: number;
  scores: AnalysisScores;
  classification: "OPEN" | "MODERATE" | "CONFINED" | "CLAUSTROPHOBIC";
  focal_point: FocalPoint;
  features: FeaturePayload;
  control: ControlPayload;
  images: AnalysisImages;
}

export interface HealthStatus {
  status: string;
  gpu_active: boolean;
  device: string;
  vram_allocated_mb: number;
  models_loaded: string[];
}

const STORAGE_KEY = "claustroscope_tunnel_url";

/**
 * Returns the active API base URL.
 * Prioritizes localStorage override, then NEXT_PUBLIC_KAGGLE_API_URL, defaulting to localhost:7860.
 */
export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && saved.trim()) {
      return saved.trim().replace(/\/+$/, "");
    }
  }

  const envUrl = process.env.NEXT_PUBLIC_KAGGLE_API_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, "");
  }

  return "http://127.0.0.1:7860";
}

/**
 * Saves a custom tunnel URL override into localStorage.
 */
export function setCustomTunnelUrl(url: string): void {
  if (typeof window !== "undefined") {
    const cleaned = url.trim().replace(/\/+$/, "");
    if (cleaned) {
      localStorage.setItem(STORAGE_KEY, cleaned);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }
}

/**
 * Clears any custom tunnel URL override from localStorage.
 */
export function clearCustomTunnelUrl(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(STORAGE_KEY);
  }
}

/**
 * Checks server health and GPU availability on the active backend URL.
 */
export async function checkServerHealth(): Promise<HealthStatus> {
  const baseUrl = getApiBaseUrl();
  const resp = await fetch(`${baseUrl}/api/health`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!resp.ok) {
    throw new Error(`Server returned HTTP ${resp.status}`);
  }

  return resp.json();
}

export interface AnalyzeOptions {
  applyLensDistortion?: boolean;
  distortionMode?: "barrel" | "tunnel";
}

/**
 * Sends an image file to the unified Kaggle server for complete 4-stage processing.
 * Note: Subject description is hardcoded to "a photograph of the scene" per design requirements.
 */
export async function analyzeImage(
  imageFile: File | Blob,
  filename = "upload.jpg",
  options?: AnalyzeOptions
): Promise<AnalysisResponse> {
  const baseUrl = getApiBaseUrl();
  const formData = new FormData();

  formData.append("image", imageFile, filename);
  formData.append(
    "apply_lens_distortion",
    String(options?.applyLensDistortion ?? true)
  );
  formData.append("distortion_mode", options?.distortionMode ?? "barrel");
  // Hardcoded default prompt context as specified in requirements
  formData.append("subject", "a photograph of the scene");

  const resp = await fetch(`${baseUrl}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    let errorDetail = `HTTP ${resp.status}`;
    try {
      const errJson = await resp.json();
      if (errJson && errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(`Inference failed: ${errorDetail}`);
  }

  return resp.json();
}
