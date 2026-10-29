# ClaustroScope

ClaustroScope is an artificial intelligence pipeline that evaluates how claustrophobic an image looks and regenerates it into a psychologically distorted canvas reflecting the state of mind of someone trapped inside.

If a photograph shows an open place (like a park or beach), the pipeline preserves its natural perspective. If a photograph shows an enclosed, crowded, or dark space (like a crawlspace, cave squeeze, narrow alley, or packed subway), the pipeline bends the walls inward, darkens the outer margins into tunnel vision, and uses generative diffusion to recreate the feeling of acute confinement.

---

## Architecture Overview

The system is organized into four sequential stages, connected through a web application.

```
[ Input Photograph ]
         |
         v
[ Stage 1: Feature Extraction ]
  Depth Anything V2, SegFormer ADE20K, YOLO11-Pose, OpenCV
         |
         v
[ Stage 2: Claustrophobia Scoring ]
  Physical Confinement (T) -> Amplifiers (R) -> S-Curve Score (C)
         |
         v
[ Stage 3: Conditioning & Spatial Modulation ]
  Radial Vignette, Depth Warping, Edge Jitter, Lens Distortion
         |
         v
[ Stage 4: Generative Multi-ControlNet Diffusion ]
  Realistic Vision v6.0 with Depth & Canny ControlNets
         |
         v
[ Final Transformed Canvas ]
```

---

## Pipeline Stages

### Stage 1: Feature Extraction
The system inspects the input photograph using four vision tools to extract eight normalized values between 0.0 and 1.0:
* **Depth Anything V2 Small:** Measures 3D distance. It calculates Border Closeness (how close ceilings and side walls are to the camera lens) and Depth Flatness (lack of distance variation, like facing a flat wall).
* **SegFormer (ADE20K):** Identifies semantic scene categories. It calculates Wall & Ceiling Ratio (enclosing structures) and Sky & Open Ratio (visible sky, water, or open fields).
* **YOLO11-Pose:** Locates people and tracks 17 bodily keypoints. It calculates Body Compression (limbs huddled close to the torso) and Crowd Density (multiple occluded people packing the frame).
* **OpenCV:** Computes Darkness Level (shadow submersion in HSV space), Edge Clutter (jagged high-frequency detail via Canny filtering), and Chromatic Warmth (warm light emitted from lamps and windows).

### Stage 2: Claustrophobia Scoring
A deterministic mathematical formula combines the extracted features into a single score C between 0.000 and 1.000:
1. **Physical Confinement Base (T):** Evaluates if you are physically blocked by walls or crowded by people, counterbalanced by visible open sky.
   ```
   T = max((0.60 * D_border + 0.40 * S_enc), P_crowd) * (1.0 - S_sky)
   ```
2. **Atmospheric Amplifiers (R):** Darkness, surface clutter, and huddled postures increase tension, while warm lamp lighting reduces it.
   ```
   R = T * (1.0 + 0.35 * L_dark + 0.25 * E_clutter + 0.20 * P_comp - 0.65 * L_warm)
   ```
3. **Logistic S-Curve Mapping (C):** Converts R into a final score C centered at threshold 0.49 with steepness 9.0.
   ```
   C = 1 / (1 + exp(-9.0 * (R - 0.49)))
   ```

* **OPEN (C < 0.35):** Wide natural spaces. Geometry remains undistorted.
* **ENCLOSED (0.35 <= C < 0.65):** Standard rooms and hallways. Mild framing is applied.
* **CLAUSTROPHOBIC (C >= 0.65):** Tight spaces and heavy crowds. Full optical and generative horror distortion is applied.

### Stage 3: Spatial Conditioning & Modulation
Before sending data to the image generator, OpenCV functions warp the guiding maps according to score C:
* **Radial Vignette:** Darkens the outer perimeter into an elliptical tunnel vision mask centered on the detected person.
* **Inward Depth Warping:** Coordinate remapping bends vertical walls and ceilings inward over the viewer.
* **Edge Jitter:** Adds random spatial displacement to background edges while preserving the primary person.
* **Barrel Lens Distortion:** Applies non-linear optical bowing that pulls walls into a fishbowl dome and adds subtle peripheral chromatic color splitting when C >= 0.35.

### Stage 4: Generative Multi-ControlNet Diffusion
The system sends the vignetted photo, warped depth map, and jittered edge lines to a Stable Diffusion v1.5 model (Realistic Vision v6.0) guided by two ControlNet adapters (Depth and Canny):
* Diffusion parameters (denoising strength, guidance scale, steps, and prompt weights) scale dynamically with score C.
* Generates an expressive visual canvas reflecting internal psychological dread.
* Can run locally for fast optical previews (under 0.5 seconds) or connect to a remote Kaggle GPU worker for full diffusion rendering.

---

## Web Dashboard

The application includes a single-column web interface designed for clarity:
* **Landing Page:** Simple upload area with five one-click sample scenes (Narrow Alley, Cave Squeeze, Crawlspace, Packed Crowd, Open Park).
* **Loading Screen:** Animated particle splashscreen with live elapsed timer while models run.
* **Results Flow:** Single vertical scroll showing the transformed canvas with an interactive Before / After split slider, the numeric fear score and classification, eight feature progress bars, four intermediate conditioning maps, and the exact diffusion prompts used.

---

## Installation & Setup

### Requirements
* Python 3.10 or 3.11
* PyTorch with CUDA support (or CPU)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/nixnord/claustroscope.git
   cd claustroscope
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the web server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
5. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Optional: Remote Kaggle GPU Acceleration
To use full Stable Diffusion rendering on a free Kaggle GPU:
1. Open a Kaggle notebook with GPU enabled.
2. Run `kaggle_gradio_server.py`.
3. Copy the generated public tunnel URL (e.g., `https://xxxx.trycloudflare.com`).
4. Add it to a `.env` file in the project root:
   ```env
   KAGGLE_TUNNEL_URL=https://xxxx.trycloudflare.com
   ```
5. Restart the server. If the remote tunnel is offline, the app automatically falls back to local preview mode.

---

## Deploying on Vercel (Hybrid Cloud Architecture)

Because Stages 1 to 3 use neural vision models (Depth Anything V2, SegFormer, YOLO11-Pose) requiring GPU or multi-gigabyte memory, you can host the interactive web dashboard on Vercel while running the primary inference engine locally on your computer forwarded via Cloudflare Tunnel.

### How the Hybrid Setup Works
```
[ User on Vercel (claustroscope.vercel.app) ]
                     |
                     | (Direct API requests /api/analyze)
                     v
[ Cloudflare Tunnel: PRIMARY_ENGINE_URL ]
                     |
                     v
[ Your Local Machine: app.py on port 8000 ]
                     |
                     +---> (Optional) Kaggle GPU: KAGGLE_TUNNEL_URL
```

### Steps to Deploy
1. Start your local processing engine:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```
2. In a separate terminal, forward your local server to a public Cloudflare tunnel:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
   Copy the generated HTTPS URL (e.g. `https://random-words.trycloudflare.com`).
3. Deploy this repository to Vercel:
   * Import the repository on Vercel.
   * Under Project Settings -> Environment Variables, add:
     * `PRIMARY_ENGINE_URL`: Your forwarded Cloudflare URL (e.g. `https://random-words.trycloudflare.com`).
     * `KAGGLE_TUNNEL_URL` (optional): Active Kaggle GPU tunnel URL.
4. Deploy. The Vercel app will automatically connect to your forwarded local processing engine. You can also adjust or test the tunnel URL anytime via the settings gear icon in the header.

---

## Repository Structure

* `app.py`: FastAPI backend application and endpoints.
* `index.html`: Web interface, comparison slider, and telemetry display.
* `vercel.json`: Vercel project configuration.
* `api/config.js`: Vercel Edge function providing environment variables to the client.
* `extractors.py`: Stage 1 perception models (Depth Anything V2, SegFormer, YOLO11-Pose, OpenCV).
* `scorer.py`: Stage 2 claustrophobia scoring formulas and parameter tables.
* `modulators.py`: Stage 3 OpenCV spatial warps, vignettes, and lens distortion.
* `pipeline.py`: Stage 4 master orchestration script and CLI runner.
* `client_pipeline.py`: Bridge client connecting local runs to remote Kaggle GPU.
* `kaggle_gradio_server.py`: Remote server script for Multi-ControlNet diffusion.
* `run_tests.py`: Diagnostic batch test suite.
* `requirements.txt`: Python package requirements.
* `Dockerfile`: Container configuration for cloud hosting.
* `test/`: Sample input photographs for testing.
