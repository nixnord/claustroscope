# ClaustroScope
https://attended-bernard-bradford-non.trycloudflare.com/
ClaustroScope is an artificial intelligence pipeline that evaluates how claustrophobic an image looks and regenerates it into a psychologically distorted canvas reflecting the state of mind of someone trapped inside.

If a photograph shows an open place (like a park or beach), the pipeline preserves its natural perspective. If a photograph shows an enclosed, crowded, or dark space (like a crawlspace, cave squeeze, narrow alley, or packed subway), the pipeline bends the walls inward, darkens the outer margins into tunnel vision, and uses generative diffusion to recreate the feeling of acute confinement.

---

## Visual Showcase

### Landing & Upload Interface
The landing page allows users to drag and drop any photograph or select one of five pre-loaded sample scenes (Narrow Alley, Cave Squeeze, Crawlspace, Packed Crowd, Open Park). Users can optionally customize the prompt description and toggle optical lens distortion.

![Landing and Upload Interface](img1.jpeg)

### Focal Anchor Detection & Confinement Scoring
Once analyzed, the dashboard presents the scene with the detected focal crosshair anchor centered on the primary subject, alongside the real-time calculated Claustrophobia Score (C = 0.857), classification state (CLAUSTROPHOBIC), and foundational confinement metrics (T, R, and dominant source).

![Interactive Results View with Original Scene](img3.jpeg)

### Transformed Claustrophobic Diffusion Canvas
An interactive Before / After split slider lets users sweep across the scene to inspect the generative transformation. Guided by Depth and Canny ControlNets, the model bends structural boundaries inward and compresses peripheral space into an intense claustrophobic canvas.

![Transformed Diffusion Canvas](img4.jpeg)

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
The system passes the vignetted photo, warped depth map, and jittered edge lines to a Stable Diffusion v1.5 model (DreamShaper 8) guided by two ControlNet adapters (Depth and Canny):
* Diffusion parameters (denoising strength, guidance scale, steps, and prompt weights) scale dynamically with score C.
* Generates an expressive visual canvas reflecting internal psychological dread.
* Can run in unified GPU mode directly inside Kaggle, locally on workstation GPU, or fallback to fast optical preview.

---

## Web Dashboard

The application features a single-column vertical web interface designed for clarity and psychological immersion:
* **Landing Page:** Minimal upload interface with five one-click sample scenes (Narrow Alley, Cave Squeeze, Crawlspace, Packed Crowd, Open Park).
* **Loading Screen:** Animated particle splashscreen with live elapsed timer while neural models execute.
* **Results Flow:** Single vertical scroll displaying the transformed canvas with an interactive Before / After split slider, numeric fear score and classification, eight feature progress bars, four intermediate conditioning maps, and synthesized diffusion prompts.

---

## Deployment Architecture: Unified Kaggle GPU + Vercel Frontend

ClaustroScope unifies the complete end-to-end processing pipeline—feature extraction, scoring, conditioning, DreamShaper 8 Multi-ControlNet diffusion, and barrel lens distortion—into a single high-performance engine hosted on a Kaggle GPU (T4 or P100), connected to the interactive frontend hosted on Vercel.

```
+-------------------------------------------------------------------------+
|                              USER CLIENT                                |
|             Browser visiting https://your-project.vercel.app            |
+-------------------------------------------------------------------------+
       |                                                   |
       | 1. GET /api/config                                | 2. Direct POST /api/analyze
       |    (Reads PRIMARY_ENGINE_URL)                     |    (Bypasses Vercel limits)
       v                                                   v
+-------------------------------+       +---------------------------------+
|       VERCEL EDGE TIER        |       |       CLOUDFLARE TUNNEL         |
|  - Static Assets (HTML/CSS/JS)|       |  https://xxxx.trycloudflare.com |
|  - Edge Route (/api/config)   |       +---------------------------------+
+-------------------------------+                          |
                                                           v
                                        +---------------------------------+
                                        |    UNIFIED KAGGLE GPU ENGINE    |
                                        |  Kaggle Notebook (T4 16GB GPU)  |
                                        |  - FastAPI (kaggle_unified_srv) |
                                        |  - Stage 1: Perception Models   |
                                        |  - Stage 2: Claustro Scoring    |
                                        |  - Stage 3: Spatial Warping     |
                                        |  - Stage 4: DreamShaper 8 SD    |
                                        |  - Multi-ControlNet (Depth/Edge)|
                                        |  - Barrel Lens Distortion       |
                                        +---------------------------------+
```

### Why This Architecture Was Chosen

1. **Zero Local Hardware Requirement:**
   The entire inference pipeline runs on Kaggle's free GPU instances (16 GB VRAM). Users and hackathon evaluators do not need an expensive local GPU to experience full generative diffusion.

2. **Unified In-Process Execution:**
   By co-locating Stages 1 through 4 inside a single process on the Kaggle GPU, intermediate images (vignette, warped depth, modulated edges) are passed directly in GPU memory to the DreamShaper 8 Multi-ControlNet pipeline, avoiding slow network hops.

3. **Bypasses Serverless Platform Limits:**
   Vercel enforces a 4.5 MB request body limit and a 10 to 60-second function timeout. In this architecture, Vercel serves the static frontend and lightweight edge discovery route (`/api/config`), while heavy image uploads and model inferences flow directly between the user browser and Kaggle via Cloudflare Tunnel.

4. **Automated Tunnel Setup:**
   The unified server script automatically provisions `cloudflared` on Linux, launches a public HTTPS tunnel to port 8000, and prints the exact live URL in the notebook output.

---

## Running on Kaggle (Step-by-Step Playbook)

### Step 1: Open Kaggle Notebook with GPU
1. Go to [kaggle.com](https://www.kaggle.com) and create a new notebook.
2. Under **Notebook Options** (right sidebar), set **Accelerator** to **GPU T4 x 2** or **GPU P100**.
3. Ensure **Internet** is toggled **ON**.

### Step 2: Cell 1 — Clone Repository & Install Dependencies
Run this in the first notebook cell:

```bash
!git clone https://github.com/nixnord/claustroscope.git /kaggle/working/claustroscope
%cd /kaggle/working/claustroscope
!pip install -q -r requirements.txt
```

### Step 3: Cell 2 — Launch Unified Server
Run this in the second notebook cell:

```bash
!python kaggle_unified_server.py
```

*(You can also directly copy and paste the code from `kaggle_unified_server.py` into Cell 2 and run it).*

### Step 4: Connect to Vercel Frontend
When the server starts, it will output a banner with a live public URL:

```
================================================================================
TUNNEL ONLINE & READY
PUBLIC URL: https://something-random.trycloudflare.com
================================================================================
```

1. Copy the `https://...trycloudflare.com` URL.
2. In your Vercel project dashboard:
   * Go to **Settings** > **Environment Variables**.
   * Set `PRIMARY_ENGINE_URL` to your Cloudflare URL.
   * Click Save and Redeploy.
3. Open your Vercel website (or visit the Cloudflare URL directly in your browser). The status indicator in the header will show **Kaggle GPU Unified Engine** in green.

---

## Alternative Deployment Options

### Standalone Local Deployment

To run both backend server and frontend locally on your machine:

1. Clone and install dependencies:
   ```bash
   git clone https://github.com/nixnord/claustroscope.git
   cd claustroscope
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start the local server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. Open `http://localhost:8000` in your browser.

---

### Containerized Cloud Hosting (Hugging Face Spaces)

The repository includes a production `Dockerfile` configured for Hugging Face Spaces (CPU Basic 16 GB RAM or ZeroGPU):

1. Create a Space on Hugging Face Spaces using the **Docker** template.
2. Push this repository to your Space Git remote.
3. Hugging Face builds the image and exposes the app on port `7860`.

---

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PRIMARY_ENGINE_URL` | Public URL of the primary engine (FastAPI / Kaggle Cloudflare Tunnel) | `""` (uses local origin `/`) |
| `KAGGLE_TUNNEL_URL` | Public URL of remote worker if running split worker topology | `""` |
| `ENABLE_UNIFIED_DIFFUSION` | Set to `1` to run local in-process DreamShaper 8 diffusion in `app.py` | `0` |

---

## Repository Structure

* `kaggle_unified_server.py`: Complete unified server script for Kaggle GPU with automated Cloudflare tunnel.
* `app.py`: FastAPI backend server managing model lifecycles and analysis routes.
* `index.html`: Web interface featuring single-column vertical layout, loading animation, and results telemetry.
* `vercel.json`: Vercel configuration for clean URL routing.
* `api/config.js`: Vercel Edge Function providing dynamic environment configuration to the browser.
* `pipeline.py`: Four-stage orchestrator supporting both in-process and remote diffusion.
* `extractors.py`: Stage 1 feature extractors (Depth Anything V2, SegFormer, YOLO11-Pose, OpenCV).
* `scorer.py`: Stage 2 claustrophobia formula calculations, classification, and prompt generation.
* `modulators.py`: Stage 3 OpenCV spatial warps, radial vignettes, and barrel lens distortion.
* `client_pipeline.py`: Asynchronous bridge client communicating with remote Gradio workers.
* `kaggle_gradio_server.py`: Standalone Gradio diffusion worker script.
* `run_tests.py`: Automated verification test suite.
* `requirements.txt`: Python package dependencies including diffusers and accelerate.
* `Dockerfile`: Container definition for Hugging Face Spaces or containerized cloud hosting.
* `test/`: Sample input photographs covering open, enclosed, and claustrophobic scenes.
