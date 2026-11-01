# ClaustroScope Frontend Architecture & Guide

This is the Next.js frontend for **ClaustroScope** — Psychological Perception & Multi-ControlNet Engine.

---

## 1. Tech Stack
- **Framework**: Next.js 16 (App Router with Turbopack)
- **Styling**: Tailwind CSS v4 (Flat, brutalist dark industrial aesthetic)
- **Components**: Custom brutalist UI components and Radix UI primitives
- **Animations**: Framer Motion and custom CSS box-warp animations
- **Background Effects**: `tsparticles` (Subtle floating geometric nodes)
- **Icons**: Lucide React
- **TypeScript**: Strict type definitions for full telemetry payloads

---

## 2. Palette and Design System
- **Colors**:
  - Dark Charcoal Background: `#201e1f`
  - Card Fill: `#2a292a`
  - Borders: `#343334`
  - Primary Flame Orange: `#ff4000`
  - Secondary Glowing Cyan: `#50b2c0`
  - Accent Coral: `#faaa8d`
  - Cream/Bone Text: `#feefdd`
- **Typography**:
  - `font-doto`: Brand Logo header.
  - `font-chivo`: Hero titles and scorecard metrics.
  - `font-fira`: Technical telemetry, labels, and metadata.
  - `font-inter`: Base readable UI copy.
- **Styling Rules**: Strict `rounded-sm` (5px border radius), high contrast, dark mode default.

---

## 3. Directory Structure

```text
frontend/
├── .env.example                 # Environment variables template
├── next.config.ts               # Next.js config (unoptimized images enabled for base64/blobs)
├── package.json                 # Next.js, React, TailwindCSS, Lucide, Framer Motion
├── public/
│   └── samples/                 # Test images (Narrow Alley, Cave Squeeze, Crawlspace, etc.)
└── src/
    ├── app/
    │   ├── layout.tsx           # Dark theme root with google fonts
    │   ├── globals.css          # CSS variables, boxWarp animation, custom scrollbars
    │   └── page.tsx             # Interactive Claustroscope Workbench and Results Dashboard
    ├── components/
    │   ├── Navbar.tsx           # Brand header + live Kaggle GPU connection status pill & modal
    │   ├── UploadWorkbench.tsx  # Full-width desktop dropzone, sample scene loader, lens toggle
    │   ├── ComparisonSlider.tsx # Interactive Before/After drag slider with focal anchor overlay
    │   ├── TelemetryPanel.tsx   # C-Score gauge, T/R breakdown, 8 feature progress bars
    │   ├── ConditioningTabs.tsx # Stage 3 viewer + Stage 4 parameters & copyable prompts
    │   ├── LoadingOverlay.tsx   # Flowing box streams, spinner, and dynamic stage indicators
    │   └── ParticleBackground.tsx # tsparticles animated dark background
    └── lib/
        └── api.ts               # Fetch wrapper communicating with Kaggle FastAPI backend
```

---

## 4. Environment Variables

| Variable | Description | Default |
|---|---|---|
| `NEXT_PUBLIC_KAGGLE_API_URL` | Public Cloudflare tunnel URL exposed by `kaggle_server.py` | `http://127.0.0.1:7860` |

### Runtime Tunnel Override
Because Cloudflare quick tunnels produce new URLs each time the Kaggle notebook restarts, you can click on the **Kaggle GPU** status badge in the header to paste and save the active tunnel URL into `localStorage`. This updates the API endpoint in real-time without requiring a redeployment or restart.

---

## 5. Development

### Install Dependencies
```bash
npm install
```

### Run Locally
```bash
npm run dev
```
Open `http://localhost:3000` in your browser.

### Build Production Bundle
```bash
npm run build
```
