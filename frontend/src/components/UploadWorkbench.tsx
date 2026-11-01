"use client";

import React, { useRef, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  Eye,
  FileImage,
  ImageIcon,
  Sparkles,
  Upload,
  X,
  Zap,
} from "lucide-react";

interface SamplePreset {
  id: string;
  name: string;
  icon: string;
  file: string;
  description: string;
}

const SAMPLE_PRESETS: SamplePreset[] = [
  {
    id: "alley",
    name: "Narrow Alley",
    icon: "🧱",
    file: "/samples/testimg3.jpeg",
    description: "High vertical masonry walls with zero lateral egress",
  },
  {
    id: "cave",
    name: "Cave Squeeze",
    icon: "🕳️",
    file: "/samples/testimg2.jpeg",
    description: "Subterranean ceiling compression with caver in tight passage",
  },
  {
    id: "crawl",
    name: "Crawlspace",
    icon: "🪜",
    file: "/samples/testimg1.jpeg",
    description: "Extreme horizontal duct enclosure with close ceiling",
  },
  {
    id: "crowd",
    name: "Packed Crowd",
    icon: "👥",
    file: "/samples/testimg6.jpeg",
    description: "Dense human body occlusions and sensory over-packing",
  },
  {
    id: "park",
    name: "Open Park",
    icon: "🌳",
    file: "/samples/testimg4.jpeg",
    description: "Expansive sunlit horizon providing tension suppression",
  },
];

interface UploadWorkbenchProps {
  onAnalyze: (file: File, options: { applyLensDistortion: boolean; distortionMode: "barrel" | "tunnel" }) => void;
  isProcessing: boolean;
}

export function UploadWorkbench({ onAnalyze, isProcessing }: UploadWorkbenchProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [applyLensDistortion, setApplyLensDistortion] = useState<boolean>(true);
  const [distortionMode, setDistortionMode] = useState<"barrel" | "tunnel">("barrel");
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<boolean>(false);
  const [imgDims, setImgDims] = useState<{ width: number; height: number } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (JPG, PNG, WEBP).");
      return;
    }

    setSelectedFile(file);
    setActiveSampleId(null);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const img = new Image();
    img.onload = () => {
      setImgDims({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleClearImage = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    setImgDims(null);
    setActiveSampleId(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleLoadSample = async (sample: SamplePreset) => {
    setLoadingSample(true);
    setActiveSampleId(sample.id);
    try {
      const resp = await fetch(sample.file);
      if (!resp.ok) throw new Error("Failed to load sample image");
      const blob = await resp.blob();
      const filename = sample.file.split("/").pop() || "sample.jpg";
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      handleFileSelect(file);
      setActiveSampleId(sample.id);
    } catch (err) {
      console.error("Error loading sample:", err);
      alert("Could not load sample image. Please upload a local photo.");
    } finally {
      setLoadingSample(false);
    }
  };

  const handleSubmit = () => {
    if (!selectedFile || isProcessing) return;
    onAnalyze(selectedFile, {
      applyLensDistortion,
      distortionMode,
    });
  };

  return (
    <div className="w-full max-w-[1440px] mx-auto px-6 md:px-12 lg:px-16 py-8">
      {/* 12-Column Responsive Widescreen Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Enlarged Desktop Dropzone & Preview Container (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#ff4000] animate-pulse"></span>
              <span className="text-xs font-fira font-bold uppercase tracking-wider text-[#feefdd]">
                Spatial Input Canvas
              </span>
            </div>
            {selectedFile && imgDims && (
              <span className="text-xs font-fira text-[#50b2c0] bg-[#201e1f] px-2.5 py-0.5 rounded-sm border border-[#343334]">
                {imgDims.width} &times; {imgDims.height} px &bull; {(selectedFile.size / 1024).toFixed(1)} KB
              </span>
            )}
          </div>

          {/* Interactive Dropzone Container with Large Desktop Height */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !previewUrl && fileInputRef.current?.click()}
            className={`relative w-full min-h-[460px] lg:min-h-[540px] rounded-sm border-2 transition-all flex flex-col items-center justify-center p-6 select-none ${
              previewUrl
                ? "border-[#343334] bg-[#1a191a]"
                : isDragging
                ? "border-[#50b2c0] bg-[#2a292a]/90 scale-[1.005]"
                : "border-dashed border-[#343334] hover:border-[#50b2c0] bg-[#201e1f]/60 hover:bg-[#201e1f] cursor-pointer"
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileSelect(e.target.files[0]);
                }
              }}
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
            />

            {!previewUrl ? (
              <div className="flex flex-col items-center text-center gap-4 max-w-md pointer-events-none">
                <div className="w-20 h-20 rounded-full bg-[#2a292a] border-2 border-[#343334] flex items-center justify-center shadow-lg transition-transform group-hover:scale-105">
                  <Upload className="w-9 h-9 text-[#50b2c0]" />
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-lg font-chivo font-bold text-[#feefdd]">
                    Drop an architectural photograph here
                  </h3>
                  <p className="text-xs font-fira text-[#736f71]">
                    or click to browse from your device
                  </p>
                </div>
                <div className="px-3 py-1 rounded-sm bg-[#2a292a] border border-[#343334] text-[11px] font-fira text-[#feefdd]/60">
                  Supports JPG, PNG, WEBP &bull; Max 20 MB &bull; Clamped to 768px for fast inference
                </div>
              </div>
            ) : (
              <div className="relative w-full h-full min-h-[460px] lg:min-h-[500px] flex items-center justify-center">
                <img
                  src={previewUrl}
                  alt="Uploaded scene"
                  className="max-h-[500px] w-auto max-w-full object-contain rounded-sm border border-[#343334] shadow-2xl"
                />

                {/* Floating Clear Button */}
                <button
                  type="button"
                  onClick={handleClearImage}
                  title="Remove image"
                  className="absolute top-2 right-2 p-2 rounded-sm bg-[#201e1f]/90 hover:bg-[#ff4000] text-[#feefdd] transition-colors border border-[#343334] cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>

                {/* Image Status Overlay Tag */}
                <div className="absolute bottom-2 left-2 px-3 py-1 rounded-sm bg-[#201e1f]/90 text-xs font-fira text-[#50b2c0] border border-[#50b2c0]/40 flex items-center gap-1.5">
                  <FileImage className="w-3.5 h-3.5" />
                  <span>Scene Staged for Analysis</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Mission Control & Parameters (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* Hero Narrative Intro */}
          <div className="space-y-3 bg-[#2a292a] border-2 border-[#343334] p-6 sm:p-7 rounded-sm">
            <div className="inline-flex items-center gap-2 text-[#ff4000] text-xs font-fira font-bold uppercase tracking-wider">
              <Sparkles className="w-4 h-4" />
              <span>Perceptual Latent Compression</span>
            </div>

            <h2 className="text-3xl sm:text-4xl font-chivo font-black leading-tight tracking-tight text-[#feefdd]">
              Experience how <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#ff4000] via-[#faaa8d] to-[#50b2c0]">
                claustrophobia feels.
              </span>
            </h2>

            <p className="text-xs sm:text-sm font-inter text-[#feefdd]/80 leading-relaxed">
              Upload any room, corridor, or outdoor setting. The pipeline extracts 8 psychological spatial
              metrics—combining Depth Anything V2, SegFormer ADE20K semantic enclosure, and YOLO11 human crowding—synthesizing an intense perceptual simulation through Stable Diffusion.
            </p>
          </div>

          {/* Sample Scene Presets Card */}
          <div className="bg-[#2a292a] border-2 border-[#343334] p-6 rounded-sm space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-fira font-bold text-[#feefdd] uppercase tracking-wider">
                Or Test With A Curated Scene:
              </span>
              <span className="text-[11px] font-fira text-[#736f71]">1-click preload</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
              {SAMPLE_PRESETS.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  onClick={() => handleLoadSample(sample)}
                  disabled={loadingSample || isProcessing}
                  className={`p-2.5 rounded-sm border text-left transition-all flex flex-col gap-1 cursor-pointer ${
                    activeSampleId === sample.id
                      ? "border-[#ff4000] bg-[#ff4000]/10 text-[#feefdd]"
                      : "border-[#343334] bg-[#201e1f] hover:border-[#50b2c0] text-[#feefdd]/80 hover:text-[#feefdd]"
                  }`}
                >
                  <div className="flex items-center gap-1.5 font-fira text-xs font-bold">
                    <span>{sample.icon}</span>
                    <span className="truncate">{sample.name}</span>
                  </div>
                  <span className="text-[10px] font-fira text-[#736f71] line-clamp-1">
                    {sample.description}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Optical Controls & Pipeline Configuration Card */}
          <div className="bg-[#2a292a] border-2 border-[#343334] p-6 rounded-sm space-y-5">
            <div className="flex items-center justify-between border-b border-[#343334] pb-3">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-[#50b2c0]" />
                <span className="text-xs font-fira font-bold text-[#feefdd] uppercase tracking-wider">
                  Post-Processing Optics
                </span>
              </div>
              <span className="text-[11px] font-fira text-[#736f71]">Non-linear optics</span>
            </div>

            {/* Lens Distortion Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <label
                  htmlFor="chkLens"
                  className="text-xs font-fira font-bold text-[#feefdd] cursor-pointer select-none block"
                >
                  Optical Lens Distortion
                </label>
                <p className="text-[11px] font-fira text-[#736f71]">
                  Simulates collapsing barrel perimeter with peripheral chromatic aberration
                </p>
              </div>
              <input
                type="checkbox"
                id="chkLens"
                checked={applyLensDistortion}
                onChange={(e) => setApplyLensDistortion(e.target.checked)}
                className="w-4 h-4 accent-[#ff4000] rounded-sm bg-[#201e1f] border-[#343334] cursor-pointer"
              />
            </div>

            {/* Distortion Mode Selector if enabled */}
            {applyLensDistortion && (
              <div className="pt-2 flex items-center justify-between text-xs font-fira">
                <span className="text-[#736f71]">Distortion Geometry:</span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setDistortionMode("barrel")}
                    className={`px-3 py-1 rounded-sm border uppercase text-[11px] font-bold transition-colors cursor-pointer ${
                      distortionMode === "barrel"
                        ? "border-[#ff4000] bg-[#ff4000]/15 text-[#ff4000]"
                        : "border-[#343334] bg-[#201e1f] text-[#736f71]"
                    }`}
                  >
                    Barrel Bowing
                  </button>
                  <button
                    type="button"
                    onClick={() => setDistortionMode("tunnel")}
                    className={`px-3 py-1 rounded-sm border uppercase text-[11px] font-bold transition-colors cursor-pointer ${
                      distortionMode === "tunnel"
                        ? "border-[#50b2c0] bg-[#50b2c0]/15 text-[#50b2c0]"
                        : "border-[#343334] bg-[#201e1f] text-[#736f71]"
                    }`}
                  >
                    Tunnel Squeeze
                  </button>
                </div>
              </div>
            )}

            {/* Primary Action Button */}
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!selectedFile || isProcessing}
              className="w-full py-4 px-6 rounded-sm font-fira font-bold uppercase tracking-tight text-sm transition-all bg-gradient-to-r from-[#ff4000] via-[#faaa8d] to-[#50b2c0] text-[#201e1f] hover:scale-[1.01] active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-[0_0_25px_rgba(255,64,0,0.35)] flex items-center justify-center gap-2.5 cursor-pointer mt-2"
            >
              <span>Initiate Perceptual Transformation</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
