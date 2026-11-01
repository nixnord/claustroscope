"use client";

import React, { useState } from "react";
import { Check, Copy, Eye, Maximize2, X } from "lucide-react";
import { AnalysisImages, ControlPayload } from "@/lib/api";

interface ConditioningTabsProps {
  images: AnalysisImages;
  control: ControlPayload;
}

interface LightboxState {
  isOpen: boolean;
  title: string;
  src: string;
}

export function ConditioningTabs({ images, control }: ConditioningTabsProps) {
  const [copiedPos, setCopiedPos] = useState(false);
  const [copiedNeg, setCopiedNeg] = useState(false);
  const [lightbox, setLightbox] = useState<LightboxState>({
    isOpen: false,
    title: "",
    src: "",
  });

  const handleCopy = (text: string, isPos: boolean) => {
    navigator.clipboard.writeText(text);
    if (isPos) {
      setCopiedPos(true);
      setTimeout(() => setCopiedPos(false), 2000);
    } else {
      setCopiedNeg(true);
      setTimeout(() => setCopiedNeg(false), 2000);
    }
  };

  const openLightbox = (title: string, src: string) => {
    setLightbox({ isOpen: true, title, src });
  };

  const mod = control.modulation_parameters;
  const hyper = control.diffusion_hyperparameters;
  const prompts = control.prompts;

  return (
    <>
      <div className="w-full flex flex-col gap-6">
        {/* Stage 3 Conditioning Maps Section */}
        <div className="bg-[#2a292a] border-2 border-[#343334] rounded-sm p-6 sm:p-8 space-y-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-[#343334] pb-3">
            <div>
              <h3 className="text-sm font-fira font-bold text-[#feefdd] uppercase tracking-wider">
                Stage 3 Conditioning Maps & Optical Warp
              </h3>
              <p className="text-xs text-[#736f71] font-fira mt-0.5">
                Spatial modulations fed as multi-modal conditioning into ControlNet & OpenCV optics.
              </p>
            </div>
            <span className="text-xs font-fira text-[#50b2c0] hidden sm:inline-block">
              Click map to inspect full-res
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-1">
            {/* 1. Radial Vignette */}
            <div
              onClick={() =>
                openLightbox(
                  "Radial Tunnel Vision Vignette",
                  images.vignette || images.original
                )
              }
              className="bg-[#201e1f] rounded-sm p-3 border border-[#343334] hover:border-[#50b2c0] transition-all cursor-pointer group flex flex-col gap-2"
            >
              <div className="aspect-square bg-black rounded-sm overflow-hidden relative flex items-center justify-center">
                <img
                  src={images.vignette || images.original}
                  alt="Radial Vignette"
                  className="w-full h-full object-contain group-hover:scale-105 transition-transform"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <Maximize2 className="w-5 h-5 text-[#50b2c0]" />
                </div>
              </div>
              <div>
                <span className="text-xs font-fira text-[#feefdd] font-bold block">
                  Radial Vignette
                </span>
                <span className="text-[10px] font-fira text-[#50b2c0]">
                  I = {mod?.vignette_intensity?.toFixed(3) ?? "0.000"}
                </span>
              </div>
            </div>

            {/* 2. Warped Depth */}
            <div
              onClick={() =>
                openLightbox(
                  "Warped Inward Depth Map (κ-Curvature)",
                  images.depth || images.original
                )
              }
              className="bg-[#201e1f] rounded-sm p-3 border border-[#343334] hover:border-[#50b2c0] transition-all cursor-pointer group flex flex-col gap-2"
            >
              <div className="aspect-square bg-black rounded-sm overflow-hidden relative flex items-center justify-center">
                <img
                  src={images.depth || images.original}
                  alt="Warped Depth"
                  className="w-full h-full object-contain group-hover:scale-105 transition-transform"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <Maximize2 className="w-5 h-5 text-[#50b2c0]" />
                </div>
              </div>
              <div>
                <span className="text-xs font-fira text-[#feefdd] font-bold block">
                  Warped Depth (&kappa;)
                </span>
                <span className="text-[10px] font-fira text-[#50b2c0]">
                  &kappa; = {mod?.depth_warp_kappa?.toFixed(3) ?? "0.000"}
                </span>
              </div>
            </div>

            {/* 3. Modulated Canny */}
            <div
              onClick={() =>
                openLightbox(
                  "Modulated Jittered Canny Edges",
                  images.canny || images.original
                )
              }
              className="bg-[#201e1f] rounded-sm p-3 border border-[#343334] hover:border-[#50b2c0] transition-all cursor-pointer group flex flex-col gap-2"
            >
              <div className="aspect-square bg-black rounded-sm overflow-hidden relative flex items-center justify-center">
                <img
                  src={images.canny || images.original}
                  alt="Modulated Canny"
                  className="w-full h-full object-contain group-hover:scale-105 transition-transform"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <Maximize2 className="w-5 h-5 text-[#50b2c0]" />
                </div>
              </div>
              <div>
                <span className="text-xs font-fira text-[#feefdd] font-bold block">
                  Modulated Canny
                </span>
                <span className="text-[10px] font-fira text-[#50b2c0]">
                  Jitter = {mod?.edge_jitter_amplitude?.toFixed(1) ?? "0.0"} px
                </span>
              </div>
            </div>

            {/* 4. Optical Warp Preview */}
            <div
              onClick={() =>
                openLightbox(
                  "Optical Barrel Lens Distortion Preview",
                  images.preview || images.original
                )
              }
              className="bg-[#201e1f] rounded-sm p-3 border border-[#343334] hover:border-[#ff4000] transition-all cursor-pointer group flex flex-col gap-2"
            >
              <div className="aspect-square bg-black rounded-sm overflow-hidden relative flex items-center justify-center">
                <img
                  src={images.preview || images.original}
                  alt="Optical Lens Preview"
                  className="w-full h-full object-contain group-hover:scale-105 transition-transform"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <Maximize2 className="w-5 h-5 text-[#ff4000]" />
                </div>
              </div>
              <div>
                <span className="text-xs font-fira text-[#feefdd] font-bold block">
                  Lens Warp Preview
                </span>
                <span className="text-[10px] font-fira text-[#ff4000]">
                  k = {mod?.lens_distortion_k?.toFixed(3) ?? "0.000"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Stage 4 Generative Parameters & Prompts */}
        <div className="bg-[#2a292a] border-2 border-[#343334] rounded-sm p-6 sm:p-8 space-y-6 shadow-xl">
          <div>
            <h3 className="text-sm font-fira font-bold text-[#feefdd] uppercase tracking-wider">
              Stage 4: Generative Prompts & Diffusion Hyperparameters
            </h3>
            <p className="text-xs text-[#736f71] font-fira mt-0.5">
              Dynamically derived hyperparameters guiding DreamShaper-8 Multi-ControlNet on Kaggle.
            </p>
          </div>

          {/* Hyperparameters Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-6 gap-2.5 text-center font-fira text-xs">
            <div className="bg-[#201e1f] p-3 rounded-sm border border-[#343334]">
              <span className="text-[#736f71] block text-[10px]">&sigma; Denoise</span>
              <span className="font-bold text-[#50b2c0] text-sm">
                {hyper?.denoising_strength?.toFixed(2) ?? "--"}
              </span>
            </div>
            <div className="bg-[#201e1f] p-3 rounded-sm border border-[#343334]">
              <span className="text-[#736f71] block text-[10px]">CFG Scale</span>
              <span className="font-bold text-[#50b2c0] text-sm">
                {hyper?.cfg_scale?.toFixed(1) ?? "--"}
              </span>
            </div>
            <div className="bg-[#201e1f] p-3 rounded-sm border border-[#343334]">
              <span className="text-[#736f71] block text-[10px]">Steps</span>
              <span className="font-bold text-[#50b2c0] text-sm">
                {hyper?.num_inference_steps ?? "--"}
              </span>
            </div>
            <div className="bg-[#201e1f] p-3 rounded-sm border border-[#343334]">
              <span className="text-[#736f71] block text-[10px]">Depth Scale</span>
              <span className="font-bold text-[#50b2c0] text-sm">
                {hyper?.controlnet_depth_scale?.toFixed(2) ?? "--"}
              </span>
            </div>
            <div className="bg-[#201e1f] p-3 rounded-sm border border-[#343334]">
              <span className="text-[#736f71] block text-[10px]">Edge Scale</span>
              <span className="font-bold text-[#50b2c0] text-sm">
                {hyper?.controlnet_edge_scale?.toFixed(2) ?? "--"}
              </span>
            </div>
            <div className="bg-[#201e1f] p-3 rounded-sm border border-[#343334]">
              <span className="text-[#736f71] block text-[10px]">Lens Warp k</span>
              <span className="font-bold text-[#ff4000] text-sm">
                {mod?.lens_distortion_k?.toFixed(3) ?? "--"}
              </span>
            </div>
          </div>

          {/* Prompts Display */}
          <div className="space-y-4 font-fira text-xs">
            <div>
              <div className="flex items-center justify-between text-[#50b2c0] mb-1.5 font-bold">
                <span>Positive Prompt:</span>
                <button
                  type="button"
                  onClick={() => handleCopy(prompts?.positive_prompt || "", true)}
                  className="text-[#736f71] hover:text-[#feefdd] text-[11px] flex items-center gap-1 transition-colors cursor-pointer"
                >
                  {copiedPos ? (
                    <>
                      <Check className="w-3 h-3 text-[#10b981]" />
                      <span className="text-[#10b981]">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
              <div className="p-3.5 bg-[#201e1f] border border-[#343334] rounded-sm text-[#feefdd]/90 leading-relaxed break-words text-xs">
                {prompts?.positive_prompt || "--"}
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between text-[#ff4000] mb-1.5 font-bold">
                <span>Negative Prompt:</span>
                <button
                  type="button"
                  onClick={() => handleCopy(prompts?.negative_prompt || "", false)}
                  className="text-[#736f71] hover:text-[#feefdd] text-[11px] flex items-center gap-1 transition-colors cursor-pointer"
                >
                  {copiedNeg ? (
                    <>
                      <Check className="w-3 h-3 text-[#10b981]" />
                      <span className="text-[#10b981]">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
              <div className="p-3.5 bg-[#201e1f] border border-[#343334] rounded-sm text-[#736f71] leading-relaxed break-words text-xs">
                {prompts?.negative_prompt || "--"}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Lightbox Modal */}
      {lightbox.isOpen && (
        <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex flex-col items-center justify-center p-4 animate-in fade-in duration-200">
          <div className="max-w-4xl w-full flex items-center justify-between pb-3">
            <h3 className="text-xs font-fira uppercase text-[#feefdd] font-bold">
              {lightbox.title}
            </h3>
            <button
              onClick={() => setLightbox({ isOpen: false, title: "", src: "" })}
              className="p-1.5 rounded-sm bg-[#2a292a] hover:bg-[#ff4000] text-[#feefdd] transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="max-w-4xl max-h-[80vh] w-full flex items-center justify-center overflow-hidden rounded-sm bg-black border-2 border-[#343334] p-2">
            <img
              src={lightbox.src}
              alt={lightbox.title}
              className="max-w-full max-h-[75vh] object-contain rounded-sm"
            />
          </div>
        </div>
      )}
    </>
  );
}
