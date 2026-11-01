"use client";

import React, { useEffect, useRef, useState } from "react";
import { Download, Eye, Layers, Maximize2, MoveHorizontal } from "lucide-react";
import { AnalysisImages, FocalPoint } from "@/lib/api";

interface ComparisonSliderProps {
  images: AnalysisImages;
  focalPoint?: FocalPoint;
  classification?: string;
  onBackToLander?: () => void;
}

export function ComparisonSlider({
  images,
  focalPoint,
  classification,
  onBackToLander,
}: ComparisonSliderProps) {
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [activeCanvasKey, setActiveCanvasKey] = useState<"final_canvas" | "preview">(
    images.final_canvas ? "final_canvas" : "preview"
  );
  const [showFocalPoint, setShowFocalPoint] = useState<boolean>(true);

  const containerRef = useRef<HTMLDivElement>(null);

  const activeAfterImage = images[activeCanvasKey] || images.final_canvas || images.preview;

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(percentage);
  };

  const handleMouseDown = () => setIsDragging(true);

  useEffect(() => {
    const handleMouseUp = () => setIsDragging(false);
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        handleMove(e.clientX);
      }
    };
    const handleTouchMove = (e: TouchEvent) => {
      if (isDragging && e.touches[0]) {
        handleMove(e.touches[0].clientX);
      }
    };

    if (isDragging) {
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("touchend", handleMouseUp);
      window.addEventListener("touchmove", handleTouchMove);
    }

    return () => {
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("touchend", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
    };
  }, [isDragging]);

  const handleExport = () => {
    if (!activeAfterImage) return;
    const link = document.createElement("a");
    link.href = activeAfterImage;
    link.download = `claustroscope_${activeCanvasKey}_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="w-full flex flex-col gap-4">
      {/* Top Slider Action Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#343334] pb-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff4000] animate-pulse"></span>
            <h3 className="text-sm font-fira font-bold uppercase tracking-wider text-[#feefdd]">
              Transformed Claustrophobic Canvas
            </h3>
          </div>

          {/* Toggle between Diffusion Canvas & Optical Preview */}
          {images.preview && images.final_canvas && images.preview !== images.final_canvas && (
            <div className="flex items-center bg-[#201e1f] p-0.5 rounded-sm border border-[#343334]">
              <button
                type="button"
                onClick={() => setActiveCanvasKey("final_canvas")}
                className={`px-2.5 py-1 text-[11px] font-fira font-bold uppercase rounded-sm transition-colors cursor-pointer ${
                  activeCanvasKey === "final_canvas"
                    ? "bg-[#ff4000] text-[#201e1f]"
                    : "text-[#736f71] hover:text-[#feefdd]"
                }`}
              >
                Diffusion Canvas
              </button>
              <button
                type="button"
                onClick={() => setActiveCanvasKey("preview")}
                className={`px-2.5 py-1 text-[11px] font-fira font-bold uppercase rounded-sm transition-colors cursor-pointer ${
                  activeCanvasKey === "preview"
                    ? "bg-[#50b2c0] text-[#201e1f]"
                    : "text-[#736f71] hover:text-[#feefdd]"
                }`}
              >
                Optical Warp
              </button>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          {focalPoint && (
            <button
              type="button"
              onClick={() => setShowFocalPoint(!showFocalPoint)}
              className={`px-2.5 py-1 text-xs font-fira rounded-sm border transition-colors cursor-pointer ${
                showFocalPoint
                  ? "border-[#50b2c0] text-[#50b2c0] bg-[#50b2c0]/10"
                  : "border-[#343334] text-[#736f71] hover:text-[#feefdd]"
              }`}
            >
              Focal Anchor: {showFocalPoint ? "Visible" : "Hidden"}
            </button>
          )}

          <button
            type="button"
            onClick={handleExport}
            className="px-3.5 py-1.5 rounded-sm bg-[#ff4000] hover:bg-[#ff551a] text-xs font-fira font-bold text-[#201e1f] uppercase tracking-tight transition-transform active:scale-95 flex items-center gap-1.5 shadow-[0_0_15px_rgba(255,64,0,0.35)] cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Canvas</span>
          </button>
        </div>
      </div>

      {/* Main Split Comparison Viewport */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onTouchStart={handleMouseDown}
        className="relative w-full aspect-[4/3] sm:aspect-[16/10] bg-black rounded-sm overflow-hidden select-none border-2 border-[#343334] cursor-ew-resize shadow-2xl"
      >
        {/* Background: Transformed Claustrophobic Canvas */}
        <img
          src={activeAfterImage}
          alt="Transformed Claustrophobic Canvas"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Foreground: Original Photograph with Dynamic Clip Path */}
        <div
          className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none"
          style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
        >
          <img
            src={images.original}
            alt="Original Photograph"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          />

          {/* Focal Anchor Crosshair Overlay */}
          {showFocalPoint && focalPoint && (
            <div
              className="absolute pointer-events-none"
              style={{
                left: `${focalPoint.x * 100}%`,
                top: `${focalPoint.y * 100}%`,
              }}
            >
              <div className="w-8 h-8 rounded-full border border-[#50b2c0] animate-ping absolute -translate-x-1/2 -translate-y-1/2 opacity-75"></div>
              <div className="w-3 h-3 rounded-full bg-[#50b2c0] absolute -translate-x-1/2 -translate-y-1/2 shadow-[0_0_10px_#50b2c0]"></div>
              <span className="absolute left-3 top-2 px-1.5 py-0.5 rounded-sm bg-[#201e1f]/90 text-[10px] font-fira text-[#50b2c0] whitespace-nowrap border border-[#50b2c0]/40">
                Focal Anchor ({Math.round(focalPoint.x * 100)}%, {Math.round(focalPoint.y * 100)}%)
              </span>
            </div>
          )}

          {/* Badge: Original */}
          <div className="absolute top-3 left-3 px-2.5 py-1 rounded-sm bg-[#201e1f]/90 text-[11px] font-fira text-[#feefdd] border border-[#343334]">
            Original Photograph
          </div>
        </div>

        {/* Badge: Transformed */}
        <div className="absolute top-3 right-3 px-2.5 py-1 rounded-sm bg-[#201e1f]/90 text-[11px] font-fira text-[#ff4000] border border-[#343334]">
          {activeCanvasKey === "final_canvas" ? "Claustrophobic Canvas" : "Optical Warp Preview"}
        </div>

        {/* Draggable Divider Handle */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-[#50b2c0] cursor-ew-resize flex items-center justify-center transition-opacity"
          style={{ left: `${sliderPos}%` }}
        >
          <div className="w-8 h-8 rounded-full bg-[#201e1f] border-2 border-[#50b2c0] flex items-center justify-center shadow-[0_0_15px_rgba(80,178,192,0.8)] -translate-x-0.5">
            <MoveHorizontal className="w-4 h-4 text-[#50b2c0]" />
          </div>
        </div>
      </div>

      <p className="text-xs font-fira text-[#736f71] text-center">
        &larr; Drag slider across the canvas to inspect latent spatial compression &rarr;
      </p>
    </div>
  );
}
