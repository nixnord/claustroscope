"use client";

import React, { useState } from "react";
import { AlertCircle, ArrowLeft, RotateCcw } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { UploadWorkbench } from "@/components/UploadWorkbench";
import { ComparisonSlider } from "@/components/ComparisonSlider";
import { TelemetryPanel } from "@/components/TelemetryPanel";
import { ConditioningTabs } from "@/components/ConditioningTabs";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import ParticleBackground from "@/components/ParticleBackground";
import { analyzeImage, AnalysisResponse } from "@/lib/api";

export default function Home() {
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (
    file: File,
    options: { applyLensDistortion: boolean; distortionMode: "barrel" | "tunnel" }
  ) => {
    setIsProcessing(true);
    setError(null);

    try {
      const response = await analyzeImage(file, file.name, {
        applyLensDistortion: options.applyLensDistortion,
        distortionMode: options.distortionMode,
      });
      setResult(response);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err: any) {
      console.error("Analysis failed:", err);
      setError(
        err.message ||
          "Failed to execute pipeline. Please ensure the Kaggle backend server is active and accessible."
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const getClassificationBadgeClass = (cls: string) => {
    switch (cls) {
      case "CLAUSTROPHOBIC":
        return "text-[#ff4000] border-[#ff4000] bg-[#ff4000]/15";
      case "CONFINED":
        return "text-[#faaa8d] border-[#faaa8d] bg-[#faaa8d]/15";
      case "MODERATE":
        return "text-[#f59e0b] border-[#f59e0b] bg-[#f59e0b]/15";
      case "OPEN":
      default:
        return "text-[#10b981] border-[#10b981] bg-[#10b981]/15";
    }
  };

  return (
    <div className="min-h-screen bg-[#201e1f] text-[#feefdd] font-inter relative flex flex-col justify-between selection:bg-[#ff4000] selection:text-white">
      {/* Background Subtle Particle Canvas */}
      <ParticleBackground />

      {/* Global Brand Navbar with Dynamic Kaggle Tunnel Status & Settings */}
      <Navbar />

      {/* Processing Animation Overlay */}
      <LoadingOverlay isProcessing={isProcessing} />

      {/* Error Alert Banner */}
      {error && (
        <div className="w-full max-w-[1440px] mx-auto px-6 md:px-12 lg:px-16 pt-6 z-30">
          <div className="p-4 rounded-sm bg-[#ff4000]/15 border-2 border-[#ff4000] text-xs font-fira text-[#feefdd] flex items-center justify-between shadow-xl">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-[#ff4000] shrink-0" />
              <div>
                <span className="font-bold text-[#ff4000] uppercase block">
                  Processing Interrupted
                </span>
                <span>{error}</span>
              </div>
            </div>
            <button
              onClick={() => setError(null)}
              className="px-3 py-1 bg-[#201e1f] hover:bg-[#ff4000] text-[#feefdd] rounded-sm transition-colors uppercase text-[11px] font-bold"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Main Interactive Workspace */}
      <main className="flex-1 w-full z-10">
        {!result ? (
          /* View 1: Upload & Parameter Workbench */
          <UploadWorkbench onAnalyze={handleAnalyze} isProcessing={isProcessing} />
        ) : (
          /* View 2: Full Results & Analysis Dashboard (Widescreen 1440px) */
          <div className="w-full max-w-[1440px] mx-auto px-6 md:px-12 lg:px-16 py-8 flex flex-col gap-10 animate-in fade-in duration-300">
            {/* Sticky Sub-Header with Return & Classification Badge */}
            <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[#343334] sticky top-20 bg-[#201e1f]/95 backdrop-blur z-30 pt-2">
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-sm bg-[#2a292a] hover:bg-[#343334] border border-[#343334] text-xs font-fira font-bold text-[#feefdd] transition-colors cursor-pointer"
              >
                <ArrowLeft className="w-4 h-4 text-[#ff4000]" />
                <span>Analyze Another Image</span>
              </button>

              <div className="flex items-center gap-3">
                <span className="text-xs font-fira text-[#736f71] hidden sm:inline-block">
                  Computed in <strong className="text-[#50b2c0]">{result.elapsed_time}s</strong>
                </span>
                <span
                  className={`px-4 py-1.5 rounded-sm text-xs font-fira font-bold uppercase tracking-wider border ${getClassificationBadgeClass(
                    result.classification
                  )}`}
                >
                  {result.classification}
                </span>
              </div>
            </div>

            {/* Section 1: Hero Transformed Visual Canvas (Comparison Slider) */}
            <ComparisonSlider
              images={result.images}
              focalPoint={result.focal_point}
              classification={result.classification}
            />

            {/* Section 2: Psychological Scorecard & 8 Stage 1 Features */}
            <TelemetryPanel
              scores={result.scores}
              classification={result.classification}
              control={result.control}
              features={result.features}
            />

            {/* Section 3: Stage 3 Conditioning Maps & Stage 4 Prompts/Parameters */}
            <ConditioningTabs images={result.images} control={result.control} />

            {/* Bottom Reset Button */}
            <div className="text-center py-6 border-t border-[#343334]">
              <button
                type="button"
                onClick={handleReset}
                className="px-8 py-3.5 rounded-sm bg-[#2a292a] hover:bg-[#343334] border border-[#343334] hover:border-[#50b2c0] text-sm font-fira font-bold text-[#feefdd] transition-all inline-flex items-center gap-2.5 shadow-lg cursor-pointer"
              >
                <RotateCcw className="w-4 h-4 text-[#50b2c0]" />
                <span>Analyze Another Photograph</span>
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Global Brutalist Footer */}
      <footer className="w-full border-t border-[#343334] bg-[#201e1f] py-6 z-10">
        <div className="w-full max-w-[1440px] mx-auto px-6 md:px-12 lg:px-16 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-fira text-[#736f71]">
          <div>
            ClaustroScope Engine &bull; Non-Linear Confinement Mathematics &bull; Code the Canvas
          </div>
          <div className="flex items-center gap-4 text-[#50b2c0]">
            <span>Depth Anything V2</span>
            <span>&bull;</span>
            <span>SegFormer B2</span>
            <span>&bull;</span>
            <span>YOLO11-Pose</span>
            <span>&bull;</span>
            <span>DreamShaper-8</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
