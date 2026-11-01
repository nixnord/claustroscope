"use client";

import React from "react";
import { Activity, AlertTriangle, ShieldAlert, Sparkles } from "lucide-react";
import { AnalysisScores, ControlPayload, FeaturePayload } from "@/lib/api";

interface TelemetryPanelProps {
  scores: AnalysisScores;
  classification: "OPEN" | "MODERATE" | "CONFINED" | "CLAUSTROPHOBIC";
  control: ControlPayload;
  features: FeaturePayload;
}

export function TelemetryPanel({
  scores,
  classification,
  control,
  features,
}: TelemetryPanelProps) {
  const getClassificationColor = (cls: string) => {
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

  const getScoreNumberColor = (c: number) => {
    if (c >= 0.75) return "text-[#ff4000]";
    if (c >= 0.5) return "text-[#faaa8d]";
    if (c >= 0.25) return "text-[#f59e0b]";
    return "text-[#10b981]";
  };

  // Feature data normalized for progress bars [0, 1]
  const telemetryMetrics = [
    {
      id: "border_closeness",
      label: "Border Closeness (Upper & Lateral Depth)",
      val: features.depth_features.border_closeness,
      color: "bg-[#50b2c0]",
      textCol: "text-[#50b2c0]",
    },
    {
      id: "wall_ceiling_ratio",
      label: "Wall & Ceiling Ratio (Semantic Enclosure)",
      val: features.semantic_features.wall_ceiling_ratio,
      color: "bg-[#ff4000]",
      textCol: "text-[#ff4000]",
    },
    {
      id: "sky_open_ratio",
      label: "Sky & Open Space Ratio (Tension Suppression)",
      val: features.semantic_features.sky_open_ratio,
      color: "bg-[#10b981]",
      textCol: "text-[#10b981]",
    },
    {
      id: "crowd_density",
      label: "Crowd Density (Occluded Human Packing)",
      val: features.pose_features.crowd_density,
      color: "bg-[#faaa8d]",
      textCol: "text-[#faaa8d]",
    },
    {
      id: "body_compression",
      label: "Body Compression (Torso-Normalized Pose)",
      val: features.pose_features.body_compression,
      color: "bg-[#c084fc]",
      textCol: "text-[#c084fc]",
    },
    {
      id: "edge_clutter",
      label: "Edge Clutter (Visual Complexity)",
      val: features.cv_features.edge_clutter,
      color: "bg-[#f59e0b]",
      textCol: "text-[#f59e0b]",
    },
    {
      id: "darkness_level",
      label: "Darkness Level (Shadow Submersion)",
      val: features.cv_features.darkness_level,
      color: "bg-[#94a3b8]",
      textCol: "text-[#94a3b8]",
    },
    {
      id: "chromatic_warmth",
      label: "Chromatic Warmth (Light Sources)",
      val: features.cv_features.chromatic_warmth,
      color: "bg-[#facc15]",
      textCol: "text-[#facc15]",
    },
  ];

  return (
    <div className="w-full flex flex-col gap-6">
      {/* 1. Psychological Scorecard */}
      <div className="bg-[#2a292a] border-2 border-[#343334] rounded-sm p-6 sm:p-8 space-y-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#343334] pb-6">
          <div>
            <span className="text-xs font-fira uppercase text-[#736f71] font-bold tracking-wider">
              Non-Linear Claustrophobia Score (C)
            </span>
            <div className="flex items-baseline gap-3 mt-1">
              <span
                className={`text-5xl font-chivo font-black tracking-tight ${getScoreNumberColor(
                  scores.claustrophobia_score_C
                )}`}
              >
                {scores.claustrophobia_score_C.toFixed(3)}
              </span>
              <span className="text-sm font-fira text-[#736f71]">/ 1.000</span>
            </div>
          </div>

          <div className="flex flex-col sm:items-end gap-1.5">
            <span className="text-xs font-fira text-[#736f71] uppercase font-bold">
              Classification State:
            </span>
            <span
              className={`px-4 py-1.5 rounded-sm text-sm font-fira font-bold uppercase tracking-wider border ${getClassificationColor(
                classification
              )}`}
            >
              {classification}
            </span>
          </div>
        </div>

        {/* Mathematical Confinement Sub-metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-fira text-xs">
          <div className="bg-[#201e1f] p-4 rounded-sm border border-[#343334] space-y-1">
            <span className="text-[#736f71] block">Physical Base Confinement (T)</span>
            <span className="text-xl font-bold text-[#feefdd]">
              {scores.trapping_base_T.toFixed(4)}
            </span>
          </div>

          <div className="bg-[#201e1f] p-4 rounded-sm border border-[#343334] space-y-1">
            <span className="text-[#736f71] block">Amplified Score (R)</span>
            <span className="text-xl font-bold text-[#feefdd]">
              {scores.raw_score_R.toFixed(4)}
            </span>
          </div>

          <div className="bg-[#201e1f] p-4 rounded-sm border border-[#343334] space-y-1">
            <span className="text-[#736f71] block">Dominant Factor</span>
            <span className="text-xl font-bold text-[#50b2c0] uppercase truncate block">
              {control.dominant_confinement_factor || "WALL_ENCLOSURE"}
            </span>
          </div>
        </div>

        {/* Psychological Narrative Explanation */}
        {(() => {
          const explanationText =
            control.narrative_explanation ||
            (typeof control.explanation === "string"
              ? control.explanation
              : typeof control.explanation === "object" && control.explanation !== null
              ? `Scene evaluated with physical trapping base T = ${scores.trapping_base_T.toFixed(3)} driven primarily by ${control.explanation.dominant_confinement_source || "structure"}.`
              : null);

          if (!explanationText) return null;

          return (
            <div className="p-4 rounded-sm bg-[#201e1f]/80 border border-[#343334] text-xs font-fira text-[#feefdd]/90 leading-relaxed">
              <div className="flex items-center gap-1.5 text-[#50b2c0] font-bold uppercase mb-1">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Perceptual Diagnostics</span>
              </div>
              {explanationText}
            </div>
          );
        })()}
      </div>

      {/* 2. Stage 1 Extracted Telemetry (8 Features) */}
      <div className="bg-[#2a292a] border-2 border-[#343334] rounded-sm p-6 sm:p-8 space-y-5 shadow-xl">
        <div>
          <h3 className="text-sm font-fira font-bold text-[#feefdd] uppercase tracking-wider">
            Stage 1: Extracted Feature Telemetry
          </h3>
          <p className="text-xs text-[#736f71] font-fira mt-0.5">
            Real-time sensory extraction from Depth Anything V2, SegFormer ADE20K, YOLO11-Pose, and OpenCV.
          </p>
        </div>

        {/* 8 Feature Bars */}
        <div className="space-y-3.5 font-fira text-xs pt-1">
          {telemetryMetrics.map((m) => {
            const pct = Math.min(100, Math.max(0, m.val * 100));
            return (
              <div key={m.id} className="space-y-1">
                <div className="flex justify-between text-[#feefdd]">
                  <span className="text-[#feefdd]/80">{m.label}</span>
                  <span className={`font-bold ${m.textCol}`}>{m.val.toFixed(3)}</span>
                </div>
                <div className="w-full bg-[#201e1f] h-2.5 rounded-sm overflow-hidden border border-[#343334]">
                  <div
                    className={`h-full ${m.color} transition-all duration-700 ease-out`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
