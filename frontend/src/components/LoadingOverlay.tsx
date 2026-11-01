"use client";

import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

interface WarpBox {
  id: number;
  top: string;
  duration: number;
  delay: number;
  color: string;
}

export function LoadingOverlay({ isProcessing }: { isProcessing: boolean }) {
  const [elapsed, setElapsed] = useState<number>(0);
  const [boxes, setBoxes] = useState<WarpBox[]>([]);

  useEffect(() => {
    if (!isProcessing) {
      setElapsed(0);
      return;
    }

    const startTime = Date.now();
    const timer = setInterval(() => {
      setElapsed(parseFloat(((Date.now() - startTime) / 1000).toFixed(1)));
    }, 100);

    // Generate 36 warp boxes
    const colors = ["bg-[#ff4000]", "bg-[#50b2c0]", "bg-[#faaa8d]"];
    const generated: WarpBox[] = [];
    for (let i = 0; i < 36; i++) {
      generated.push({
        id: i,
        top: `${Math.random() * 100}%`,
        duration: Math.random() * 0.9 + 0.6,
        delay: Math.random() * 0.6,
        color: colors[Math.floor(Math.random() * colors.length)],
      });
    }
    setBoxes(generated);

    return () => clearInterval(timer);
  }, [isProcessing]);

  if (!isProcessing) return null;

  // Dynamic status text based on elapsed time
  let statusText = "Stage 1: Extracting spatial depth, semantics & pose crowding...";
  if (elapsed > 3.0 && elapsed <= 5.0) {
    statusText = "Stage 2: Calculating non-linear claustrophobia scoring (T -> R -> C)...";
  } else if (elapsed > 5.0 && elapsed <= 8.0) {
    statusText = "Stage 3: Applying radial vignette, inward depth warp & edge jitter...";
  } else if (elapsed > 8.0) {
    statusText = "Stage 4: Synthesizing Multi-ControlNet diffusion & optical lens distortion on Kaggle...";
  }

  return (
    <div className="fixed inset-0 z-50 bg-[#201e1f]/90 backdrop-blur-md overflow-hidden flex items-center justify-center animate-in fade-in duration-300">
      {/* Animated Flowing Warp Boxes */}
      <div className="absolute inset-0 pointer-events-none">
        {boxes.map((box) => (
          <div
            key={box.id}
            style={{
              top: box.top,
              animationDuration: `${box.duration}s`,
              animationDelay: `${box.delay}s`,
            }}
            className={`warp-box ${box.color}/80 shadow-[0_0_12px_rgba(80,178,192,0.6)]`}
          />
        ))}
      </div>

      {/* Center Modal Card */}
      <div className="z-10 bg-[#201e1f]/95 p-8 rounded-sm border-2 border-[#50b2c0] flex flex-col items-center gap-5 shadow-[0_0_45px_rgba(80,178,192,0.35)] max-w-md w-full mx-4 text-center">
        <div className="w-14 h-14 rounded-full border-2 border-[#50b2c0] flex items-center justify-center">
          <Loader2 className="w-7 h-7 text-[#50b2c0] animate-spin" />
        </div>

        <div className="space-y-1.5">
          <h2 className="text-2xl font-doto font-bold text-[#feefdd] uppercase tracking-tight">
            Transforming Space
          </h2>
          <p className="text-[#50b2c0] font-fira tracking-tight text-xs uppercase font-bold min-h-[36px] flex items-center justify-center">
            {statusText}
          </p>
        </div>

        {/* Live Elapsed Timer */}
        <div className="px-4 py-1.5 rounded-sm bg-[#2a292a] border border-[#343334] text-xs font-fira text-[#feefdd]/90">
          Elapsed: <span className="font-bold text-[#faaa8d]">{elapsed.toFixed(1)}s</span>
        </div>
      </div>
    </div>
  );
}
