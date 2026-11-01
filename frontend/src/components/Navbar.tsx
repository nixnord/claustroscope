"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  CheckCircle2,
  Cpu,
  ExternalLink,
  Globe,
  RefreshCw,
  Settings,
  XCircle,
} from "lucide-react";
import {
  checkServerHealth,
  clearCustomTunnelUrl,
  getApiBaseUrl,
  HealthStatus,
  setCustomTunnelUrl,
} from "@/lib/api";

export function Navbar() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [currentUrl, setCurrentUrl] = useState<string>("");
  const [inputUrl, setInputUrl] = useState<string>("");
  const [modalStatus, setModalStatus] = useState<string>("");
  const [testingConnection, setTestingConnection] = useState<boolean>(false);

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const data = await checkServerHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const active = getApiBaseUrl();
    setCurrentUrl(active);
    setInputUrl(active);
    fetchHealth();

    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleOpenModal = () => {
    const active = getApiBaseUrl();
    setCurrentUrl(active);
    setInputUrl(active);
    setModalStatus("");
    setIsModalOpen(true);
  };

  const handleTestAndSave = async () => {
    setTestingConnection(true);
    setModalStatus("Pinging endpoint...");
    try {
      setCustomTunnelUrl(inputUrl);
      const data = await checkServerHealth();
      setHealth(data);
      setCurrentUrl(getApiBaseUrl());
      setModalStatus(`Connected to ${data.device || "GPU"} successfully!`);
      setTimeout(() => {
        setIsModalOpen(false);
      }, 1000);
    } catch (err: any) {
      setModalStatus(`Connection failed: ${err.message}`);
    } finally {
      setTestingConnection(false);
    }
  };

  const handleResetDefault = () => {
    clearCustomTunnelUrl();
    const active = getApiBaseUrl();
    setCurrentUrl(active);
    setInputUrl(active);
    setModalStatus("Reset to default URL.");
    fetchHealth();
  };

  return (
    <>
      <header className="w-full border-b border-[#343334] bg-[#201e1f]/95 backdrop-blur sticky top-0 z-40 transition-all">
        <div className="w-full max-w-[1440px] mx-auto px-6 md:px-12 lg:px-16 h-20 flex items-center justify-between">
          {/* Logo & Tagline */}
          <div className="flex items-center gap-3.5">
            <div className="bg-[#ff4000] text-[#201e1f] p-2.5 rounded-sm rotate-3 shadow-[0_0_15px_rgba(255,64,0,0.3)] select-none">
              <Cpu className="w-6 h-6 -rotate-3 text-[#201e1f]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-doto font-bold tracking-tight text-[#feefdd] uppercase">
                  Claustro<span className="text-[#ff4000]">Scope</span>
                </span>
                <span className="px-2 py-0.5 rounded-sm bg-[#ff4000]/10 border border-[#ff4000]/30 text-[10px] font-fira font-bold text-[#ff4000] uppercase tracking-wider hidden sm:inline-block">
                  v2.0 Cloud
                </span>
              </div>
              <p className="text-[11px] font-fira tracking-tight text-[#50b2c0] font-bold uppercase">
                Psychological Perception & Multi-ControlNet Engine
              </p>
            </div>
          </div>

          {/* Right Status Badge & Actions */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleOpenModal}
              title="Click to configure Kaggle Cloudflare Tunnel URL"
              className="flex items-center gap-2.5 px-3.5 py-1.5 rounded-sm border border-[#343334] hover:border-[#50b2c0] bg-[#2a292a] hover:bg-[#343334] text-xs font-fira text-[#feefdd] transition-all cursor-pointer group"
            >
              <span className="relative flex h-2 w-2">
                {health?.gpu_active ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#10b981] opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[#10b981]"></span>
                  </>
                ) : loading ? (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#f59e0b] animate-pulse"></span>
                ) : (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#ef4444]"></span>
                )}
              </span>

              <span className="font-bold tracking-tight uppercase">
                {loading
                  ? "Checking GPU..."
                  : health?.gpu_active
                  ? `Kaggle GPU Online (${health.device || "T4"})`
                  : "Kaggle GPU Offline"}
              </span>

              <Settings className="w-3.5 h-3.5 text-[#736f71] group-hover:text-[#50b2c0] transition-colors ml-1" />
            </button>
          </div>
        </div>
      </header>

      {/* Cloudflare Tunnel Configuration Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200">
          <div className="bg-[#2a292a] border-2 border-[#343334] rounded-sm max-w-lg w-full p-6 sm:p-8 space-y-5 shadow-2xl relative">
            <div className="flex items-center justify-between border-b border-[#343334] pb-4">
              <div className="flex items-center gap-2 text-[#50b2c0]">
                <Globe className="w-5 h-5" />
                <h3 className="font-doto text-lg font-bold text-[#feefdd] uppercase tracking-tight">
                  Kaggle Backend Tunnel Setup
                </h3>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="text-[#736f71] hover:text-[#feefdd] transition-colors p-1"
              >
                ✕
              </button>
            </div>

            <p className="text-xs font-fira text-[#feefdd]/80 leading-relaxed">
              When running <code className="text-[#50b2c0]">kaggle_server.py</code> in your Kaggle notebook, a temporary Cloudflare URL is generated (e.g. <span className="text-[#faaa8d]">https://xxxx.trycloudflare.com</span>). Paste it below to re-link your frontend in real-time without restarting or redeploying:
            </p>

            <div className="space-y-2">
              <label className="block text-xs font-fira text-[#736f71] uppercase font-bold">
                Active Tunnel Endpoint:
              </label>
              <input
                type="text"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="https://xxxx.trycloudflare.com"
                className="w-full bg-[#201e1f] border border-[#343334] focus:border-[#50b2c0] rounded-sm px-3.5 py-2.5 text-xs font-fira text-[#feefdd] focus:outline-none transition-colors"
              />
              <div className="text-[11px] font-fira text-[#736f71] flex justify-between">
                <span>Default from .env: {process.env.NEXT_PUBLIC_KAGGLE_API_URL || "None"}</span>
                <button
                  type="button"
                  onClick={handleResetDefault}
                  className="text-[#50b2c0] hover:underline cursor-pointer"
                >
                  Reset Default
                </button>
              </div>
            </div>

            {/* Test Result Message */}
            {modalStatus && (
              <div className="p-3 rounded-sm bg-[#201e1f] border border-[#343334] text-xs font-fira text-[#feefdd]">
                {modalStatus}
              </div>
            )}

            {/* Model Inventory if online */}
            {health && (
              <div className="bg-[#201e1f] p-3 rounded-sm border border-[#343334] space-y-1.5 text-xs font-fira">
                <div className="flex justify-between text-[#feefdd]">
                  <span className="text-[#736f71]">Active Device:</span>
                  <span className="text-[#10b981] font-bold">{health.device}</span>
                </div>
                <div className="flex justify-between text-[#feefdd]">
                  <span className="text-[#736f71]">VRAM Allocated:</span>
                  <span className="text-[#50b2c0] font-bold">{health.vram_allocated_mb} MB</span>
                </div>
                <div className="text-[#736f71] pt-1 border-t border-[#343334]">
                  Loaded Modules:{" "}
                  <span className="text-[#feefdd]">{health.models_loaded?.join(", ") || "None"}</span>
                </div>
              </div>
            )}

            {/* Modal Actions */}
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="px-4 py-2 rounded-sm bg-[#201e1f] hover:bg-[#343334] text-xs font-fira text-[#feefdd] transition-colors border border-[#343334]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleTestAndSave}
                disabled={testingConnection}
                className="px-5 py-2 rounded-sm bg-[#50b2c0] hover:bg-[#3da1b0] text-[#201e1f] text-xs font-fira font-bold uppercase tracking-tight transition-transform active:scale-95 disabled:opacity-50 flex items-center gap-1.5"
              >
                {testingConnection ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Test & Connect
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
