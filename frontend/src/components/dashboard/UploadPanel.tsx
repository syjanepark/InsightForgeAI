"use client";
import { useState } from "react";
import { analyzeCSV } from "@/lib/api";

export function UploadPanel({ onAnalyzed }: { onAnalyzed: (runId: string, summary: any, charts: any[]) => void }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onUpload(file: File) {
    setError(null);
    setLoading(true);
    try {
      const res = await analyzeCSV(file);
      onAnalyzed(res.run_id, res.summary, res.charts);
    } catch (e: any) {
      setError(e?.message || "Failed to analyze file");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-xl p-6 bg-gradient-to-br from-[#EAE6FF] to-white border">
      <div className="font-semibold mb-2">Upload CSV</div>
      <input
        type="file"
        accept=".csv,text/csv"
        onChange={e => e.target.files?.[0] && onUpload(e.target.files[0])}
        className="block"
      />
      {loading && <div className="mt-3 text-[#A18AFF]">Analyzing dataset…</div>}
      {error && <div className="mt-3 text-red-500 text-sm">{error}</div>}
    </div>
  );
}


