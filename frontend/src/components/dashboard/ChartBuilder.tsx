"use client";
import { useEffect, useState } from "react";
import { getChartColumns, getChartPreview, summarizeCharts } from "@/lib/api";
import Markdown from "react-markdown";
import { ChartRenderer } from "./ChartGrid";

type SlotState = { chart_type: 'bar'|'line'|'scatter'|'pie'; x: string; y: string; agg: 'sum'|'mean'|'count'; preview?: { type: string; spec: any } };

export function ChartBuilder({ runId }: { runId: string }) {
  const [cols, setCols] = useState<{ numeric: string[]; categorical: string[]; datetime: string[] }>({ numeric: [], categorical: [], datetime: [] });
  const [summary, setSummary] = useState<string>("");
  const [slots, setSlots] = useState<SlotState[]>([
    { chart_type: 'line', x: '', y: '', agg: 'sum' },
    { chart_type: 'bar', x: '', y: '', agg: 'sum' },
    { chart_type: 'scatter', x: '', y: '', agg: 'sum' },
  ]);

  useEffect(() => {
    (async () => {
      const c = await getChartColumns(runId);
      setCols(c);
      const defX = c.datetime[0] || c.categorical[0] || c.numeric[0] || '';
      const defY = c.numeric[0] || '';
      setSlots(s => s.map((sl, i) => ({ ...sl, x: sl.x || defX, y: sl.y || defY })));
    })();
  }, [runId]);

  async function generate(i: number) {
    const s = slots[i];
    if (!s.x || !s.y) return;
    const res = await getChartPreview({ run_id: runId, x: s.x, y: s.y, agg: s.agg, chart_type: s.chart_type });
    setSlots(prev => prev.map((p, idx) => idx === i ? { ...p, preview: res } : p));
  }

  function update(i: number, patch: Partial<SlotState>) {
    setSlots(prev => prev.map((p, idx) => idx === i ? { ...p, ...patch } : p));
  }

  const uniq = (arr: string[]) => Array.from(new Set((arr || []).map(String)));
  const allX = uniq([...(cols.datetime || []), ...(cols.categorical || []), ...(cols.numeric || [])]);
  const yNums = uniq(cols.numeric || []);

  return (
    <div className="rounded-xl p-6 bg-white shadow-sm border mt-4">
      <div className="font-semibold mb-3">Build up to 3 charts</div>
      <div className="grid md:grid-cols-3 gap-4">
        {slots.map((s, i) => (
          <div key={i} className="border rounded-xl p-4">
            <div className="font-medium mb-2">Chart {i+1} {i===0 ? "(required)" : "(optional)"}</div>
            <div className="flex flex-col gap-2 mb-3">
              <select className="border rounded px-2 py-1" value={s.chart_type} onChange={e => update(i, { chart_type: e.target.value as any })}>
                <option value="line">line</option>
                <option value="bar">bar</option>
                <option value="scatter">scatter</option>
                <option value="pie">pie</option>
              </select>
              <select className="border rounded px-2 py-1" value={s.x} onChange={e => update(i, { x: e.target.value })}>
                {allX.map((c, idx) => <option key={`${c}-${idx}`} value={c}>{c}</option>)}
              </select>
              <select className="border rounded px-2 py-1" value={s.y} onChange={e => update(i, { y: e.target.value })}>
                {yNums.map((c, idx) => <option key={`${c}-${idx}`} value={c}>{c}</option>)}
              </select>
              <select className="border rounded px-2 py-1" value={s.agg} onChange={e => update(i, { agg: e.target.value as any })}>
                <option value="sum">sum</option>
                <option value="mean">mean</option>
                <option value="count">count</option>
              </select>
            </div>
            <button className="bg-[#A18AFF] text-white px-3 py-2 rounded" onClick={() => generate(i)}>Generate</button>
            {s.preview && (
              <div className="mt-3"><ChartRenderer type={s.preview.type} spec={s.preview.spec} /></div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 flex items-center gap-3">
        <button
          className="bg-[#6B5AE0] text-white px-3 py-2 rounded"
          onClick={async () => {
            const selected = slots.filter(s => s.preview).slice(0,3).map(s => s.preview!) as any;
            if (selected.length === 0) return;
            const res = await summarizeCharts({ run_id: runId, charts: selected });
            setSummary(res.answer || "");
          }}
        >
          Summarize current charts
        </button>
      </div>
      {summary && (
        <div className="mt-3 border rounded-xl p-4 bg-[#F7F7FA]">
          <Markdown>{summary}</Markdown>
        </div>
      )}
    </div>
  );
}


