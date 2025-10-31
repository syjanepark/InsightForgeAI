"use client";
import { useEffect, useState } from "react";
import { getChartColumns, getChartPreview, summarizeCharts, getDistinctValues } from "@/lib/api";
import Markdown from "react-markdown";
import { ChartRenderer } from "./ChartGrid";
import {showInfo, showWarning } from "@/components/ui/toast";
import { useScreenLoader } from "@/components/ui/ScreenLoader";
import { focusElement } from "@/components/ui/FocusBody";

type SlotState = { chart_type: 'bar'|'line'|'pie'; x: string; y: string; yFields?: string[]; agg: 'sum'|'mean'|'count'; filterCol?: string; filterValues?: string[]; preview?: { type: string; spec: any } };

export function ChartBuilder({ runId }: { runId: string }) {
  const { showLoader, hideLoader } = useScreenLoader();
  const [cols, setCols] = useState<{ numeric: string[]; categorical: string[]; datetime: string[] }>({ numeric: [], categorical: [], datetime: [] });
  const [summary, setSummary] = useState<string>("");
  const [slots, setSlots] = useState<SlotState[]>([
    { chart_type: 'line', x: '', y: '', agg: 'sum' },
    { chart_type: 'bar', x: '', y: '', agg: 'sum' },
    { chart_type: 'bar', x: '', y: '', agg: 'sum' },
  ]);

  useEffect(() => {
    (async () => {
      const c = await getChartColumns(runId);
      setCols(c);
      const defLineX = c.datetime[0] || c.categorical[0] || c.numeric[0] || '';
      const defBarX = c.categorical[0] || c.datetime[0] || c.numeric[0] || '';
      const defY = c.numeric[0] || '';
      setSlots(s => s.map((sl, i) => {
        const t = sl.chart_type;
        const x = sl.x || (t === 'bar' ? defBarX : defLineX);
        return { ...sl, x, y: sl.y || defY };
      }));
    })();
  }, [runId]);

  async function generate(i: number) {
    const s = slots[i];
    if (!s.x || (!s.y && !(s.chart_type === 'pie' && s.yFields && s.yFields.length))) return;
    if(s.x === s.y){
      showWarning("X and Y axis cannot be the same");
      return;
    }
    showLoader();
    try {
      const payload: any = { run_id: runId, x: s.x, y: s.y, agg: s.agg, chart_type: s.chart_type };
      if ((s.chart_type === 'pie' || s.chart_type === 'line' || s.chart_type === 'bar') && s.yFields && s.yFields.length) {
        if (s.chart_type === 'pie') {
          payload.y_fields = s.yFields;
        }
        if (!payload.y) payload.y = s.yFields[0];
      }
      if (s.filterCol && s.filterValues && s.filterValues.length) {
        payload.filter_col = s.filterCol;
        payload.filter_values = s.filterValues;
      }
      let res = await getChartPreview(payload);
      if(null == res){
        res = { type: 'error', spec: { message: 'No preview available' } };
      }
      setSlots(prev => prev.map((p, idx) => idx === i ? { ...p, preview: res } : p));
    } catch (err: any) {
      const msg = (err?.message || '').toString();
      showWarning(msg.includes('400') ? msg.replace(/^Error:\s*/,'') : (msg || 'Preview failed. Please check your selections.'));
    } finally {
      hideLoader();
    }
  }

  function update(i: number, patch: Partial<SlotState>) {
    setSlots(prev => prev.map((p, idx) => {
      if (idx !== i) return p;
      const next = { ...p, ...patch } as SlotState;
      // Auto-correct axes based on chart type to avoid invalid combos
      if (patch.chart_type) {
        if (patch.chart_type === 'pie') {
          const nx = (cols.categorical[0] || cols.datetime[0] || next.x);
          const ny = (cols.numeric[0] || next.y);
          return { ...next, x: nx, y: ny };
        }
        if (patch.chart_type === 'bar') {
          const nx = (cols.categorical[0] || cols.datetime[0] || next.x);
          const ny = (cols.numeric[0] || next.y);
          return { ...next, x: nx, y: ny };
        }
        if (patch.chart_type === 'line') {
          const nx = (cols.datetime[0] || cols.categorical[0] || next.x);
          const ny = (cols.numeric[0] || next.y);
          return { ...next, x: nx, y: ny };
        }
      }
      return next;
    }));
  }

  const uniq = (arr: string[]) => Array.from(new Set((arr || []).map(String)));
  const allX = uniq([...(cols.datetime || []), ...(cols.categorical || []), ...(cols.numeric || [])]);
  const pieX = uniq([...(cols.datetime || []), ...(cols.categorical || [])]); // pie accepts date-like or categorical
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
                <option value="pie">pie</option>
              </select>
              <label className="text-sm text-gray-500">X-axis (categories or time)</label>
              <select className="border rounded px-2 py-1" value={s.x} onChange={e => update(i, { x: e.target.value })}>
                {(s.chart_type === 'pie' ? pieX : allX).map((c, idx) => <option key={`${c}-${idx}`} value={c}>{c}</option>)}
              </select>
              <label className="text-sm text-gray-500">{s.chart_type === 'pie' ? 'Metric(s) (Y) – select one or more' : 'Y-axis (numeric metric)'}</label>
              {s.chart_type === 'pie' || s.chart_type === 'line' || s.chart_type === 'bar' ? (
                <select
                  multiple
                  className="border rounded px-2 py-1 h-24"
                  value={s.yFields || []}
                  onChange={e => {
                    const opts = Array.from(e.target.selectedOptions).map(o => o.value);
                    update(i, { yFields: opts, y: opts[0] || s.y });
                  }}
                >
                  {yNums.map((c, idx) => <option key={`${c}-${idx}`} value={c}>{c}</option>)}
                </select>
              ) : (
                <select className="border rounded px-2 py-1" value={s.y} onChange={e => update(i, { y: e.target.value })}>
                  {yNums.map((c, idx) => <option key={`${c}-${idx}`} value={c}>{c}</option>)}
                </select>
              )}

              {/* Filter controls */}
              <label className="text-sm text-gray-500 mt-2">Filter by (optional)</label>
              <select className="border rounded px-2 py-1" value={s.filterCol || ''} onChange={async e => {
                const col = e.target.value;
                if (!col) { update(i, { filterCol: undefined, filterValues: [] }); return; }
                update(i, { filterCol: col, filterValues: [] });
                try {
                  const distinct = await getDistinctValues(runId, col, 100);
                  // stash on slot preview spec meta so we don't add new state shape; or we can track locally
                  (window as any).__distinct ||= {};
                  (window as any).__distinct[`${runId}:${col}`] = distinct.values;
                } catch {}
              }}>
                <option value="">-- none --</option>
                {uniq([...(cols.categorical||[])]).map((c, idx) => <option key={`${c}-${idx}`} value={c}>{c}</option>)}
              </select>
              <select
                multiple
                className="border rounded px-2 py-1 h-24"
                value={s.filterValues || []}
                onChange={e => {
                  const opts = Array.from(e.target.selectedOptions).map(o => o.value);
                  update(i, { filterValues: opts });
                }}
              >
                {(() => {
                  const vals: string[] = (s.filterCol && (window as any).__distinct?.[`${runId}:${s.filterCol}`]) || [];
                  return vals.map((v, idx) => <option key={`${v}-${idx}`} value={v}>{v}</option>);
                })()}
              </select>
              <select className="border rounded px-2 py-1" value={s.agg} onChange={e => update(i, { agg: e.target.value as any })}>
                <option value="sum">sum</option>
                <option value="mean">mean</option>
                <option value="count">count</option>
              </select>
            </div>
            <div className="content-end">
              <button className="bg-[#A18AFF] text-white px-3 py-2 rounded" onClick={() => generate(i)}>Generate</button>
            </div>
            {s.preview && (
              <div className="mt-3"><ChartRenderer type={s.preview.type} spec={s.preview.spec} /></div>
            )}
            {s.preview && s.preview.spec?.meta?.available && s.chart_type === 'pie' && (
              <div className="mt-2 flex flex-wrap gap-2">
                <span className="text-xs text-gray-500 mr-2">Metric toggle:</span>
                {s.preview.spec.meta.available.map((m: string) => (
                  <button
                    key={m}
                    className={`text-xs px-2 py-1 rounded border ${m === s.preview.spec.meta.selected ? 'bg-[#A18AFF] text-white border-[#A18AFF]' : 'border-gray-300 text-gray-700'}`}
                    onClick={() => { update(i, { y: m }); generate(i); }}
                  >
                    {m}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 flex items-center gap-3">
        <button
          className="bg-[#6B5AE0] text-white px-3 py-2 rounded"
          onClick={async () => {
            showLoader();
            try {
              const selected = slots.filter(s => s.preview).slice(0,3).map(s => s.preview!) as any;
              if (selected.length === 0) {
                showWarning("Generate at least one chart first");
                return;
              }
              const res = await summarizeCharts({ run_id: runId, charts: selected });
              setSummary(res.answer || "");
              setTimeout(() => focusElement(".pageSummaryBlock"), 300);
            } catch (err: any) {
              showWarning((err?.message || "Summarize failed").toString());
            } finally {
              hideLoader();
            }
          }}
        >
          Summarize current charts
        </button>
      </div>
      {summary && (
        <div className="mt-3 border pageSummaryBlock rounded-xl p-4 bg-[#F7F7FA]">
          <Markdown>{summary}</Markdown>
        </div>
      )}
    </div>
  );
}


