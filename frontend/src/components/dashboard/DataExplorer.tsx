"use client";
import { useEffect, useState } from "react";
import { getChartColumns, getChartPreview } from "@/lib/api";
import { ChartRenderer } from "./ChartGrid";

export function DataExplorer({ runId }: { runId: string }) {
  const [cols, setCols] = useState<{ numeric: string[]; categorical: string[]; datetime: string[] }>({ numeric: [], categorical: [], datetime: [] });
  const [x, setX] = useState<string>("");
  const [y, setY] = useState<string>("");
  const [agg, setAgg] = useState<string>("sum");
  const [chart, setChart] = useState<{ type: string; spec: any } | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const c = await getChartColumns(runId);
        setCols(c);
        const defX = c.datetime[0] || c.categorical[0] || c.numeric[0] || "";
        const defY = c.numeric[0] || "";
        setX(defX);
        setY(defY);
      } catch (e) {}
    })();
  }, [runId]);

  useEffect(() => {
    if (!x || !y) return;
    (async () => {
      try {
        const res = await getChartPreview({ run_id: runId, x, y, agg });
        setChart(res);
      } catch (e) {
        setChart(null);
      }
    })();
  }, [x, y, agg, runId]);

  const uniq = (arr: string[]) => Array.from(new Set((arr || []).map(String)));
  const allX = uniq([...(cols.datetime || []), ...(cols.categorical || []), ...(cols.numeric || [])]);
  const yNums = uniq(cols.numeric || []);
  return (
    <div className="rounded-xl p-6 bg-white shadow-sm border mt-4">
      <div className="font-semibold mb-3">Explore your data</div>
      <div className="flex flex-wrap gap-3 mb-3">
        <select className="border rounded px-2 py-1" value={x} onChange={e => setX(e.target.value)}>
          {allX.map((c, i) => <option key={`${c}-${i}`} value={c}>{c}</option>)}
        </select>
        <select className="border rounded px-2 py-1" value={y} onChange={e => setY(e.target.value)}>
          {yNums.map((c, i) => <option key={`${c}-${i}`} value={c}>{c}</option>)}
        </select>
        <select className="border rounded px-2 py-1" value={agg} onChange={e => setAgg(e.target.value)}>
          <option value="sum">sum</option>
          <option value="mean">mean</option>
          <option value="count">count</option>
        </select>
      </div>
      {chart ? (
        <ChartRenderer type={chart.type} spec={chart.spec} />
      ) : (
        <div className="text-sm text-slate-500">Pick x and y columns to preview a chart.</div>
      )}
    </div>
  );
}


