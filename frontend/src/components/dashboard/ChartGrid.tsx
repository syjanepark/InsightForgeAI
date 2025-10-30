"use client";

export function ChartGrid({ charts }: { charts: Array<{ type: string; spec: any }> }) {
  const list = Array.isArray(charts) ? charts.slice(0, 3) : [];
  return (
    <div className="grid md:grid-cols-3 gap-4">
      {list.map((c, idx) => (
        <div key={idx} className="rounded-xl p-4 bg-white shadow-sm border">
          <ChartRenderer type={c.type} spec={c.spec} />
        </div>
      ))}
    </div>
  );
}

import { Chart as GenericChart } from "@/components/charts";

function toRechartsData(type: string, spec: any): { title: string; data: any[]; xKey?: string; yKey?: string } {
  const title = spec?.title || "Chart";
  if (type === 'scatter' && spec?.data?.points) {
    return { title, data: (spec.data.points || []).map((p: any) => ({ x: Number(p.x), y: Number(p.y) })), xKey: 'x', yKey: 'y' };
  }
  if (spec?.data?.labels && spec?.data?.datasets?.[0]?.data) {
    const labels = spec.data.labels as any[];
    const series = spec.data.datasets[0].data as any[];
    const data = labels.map((l, i) => ({ name: String(l), value: Number(series[i] ?? 0) }));
    return { title, data, xKey: 'name', yKey: 'value' };
  }
  return { title, data: [], xKey: 'name', yKey: 'value' };
}

export function ChartRenderer({ type, spec }: { type: string; spec: any }) {
  const cfg = toRechartsData(type, spec);
  return (
    <GenericChart title={cfg.title} data={cfg.data} type={type as any} xKey={cfg.xKey} yKey={cfg.yKey} />
  );
}


