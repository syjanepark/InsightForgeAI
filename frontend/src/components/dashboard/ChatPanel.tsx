"use client";
import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import { sendChatMessage } from "@/lib/api";
import { ChartRenderer } from "./ChartGrid";

export function ChatPanel({ runId, context }: { runId: string; context: string }) {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState<Array<{ role: "user" | "ai"; content: string; viz?: any; citations?: string[] }>>([]);

  useEffect(() => {
    const onSeed = (e: any) => setOpen(true) || onAsk(e.detail);
    window.addEventListener("openChatSeed", onSeed);
    return () => window.removeEventListener("openChatSeed", onSeed);
  }, []);

  async function onAsk(question: string) {
    setMsgs(prev => [...prev, { role: "user", content: question }]);
    setMsgs(prev => [...prev, { role: "ai", content: "🌐 Gathering external context…" }]);
    const res = await sendChatMessage({ question, run_id: runId });
    setMsgs(prev => [...prev.slice(0, -1), { role: "ai", content: res.answer, viz: res.visualizations?.[0], citations: res.citations }]);
  }

  if (!open) return null;

  return (
    <div className="fixed right-0 top-0 h-full w-full md:w-[420px] bg-white border-l shadow-2xl">
      <div className="p-3 border-b flex justify-between">
        <div className="font-semibold">Analyst Copilot</div>
        <button onClick={() => setOpen(false)}>✕</button>
      </div>
      <div className="p-3 space-y-3 overflow-auto h-[calc(100%-110px)]">
        {msgs.map((m, i) => (
          <div key={i} className={`${m.role === "ai" ? "bg-[#F6F3FF] border" : "bg-[#EAE6FF]"} p-3 rounded-xl`}>
            <Markdown>{m.content}</Markdown>
            {m.viz && <div className="mt-2"><ChartRenderer type={m.viz.type} spec={m.viz.spec} /></div>}
            {m.citations && m.citations.length > 0 && (
              <div className="mt-2 text-xs text-slate-500">
                Sources: {m.citations.map((c, idx) => <a key={idx} className="underline mr-2" href={c} target="_blank" rel="noreferrer">{new URL(c).hostname}</a>)}
              </div>
            )}
          </div>
        ))}
      </div>
      <ChatInput onSend={onAsk} />
    </div>
  );
}

function ChatInput({ onSend }: { onSend: (q: string) => void }) {
  const [q, setQ] = useState("");
  return (
    <div className="p-3 border-t flex gap-2">
      <input className="flex-1 border rounded-lg p-2" value={q} onChange={e => setQ(e.target.value)} placeholder="Ask a deeper question…" />
      <button className="bg-[#A18AFF] text-white px-3 py-2 rounded-lg" onClick={() => q && onSend(q)}>Send</button>
    </div>
  );
}


