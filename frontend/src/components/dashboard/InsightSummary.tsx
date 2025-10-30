"use client";
import { motion } from "framer-motion";

export function InsightSummary({ iri }: { iri: { insight?: string; reasoning?: string; implication?: string } }) {
  const items = [
    { label: "Insight", text: iri?.insight || "" },
    { label: "Reasoning", text: iri?.reasoning || "" },
    { label: "Implication", text: iri?.implication || "" },
  ].filter(i => i.text);

  return (
    <div className="rounded-xl p-6 bg-white shadow-sm border">
      {items.map((it, i) => (
        <motion.div key={it.label} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.2 }}>
          <div className="font-semibold text-[#6B5AE0]">{it.label}:</div>
          <div className="mb-3">{it.text}</div>
        </motion.div>
      ))}
      <button
        aria-label="Ask a deeper question"
        className="mt-2 text-white bg-[#A18AFF] px-3 py-2 rounded-lg"
        onClick={() => window.dispatchEvent(new CustomEvent("openChatSeed", { detail: "Why did this trend happen?" }))}
      >
        Ask deeper question
      </button>
    </div>
  );
}


