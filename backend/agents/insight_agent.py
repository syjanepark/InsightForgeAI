import json
from jinja2 import Template
from services.you_smart import call_smart
from core.schemas import Insight, Evidence

PROMPT_TEMPLATE = """
DATA SUMMARY:
{{ summary }}

KPIS:
{{ kpis }}

EVIDENCE:
{{ evidence }}

TASK:
1) Explain likely causes of the top trends.
2) Produce 2–3 actionable insights.
Return JSON like:
{"insights":[{"title":"","why":"","recommendations":[]}]}
"""

class InsightAgent:
    async def synthesize(self, data_result, evidence):
        tmpl = Template(PROMPT_TEMPLATE)
        prompt = tmpl.render(
            summary="Dataset with numeric KPIs and market evidence",
            kpis=[k.model_dump() for k in data_result.kpis],
            evidence=evidence
        )
        response = call_smart(prompt)
        try:
            parsed = json.loads(response)
            insights = []
            for i in parsed.get("insights", []):
                insights.append(
                    Insight(
                        title=i.get("title",""),
                        why=i.get("why",""),
                        recommendations=i.get("recommendations", []),
                        evidence=[Evidence(**e) for e in evidence[:2]]
                    )
                )
            return insights
        except Exception:
            # fallback
            return [
                Insight(
                    title="Market signals summary",
                    why="Unable to parse Smart API JSON.",
                    recommendations=["Check prompt", "Retry later"]
                )
            ]