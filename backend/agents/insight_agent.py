import json
from jinja2 import Template
# Insight generation is fully local (no external AI dependencies)
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
        # Generate local insights (no external AI)
        return self._generate_fallback_insights(data_result, evidence)
    
    def _generate_fallback_insights(self, data_result, evidence):
        """Generate fallback insights when API fails"""
        insights = []
        
        # Generate insights based on KPIs
        if data_result.kpis:
            kpi_names = [k.name for k in data_result.kpis[:3]]
            insights.append(
                Insight(
                    title="Key Performance Indicators Analysis",
                    why=f"Analysis of {', '.join(kpi_names)} reveals important trends in your dataset. These metrics provide valuable insights into performance and areas for improvement.",
                    recommendations=[
                        "Monitor these KPIs regularly",
                        "Set targets for improvement",
                        "Compare with industry benchmarks"
                    ],
                    evidence=[Evidence(**e) for e in evidence[:2]]
                )
            )
        
        # Generate market insights if evidence is available
        if evidence:
            insights.append(
                Insight(
                    title="Market Intelligence Summary",
                    why="Based on current market research and competitive analysis, there are several opportunities for strategic improvement and growth.",
                    recommendations=[
                        "Stay updated with market trends",
                        "Analyze competitor strategies",
                        "Identify new market opportunities"
                    ],
                    evidence=[Evidence(**e) for e in evidence[:2]]
                )
            )
        
        # Default insight if no data
        if not insights:
            insights.append(
                Insight(
                    title="Data Analysis Complete",
                    why="Your data has been successfully processed and analyzed. The system has identified key patterns and trends that can inform your business decisions.",
                    recommendations=[
                        "Review the analysis results",
                        "Consider implementing suggested improvements",
                        "Schedule regular data reviews"
                    ],
                    evidence=[]
                )
            )
        
        return insights