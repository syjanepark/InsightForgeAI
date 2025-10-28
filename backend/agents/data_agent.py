import pandas as pd
import io, uuid
from typing import Dict, Any
from core.schemas import AnalyzeResponse, KPI, Chart, Insight, Evidence

class DataAgent:
    def analyze(self, file_bytes: bytes) -> AnalyzeResponse:
        df = pd.read_csv(io.BytesIO(file_bytes))

        # --- basic profiling ---
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        kpis = []
        for col in numeric_cols[:3]:
            val = df[col].mean()
            kpis.append(KPI(name=col, value=round(val, 2)))

        # --- simple chart spec (mock for frontend) ---
        chart = Chart(
            type="bar",
            spec={
                "x": numeric_cols[:3],
                "y": [float(df[col].mean()) for col in numeric_cols[:3]],
                "title": "Average of top 3 numeric columns"
            }
        )

        # --- keywords from column names ---
        keywords = [c for c in df.columns if c not in numeric_cols][:5]

        # --- mock insight (placeholder) ---
        insights = [
            Insight(
                title="Preliminary data summary",
                why="Shows average metrics for key columns.",
                recommendations=["Upload full dataset", "Check data consistency"],
                evidence=[Evidence(source="data_agent", title="local analysis", url="N/A")]
            )
        ]

        return AnalyzeResponse(
            run_id=str(uuid.uuid4()),
            kpis=kpis,
            charts=[chart],
            keywords=keywords,
            insights=insights
        )