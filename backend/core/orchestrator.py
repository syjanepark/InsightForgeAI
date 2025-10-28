import uuid
import pandas as pd
import io
from agents.data_agent import DataAgent
from agents.research_agent import ResearchAgent
from agents.insight_agent import InsightAgent
from core.schemas import AnalyzeResponse
from core.cache import cache_get, cache_set
from core.data_store import data_store

class Orchestrator:
    def __init__(self):
        self.data_agent = DataAgent()
        self.research_agent = ResearchAgent()
        self.insight_agent = InsightAgent()

    async def run(self, file_bytes: bytes) -> AnalyzeResponse:
        run_id = str(uuid.uuid4())
        
        # Parse the CSV and store it for future chat queries
        df = pd.read_csv(io.BytesIO(file_bytes))

        # Step 1: local data analysis
        data_result = self.data_agent.analyze(file_bytes)

        # Step 2: research (cached by keyword)
        evidence = []
        for kw in data_result.keywords:
            cached = cache_get(kw)
            if cached:
                evidence.extend(cached)
                continue
            res = await self.research_agent.query(kw)
            cache_set(kw, res)
            evidence.extend(res)

        # Step 3: insight generation
        insights = await self.insight_agent.synthesize(
            data_result, evidence
        )

        data_result.insights = insights
        data_result.run_id = run_id
        
        # Store the data and analysis results for chat queries
        data_store.store_data(run_id, df, {
            'kpis': data_result.kpis,
            'charts': data_result.charts,
            'keywords': data_result.keywords,
            'insights': insights
        })
        
        return data_result