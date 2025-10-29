"""
Temporary simplified orchestrator for testing CORS and basic functionality
"""
import uuid
import pandas as pd
import io
from typing import List, Dict, Any
from agents.data_agent import DataAgent
from core.schemas import AnalyzeResponse
from core.data_store import data_store

class SimpleOrchestrator:
    def __init__(self):
        self.data_agent = DataAgent()

    async def run(self, file_bytes: bytes, filename: str = "dataset.csv") -> AnalyzeResponse:
        """Simplified analysis pipeline for testing"""
        print(f"🚀 Starting simple analysis for {filename}")
        
        # Parse the CSV and store it for future chat queries
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            print(f"📊 Successfully parsed CSV: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"❌ CSV parsing failed: {e}")
            raise e

        # Use the legacy data agent
        try:
            data_result = self.data_agent.analyze(file_bytes)
            print("✅ Data analysis completed")
        except Exception as e:
            print(f"❌ Data analysis failed: {e}")
            raise e
        
        # Store the data for chat queries
        run_id = data_result.run_id
        try:
            data_store.store_data(run_id, df, {
                'kpis': [kpi.dict() for kpi in data_result.kpis],
                'charts': [chart.dict() for chart in data_result.charts],
                'keywords': data_result.keywords,
                'insights': [insight.dict() for insight in data_result.insights]
            })
            print("✅ Data stored successfully")
        except Exception as e:
            print(f"❌ Data storage failed: {e}")
            # Don't fail the whole request for storage issues
        
        print(f"✅ Simple analysis complete for {filename}")
        return data_result
