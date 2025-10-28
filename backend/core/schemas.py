from pydantic import BaseModel
from typing import List, Optional

class KPI(BaseModel):
    name: str
    value: float
    delta_pct: Optional[float] = None

class Chart(BaseModel):
    type: str
    spec: dict

class Evidence(BaseModel):
    source: str
    title: str
    url: str

class Insight(BaseModel):
    title: str
    why: str
    recommendations: List[str]
    evidence: Optional[List[Evidence]] = None

class AnalyzeResponse(BaseModel):
    run_id: str
    kpis: List[KPI]
    charts: List[Chart]
    keywords: List[str]
    insights: List[Insight]

class ChatMessage(BaseModel):
    question: str
    run_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    visualizations: Optional[List[Chart]] = None
    suggested_actions: Optional[List[dict]] = None
    citations: Optional[List[str]] = None