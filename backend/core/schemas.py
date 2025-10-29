from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class KPI(BaseModel):
    name: str
    value: float
    delta_pct: Optional[float] = None
    trend: Optional[str] = None
    format: Optional[str] = None
    business_impact: Optional[str] = None

class Chart(BaseModel):
    type: str
    spec: dict
    # Support both legacy and new chart formats
    chart_type: Optional[str] = None
    title: Optional[str] = None
    subtitle: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    insights: Optional[List[str]] = None
    business_context: Optional[str] = None

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
    kpis: Union[List[KPI], List[Dict[str, Any]]]  # Support both formats
    charts: Union[List[Chart], List[Dict[str, Any]]]  # Support both formats
    keywords: List[str]
    insights: Union[List[Insight], List[str]]  # Support both string and object insights
    summary: Optional[str] = None  # New executive summary field

class ChatMessage(BaseModel):
    question: str
    run_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    visualizations: Optional[List[Chart]] = None
    suggested_actions: Optional[List[dict]] = None
    citations: Optional[List[str]] = None