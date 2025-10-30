from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
from core.data_store import data_store
from services.you_smart import call_smart

router = APIRouter(prefix="/chart", tags=["chart"])

class PreviewRequest(BaseModel):
    run_id: str
    x: str
    y: str
    agg: Optional[str] = "sum"  # sum|mean|count
    chart_type: Optional[str] = None  # 'bar'|'line'|'scatter'|'pie'

@router.get("/columns")
def list_columns(run_id: str):
    df: pd.DataFrame = data_store.get_data(run_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Run not found")
    # detect types
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(exclude=['number']).columns.tolist()
    # try detect datetime-like categoricals
    datetime_like = []
    for c in categorical:
        try:
            dt = pd.to_datetime(df[c], errors='coerce')
            if dt.notna().sum() > max(10, int(0.2*len(df))):
                datetime_like.append(c)
        except Exception:
            pass
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime_like}

@router.post("/preview")
def chart_preview(req: PreviewRequest):
    df: pd.DataFrame = data_store.get_data(req.run_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if req.x not in df.columns or req.y not in df.columns:
        raise HTTPException(status_code=400, detail="Columns not found")

    # Light cleanup for selected columns (handle commas/$ and date strings)
    try:
        if not pd.api.types.is_numeric_dtype(df[req.y]):
            col = df[req.y].astype(str).str.replace(r'[\$,\s,]', '', regex=True)
            df[req.y] = pd.to_numeric(col, errors='coerce')
    except Exception:
        pass

    # Decide chart type
    x_is_num = pd.api.types.is_numeric_dtype(df[req.x])
    y_is_num = pd.api.types.is_numeric_dtype(df[req.y])

    # If x is date-like parse year
    year_series = None
    if not x_is_num:
        try:
            dt = pd.to_datetime(df[req.x], errors='coerce')
            if dt.notna().sum() > max(10, int(0.2*len(df))):
                year_series = dt.dt.year
        except Exception:
            pass

    if (req.chart_type in (None, 'line')) and year_series is not None and pd.api.types.is_numeric_dtype(df[req.y]):
        tmp = pd.DataFrame({'year': year_series, 'val': df[req.y]}).dropna()
        if req.agg == 'mean':
            agg = tmp.groupby('year')['val'].mean().sort_index()
        elif req.agg == 'count':
            agg = tmp.groupby('year')['val'].count().sort_index()
        else:
            agg = tmp.groupby('year')['val'].sum().sort_index()
        return {
            "type": "line",
            "spec": {
                "title": f"{req.y} by Year",
                "data": {
                    "labels": agg.index.astype(str).tolist(),
                    "datasets": [{
                        "label": req.y,
                        "data": [float(v) for v in agg.values],
                        "borderColor": "#6B5AE0",
                        "backgroundColor": "rgba(107,90,224,0.10)",
                        "fill": True
                    }]
                },
                "format": {"y": "comma", "tooltip": "comma"}
            }
        }

    if (req.chart_type in (None, 'bar')) and (not x_is_num) and pd.api.types.is_numeric_dtype(df[req.y]):
        # categorical bar
        grp = df[[req.x, req.y]].dropna()
        if req.agg == 'mean':
            agg = grp.groupby(req.x)[req.y].mean().sort_values(ascending=False).head(25)
        elif req.agg == 'count':
            agg = grp.groupby(req.x)[req.y].count().sort_values(ascending=False).head(25)
        else:
            agg = grp.groupby(req.x)[req.y].sum().sort_values(ascending=False).head(25)
        return {
            "type": "bar",
            "spec": {
                "title": f"{req.y} by {req.x}",
                "data": {
                    "labels": [str(i) for i in agg.index.tolist()],
                    "datasets": [{"label": req.y, "data": [float(v) for v in agg.values], "backgroundColor": "#3B82F6"}]
                },
                "format": {"y": "comma", "tooltip": "comma"}
            }
        }

    if (req.chart_type in (None, 'scatter')) and x_is_num and y_is_num:
        sample = df[[req.x, req.y]].dropna()
        if len(sample) > 1500:
            sample = sample.sample(1500, random_state=42)
        points = [{"x": float(a), "y": float(b)} for a, b in sample[[req.x, req.y]].to_numpy()]
        return {
            "type": "scatter",
            "spec": {
                "title": f"{req.y} vs {req.x}",
                "data": {"points": points},
                "format": {"x": "comma", "y": "comma", "tooltip": "comma"}
            }
        }

    # pie: categorical x, numeric y (top 10)
    if req.chart_type == 'pie' and (not x_is_num) and y_is_num:
        grp = df[[req.x, req.y]].dropna()
        if req.agg == 'mean':
            agg = grp.groupby(req.x)[req.y].mean().sort_values(ascending=False).head(10)
        elif req.agg == 'count':
            agg = grp.groupby(req.x)[req.y].count().sort_values(ascending=False).head(10)
        else:
            agg = grp.groupby(req.x)[req.y].sum().sort_values(ascending=False).head(10)
        return {
            "type": "pie",
            "spec": {
                "title": f"{req.y} share by {req.x}",
                "data": {
                    "labels": [str(i) for i in agg.index.tolist()],
                    "datasets": [{"label": req.y, "data": [float(v) for v in agg.values]}]
                },
                "format": {"tooltip": "comma"}
            }
        }

    # Additional LINE support when X is numeric or categorical
    if req.chart_type == 'line' and pd.api.types.is_numeric_dtype(df[req.y]):
        if x_is_num:
            series = df[[req.x, req.y]].dropna().sort_values(req.x)
            return {
                "type": "line",
                "spec": {
                    "title": f"{req.y} vs {req.x}",
                    "data": {
                        "labels": [float(v) for v in series[req.x].tolist()],
                        "datasets": [{
                            "label": req.y,
                            "data": [float(v) for v in series[req.y].tolist()],
                            "borderColor": "#6B5AE0",
                            "backgroundColor": "rgba(107,90,224,0.10)",
                            "fill": False
                        }]
                    },
                    "format": {"x": "comma", "y": "comma", "tooltip": "comma"}
                }
            }
        if not x_is_num:
            grp = df[[req.x, req.y]].dropna()
            if req.agg == 'mean':
                agg = grp.groupby(req.x)[req.y].mean().head(50)
            elif req.agg == 'count':
                agg = grp.groupby(req.x)[req.y].count().head(50)
            else:
                agg = grp.groupby(req.x)[req.y].sum().head(50)
            return {
                "type": "line",
                "spec": {
                    "title": f"{req.y} by {req.x}",
                    "data": {
                        "labels": [str(i) for i in agg.index.tolist()],
                        "datasets": [{
                            "label": req.y,
                            "data": [float(v) for v in agg.values],
                            "borderColor": "#6B5AE0",
                            "backgroundColor": "rgba(107,90,224,0.10)",
                            "fill": True
                        }]
                    },
                    "format": {"y": "comma", "tooltip": "comma"}
                }
            }

class SummarizeRequest(BaseModel):
    run_id: str
    charts: List[Dict[str, Any]]  # each has type and spec

@router.post("/summarize")
def summarize_charts(req: SummarizeRequest):
    meta = data_store.get_metadata(req.run_id) or {}
    context = meta.get('qa_context', '')
    # Construct a concise description of chart specs
    def brief(spec: Dict[str, Any]) -> str:
        title = spec.get('title') or 'Untitled'
        labels = spec.get('data', {}).get('labels')
        return f"- {title} with {len(labels) if labels else 'N/A'} points"
    chart_summaries = "\n".join([f"{c.get('type', 'chart').upper()}: {brief(c.get('spec', {}))}" for c in req.charts[:3]])

    prompt = f"""
You are a business analyst. Given dataset context and up to three charts, synthesize a concise analysis.
Return exactly three sections:
**Insight:** one-sentence key takeaway
**Reasoning:** short explanation connecting trends/patterns in the charts
**Implication:** actionable recommendation

DATA CONTEXT:
{context}

CHARTS:
{chart_summaries}
"""
    out = call_smart(prompt, use_tools=False)
    return {"answer": out}

    # default fallback
    raise HTTPException(status_code=400, detail="Unsupported column types for preview")


