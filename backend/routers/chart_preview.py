from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
from core.data_store import data_store

router = APIRouter(prefix="/chart", tags=["chart"])

class PreviewRequest(BaseModel):
    run_id: str
    x: str
    y: str
    agg: Optional[str] = "sum"  # sum|mean|count
    chart_type: Optional[str] = None  # 'bar'|'line'|'scatter'|'pie'
    y_fields: Optional[List[str]] = None  # only used for pie (multi-metric)
    filter_col: Optional[str] = None  # optional categorical/entity column to filter on
    filter_values: Optional[List[str]] = None  # values to include from filter_col

@router.get("/columns")
def list_columns(run_id: str):
    df: pd.DataFrame = data_store.get_data(run_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Run not found")
    # detect types
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(exclude=['number']).columns.tolist()
    # Promote numeric-like strings (e.g., "$3,200", "1,234") to numeric
    numeric_like: list[str] = []
    for c in list(categorical):
        try:
            sample = df[c].astype(str).str.replace(r'[\$%,\s]', '', regex=True)
            coerced = pd.to_numeric(sample, errors='coerce')
            ratio = coerced.notna().mean()
            if ratio >= 0.3:  # relaxed threshold to capture numeric-like columns
                numeric_like.append(c)
        except Exception:
            continue
    # Update lists: add numeric-like to numeric, drop from categorical
    numeric_all = list(dict.fromkeys(numeric + numeric_like))
    categorical = [c for c in categorical if c not in numeric_like]
    # numeric columns with few uniques can be treated as categorical too
    try:
        few_unique_as_cat = [c for c in numeric_all if df[c].nunique(dropna=True) <= 50]
    except Exception:
        few_unique_as_cat = []
    cat_union = list(dict.fromkeys(categorical + few_unique_as_cat))
    # detect datetime-like from any column (including strings)
    datetime_like = []
    for c in df.columns:
        try:
            # numeric year detection
            series = df[c]
            if not pd.api.types.is_numeric_dtype(series) and c in numeric_like:
                # use coerced numeric for year test
                series = pd.to_numeric(df[c].astype(str).str.replace(r'[\$%,\s]', '', regex=True), errors='coerce')
            if pd.api.types.is_numeric_dtype(series):
                s = pd.to_numeric(series, errors='coerce')
                if s.dropna().between(1900, 2100).sum() >= max(5, int(0.1*len(df))):
                    datetime_like.append(c)
                    continue
            dt = pd.to_datetime(series, errors='coerce')
            if dt.notna().sum() > max(10, int(0.2*len(df))):
                datetime_like.append(c)
        except Exception:
            pass
    return {"numeric": numeric_all, "categorical": cat_union, "datetime": list(dict.fromkeys(datetime_like))}

@router.get("/distinct")
def list_distinct(run_id: str, column: str, limit: int = 50):
    df: pd.DataFrame = data_store.get_data(run_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    try:
        vc = df[column].astype(str).value_counts().head(max(1, min(limit, 200)))
        return {"values": vc.index.tolist(), "counts": [int(x) for x in vc.values.tolist()]}
    except Exception:
        vals = df[column].astype(str).dropna().unique().tolist()[:limit]
        return {"values": vals, "counts": None}

@router.post("/preview")
def chart_preview(req: PreviewRequest):
    df: pd.DataFrame = data_store.get_data(req.run_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if req.x not in df.columns or req.y not in df.columns:
        raise HTTPException(status_code=400, detail="Columns not found")

    # Optional filtering by entity/category (e.g., model == i8 or ticker == AAPL)
    if req.filter_col and req.filter_col in df.columns and req.filter_values:
        try:
            vals = set([str(v).lower() for v in req.filter_values])
            sub = df[df[req.filter_col].astype(str).str.lower().isin(vals)]
            if sub.empty:
                raise HTTPException(status_code=400, detail=f"No rows match {req.filter_col} in {req.filter_values}.")
            df = sub
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to apply filter. Check filter_col and filter_values.")

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

    # If x is date-like parse year (robust parsing incl. day-first and common formats)
    year_series = None
    def infer_year_series(series: pd.Series) -> pd.Series:
        thresholds = [max(5, int(0.10*len(series))), max(10, int(0.20*len(series)))]
        # 0) numeric year-like column
        try:
            if pd.api.types.is_numeric_dtype(series):
                s = pd.to_numeric(series, errors='coerce')
                valid = s.dropna().astype(float)
                mask = (valid >= 1900) & (valid <= 2100)
                if mask.sum() >= thresholds[0]:
                    return valid.astype(int)
        except Exception:
            pass
        # 1) default parsing
        dt = pd.to_datetime(series, errors='coerce')
        if dt.notna().sum() >= thresholds[0]:
            return dt.dt.year
        # 2) day-first
        dt = pd.to_datetime(series, errors='coerce', dayfirst=True)
        if dt.notna().sum() >= thresholds[0]:
            return dt.dt.year
        # 3) try common explicit formats
        fmts = ['%Y-%m-%d','%d-%m-%Y','%m/%d/%Y','%d/%m/%Y','%Y/%m/%d','%d.%m.%Y','%Y.%m.%d']
        for f in fmts:
            try:
                dt = pd.to_datetime(series, errors='coerce', format=f)
                if dt.notna().sum() >= thresholds[0]:
                    return dt.dt.year
            except Exception:
                continue
        # 4) regex extract 4-digit year from strings
        try:
            import re
            years = series.astype(str).str.extract(r'(19\d{2}|20\d{2})', expand=False)
            years = pd.to_numeric(years, errors='coerce')
            if years.notna().sum() >= thresholds[0]:
                return years.astype(int)
        except Exception:
            pass
        return None

    if not x_is_num:
        try:
            year_series = infer_year_series(df[req.x])
        except Exception:
            year_series = None

    if (req.chart_type in (None, 'line')) and year_series is not None and pd.api.types.is_numeric_dtype(df[req.y]):
        tmp = pd.DataFrame({'year': year_series, 'val': df[req.y]}).dropna()
        if req.agg == 'mean':
            agg = tmp.groupby('year')['val'].mean().sort_index(ascending=True)
        elif req.agg == 'count':
            agg = tmp.groupby('year')['val'].count().sort_index(ascending=True)
        else:
            agg = tmp.groupby('year')['val'].sum().sort_index(ascending=True)
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

    if (req.chart_type in (None, 'bar')) and pd.api.types.is_numeric_dtype(df[req.y]):
        # bar: allow categorical or numeric X (treat numeric X like Year as categories)
        grp = df[[req.x, req.y]].dropna()
        if grp.empty:
            raise HTTPException(status_code=400, detail="No data after filtering for bar chart.")
        x_key = req.x
        if x_is_num:
            # Cast to int if near-integer, then to string labels
            try:
                x_vals = grp[req.x]
                if (x_vals.dropna() % 1).abs().mean() < 1e-6:
                    grp[req.x] = x_vals.astype(int)
            except Exception:
                pass
        # Sort by X-axis (increasing order) for numeric/time X; by values (descending) for categorical
        if x_is_num or year_series is not None:
            # For numeric/time X: sort by index (X-axis) in increasing order
            if req.agg == 'mean':
                agg = grp.groupby(req.x)[req.y].mean().sort_index(ascending=True).head(25)
            elif req.agg == 'count':
                agg = grp.groupby(req.x)[req.y].count().sort_index(ascending=True).head(25)
            else:
                agg = grp.groupby(req.x)[req.y].sum().sort_index(ascending=True).head(25)
        else:
            # For categorical X: sort by values (descending) to show top categories
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

    # pie: X can be date-like (aggregated by Year) OR categorical (top 10). Support multi-metric via y_fields.
    if req.chart_type == 'pie':
        # Determine which Y metrics to use
        y_list = [req.y]
        if req.y_fields and isinstance(req.y_fields, list) and len(req.y_fields) > 0:
            y_list = [y for y in req.y_fields if y in df.columns]
        # choose first numeric y for rendering
        render_y = next((y for y in y_list if pd.api.types.is_numeric_dtype(df[y])), None)
        if not render_y or not pd.api.types.is_numeric_dtype(df[render_y]):
            # fallback to provided y if numeric
            render_y = req.y if pd.api.types.is_numeric_dtype(df[req.y]) else None
        if not render_y:
            raise HTTPException(status_code=400, detail="Pie requires a numeric metric (Y)")

        # If X is date-like, aggregate by year; else treat X as categorical
        if year_series is not None:
            tmp = pd.DataFrame({'year': year_series, 'val': df[render_y]}).dropna()
            if tmp.empty:
                raise HTTPException(status_code=400, detail="No data available after date parsing for pie.")
            if req.agg == 'mean':
                agg = tmp.groupby('year')['val'].mean().sort_values(ascending=False).head(10)
            elif req.agg == 'count':
                agg = tmp.groupby('year')['val'].count().sort_values(ascending=False).head(10)
            else:
                agg = tmp.groupby('year')['val'].sum().sort_values(ascending=False).head(10)
            spec = {
                "title": f"{render_y} share by Year",
                "data": {
                    "labels": [str(i) for i in agg.index.tolist()],
                    "datasets": [{"label": render_y, "data": [float(v) for v in agg.values]}]
                },
                "format": {"tooltip": "comma"},
                "meta": {"selected": render_y, "available": y_list}
            }
            return {"type": "pie", "spec": spec}
        else:
            # categorical pie
            # If X is numeric with few uniques, treat as categorical
            treat_as_cat = (not pd.api.types.is_numeric_dtype(df[req.x])) or (df[req.x].nunique(dropna=True) <= 50)
            if not treat_as_cat:
                raise HTTPException(status_code=400, detail=f"Pie requires X to be date-like or categorical. '{req.x}' looks too continuous.")
            grp = df[[req.x, render_y]].dropna()
            if grp.empty:
                raise HTTPException(status_code=400, detail="No data available for the selected fields.")
            if req.agg == 'mean':
                agg = grp.groupby(req.x)[render_y].mean().sort_values(ascending=False).head(10)
            elif req.agg == 'count':
                agg = grp.groupby(req.x)[render_y].count().sort_values(ascending=False).head(10)
            else:
                agg = grp.groupby(req.x)[render_y].sum().sort_values(ascending=False).head(10)
            spec = {
                "title": f"{render_y} share by {req.x}",
                "data": {
                    "labels": [str(i) for i in agg.index.tolist()],
                    "datasets": [{"label": render_y, "data": [float(v) for v in agg.values]}]
                },
                "format": {"tooltip": "comma"},
                "meta": {"selected": render_y, "available": y_list}
            }
            return {"type": "pie", "spec": spec}

    # fall through to other handlers below

    # Additional LINE support when X is numeric or categorical
    if req.chart_type == 'line' and pd.api.types.is_numeric_dtype(df[req.y]):
        if x_is_num:
            # Aggregate by numeric X (e.g., Year)
            grp = df[[req.x, req.y]].dropna()
            if grp.empty:
                raise HTTPException(status_code=400, detail="No data after filtering for line chart.")
            if req.agg == 'mean':
                agg = grp.groupby(req.x)[req.y].mean().sort_index(ascending=True)
            elif req.agg == 'count':
                agg = grp.groupby(req.x)[req.y].count().sort_index(ascending=True)
            else:
                agg = grp.groupby(req.x)[req.y].sum().sort_index(ascending=True)
            return {
                "type": "line",
                "spec": {
                    "title": f"{req.y} over {req.x}",
                    "data": {
                        "labels": [float(v) for v in agg.index.tolist()],
                        "datasets": [{
                            "label": req.y,
                            "data": [float(v) for v in agg.values.tolist()],
                            "borderColor": "#6B5AE0",
                            "backgroundColor": "rgba(107,90,224,0.10)",
                            "fill": True
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
                    "title": f"{req.y} over {req.x}",
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

    # If we got here, the selection is unsupported
    raise HTTPException(status_code=400, detail="Unsupported selection. Line: X must be Date/Year/categorical (or numeric) and Y must be numeric. Bar: X categorical and Y numeric. Pie: X Date/Year or categorical and Y numeric.")

class SummarizeRequest(BaseModel):
    run_id: str
    charts: List[Dict[str, Any]]  # each has type and spec

@router.post("/summarize")
def summarize_charts(req: SummarizeRequest):
    # Observability: print a concise log so you can see requests in the backend console
    try:
        print(f"🧾 /chart/summarize run_id={req.run_id} charts={len(req.charts) if req.charts else 0}")
    except Exception:
        pass
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
    # Smart API disabled for summarize to avoid external dependency; generate local summary
    print("ℹ️ Summarize: using local synthesis (no Smart API)")
    # Data-aware local synthesis from chart specs (no external calls)
    def summarize_chart(c: Dict[str, Any]) -> Dict[str, Any]:
        try:
            typ = (c.get('type') or '').upper()
            spec = c.get('spec') or {}
            title = spec.get('title') or 'Untitled'
            data = (spec.get('data') or {})
            labels = data.get('labels') or []
            series = []
            if isinstance(data.get('datasets'), list) and data['datasets']:
                series = data['datasets'][0].get('data') or []
            # compute simple trend stats
            delta_text = ""
            peak_text = ""
            first = last = change = pct = None
            peak_label = None
            if labels and series and len(labels) == len(series):
                try:
                    first, last = float(series[0]), float(series[-1])
                    change = last - first
                    pct = (change / first * 100.0) if first not in (0, None) else None
                    delta_text = f"{labels[0]}→{labels[-1]}: {last:,.0f} ({'+' if change>=0 else ''}{change:,.0f}" + (f", {pct:+.1f}%" if pct is not None else "") + ")"
                    # peak
                    mx = max((float(v), i) for i, v in enumerate(series))
                    peak_label = labels[int(mx[1])]
                    peak_text = f"peak {peak_label} ({mx[0]:,.0f})"
                except Exception:
                    pass
            return {
                "title": title,
                "labels": labels,
                "series": series,
                "first": first,
                "last": last,
                "change": change,
                "pct": pct,
                "peak_label": peak_label,
                "delta_text": delta_text,
                "peak_text": peak_text,
            }
        except Exception:
            return {"title": "Untitled"}

    per_chart = [summarize_chart(c) for c in req.charts[:3]]
    # Enrich stats: YoY deltas, trough, CAGR, CV
    def enrich(pc: Dict[str, Any]):
        labels = pc.get("labels") or []
        series = pc.get("series") or []
        if not labels or not series or len(labels) != len(series):
            return
        # Trough
        try:
            mn = min((float(v), i) for i, v in enumerate(series))
            pc["trough_text"] = f"trough {labels[int(mn[1])]} ({mn[0]:,.0f})"
        except Exception:
            pass
        # Largest YoY delta (for ordered labels)
        try:
            yoy = []
            for i in range(1, len(series)):
                a, b = float(series[i-1]), float(series[i])
                d = b - a
                p = (d/a*100.0) if a else None
                yoy.append((abs(d), i, d, p))
            if yoy:
                best = max(yoy, key=lambda x: x[0])
                i = best[1]
                sign = '+' if best[2] >= 0 else ''
                pct = f", {best[3]:+.1f}%" if best[3] is not None else ""
                pc["largest_yoy_text"] = f"largest Δ {labels[i-1]}→{labels[i]}: {sign}{best[2]:,.0f}{pct}"
        except Exception:
            pass
        # CV and CAGR
        try:
            import math
            vals = [float(v) for v in series]
            mean = sum(vals)/len(vals)
            if mean:
                var = sum((v-mean)**2 for v in vals)/len(vals)
                pc["cv_text"] = f"volatility CV {math.sqrt(var)/abs(mean)*100.0:.1f}%"
        except Exception:
            pass
        try:
            if str(labels[0]).isdigit() and str(labels[-1]).isdigit() and pc.get('first') not in (None, 0):
                years = int(labels[-1]) - int(labels[0])
                if years > 0 and pc.get('last') and pc.get('first'):
                    cagr = ((pc['last']/pc['first'])**(1/years)-1)*100.0
                    pc["cagr_text"] = f"CAGR {cagr:+.1f}%"
        except Exception:
            pass
    for pc in per_chart:
        enrich(pc)
    # Build I/R/I: concise insight (primary chart), reasoning (bullets), implication (actionable)
    main = per_chart[0] if per_chart else {}
    if main and main.get("first") is not None and main.get("last") is not None:
        pct_txt = f" ({main['pct']:+.1f}%)" if isinstance(main.get('pct'), (int, float)) else ""
        labels_list = main.get('labels') or ['?']
        start_label = labels_list[0]
        end_label = labels_list[-1]
        last_val = main.get('last') or 0.0
        insight = f"{main.get('title','')}: {start_label}→{end_label} changed to {last_val:,.0f}{pct_txt}."
    else:
        insight = "Notable change detected in the primary chart."

    reasoning_lines = []
    for pc in per_chart:
        title = pc.get("title","")
        bullets = []
        if pc.get("delta_text"):
            bullets.append(pc["delta_text"])
        if pc.get("peak_text"):
            bullets.append(pc["peak_text"])
        if pc.get("trough_text"):
            bullets.append(pc["trough_text"])
        if pc.get("largest_yoy_text"):
            bullets.append(pc["largest_yoy_text"])
        if pc.get("cagr_text"):
            bullets.append(pc["cagr_text"])
        if pc.get("cv_text"):
            bullets.append(pc["cv_text"])
        if bullets:
            reasoning_lines.append(f"- {title}: " + "; ".join(bullets))
        else:
            reasoning_lines.append(f"- {title}")
    reasoning = "\n".join(["Based on the charted series:"] + reasoning_lines)
    if context:
        reasoning += "\n\nData context considered."
    # Implication tailored to direction of main chart
    implication = ""
    try:
        if isinstance(main.get('pct'), (int, float)):
            if main['pct'] <= -5:
                implication = "Decline detected: investigate price/mix, regional shifts, and retention; focus on reversing key-year drops."
            elif main['pct'] >= 5:
                implication = "Growth detected: double down on winning segments/periods; secure supply and scale channels."
    except Exception:
        pass
    if not implication:
        implication = "Focus on periods with largest swings; segment further (filters) to isolate drivers and validate with recent news when relevant."
    local = f"**Insight:** {insight}\n\n**Reasoning:**\n{reasoning}\n\n**Implication:** {implication}"
    # Return local summary (no external AI dependencies)
    return {"answer": local}

    # default fallback
    raise HTTPException(status_code=400, detail="Unsupported column types for preview")


