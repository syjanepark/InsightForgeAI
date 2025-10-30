# analyze.py - Professional Business Intelligence Flow
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import re

def _top_kpi_summaries(df: pd.DataFrame) -> Dict[str, Any]:
    num = df.select_dtypes(include=[np.number])
    kpis = []
    if not num.empty:
        desc = num.describe().T
        for col, row in desc.iterrows():
            # Skip obvious index columns and normalized columns
            if col.lower() in ['unnamed_0', 'index'] or col.endswith('_Normalized'):
                continue
            kpis.append({
                "metric": col,
                "mean": float(round(row["mean"], 3)),
                "min": float(round(row["min"], 3)),
                "max": float(round(row["max"], 3)),
                "std": float(round(row["std"], 3))
            })
    return {"kpis": kpis[:6]}

def _time_trends(df: pd.DataFrame) -> Dict[str, Any]:
    # find a date/year column
    time_col = None
    for c in df.columns:
        lc = c.lower()
        if any(x in lc for x in ("date","year")) and np.issubdtype(df[c].dtype, np.datetime64) or np.issubdtype(df[c].dtype, np.number):
            time_col = c; break
    trends = []
    if time_col:
        # pick top numeric series by variance for YoY/period deltas
        num = df.select_dtypes(include=[np.number]).copy()
        if time_col in num.columns: num = num.drop(columns=[time_col])
        if not num.empty:
            # aggregate by time (year if date)
            g = df.copy()
            if np.issubdtype(df[time_col].dtype, np.datetime64):
                g["__year__"] = pd.to_datetime(g[time_col]).dt.year
                tcol = "__year__"
            else:
                tcol = time_col
            agg = g.groupby(tcol)[num.columns].sum().sort_index()
            for col in agg.columns[:4]:
                s = agg[col]
                yoy = s.pct_change().replace([np.inf,-np.inf], np.nan).dropna()
                if not yoy.empty:
                    trends.append({
                        "series": col,
                        "time_index": s.index.tolist(),
                        "values": [float(x) for x in s.values],
                        "last_yoy_pct": round(float(yoy.iloc[-1]*100), 2)
                    })
    return {"trends": trends}

def run_etl(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: ETL - Clean, type-cast, normalize data"""
    
    print("🧹 Starting ETL process...")
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = [
        re.sub(r'[^\w\s]', '', col).strip().replace(' ', '_').title()
        for col in df_clean.columns
    ]
    
    # Handle missing data
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('Unknown')
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Type casting - convert string numbers to numeric
    for col in df_clean.select_dtypes(include=['object']).columns:
        # Try to convert string numbers to numeric
        sample = df_clean[col].dropna().head(100)
        if len(sample) > 0:
            # Remove common formatting
            clean_sample = sample.astype(str).str.replace(r'[$,\s]', '', regex=True)
            try:
                pd.to_numeric(clean_sample, errors='raise')
                df_clean[col] = pd.to_numeric(
                    df_clean[col].astype(str).str.replace(r'[$,\s]', '', regex=True),
                    errors='coerce'
                )
                print(f"  ✅ Converted {col} to numeric")
            except:
                pass  # Keep as categorical
    
    # Normalize numeric columns (optional - for correlation analysis)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].std() > 0:  # Only normalize if there's variation
            df_clean[f"{col}_Normalized"] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    
    print(f"  ✅ ETL complete: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    return df_clean

def generate_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Step 2-4: Complete Business Intelligence Analysis Flow"""
    
    # Step 1: ETL
    df_clean = run_etl(df)
    
    # Step 2: Analyze (KPIs, trends, deltas, correlations)
    print("📊 Analyzing data patterns...")
    
    basic = {
        "rows": int(len(df_clean)),
        "cols": int(len(df_clean.columns)),
        "columns": df_clean.columns.tolist(),
        "cleaned_cols": [col for col in df_clean.columns if not col.endswith('_Normalized')],
        "sample": _convert_numpy_types(df_clean.head(2).to_dict("records"))
    }
    
    kpis = _top_kpi_summaries(df_clean)
    trends = _time_trends(df_clean)
    deltas = _calculate_deltas(df_clean)
    correlations = _analyze_correlations(df_clean)
    
    # Step 3: Generate precomputed chart specs
    print("📈 Generating chart specifications...")
    charts = _generate_simple_charts(df_clean, kpis, trends)
    
    # Step 4: Create summary section
    summary = _generate_executive_summary(basic, kpis, trends, deltas, correlations)

    # Create a compact QA context string the chat can use
    context_lines = [
        f"Rows: {basic['rows']}, Cols: {basic['cols']}",
        f"Columns: {', '.join(basic['columns'][:15])}",
        f"Top KPIs: " + "; ".join([f"{k['metric']} mean={k['mean']}" for k in kpis.get("kpis", [])]),
    ]
    for t in trends.get("trends", []):
        context_lines.append(f"Trend {t['series']}: last YoY change {t['last_yoy_pct']}%.")

    return _convert_numpy_types({
        "basic": basic,
        "kpis": kpis["kpis"],
        "trends": trends["trends"],
        "deltas": deltas,
        "correlations": correlations,
        "charts": charts,
        "summary": summary,
        "qa_context": "\n".join(context_lines)
    })

def _detect_binary_target(df: pd.DataFrame) -> tuple:
    """Detect binary target variable in the dataset - completely dynamic"""
    
    # Find all binary columns (exactly 2 unique values)
    binary_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            binary_cols.append(col)
    
    if not binary_cols:
        return None, None
    
    # Score columns by how likely they are to be targets
    # Higher score = more likely to be target
    scored_cols = []
    
    for col in binary_cols:
        score = 0
        col_lower = col.lower()
        
        # Target-like keywords (generic, not hardcoded)
        target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'flag', 'status', 'type', 'category']
        for keyword in target_keywords:
            if keyword in col_lower:
                score += 10
        
        # Binary pattern keywords
        binary_keywords = ['is_', 'has_', 'can_', 'should_', 'will_', 'did_', 'was_', 'were_']
        for keyword in binary_keywords:
            if col_lower.startswith(keyword):
                score += 5
        
        # Avoid obvious non-targets
        avoid_keywords = ['id', 'index', 'count', 'num_', 'total_', 'sum_', 'avg_', 'mean_', 'max_', 'min_']
        for keyword in avoid_keywords:
            if keyword in col_lower:
                score -= 5
        
        # Prefer columns with balanced distribution (not too skewed)
        value_counts = df[col].value_counts()
        if len(value_counts) == 2:
            ratio = min(value_counts) / max(value_counts)
            if 0.2 <= ratio <= 0.8:  # Balanced distribution
                score += 3
            elif ratio < 0.1:  # Very skewed
                score -= 2
        
        scored_cols.append((col, score))
    
    # Sort by score (highest first) and pick the best one
    scored_cols.sort(key=lambda x: x[1], reverse=True)
    
    if not scored_cols or scored_cols[0][1] < 0:
        return None, None
    
    best_col = scored_cols[0][0]
    unique_vals = df[best_col].dropna().unique()
    
    # Determine positive class
    if df[best_col].dtype in ['int64', 'float64'] and set(unique_vals) == {0, 1}:
        return best_col, 1
    elif df[best_col].dtype == 'object':
        # Check for common binary text patterns
        unique_lower = [str(v).lower() for v in unique_vals]
        if 'true' in unique_lower and 'false' in unique_lower:
            return best_col, 'true'
        elif 'yes' in unique_lower and 'no' in unique_lower:
            return best_col, 'yes'
        elif 'positive' in unique_lower and 'negative' in unique_lower:
            return best_col, 'positive'
        else:
            # Generic binary: pick the less frequent one as positive
            counts = df[best_col].value_counts()
            if len(counts) == 2:
                positive_class = counts.idxmin()  # Less frequent
                return best_col, positive_class
    
    return None, None

def _generate_simple_charts(df: pd.DataFrame, kpis: Dict[str, Any], trends: Dict[str, Any]) -> list:
    """Generate simple, focused charts matching your clean approach"""
    
    charts = []
    
    # Check for binary target variable first
    target_col, positive_class = _detect_binary_target(df)
    
    if target_col:
        # Generate target-aware charts only
        print(f"🎯 Detected binary target: {target_col} (positive class: {positive_class})")
        
        # 1. Target distribution
        target_counts = df[target_col].value_counts()
        charts.append({
            "type": "pie",
            "title": f"Distribution of {target_col}",
            "data": {
                "labels": [str(label) for label in target_counts.index],
                "datasets": [{
                    "data": [float(x) for x in target_counts.values],
                    "backgroundColor": ["#EF4444", "#10B981"][:len(target_counts)]
                }]
            }
        })
        
        # 2. Target by category (prefer platform, country, region, author_verified)
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and not col.lower().startswith('unnamed')]
        preferred_cats = ['platform', 'country', 'region', 'author_verified', 'source_domain_reliability']
        
        category_col = categorical_cols[0]  # Default
        for pref in preferred_cats:
            for cat in categorical_cols:
                if cat.lower() == pref:
                    category_col = cat
                    break
            if category_col != categorical_cols[0]:
                break
        
        # Count positive cases by category
        positive_data = df[df[target_col] == positive_class]
        if not positive_data.empty:
            category_counts = positive_data[category_col].value_counts().head(10)
            charts.append({
                "type": "bar",
                "title": f"{target_col}={positive_class} by {category_col}",
                "data": {
                    "labels": [str(label) for label in category_counts.index],
                    "datasets": [{
                        "label": f"{target_col}={positive_class}",
                        "data": [float(x) for x in category_counts.values],
                        "backgroundColor": "#EF4444"
                    }]
                }
            })
        
        # 3. Target rate over time (if time column exists)
        time_cols = [col for col in df.columns if any(x in col.lower() for x in ['timestamp', 'date', 'time', 'month', 'weekday'])]
        if time_cols:
            time_col = time_cols[0]
            # Calculate target rate by time period
            time_groups = df.groupby(time_col)[target_col].agg(['count', 'sum']).reset_index()
            time_groups['rate'] = (time_groups['sum'] / time_groups['count'] * 100).round(2)
            
            charts.append({
                "type": "line",
                "title": f"{target_col} rate over {time_col} (%)",
                "data": {
                    "labels": [str(x) for x in time_groups[time_col]],
                    "datasets": [{
                        "label": f"{target_col} rate (%)",
                        "data": [float(x) for x in time_groups['rate']],
                        "borderColor": "#EF4444",
                        "backgroundColor": "rgba(239, 68, 68, 0.1)",
                        "fill": True
                    }]
                }
            })
        
        return charts
    
    # Original chart generation for non-target datasets
    # 1. KPI Overview Chart
    if kpis.get('kpis'):
        kpi_data = kpis['kpis'][:5]  # Top 5 KPIs
        charts.append({
            "type": "bar",
            "title": "Key Performance Indicators",
            "data": {
                "labels": [k['metric'] for k in kpi_data],
                "datasets": [{
                    "label": "Average Values",
                    "data": [float(k['mean']) for k in kpi_data],
                    "backgroundColor": "#3B82F6"
                }]
            }
        })
    
    # 2. Trend Chart (if time series available)
    if trends.get('trends'):
        trend_data = trends['trends'][0]  # Primary trend
        charts.append({
            "type": "line", 
            "title": f"{trend_data['series']} Over Time",
            "data": {
                "labels": [str(x) for x in trend_data['time_index']],
                "datasets": [{
                    "label": trend_data['series'],
                    "data": [float(x) for x in trend_data['values']],
                    "borderColor": "#10B981",
                    "backgroundColor": "rgba(16, 185, 129, 0.1)",
                    "fill": True
                }]
            }
        })
    
    # 3. Distribution Chart (smart categorical breakdown)
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and not col.lower().startswith('unnamed')]
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() 
                   if not col.endswith('_Normalized') and not col.lower() in ['unnamed_0', 'index']]
    
    if categorical_cols and numeric_cols and len(df) <= 50:  # Only for smaller datasets
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        # Get top categories by sum (limit to prevent overcrowded charts)
        grouped = df.groupby(cat_col)[num_col].sum().nlargest(5)
        
        # Only create chart if we have meaningful data spread
        if len(grouped) >= 2 and grouped.iloc[0] > 0:
            charts.append({
                "type": "doughnut",
                "title": f"Top 5: {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
                "data": {
                    "labels": [str(label) for label in grouped.index],
                    "datasets": [{
                        "data": [float(x) for x in grouped.values],
                        "backgroundColor": ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444"][:len(grouped)]
                    }]
                }
            })
    elif len(numeric_cols) >= 2:
        # For larger datasets, show correlation between top 2 numeric columns
        col1, col2 = numeric_cols[0], numeric_cols[1]
        
        # Create scatter plot data (sample if too many points)
        sample_df = df.sample(min(20, len(df))) if len(df) > 20 else df
        
        charts.append({
            "type": "scatter",
            "title": f"{col1.replace('_', ' ').title()} vs {col2.replace('_', ' ').title()}",
            "data": {
                "datasets": [{
                    "label": "Data Points",
                    "data": [{"x": float(row[col1]), "y": float(row[col2])} for _, row in sample_df.iterrows()],
                    "backgroundColor": "#3B82F6",
                    "borderColor": "#1E40AF"
                }]
            }
        })
    
    return charts

def _calculate_deltas(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate period-over-period deltas for key metrics"""
    
    deltas = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Find time column
    time_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['year', 'date', 'period', 'time']):
            time_col = col
            break
    
    if time_col and len(numeric_cols) > 0:
        try:
            # Group by time and calculate period deltas
            grouped = df.groupby(time_col)[numeric_cols].sum().sort_index()
            
            for col in numeric_cols[:5]:  # Top 5 metrics
                if col in grouped.columns:
                    values = grouped[col]
                    if len(values) >= 2:
                        # Calculate absolute and percentage deltas
                        abs_delta = values.iloc[-1] - values.iloc[-2] if len(values) >= 2 else 0
                        pct_delta = ((values.iloc[-1] / values.iloc[-2]) - 1) * 100 if values.iloc[-2] != 0 and len(values) >= 2 else 0
                        
                        deltas[col] = {
                            "absolute": float(round(abs_delta, 2)),
                            "percentage": float(round(pct_delta, 2)),
                            "direction": "up" if abs_delta > 0 else "down" if abs_delta < 0 else "stable"
                        }
        except Exception as e:
            print(f"  ⚠️ Could not calculate deltas: {e}")
    
    return deltas

def _analyze_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze correlations between numeric variables"""
    
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if not col.endswith('_Normalized')][:8]  # Limit to prevent overload
    
    correlations = {
        "matrix": {},
        "strong_pairs": [],
        "insights": []
    }
    
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Convert to dict for JSON serialization (convert numpy floats to Python floats)
            matrix_dict = corr_matrix.round(3).to_dict()
            correlations["matrix"] = {
                k1: {k2: float(v2) for k2, v2 in v1.items()} 
                for k1, v1 in matrix_dict.items()
            }
            
            # Find strong correlations (> 0.7 or < -0.7)
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.7:
                        correlations["strong_pairs"].append({
                            "var1": col1,
                            "var2": col2,
                            "correlation": round(float(corr_val), 3),
                            "strength": "strong positive" if corr_val > 0.7 else "strong negative"
                        })
            
            # Generate insights
            if correlations["strong_pairs"]:
                correlations["insights"].append(f"Found {len(correlations['strong_pairs'])} strong correlations")
            else:
                correlations["insights"].append("No strong correlations detected - variables operate independently")
                
        except Exception as e:
            print(f"  ⚠️ Could not calculate correlations: {e}")
            correlations["insights"].append("Correlation analysis unavailable")
    
    return correlations

def _generate_executive_summary(basic: Dict, kpis: Dict, trends: Dict, deltas: Dict, correlations: Dict) -> Dict[str, Any]:
    """Generate executive summary with specific insights from the data"""
    
    summary_points = []
    
    # Data overview
    summary_points.append(f"📊 **Dataset Overview**: {basic['rows']:,} records across {len(basic['cleaned_cols'])} business dimensions")
    
    # KPI insights
    if kpis.get('kpis'):
        top_kpis = kpis['kpis'][:3]
        kpi_names = [k['metric'] for k in top_kpis]
        summary_points.append(f"📈 **Key Metrics**: Primary focus on {', '.join(kpi_names)}")
    
    # Trend insights
    if trends.get('trends'):
        positive_trends = [t for t in trends['trends'] if t.get('last_yoy_pct', 0) > 5]
        if positive_trends:
            summary_points.append(f"📈 **Growth Trends**: {len(positive_trends)} metrics showing positive momentum")
        else:
            summary_points.append("📊 **Performance**: Mixed trends across key metrics")
    
    # Delta insights
    if deltas:
        growth_metrics = [k for k, v in deltas.items() if v.get('direction') == 'up']
        if growth_metrics:
            summary_points.append(f"🚀 **Recent Performance**: {len(growth_metrics)} metrics trending upward")
    
    # Correlation insights
    if correlations.get('strong_pairs'):
        summary_points.append(f"🔗 **Data Relationships**: {len(correlations['strong_pairs'])} strong correlations identified")
    
    # Generate data-driven overall assessment paragraph
    overall = _create_insight_paragraph(basic, kpis, trends, deltas, correlations)
    
    return {
        "highlights": summary_points,
        "overall_assessment": overall,
        "data_quality": "High" if basic['rows'] > 100 else "Moderate",
        "analysis_completeness": "Complete" if all([kpis, trends, deltas]) else "Partial"
    }

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def _create_insight_paragraph(basic: Dict, kpis: Dict, trends: Dict, deltas: Dict, correlations: Dict) -> str:
    """Create a specific, data-driven insight paragraph"""
    
    insights = []
    
    # Data scale context
    if basic['rows'] > 1000:
        insights.append(f"Analysis of {basic['rows']:,} records reveals")
    else:
        insights.append(f"This {basic['rows']}-record dataset shows")
    
    # KPI findings
    if kpis.get('kpis'):
        top_kpi = kpis['kpis'][0]
        if top_kpi['mean'] > 1000000:
            scale = "million-scale" if top_kpi['mean'] < 1000000000 else "billion-scale"
            insights.append(f"{scale} values in {top_kpi['metric']} (avg: {top_kpi['mean']:,.0f})")
        else:
            insights.append(f"key metric {top_kpi['metric']} averaging {top_kpi['mean']:,.1f}")
    
    # Trend findings  
    if trends.get('trends'):
        growth_trends = [t for t in trends['trends'] if t.get('last_yoy_pct', 0) > 5]
        decline_trends = [t for t in trends['trends'] if t.get('last_yoy_pct', 0) < -5]
        
        if growth_trends:
            best_trend = max(growth_trends, key=lambda x: x.get('last_yoy_pct', 0))
            insights.append(f"with {best_trend['series']} showing strong growth ({best_trend['last_yoy_pct']:+.1f}% YoY)")
        elif decline_trends:
            worst_trend = min(decline_trends, key=lambda x: x.get('last_yoy_pct', 0))
            insights.append(f"with {worst_trend['series']} declining ({worst_trend['last_yoy_pct']:+.1f}% YoY)")
        else:
            insights.append("with stable performance across time series")
    
    # Correlation findings
    if correlations.get('strong_pairs'):
        strongest = max(correlations['strong_pairs'], key=lambda x: abs(x.get('correlation', 0)))
        relationship = "positive" if strongest['correlation'] > 0 else "negative"
        insights.append(f"Notable {relationship} correlation between {strongest['var1']} and {strongest['var2']} ({strongest['correlation']:+.2f})")
    
    # Performance assessment
    if deltas:
        up_count = sum(1 for v in deltas.values() if v.get('direction') == 'up')
        down_count = sum(1 for v in deltas.values() if v.get('direction') == 'down')
        
        if up_count > down_count:
            insights.append(f"Overall momentum is positive with {up_count} metrics trending upward")
        elif down_count > up_count:
            insights.append(f"Mixed performance with {down_count} metrics declining recently")
        else:
            insights.append("Performance indicators show balanced, stable patterns")
    
    # Join insights into a coherent paragraph
    if len(insights) >= 2:
        return f"{insights[0]} {insights[1]}. {' '.join(insights[2:])}."
    elif len(insights) == 1:
        return f"{insights[0]} clear patterns for strategic analysis."
    else:
        return "Dataset provides comprehensive foundation for business intelligence analysis."