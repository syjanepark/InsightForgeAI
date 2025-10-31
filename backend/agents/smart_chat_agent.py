# agents/smart_chat_agent.py
from services.you_search import search_web
import pandas as pd
import asyncio

class SmartChatAgent:
    def __init__(self):
        pass
    
    async def process_query(self, question: str, context: str, df: pd.DataFrame = None):
        # Check if question needs external context search
        needs_search = self._needs_contextual_search(question)
        print(f"🤔 Question: '{question}' - Needs contextual search: {needs_search}")
        search_context = ""
        
        if needs_search:
            print(f"🔍 Detected contextual question, searching for background info...")
            search_results = await self._get_contextual_search(question)
            print(f"📱 Search results: {len(search_results) if search_results else 0} results")
            if search_results:
                search_context = f"\n\nEXTERNAL CONTEXT:\n{search_results}"
                print(f"✅ Found external context: {len(search_context)} chars")
                print(f"🔍 Search context preview: {search_context[:200]}...")
            else:
                print(f"❌ No search results returned")
        else:
            print(f"ℹ️ Question doesn't require external context - using local analysis only")
        
        # Enhanced prompt with both data context and search context
        prompt = f"""
You are a senior business analyst with access to both internal data and real-time market intelligence.

DATA CONTEXT:
{context}
{search_context}

QUESTION:
{question}

REQUIREMENTS:
- Analyze the actual data provided in the context
- Be specific and cite the exact metrics/columns from the dataset
- Provide actionable insights based on the real data
- If you found external context, reference it to explain "why" behind the data patterns
- Do NOT make assumptions about what the data contains - only analyze what's actually provided
"""
        
        # Local synthesis (no external AI dependencies)
        local_answer = self._create_local_analysis_with_context(question, context, df, search_context)
        local_visuals = self._generate_visualizations(question, df)
        
        return {
            "answer": local_answer,
            "visualizations": local_visuals,
            "suggested_actions": None,  # Simplified for now
            "citations": self._extract_citations(search_context)
        }
    
    def _create_local_analysis(self, question: str, context: str, df: pd.DataFrame = None) -> str:
        """Create intelligent local analysis without hardcoded phrase checks."""
        if df is None:
            return "I need access to the raw data to answer specific questions. Please re-upload your CSV file."
        
        print(f"🔍 Analyzing question: '{question}'")
        print(f"🔍 Dataframe info: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"🔍 Columns: {list(df.columns)}")

        intent = self._detect_intent(question, df)
        intent['_question'] = question  # Store question for later use
        answer = self._handle_intent(intent, df)
        if answer:
            return answer
        return self._analyze_mentioned_metrics(question.lower(), df)

    def _detect_intent(self, question: str, df: pd.DataFrame) -> dict:
        q = question.lower()
        entity_col = self._find_entity_column(df)
        entity = None
        if entity_col:
            # match multi-token, case-insensitive
            tokens = q.split()
            values = df[entity_col].dropna().astype(str).unique()
            lower_map = {v.lower(): v for v in values}
            # try 2-gram then 1-gram
            for i in range(len(tokens)-1):
                two = f"{tokens[i]} {tokens[i+1]}"
                if two in lower_map:
                    entity = lower_map[two]
                    break
            if not entity:
                for t in tokens:
                    if t in lower_map:
                        entity = lower_map[t]
                        break

        # years
        import re
        years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', q)]

        # time/value columns
        time_col = self._find_time_column(df)
        # Try to extract value column from question first (e.g., "travel rate" -> "booking_rate" or "trip_count")
        mentioned_cols = self._extract_columns_from_question(q, df)
        value_col = None
        if mentioned_cols:
            # Find first numeric column from mentioned columns
            for col in mentioned_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    value_col = col
                    break
            # If no numeric found in mentioned cols, try all numeric cols and pick best match
            if not value_col:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                q_tokens = set(q.lower().split())
                best_match = None
                best_score = 0
                for col in numeric_cols:
                    col_words = set(col.lower().replace('_', ' ').replace('-', ' ').split())
                    overlap = len(q_tokens & col_words)
                    if overlap > best_score:
                        best_score = overlap
                        best_match = col
                if best_match:
                    value_col = best_match
        # Fallback to auto-detection
        if not value_col:
            value_col = self._pick_value_column(df)

        # mode inference (no phrase hardcoding beyond minimal intent cues)
        mode = 'non_time_trend'
        # pairwise correlation intent
        pair_cols = self._extract_columns_from_question(q, df)
        if len(pair_cols) == 2:
            mode = 'correlation_pair'
            return {
                'entity_col': entity_col,
                'entity': entity,
                'years': years,
                'time_col': time_col,
                'value_col': value_col,
                'mode': mode,
                'pair_cols': pair_cols
            }
        # Year comparison can work with or without entity (e.g., "global revenue" questions)
        if time_col and value_col and len(years) >= 1:
            if len(years) >= 2 or ('compare' in q or 'vs' in q or 'versus' in q or 'compared to' in q):
                # Year comparison (with or without "why")
                mode = 'compare_years'
            elif 'why' in q or 'reason' in q or 'cause' in q or 'drop' in q or 'decline' in q:
                # "Why" with year mentioned - treat as compare with previous year
                mode = 'compare_years'
        elif entity and time_col and value_col:
            if 'why' in q or 'reason' in q or 'cause' in q:
                mode = 'factor_explanation'
            else:
                mode = 'over_time_summary'
        elif time_col and value_col and ('why' in q or 'reason' in q or 'cause' in q) and ('increase' in q or 'grow' in q or 'improve' in q or 'keep' in q):
            # "Why does X keep increasing?" - analyze trend and explain
            mode = 'factor_explanation'
        elif 'increase' in q or 'improve' in q or 'grow' in q or 'boost' in q or 'strategy' in q or 'what should we do' in q:
            mode = 'strategy'

        return {
            'entity_col': entity_col,
            'entity': entity,
            'years': years,
            'time_col': time_col,
            'value_col': value_col,
            'mode': mode
        }

    def _handle_intent(self, intent: dict, df: pd.DataFrame) -> str:
        entity_col = intent.get('entity_col')
        entity = intent.get('entity')
        years = intent.get('years') or []
        time_col = intent.get('time_col')
        value_col = intent.get('value_col')
        mode = intent.get('mode')

        if mode == 'compare_years' and time_col and value_col:
            # Handle year comparison - can work with or without entity filter
            if entity_col and entity:
                sub_df = df[df[entity_col] == entity]
            else:
                sub_df = df
            
            # Extract year from time column (handle date parsing)
            time_series = sub_df[time_col]
            if not pd.api.types.is_datetime64_any_dtype(time_series):
                # Try to parse as dates
                try:
                    time_series = pd.to_datetime(time_series, errors='coerce')
                except:
                    pass
            
            # Extract year
            if pd.api.types.is_datetime64_any_dtype(time_series):
                sub_df = sub_df.copy()
                sub_df['_year'] = time_series.dt.year
            else:
                # Assume time_col is already year-like numeric
                sub_df = sub_df.copy()
                sub_df['_year'] = pd.to_numeric(sub_df[time_col], errors='coerce')
            
            series = sub_df.groupby('_year')[value_col].sum().sort_index()
            
            if len(series) < 2:
                return f"I need at least two years of data to compare."
            
            # Determine years to compare
            y1 = years[0] if years else series.index[-2]
            y2 = years[1] if len(years) > 1 else (years[0] + 1 if years and (years[0] + 1) in series.index else series.index[-1])
            
            if y1 not in series.index or y2 not in series.index:
                available = ', '.join(map(str, sorted(series.index.tolist())))
                return f"I couldn't find both {y1} and {y2} in the data. Available years: {available}"
            
            v1, v2 = series.loc[y1], series.loc[y2]
            delta = v2 - v1
            pct = (delta / v1 * 100.0) if v1 != 0 else float('inf')
            
            # Format the comparison
            metric_name = value_col.replace('_', ' ')
            if entity:
                result = f"{entity}'s {metric_name} changed from {v1:,.0f} in {y1} to {v2:,.0f} in {y2} ({pct:+.1f}%)."
            else:
                result = f"{metric_name.capitalize()} changed from {v1:,.0f} in {y1} to {v2:,.0f} in {y2} ({pct:+.1f}%)."
            
            return result

        if mode == 'over_time_summary' and entity and time_col and value_col:
            sub = df[df[entity_col] == entity]
            if sub.empty:
                return f"I couldn't find {entity} in the dataset."
            by_year = sub.groupby(time_col)[value_col].sum().sort_index()
            peak_year = by_year.idxmax()
            peak_val = by_year.max()
            return f"**{entity}** {value_col.replace('_',' ')} peaks at **{peak_year}** with **{peak_val:,.0f}**. Range: {by_year.index.min()}–{by_year.index.max()} ({len(by_year)} periods)."

        if mode == 'factor_explanation':
            if entity and time_col and value_col:
                # Check if question is about decline or increase
                question_lower = str(intent.get('_question', '')).lower()
                if 'drop' in question_lower or 'decline' in question_lower or 'decrease' in question_lower or 'fall' in question_lower:
                    return self._analyze_sales_decline("why " + entity, df)
                else:
                    # Analyze trend/growth - show data and trend
                    return self._analyze_trend_over_time(entity, time_col, value_col, df)
            elif time_col and value_col:
                # No entity, just analyze the value column trend
                question_lower = str(intent.get('_question', '')).lower()
                if 'increase' in question_lower or 'grow' in question_lower or 'keep' in question_lower:
                    return self._analyze_trend_over_time(None, time_col, value_col, df)

        if mode == 'strategy':
            return self._handle_strategy(df, entity_col, entity, time_col, value_col)

        if mode == 'non_time_trend':
            return self._dataset_trends_and_correlations(df)

        if mode == 'correlation_pair':
            a, b = intent.get('pair_cols')
            series = df[[a, b]].dropna()
            if series.empty:
                return f"I couldn't compute correlation; {a} and {b} have no overlapping data."
            corr_matrix = series.corr(numeric_only=True)
            # Check if correlation matrix has both dimensions
            if corr_matrix.shape[0] < 2 or corr_matrix.shape[1] < 2:
                return f"I couldn't compute correlation between {a} and {b}; insufficient numeric data."
            corr = corr_matrix.iloc[0, 1]
            return f"Correlation between {a} and {b}: r={corr:+.3f}."

        return None

    def _find_time_column(self, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # prefer explicit 'year'
        for c in numeric_cols:
            if 'year' in c.lower():
                return c
        # year-like range
        for c in numeric_cols:
            try:
                mn, mx = df[c].min(), df[c].max()
                if pd.notna(mn) and pd.notna(mx) and 1900 <= float(mn) <= 2100 and 1900 <= float(mx) <= 2100:
                    return c
            except Exception:
                continue
        return None

    def _pick_value_column(self, df: pd.DataFrame) -> str:
        """Select KPI dynamically (no hardcode to a specific dataset):
        1) Name priority (sales-like)
        2) Temporal relevance with detected year column
        3) Variance fallback
        Exclude common spec-like columns.
        Optionally use Claude ranking when available.
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return None

        # Exclusion patterns (generic specs)
        deny = ['transmission', 'engine', 'color', 'doors', 'trim', 'drivetrain', 'fuel']
        numeric_cols = [c for c in numeric_cols if not any(d in c.lower() for d in deny)]
        if not numeric_cols:
            return None

        # 1) Name priority
        priority = ['sales', 'revenue', 'units', 'volume', 'qty', 'amount', 'engagement', 'kpi']
        for c in numeric_cols:
            cl = c.lower()
            if any(p in cl for p in priority):
                return c

        # 2) Temporal relevance
        time_col = self._find_time_column(df)
        if time_col and time_col in df.columns:
            try:
                # Spearman correlation absolute value
                import numpy as np
                from scipy.stats import spearmanr
                best = None
                best_abs = -1
                years = df[time_col]
                for c in numeric_cols:
                    series = df[[time_col, c]].dropna()
                    if len(series) >= 8:
                        r, _ = spearmanr(series[time_col], series[c])
                        val = abs(r) if r is not None else 0
                        if np.isfinite(val) and val > best_abs:
                            best = c
                            best_abs = val
                if best:
                    return best
            except Exception:
                pass

        # 3) Variance fallback
        variances = [(c, float(df[c].std())) for c in numeric_cols]
        variances = [(c, v) for c, v in variances if not pd.isna(v)]
        if variances:
            variances.sort(key=lambda x: -x[1])
            return variances[0][0]
        return None

    def _select_kpi_column(self, df: pd.DataFrame) -> str:
        """Try Claude ranking first (if available), then fallback to heuristic."""
        try:
            # Prepare small preview stats to keep prompt small
            cols = df.select_dtypes(include=['number']).columns.tolist()
            if not cols:
                return None
            preview = {}
            for c in cols[:20]:  # cap
                s = df[c]
                preview[c] = {
                    'missing_pct': float(s.isna().mean()),
                    'mean': float(s.dropna().mean()) if s.notna().any() else 0.0,
                    'std': float(s.dropna().std()) if s.notna().any() else 0.0,
                }
            # Heuristic KPI ranking (no Claude)
            deny = ['transmission', 'engine', 'color', 'doors', 'trim', 'drivetrain', 'fuel']
            kpi_keywords = ['sales', 'revenue', 'volume', 'quantity', 'amount', 'units', 'engagement', 'members', 'users']
            # Prioritize columns with KPI-like names
            for c in cols:
                c_lower = c.lower()
                if not any(d in c_lower for d in deny):
                    if any(k in c_lower for k in kpi_keywords):
                        return c
        except Exception:
            pass
        return self._pick_value_column(df)

    def _dataset_trends_and_correlations(self, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return "I don't see numeric columns to summarize trends."
        insights = [f"Dataset has {len(df):,} rows and {len(df.columns)} columns."]
        # top 3 by variance
        scored = []
        for c in numeric_cols:
            col = df[c].dropna()
            if len(col) >= 3:
                mean = col.mean()
                std = col.std()
                if std and abs(mean) > 0:
                    cv = std/abs(mean)
                    scored.append((c, cv))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            insights.append("Most variable metrics: " + ", ".join([f"{c.replace('_',' ')} (CV {cv:.2f})" for c, cv in scored[:3]]))
        # simple correlation matrix top pairs
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            pairs = []
            for i, a in enumerate(numeric_cols):
                for b in numeric_cols[i+1:]:
                    val = corr.at[a, b]
                    if pd.notna(val):
                        pairs.append((a, b, abs(val), val))
            pairs.sort(key=lambda x: x[2], reverse=True)
            if pairs:
                top = pairs[0]
                insights.append(f"Strongest correlation: {top[0].replace('_',' ')} vs {top[1].replace('_',' ')} (r={top[3]:+.2f})")
        return "\n".join(insights)

    def _extract_columns_from_question(self, q: str, df: pd.DataFrame) -> list:
        """Extract column names from question using flexible matching (no hardcoding)"""
        cols = [c for c in df.columns]
        lower_map = {c.lower(): c for c in cols}
        found = []
        q_lower = q.lower()
        
        # Split question into tokens (words)
        q_tokens = set(q_lower.split())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                     'why', 'what', 'when', 'where', 'how', 'which', 'who', 'did', 'does',
                     'drop', 'drops', 'dropped', 'decline', 'declined', 'decrease', 'decreased',
                     'increase', 'increased', 'rise', 'rose', 'fall', 'fell', 'change', 'changed'}
        q_tokens -= stop_words
        
        # Direct substring match (highest priority)
        for name_lower, orig in lower_map.items():
            if name_lower in q_lower or q_lower in name_lower:
                found.append(orig)
                if len(found) == 2:
                    break
        
        # If not enough matches, try word-level matching
        if len(found) < 2:
            for col in cols:
                if col in found:
                    continue
                col_lower = col.lower()
                col_words = set(col_lower.replace('_', ' ').replace('-', ' ').split())
                # Remove common generic words from column names
                col_words -= {'total', 'sum', 'avg', 'average', 'mean', 'count', 'number', 
                             'amount', 'value', 'rate', 'ratio', 'percent', 'percentage'}
                
                # Check if any meaningful words from column appear in question
                overlap = q_tokens & col_words
                if overlap:
                    # Also check reverse: if question word appears in column name
                    for token in q_tokens:
                        if token in col_lower or col_lower in token:
                            found.append(col)
                            break
                    if len(found) == 2:
                        break
        
        # If still not enough, try partial word matching (e.g., "travel" matches "travel_rate")
        if len(found) < 2:
            for col in cols:
                if col in found:
                    continue
                col_lower = col.lower().replace('_', ' ').replace('-', ' ')
                for token in q_tokens:
                    if len(token) >= 4 and (token in col_lower or col_lower.startswith(token) or col_lower.endswith(token)):
                        found.append(col)
                        break
                if len(found) == 2:
                    break
        
        return found[:2]

    def _handle_strategy(self, df: pd.DataFrame, entity_col: str, entity: str, time_col: str, value_col: str) -> str:
        # Strategy recommendations based on correlations/segments (data-driven)
        lines = ["Strategy recommendations based on your data:"]
        # If entity specified, use its slice; else whole dataset
        scope = df
        label = "overall"
        if entity_col and entity:
            scope = df[df[entity_col] == entity]
            label = entity
        # price elasticity hint
        numeric_cols = scope.select_dtypes(include=['number']).columns.tolist()
        price_cols = [c for c in numeric_cols if any(k in c.lower() for k in ['price', 'usd', 'amount'])]
        if value_col and price_cols:
            pc = price_cols[0]
            if time_col:
                agg = scope.groupby(time_col).agg({value_col: 'sum', pc: 'mean'})
            else:
                agg = scope[[value_col, pc]].dropna()
            if len(agg) >= 3:
                corr = agg.corr(numeric_only=True)[value_col].get(pc, None)
                if pd.notna(corr):
                    if corr < -0.2:
                        lines.append(f"- Consider price testing: {value_col.replace('_',' ')} is negatively correlated with {pc.replace('_',' ')} (r={corr:+.2f}) for {label}.")
                    elif corr > 0.2:
                        lines.append(f"- Pricing not a headwind: positive correlation with {pc.replace('_',' ')} (r={corr:+.2f}).")
        # region focus
        cat_cols = scope.select_dtypes(include=['object']).columns.tolist()
        region_cols = [c for c in cat_cols if any(k in c.lower() for k in ['region', 'market', 'area'])]
        if value_col and region_cols:
            rc = region_cols[0]
            perf = scope.groupby(rc)[value_col].sum().sort_values(ascending=False)
            if len(perf) >= 2:
                lines.append(f"- Double down on top {rc}: {perf.index[0]} (best), explore improving {perf.index[-1]} (lowest).")
        # segment optimization generic
        if cat_cols:
            seg = cat_cols[0]
            seg_perf = scope.groupby(seg).size().sort_values(ascending=False).head(3)
            if len(seg_perf) > 0:
                lines.append(f"- Focus on leading {seg}: " + ", ".join([str(i) for i in seg_perf.index.tolist()]))
        return "\n".join(lines)
    
    def _generate_visualizations(self, question: str, df: pd.DataFrame) -> list:
        """Generate relevant visualizations based on the question and data - DYNAMIC"""
        if df is None:
            return None
        
        visualizations = []
        question_lower = question.lower()
        
        # Find categorical and numeric columns dynamically
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Derive year from date-like columns or datetime index (to enable trends without hardcoding)
        year_series = None
        try:
            if hasattr(df.index, 'inferred_type') and 'date' in str(df.index.inferred_type):
                year_series = pd.to_datetime(df.index, errors='coerce').year
            if year_series is None:
                # Prefer columns containing 'date'
                for c in list(df.columns):
                    if 'date' in c.lower():
                        dt = pd.to_datetime(df[c], errors='coerce')
                        if dt.notna().sum() > max(50, int(len(df)*0.2)):
                            year_series = dt.dt.year
                            break
            if year_series is None:
                # Fallback: try first object col that parses to many dates
                for c in categorical_cols:
                    dt = pd.to_datetime(df[c], errors='coerce')
                    if dt.notna().sum() > max(50, int(len(df)*0.2)):
                        year_series = dt.dt.year
                        break
        except Exception:
            year_series = None

        # Find potential entity columns (categorical with reasonable unique count)
        entity_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 50]
        
        # Find potential value columns (numeric)
        value_cols = [col for col in numeric_cols if col.lower() not in ['year', 'id', 'index']]
        
        # Find time columns
        time_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['year', 'date', 'time'])]
        
        # Correlation request: scatter plot when two columns are referenced
        pair_cols = self._extract_columns_from_question(question_lower, df)
        if len(pair_cols) == 2:
            a, b = pair_cols
            sample = df[[a, b]].dropna()
            if not sample.empty:
                # downsample to reasonable size
                if len(sample) > 1000:
                    sample = sample.sample(1000, random_state=42)
                points = [{ 'x': float(x), 'y': float(y) } for x, y in sample[[a, b]].to_numpy() if pd.notna(x) and pd.notna(y)]
                visualizations.append({
                    'type': 'scatter',
                    'spec': {
                        'title': f'{a} vs {b}',
                        'data': { 'points': points },
                        'format': { 'x': 'comma', 'y': 'comma', 'tooltip': 'comma' }
                    }
                })
                return visualizations

        # Generate visualizations based on available data
        # Prefer KPI column selection using enhanced selector
        kpi_col = self._select_kpi_column(df) if df is not None else None

        if entity_cols and (value_cols or kpi_col):
            # Top entities by value
            entity_col = entity_cols[0]
            value_col = kpi_col or value_cols[0]
            
            entity_values = df.groupby(entity_col)[value_col].sum().nlargest(5)
            if not entity_values.empty:
                visualizations.append({
                    "type": "bar",
                    "spec": {
                        "title": f"Top 5 {entity_col} by {value_col}",
                        "data": {
                            "labels": entity_values.index.tolist(),
                            "datasets": [{
                                "label": value_col,
                                "data": entity_values.values.tolist(),
                                "backgroundColor": "#3B82F6"
                            }]
                        },
                        "format": { "y": "comma", "tooltip": "comma" }
                    }
                })
        
        # Prefer using derived year when available; else fall back to numeric time column
        if (year_series is not None) and (value_cols or kpi_col):
            value_col = kpi_col or value_cols[0]
            tmp = pd.DataFrame({ 'year': year_series, 'val': df[value_col] }).dropna()
            yearly = tmp.groupby('year')['val'].mean().sort_index()
            if len(yearly) > 1:
                visualizations.append({
                    "type": "line",
                    "spec": {
                        "title": f"{value_col} Trend by Year",
                        "data": {
                            "labels": yearly.index.astype(str).tolist(),
                            "datasets": [{
                                "label": value_col,
                                "data": yearly.values.tolist(),
                                "borderColor": "#6B5AE0",
                                "backgroundColor": "rgba(107,90,224,0.10)",
                                "fill": True
                            }]
                        },
                        "format": { "y": "comma", "tooltip": "comma" }
                    }
                })
        elif time_cols and (value_cols or kpi_col):
            time_col = time_cols[0]
            value_col = kpi_col or value_cols[0]
            time_values = df.groupby(time_col)[value_col].mean().sort_index()
            if len(time_values) > 1:
                visualizations.append({
                    "type": "line",
                    "spec": {
                        "title": f"{value_col} Over Time",
                        "data": {
                            "labels": time_values.index.astype(str).tolist(),
                            "datasets": [{
                                "label": value_col,
                                "data": time_values.values.tolist(),
                                "borderColor": "#10B981",
                                "backgroundColor": "rgba(16,185,129,0.10)",
                                "fill": True
                            }]
                        }
                    }
                })

        # Add volume-by-year if volume column exists
        if year_series is not None:
            vol_col = next((c for c in numeric_cols if 'volume' in c.lower()), None)
            if vol_col is not None:
                vt = pd.DataFrame({ 'year': year_series, 'val': df[vol_col] }).dropna()
                v_yearly = vt.groupby('year')['val'].sum().sort_index()
                if len(v_yearly) > 1:
                    visualizations.append({
                        "type": "bar",
                        "spec": {
                            "title": f"{vol_col} by Year",
                            "data": {
                                "labels": v_yearly.index.astype(str).tolist(),
                                "datasets": [{
                                    "label": vol_col,
                                    "data": v_yearly.values.tolist(),
                                    "backgroundColor": "#A18AFF"
                                }]
                            },
                            "format": { "y": "comma", "tooltip": "comma" }
                        }
                    })

        # Add high-low range% trend if columns exist
        if year_series is not None:
            high_col = next((c for c in numeric_cols if 'high' in c.lower()), None)
            low_col = next((c for c in numeric_cols if 'low' in c.lower()), None)
            close_col = next((c for c in numeric_cols if 'close' in c.lower()), None)
            if high_col and low_col and close_col:
                rng_pct = (df[high_col] - df[low_col]) / df[close_col].replace(0, pd.NA) * 100.0
                rt = pd.DataFrame({ 'year': year_series, 'val': rng_pct }).dropna()
                r_yearly = rt.groupby('year')['val'].mean().sort_index()
                if len(r_yearly) > 1:
                    visualizations.append({
                        "type": "line",
                        "spec": {
                            "title": "Daily Range % (avg) by Year",
                            "data": {
                                "labels": r_yearly.index.astype(str).tolist(),
                                "datasets": [{
                                    "label": "Range %",
                                    "data": r_yearly.values.tolist(),
                                    "borderColor": "#F59E0B",
                                    "backgroundColor": "rgba(245,158,11,0.10)",
                                    "fill": True
                                }]
                            },
                            "format": { "y": "comma", "tooltip": "comma" }
                        }
                    })
        
        return visualizations if visualizations else None
    
    def _generate_suggested_actions(self, question: str, df: pd.DataFrame) -> list:
        """Generate suggested follow-up actions based on the question - DYNAMIC"""
        if df is None:
            return None
        
        actions = []
        question_lower = question.lower()
        
        # Get actual column names from the dataset
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Find potential entity columns
        entity_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 50]
        
        # Find time columns
        time_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['year', 'date', 'time'])]
        
        # Generate generic actions based on available data
        if entity_cols:
            entity_col = entity_cols[0]
            actions.append({
                "label": f"Top {entity_col}", 
                "action": f"Show me the top {entity_col} by performance"
            })
        
        if len(entity_cols) > 1:
            actions.append({
                "label": "Compare categories", 
                "action": f"Compare {entity_cols[0]} vs {entity_cols[1]}"
            })
        
        if time_cols:
            time_col = time_cols[0]
            actions.append({
                "label": "Time analysis", 
                "action": f"Show trends over {time_col}"
            })
        
        # Generic actions
        actions.extend([
            {"label": "Data summary", "action": "Give me a summary of this dataset"},
            {"label": "Correlation analysis", "action": "What correlations exist in the data?"},
            {"label": "Outlier detection", "action": "Are there any unusual patterns?"}
        ])
        
        return actions[:5]  # Limit to 5 actions
    
    def _extract_citations(self, text: str) -> list:
        """Extract citations from text that contains external context"""
        if not text or "Source:" not in text:
            return None
        
        citations = []
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith('Source:'):
                url = line.replace('Source:', '').strip()
                if url:
                    citations.append(url)
        
        return citations if citations else None
    
    def _analyze_entity_over_time(self, question: str, df: pd.DataFrame) -> str:
        """Analyze entity performance over time (generic, dynamic)."""
        
        try:
            # Dynamically find categorical and time columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Find entity columns (categorical with reasonable unique count)
            entity_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 100]
            
            # Find time columns (numeric that look like years/dates)
            time_cols = [col for col in numeric_cols if 
                        any(word in col.lower() for word in ['year', 'date', 'time']) or
                        (df[col].min() >= 1900 and df[col].max() <= 2030)]  # Year range detection
            
            # Find value columns (numeric, not time/id columns) - prioritize sales/volume/revenue
            value_priority = ['sales', 'volume', 'revenue', 'amount', 'total', 'count', 'price', 'value']
            
            priority_cols = []
            other_cols = []
            
            for col in numeric_cols:
                if col in time_cols or any(word in col.lower() for word in ['id', 'index', 'row']):
                    continue
                
                if any(priority_word in col.lower() for priority_word in value_priority):
                    priority_cols.append(col)
                else:
                    other_cols.append(col)
            
            value_cols = priority_cols + other_cols  # Prioritize sales/volume columns
            
            if not entity_cols:
                return f"I can't find categorical columns to analyze. Available columns: {', '.join(df.columns.tolist())}"
            
            if not time_cols:
                return f"I can't find time-based columns to analyze trends over time. Available columns: {', '.join(df.columns.tolist())}"
            
            if not value_cols:
                return f"I can't find numeric value columns to analyze. Available columns: {', '.join(df.columns.tolist())}"
            
            # Use the first available columns of each type
            entity_col = entity_cols[0]
            time_col = time_cols[0]
            value_col = value_cols[0]
            
            print(f"🔍 Dynamic analysis: {entity_col} over {time_col} measuring {value_col}")
            
            if 'top' in question and ('by' in question or 'each' in question):
                # Find top entity for each time period
                yearly_winners = df.loc[df.groupby(time_col)[value_col].idxmax()]
                
                result = f"**Top {entity_col} by {time_col}:**\n\n"
                for _, row in yearly_winners.iterrows():
                    time_val = row[time_col]
                    entity_val = row[entity_col]
                    value_val = row[value_col]
                    
                    # Format time value appropriately
                    if isinstance(time_val, float) and time_val.is_integer():
                        time_str = str(int(time_val))
                    else:
                        time_str = str(time_val)
                    
                    result += f"• **{time_str}**: {entity_val} ({value_val:,.0f})\n"
                
                return result
            
            elif 'peak' in question or 'when did' in question:
                # Find when a specific entity peaked
                # Extract entity name from question (case-insensitive)
                entity_name = None
                words = question.split()
                unique_entities = df[entity_col].unique()
                
                for i, word in enumerate(words):
                    # Look for entity names mentioned in the question (case-insensitive)
                    for entity in unique_entities:
                        if word.lower() == entity.lower():
                            entity_name = entity
                            break
                    if entity_name:
                        break
                    
                    # Handle multi-word entities like "7 series"
                    if i < len(words) - 1:
                        two_word = f"{word} {words[i+1]}"
                        for entity in unique_entities:
                            if two_word.lower() == entity.lower():
                                entity_name = entity
                                break
                        if entity_name:
                            break
                
                if entity_name:
                    entity_data = df[df[entity_col] == entity_name]
                    if not entity_data.empty:
                        peak_row = entity_data.loc[entity_data[value_col].idxmax()]
                        peak_time = peak_row[time_col]
                        peak_value = peak_row[value_col]
                        
                        return f"**{entity_name}** peaked in **{peak_time}** with {value_col} of **{peak_value:,.0f}**."
                
                return f"I couldn't find a specific {entity_col.lower()} mentioned in your question. Available {entity_col.lower()}s: {', '.join(df[entity_col].unique()[:10])}"
            
            else:
                # Show entity performance across time periods
                entity_time = df.groupby([entity_col, time_col])[value_col].sum().reset_index()
                
                result = f"**{entity_col} Performance Across {time_col}:**\n\n"
                
                # Get top 5 entities by total value
                top_entities = df.groupby(entity_col)[value_col].sum().nlargest(5)
                
                for entity in top_entities.index:
                    entity_data = entity_time[entity_time[entity_col] == entity]
                    time_values = entity_data[time_col].tolist()
                    total_value = top_entities[entity]
                    
                    result += f"**{entity}**: {total_value:,.0f} total\n"
                    result += f"  Active periods: {min(time_values)}-{max(time_values)} ({len(time_values)} periods)\n"
                    
                    # Show peak period
                    peak_data = entity_data.loc[entity_data[value_col].idxmax()]
                    peak_time = peak_data[time_col]
                    peak_value = peak_data[value_col]
                    
                    result += f"  Peak: {peak_time} ({peak_value:,.0f})\n\n"
                
                return result
                
        except Exception as e:
            print(f"Error in dynamic analysis: {e}")
            return f"I encountered an error analyzing the data: {str(e)}. Available columns: {', '.join(df.columns.tolist())}"
    
    def _find_top_entity_by_metric(self, question: str, df: pd.DataFrame) -> str:
        """Find the top entity (model/product/brand) by a sales-like metric (generic)."""
        
        try:
            # Look for model/product/brand columns
            item_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['model', 'product', 'item', 'brand', 'name', 'type']):
                    item_col = col
                    break
            
            if not item_col:
                return "I can't find a model/product column in your dataset. Available columns: " + ", ".join(df.columns.tolist())
            
            # Look for sales/quantity/value columns
            sales_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['sales', 'quantity', 'amount', 'value', 'revenue', 'units', 'sold']):
                    sales_col = col
                    break
            
            if not sales_col:
                # Use the first numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    sales_col = numeric_cols[0]
                else:
                    return "I can't find a sales/quantity column to analyze."
            
            # Find the item with highest sales
            if any(year in question for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']):
                # Extract year from question
                year = None
                for y in ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']:
                    if y in question:
                        year = int(y)
                        break
                
                # Filter for specific year data if year column exists
                year_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    if any(word in col_lower for word in ['year', 'date', 'time']):
                        year_col = col
                        break
                
                if year_col and year:
                    df_filtered = df[df[year_col] == year] if year_col in df.columns else df
                    print(f"🔍 Filtered data for year {year}: {len(df_filtered)} rows")
                else:
                    df_filtered = df
            else:
                df_filtered = df
            
            # Group by item and sum sales
            grouped = df_filtered.groupby(item_col)[sales_col].sum().sort_values(ascending=False)
            
            if grouped.empty:
                return f"No data found for the specified criteria. Dataset has {len(df)} rows."
            
            best_item = grouped.index[0]
            best_sales = grouped.iloc[0]
            
            year_text = f" in {year}" if 'year' in locals() and year else ""
            return f"Based on your data, **{best_item}** was the best selling model{year_text} with {sales_col} of {best_sales:,.0f}. The top 3 models were: " + ", ".join([f"{item} ({sales:,.0f})" for item, sales in grouped.head(3).items()])
            
        except Exception as e:
            print(f"Error in best selling analysis: {e}")
            return f"I encountered an error analyzing your data: {str(e)}. Available columns: {', '.join(df.columns.tolist())}"
    
    def _find_max_by_location(self, question: str, df: pd.DataFrame) -> str:
        """Find location (country/region/state/city) with maximum value for a metric (generic)."""
        
        try:
            # Find location column generically
            country_col = self._find_location_column(df)
            if not country_col:
                return "I can't identify a country or location column in your dataset."
            
            # Find relevant numeric columns based on question keywords
            target_col = self._find_target_column(question, df)
            if not target_col:
                # Fallback: use the largest numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]
                else:
                    return "I can't find any numeric columns to analyze in your dataset."
            
            # Find the maximum value
            max_row = df.loc[df[target_col].idxmax()]
            location = max_row[country_col]
            value = max_row[target_col]
            
            # Format the response generically
            return self._format_max_response(location, target_col, value, question)
                    
        except Exception as e:
            print(f"Error in max analysis: {e}")
            return "I encountered an error analyzing your data. Please check that your dataset has both location and numeric columns."
    
    def _find_min_by_location(self, question: str, df: pd.DataFrame) -> str:
        """Find location (country/region/state/city) with minimum value for a metric (generic)."""
        
        try:
            country_col = self._find_location_column(df)
            if not country_col:
                return "I can't identify a country or location column in your dataset."
            
            target_col = self._find_target_column(question, df)
            if not target_col:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]
                else:
                    return "I can't find any numeric columns to analyze in your dataset."
            
            min_row = df.loc[df[target_col].idxmin()]
            location = min_row[country_col]
            value = min_row[target_col]
            
            return self._format_min_response(location, target_col, value, question)
            
        except Exception as e:
            print(f"Error in min analysis: {e}")
            return "I encountered an error analyzing your data. Please check that your dataset has both location and numeric columns."
    
    def _analyze_trends(self, df: pd.DataFrame) -> str:
        """Analyze trends in the data - GENERIC approach"""
        
        insights = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return "I don't see any numeric columns in your dataset to analyze for trends."
        
        try:
            # Analyze top 3 numeric columns
            for col in numeric_cols[:3]:
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                
                # Determine if values are percentages, currency, or regular numbers
                col_clean = col.replace('_', ' ').title()
                
                if std_val > 0:  # Only analyze columns with variation
                    cv = std_val / abs(mean_val) if mean_val != 0 else 0
                    
                    if cv > 1:
                        insights.append(f"• **{col_clean}**: High variation (CV: {cv:.1f}) - values range widely")
                    elif cv > 0.5:
                        insights.append(f"• **{col_clean}**: Moderate variation - some diversity in values") 
                    else:
                        insights.append(f"• **{col_clean}**: Low variation - values are relatively similar")
            
            # Add overall dataset insights
            total_records = len(df)
            if total_records > 100:
                insights.insert(0, f"**Dataset Overview:** Large dataset with {total_records} records")
            else:
                insights.insert(0, f"**Dataset Overview:** {total_records} records analyzed")
                
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return "I can see your dataset has numeric data, but encountered an issue analyzing the trends."
        
        if insights:
            return '\n'.join(insights)
        else:
            return "I can see numeric data in your dataset but need more information to identify specific trends."
    
    def _compare_metrics(self, question: str, df: pd.DataFrame) -> str:
        """Compare different metrics or countries"""
        return "I can help compare specific metrics or countries. Try asking something like 'compare USA and China GDP' or 'which countries have similar population sizes'."
    
    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate a data-driven summary based on actual data"""
        insights = []
        
        insights.append(f"**Dataset Summary:** {len(df)} records with {len(df.columns)} columns")
        
        # Analyze numeric columns dynamically
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            insights.append(f"**Numeric Analysis:**")
            for col in numeric_cols[:3]:  # Top 3 numeric columns
                total = df[col].sum()
                mean = df[col].mean()
                if total > 1e12:
                    insights.append(f"• {col}: Total ${total/1e12:.1f}T, Average ${mean/1e9:.1f}B")
                elif total > 1e9:
                    insights.append(f"• {col}: Total ${total/1e9:.1f}B, Average ${mean/1e6:.1f}M")
                else:
                    insights.append(f"• {col}: Total {total:,.0f}, Average {mean:,.0f}")
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            insights.append(f"**Categories:**")
            for col in categorical_cols[:2]:  # Top 2 categorical columns
                unique_count = df[col].nunique()
                insights.append(f"• {col}: {unique_count} unique values")
        
        return '\n'.join(insights)
    
    def _analyze_mentioned_metrics(self, question: str, df: pd.DataFrame) -> str:
        """Analyze any specific metrics mentioned in the question"""
        
        # First, let's provide basic dataset info
        basic_info = f"Your dataset has {len(df)} rows and {len(df.columns)} columns: {', '.join(df.columns.tolist())}"
        
        # Look for column names mentioned in the question
        mentioned_cols = []
        for col in df.columns:
            if col.lower() in question:
                mentioned_cols.append(col)
        
        if mentioned_cols:
            # Analyze the mentioned columns
            results = []
            for col in mentioned_cols[:3]:  # Limit to 3 columns
                if df[col].dtype in ['int64', 'float64']:
                    stats = df[col].describe()
                    results.append(f"{col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
                else:
                    # Categorical analysis
                    top_values = df[col].value_counts().head(3)
                    results.append(f"{col}: top values are {', '.join([f'{val} ({count})' for val, count in top_values.items()])}")
            
            return f"{basic_info}\n\nI found these metrics in your question: {'; '.join(results)}"
        
        # Fallback: provide basic dataset info with sample data
        sample_data = df.head(2).to_dict('records')
        return f"{basic_info}\n\nSample data:\n{str(sample_data)}\n\nI can analyze any of these columns. What specific question would you like me to answer about your data?"
    
    def _find_location_column(self, df: pd.DataFrame) -> str:
        """Find the column that contains location names (country/region/state/city)."""
        possible_names = ['country', 'nation', 'location', 'place', 'region', 'state', 'city']
        
        for col in df.columns:
            if col.lower() in possible_names or any(name in col.lower() for name in possible_names):
                return col
        
        # Fallback: look for object columns with reasonable number of unique values
        for col in df.select_dtypes(include=['object']).columns:
            unique_count = df[col].nunique()
            if 10 <= unique_count <= 300:  # Reasonable range for locations
                return col
        
        return None
    
    def _find_target_column(self, question: str, df: pd.DataFrame) -> str:
        """Find the column the user is asking about based on keywords"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Create keyword mappings for different types of data
        keywords = {
            'population': ['population', 'people', 'inhabitants', 'citizens'],
            'economy': ['gdp', 'economy', 'economic', 'income', 'wealth', 'revenue'],
            'growth': ['growth', 'rate', 'increase', 'change', 'percent'],
            'per_capita': ['capita', 'per person', 'average income', 'individual'],
            'share': ['share', 'percentage', 'portion', 'part']
        }
        
        # Score each numeric column based on question keywords
        best_col = None
        best_score = 0
        
        for col in numeric_cols:
            col_lower = col.lower()
            score = 0
            
            for category, words in keywords.items():
                if any(word in question.lower() for word in words):
                    if any(word in col_lower for word in words):
                        score += 10  # Exact match
                    elif category == 'population' and any(pop_word in col_lower for pop_word in ['pop', 'people']):
                        score += 8
                    elif category == 'economy' and any(econ_word in col_lower for econ_word in ['gdp', 'income', 'revenue']):
                        score += 8
                    elif category == 'growth' and any(growth_word in col_lower for growth_word in ['growth', 'rate', 'change']):
                        score += 8
            
            if score > best_score:
                best_score = score
                best_col = col
        
        return best_col
    
    def _format_max_response(self, location: str, column: str, value: float, question: str) -> str:
        """Format a detailed response for maximum value queries"""
        
        col_clean = column.replace('_', ' ').title()
        
        # Determine appropriate formatting based on value magnitude and column name
        if value >= 1e12:  # Trillions
            formatted_value = f"${value/1e12:.1f} trillion"
        elif value >= 1e9:  # Billions
            formatted_value = f"${value/1e9:.1f} billion"
        elif value >= 1e6:  # Millions
            formatted_value = f"{value/1e6:.1f} million"
        elif 'percent' in column.lower() or '%' in str(value):
            formatted_value = f"{value:.1f}%"
        elif value >= 1000:
            formatted_value = f"{value:,.0f}"
        else:
            formatted_value = f"{value:.2f}"
        
        # Create intelligent response based on the data
        if 'population' in question.lower():
            return f"Based on my analysis of your dataset, **{location}** has the highest population with **{formatted_value}** people. This represents the largest demographic concentration in your data."
        elif any(word in question.lower() for word in ['gdp', 'economy', 'economic']):
            return f"According to your economic data, **{location}** leads with the highest GDP of **{formatted_value}**. This makes it the largest economy in your dataset by nominal GDP."
        elif 'growth' in question.lower():
            return f"**{location}** shows the strongest economic performance with a growth rate of **{formatted_value}**. This is the highest growth rate among all countries in your dataset."
        else:
            return f"**{location}** ranks highest in {col_clean} with a value of **{formatted_value}**. This represents the maximum value for this metric in your dataset."
    
    def _format_min_response(self, location: str, column: str, value: float, question: str) -> str:
        """Format a detailed response for minimum value queries"""
        
        col_clean = column.replace('_', ' ').title()
        
        # Format value appropriately
        if value >= 1e9:
            formatted_value = f"{value/1e9:.1f} billion"
        elif value >= 1e6:
            formatted_value = f"{value/1e6:.1f} million"
        elif 'percent' in column.lower():
            formatted_value = f"{value:.1f}%"
        elif value >= 1000:
            formatted_value = f"{value:,.0f}"
        else:
            formatted_value = f"{value:.2f}"
        
        return f"**{location}** has the lowest {col_clean.lower()} at **{formatted_value}**. This represents the minimum value for this metric in your dataset."
    
    def _needs_contextual_search(self, question: str) -> bool:
        """Detect if a question needs external context search - ENHANCED for reasoning queries"""
        
        question_lower = question.lower()
        
        # Strong reasoning indicators that require external context
        reasoning_keywords = [
            'why', 'because', 'reason', 'cause', 'due to', 'behind',
            'what caused', 'how come', 'explain', 'what happened',
            'what led to', 'what drove', 'what influenced', 'what affected'
        ]
        
        # Market/economic context indicators
        market_keywords = [
            'market', 'economic', 'economy', 'financial', 'stock', 'trading',
            'investor', 'investment', 'recession', 'boom', 'crash', 'volatility',
            'policy', 'regulation', 'government', 'fed', 'interest rate'
        ]
        
        # Time-based indicators that often need context (dynamic year detection)
        import re
        years_in_question = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
        time_indicators = [
            'last year', 'this year', 'recent', 'lately', 'currently',
            'pandemic', 'covid', 'crisis', 'recovery', 'recession', 'boom'
        ] + years_in_question  # Add any years found in the question
        
        # Company/industry context - look for capitalized words and business terms
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', question)
        company_keywords = []
        for word in capitalized_words:
            if word.lower() not in ['Why', 'What', 'When', 'Where', 'How', 'The', 'This', 'That', 'These', 'Those']:
                company_keywords.append(word.lower())
        
        # Add common business/industry terms
        business_terms = ['tech', 'technology', 'smartphone', 'software', 'hardware', 'finance', 'banking', 'retail', 'manufacturing']
        company_keywords.extend([term for term in business_terms if term in question_lower])
        
        # Check for reasoning patterns
        has_reasoning = any(keyword in question_lower for keyword in reasoning_keywords)
        
        # Check for market/economic context
        has_market_context = any(keyword in question_lower for keyword in market_keywords)
        
        # Check for time-based context
        has_time_context = any(keyword in question_lower for keyword in time_indicators)
        
        # Check for company/industry context
        has_company_context = any(keyword in question_lower for keyword in company_keywords)
        
        # Strategy/recommendation keywords that benefit from external context
        strategy_keywords = [
            'what should', 'how to', 'how can', 'what can', 'recommendation', 'recommend',
            'strategy', 'strategic', 'solution', 'solutions', 'action', 'actions',
            'best practice', 'best practices', 'improve', 'optimize', 'reduce', 'decrease',
            'increase', 'enhance', 'address', 'tackle', 'solve'
        ]
        has_strategy = any(keyword in question_lower for keyword in strategy_keywords)
        
        # Trigger search if any combination suggests need for external context
        needs_search = (
            has_reasoning or  # Any "why" type question
            has_strategy or  # Strategy/recommendation questions benefit from external best practices
            (has_market_context and has_time_context) or  # Market questions with time
            (has_company_context and has_reasoning) or  # Company-specific reasoning
            (has_time_context and has_reasoning)  # Time-based reasoning
        )
        
        print(f"🔍 Search detection: reasoning={has_reasoning}, strategy={has_strategy}, market={has_market_context}, time={has_time_context}, company={has_company_context} -> {needs_search}")
        
        return needs_search
    
    async def _get_contextual_search(self, question: str) -> str:
        """Get contextual information from web search"""
        
        try:
            from services.you_search import search_web
            
            # Try exact user question first (most reliable, avoids 403s from complex queries)
            print("🔍 Trying exact question first…")
            loop = asyncio.get_event_loop()
            direct_result = await loop.run_in_executor(None, search_web, question, 4)
            
            combined_results = []
            if direct_result and isinstance(direct_result, list) and len(direct_result) > 0:
                print(f"✅ Direct search succeeded: {len(direct_result)} results")
                combined_results = direct_result
            
            # ALWAYS also try generated queries for better results (especially for growth/increase questions)
            print("🔍 Also trying targeted queries for better context…")
            search_queries = self._generate_search_queries(question)
            
            # Perform searches with generated queries
            search_tasks = [
                loop.run_in_executor(None, search_web, query, 3) 
                for query in search_queries[:2]  # Limit to top 2 queries
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results from generated queries
            for result_group in results:
                if isinstance(result_group, list):
                    # Deduplicate by URL to avoid showing same result twice
                    for item in result_group:
                        url = item.get('url', '')
                        if url and not any(r.get('url') == url for r in combined_results):
                            combined_results.append(item)
            
            print(f"📱 Total combined results: {len(combined_results)}")
            
            if not combined_results:
                # Direct News fallback if Search yielded nothing
                try:
                    print("📰 Web search empty; trying News fallback directly…")
                    from services.you_news import call_news
                    # Build multiple focused queries using simple heuristics (no Claude, no hardcoding)
                    news_queries = []
                    try:
                        import re
                        years = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
                        
                        # Extract entities dynamically (companies, products, etc.)
                        question_lower = question.lower()
                        all_words = re.findall(r'\b[a-zA-Z]+\b', question_lower)
                        stop_words = {
                            'why', 'what', 'when', 'where', 'how', 'the', 'this', 'that', 'these', 'those',
                            'did', 'does', 'do', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
                            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                            'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
                            'revenue', 'sales', 'profit', 'earnings', 'growth', 'decline', 'drop', 'performance'
                        }
                        entities = []
                        for word in all_words:
                            if word not in stop_words and len(word) > 2:
                                # Pattern: "why did [word] ..." or "[word] revenue"
                                if (re.search(rf'\b(why did|what caused|how did)\s+{re.escape(word)}\b', question_lower) or
                                    re.search(rf'\b{re.escape(word)}\s+(revenue|sales|growth|decline|performance)\b', question_lower)):
                                    entities.append(word)
                        
                        entities = list(dict.fromkeys(entities))[:2]  # Max 2 entities
                        entity = entities[0] if entities else ""
                        
                        # Extract key metrics/concepts
                        concepts = re.findall(r'\b(revenue|sales|profit|earnings|growth|decline|drop|performance)\b', question_lower)
                        concept = concepts[0] if concepts else "performance"
                        
                        # Build dynamic variant queries
                        news_queries = [question]  # Always start with original
                        
                        if entity and years:
                            year1 = years[0]
                            year2 = years[1] if len(years) > 1 else str(int(year1) - 1)
                            news_queries.extend([
                                f"{entity} {year1} {concept}",
                                f"{entity} {year1} vs {year2} {concept}",
                                f"{entity} earnings {year1}",
                            ])
                        elif entity:
                            news_queries.append(f"{entity} {concept}")
                        elif years:
                            news_queries.append(f"{years[0]} {concept}")
                    except Exception:
                        news_queries = [question]

                    # Deduplicate and try in order
                    seen = set()
                    deduped = []
                    for q in news_queries:
                        if q not in seen:
                            seen.add(q)
                            deduped.append(q)
                    news_queries = deduped

                    for nq in news_queries:
                        items = call_news(nq, count=4) or []
                        if items:
                            print(f"📰 News fallback returned {len(items)} items for query: {nq}")
                            formatted_context = []
                            for i, item in enumerate(items[:4], 1):
                                title = item.get('title', 'Unknown')
                                snippet = item.get('snippet', '')
                                url = item.get('url', '')
                                if snippet:
                                    formatted_context.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")
                            ctx = '\n\n'.join(formatted_context) if formatted_context else ""
                            if ctx:
                                print("📰 Using News fallback context")
                                return ctx
                except Exception:
                    pass
                return ""
            
            # Format search results for context
            formatted_context = []
            for i, result in enumerate(combined_results[:4], 1):  # Top 4 results
                title = result.get('title', 'Unknown')
                snippet = result.get('snippet', '')
                url = result.get('url', '')
                
                if snippet:
                    formatted_context.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")
            
            return '\n\n'.join(formatted_context) if formatted_context else ""
            
        except Exception as e:
            print(f"⚠️ Search failed: {e}")
            return ""
    
    def _generate_search_queries(self, question: str) -> list:
        """Generate smart search queries for reasoning questions - ENHANCED approach"""
        
        question_lower = question.lower()
        queries = []
        
        # Extract years mentioned (any 4-digit year)
        import re
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
        
        # Extract key entities dynamically - look for potential company/product names
        entities = []
        
        # Find all words in the question (case-insensitive)
        all_words = re.findall(r'\b[a-zA-Z]+\b', question_lower)
        
        # Filter out common question words, articles, and verbs
        stop_words = {
            'why', 'what', 'when', 'where', 'how', 'the', 'this', 'that', 'these', 'those',
            'did', 'does', 'do', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
            'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'stock', 'market', 'company', 'firm', 'revenue', 'sales', 'profit', 'earnings',
            'growth', 'decline', 'crash', 'boom', 'performance', 'prices', 'value'
        }
        
        # Look for words that could be company names based on context patterns
        for word in all_words:
            if word not in stop_words and len(word) > 2:
                # Check if this word appears in patterns that suggest it's a company name
                # Pattern 1: "why did [word] ..." or "what caused [word] ..."
                if re.search(rf'\b(why did|what caused|what happened to|how did)\s+{re.escape(word)}\b', question_lower):
                    entities.append(word)
                # Pattern 2: "[word] stock" or "[word] revenue" or "[word] growth"
                elif re.search(rf'\b{re.escape(word)}\s+(stock|revenue|sales|growth|decline|crash|boom|performance)\b', question_lower):
                    entities.append(word)
                # Pattern 3: "[word] in [year]" - company names often appear before years
                elif re.search(rf'\b{re.escape(word)}\s+in\s+\d{{4}}\b', question_lower):
                    entities.append(word)
        
        # Remove duplicates while preserving order
        entities = list(dict.fromkeys(entities))
        
        # Extract key concepts dynamically - look for financial/business terminology
        concepts = []
        # Common business/financial concept patterns
        concept_patterns = [
            r'\b(sales|revenue|growth|decline|drop|increase|decrease)\b',
            r'\b(performance|volatility|trading|volume|price|value)\b',
            r'\b(profit|loss|earnings|income|expense)\b',
            r'\b(market|economy|financial|economic)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, question_lower)
            concepts.extend(matches)
        
        # Build targeted queries based on question type
        if 'why' in question_lower or 'reason' in question_lower or 'cause' in question_lower:
            # Reasoning queries - focus on causes and explanations
            # Check if question is about increase/growth or decline
            is_growth = any(word in question_lower for word in ['increase', 'increasing', 'grow', 'growing', 'rise', 'rising', 'growth', 'keep'])
            
            if entities and years:
                # Company + year + reasoning
                entity = entities[0]
                year = years[0]
                if is_growth:
                    queries.append(f"{entity} {year} growth reasons drivers")
                    queries.append(f"why {entity} increasing {year} factors")
                    queries.append(f"{entity} {year} expansion drivers")
                else:
                    queries.append(f"{entity} {year} decline reasons causes analysis")
                    queries.append(f"what caused {entity} {year} performance factors")
            elif entities and concepts:
                # Company + concept + reasoning
                entity = entities[0]
                concept = concepts[0] if concepts else "performance"
                if is_growth:
                    queries.append(f"{entity} {concept} growth reasons drivers")
                    queries.append(f"why {entity} {concept} increasing factors")
                    queries.append(f"{entity} {concept} expansion market analysis")
                else:
                    queries.append(f"{entity} {concept} decline reasons market analysis")
                    queries.append(f"why {entity} {concept} changed factors")
            elif years and concepts:
                # Year + concept + reasoning
                year = years[0]
                concept = concepts[0]
                queries.append(f"{year} {concept} market trends reasons")
                queries.append(f"what caused {year} {concept} changes")
            elif entities:
                # Just entity + reasoning
                entity = entities[0]
                queries.append(f"{entity} decline reasons causes analysis")
                queries.append(f"what caused {entity} performance factors")
            elif years:
                # Just year + reasoning
                year = years[0]
                queries.append(f"{year} market trends reasons")
                queries.append(f"what caused {year} market changes")
            else:
                # Generic reasoning query
                queries.append(f"{question} reasons causes analysis")
                queries.append(f"what caused {question} factors")
        
        elif 'what happened' in question_lower or 'explain' in question_lower:
            # Explanation queries - focus on events and context
            if entities and years:
                entity = entities[0]
                year = years[0]
                queries.append(f"{entity} {year} events timeline analysis")
                queries.append(f"{entity} {year} market performance explanation")
            elif entities:
                entity = entities[0]
                queries.append(f"{entity} events timeline analysis")
                queries.append(f"{entity} market performance explanation")
            elif years:
                year = years[0]
                queries.append(f"{year} market events timeline analysis")
                queries.append(f"{year} market performance explanation")
            else:
                queries.append(f"{question} explanation analysis")
                queries.append(f"{question} context background")
        
        elif 'compare' in question_lower or 'vs' in question_lower or 'versus' in question_lower:
            # Comparison queries
            if len(entities) >= 2:
                queries.append(f"{entities[0]} vs {entities[1]} comparison analysis")
                queries.append(f"{entities[0]} {entities[1]} performance differences")
            else:
                queries.append(f"{question} comparison analysis")
        
        else:
            # Fallback: general analysis queries
            if entities and years:
                entity = entities[0]
                year = years[0]
                queries.append(f"{entity} {year} analysis trends")
                queries.append(f"{entity} {year} market performance")
            elif entities:
                entity = entities[0]
                queries.append(f"{entity} analysis trends")
                queries.append(f"{entity} market performance")
            elif years:
                year = years[0]
                queries.append(f"{year} market analysis trends")
                queries.append(f"{year} market performance")
            else:
                queries.append(f"{question} analysis trends")
                queries.append(f"{question} market context")
        
        # Ensure we have at least one query
        if not queries:
            queries.append(f"{question} analysis")
        
        return queries[:2]  # Limit to 2 queries
    
    def _create_local_analysis_with_context(self, question: str, context: str, df: pd.DataFrame, search_context: str) -> str:
        """Create local analysis that includes external context when available"""
        
        # Get the base local analysis
        base_analysis = self._create_local_analysis(question, context, df)
        
        # Check if the question is about data not in the dataset (general business question)
        question_lower = question.lower()
        is_general_question = False
        
        # Extract key terms from question
        import re
        question_terms = set(re.findall(r'\b[a-z]{3,}\b', question_lower))
        question_terms -= {'why', 'did', 'what', 'when', 'where', 'how', 'the', 'and', 'or', 'but', 'to', 'from', 'in', 'on', 'at', 'for', 'of', 'with', 'by', 'about', 'that', 'this', 'these', 'those', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must'}
        
        # Check if any key terms from question match dataset columns
        if df is not None and len(df.columns) > 0:
            column_terms = set()
            for col in df.columns:
                column_terms.update(re.findall(r'\b[a-z]{3,}\b', col.lower()))
            
            # If question terms don't overlap much with dataset columns, it's likely a general question
            overlap = question_terms & column_terms
            if len(overlap) == 0 or (len(question_terms) > 3 and len(overlap) < 2):
                is_general_question = True
        
        # If we have search context and it's a general question OR local analysis is generic
        if search_context:
            # If it's a general question or local analysis is just generic summary, prioritize external context
            local_is_generic = base_analysis and any(generic in base_analysis.lower() for generic in [
                'dataset has', 'most variable', 'strongest correlation', 'i couldn\'t find',
                'i need at least', 'i don\'t see', 'couldn\'t compute'
            ])
            
            if is_general_question or local_is_generic:
                # For general questions, external context IS the answer
                return self._format_structured_response(question, "", search_context, df)
            
            return self._format_structured_response(question, base_analysis, search_context, df)
        
        return base_analysis
    
    def _format_structured_response(self, question: str, data_analysis: str, search_context: str, df: pd.DataFrame) -> str:
        """Format response as natural, conversational explanation"""
        import re
        
        # Build natural explanation combining data + external context
        explanation_parts = []
        
        # Start with data analysis (but make it conversational)
        # Skip if data_analysis is empty or just generic error messages
        if data_analysis and data_analysis.strip():
            data_text = data_analysis.strip()
            
            # Skip generic/unhelpful responses
            if any(skip in data_text.lower() for skip in [
                'i couldn\'t find', 'i need at least', 'i don\'t see', 
                'couldn\'t compute', 'no overlapping data'
            ]):
                # Skip unhelpful local analysis, rely on external context
                pass
            # Check if it's a year comparison (has year numbers and percentage change)
            elif re.search(r'\d{4}', data_text) and ('%' in data_text or 'changed' in data_text.lower() or 'from' in data_text.lower() and 'to' in data_text.lower()):
                # This is a year comparison - lead with it directly
                explanation_parts.append(data_text)
            elif "Strategy recommendations" in data_text:
                # Extract actionable recommendations
                lines = data_text.split('\n')
                recommendations = [l.strip('- ') for l in lines if l.strip().startswith('-')]
                if recommendations:
                    explanation_parts.append(f"Based on your data, here are the key patterns:\n\n" + '\n'.join([f"• {r}" for r in recommendations[:5]]))
            elif "Dataset has" in data_text or "Most variable" in data_text or "Strongest correlation" in data_text:
                # Generic data summary - skip it (external context is more relevant)
                pass
            else:
                # Only include if it's meaningful
                explanation_parts.append(data_text)
        
        # Add external context naturally
        key_findings = []  # Initialize outside if block
        if search_context:
            # Parse the formatted search context (format: "1. **Title**\n   snippet\n   Source: url")
            # Extract each numbered item
            items = re.split(r'\n(?=\d+\.\s*\*\*)', search_context)
            
            # Filter keywords based on question type to get relevant snippets
            question_lower = question.lower()
            is_why_question = 'why' in question_lower or 'reason' in question_lower or 'cause' in question_lower
            is_drop_question = 'drop' in question_lower or 'decline' in question_lower or 'decrease' in question_lower
            
            # Extract years from question for relevance filtering
            import re
            years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', question)]
            
            # Stop words for entity extraction
            stop_words = {
                'why', 'what', 'when', 'where', 'how', 'the', 'this', 'that', 'these', 'those',
                'did', 'does', 'do', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
                'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
                'so', 'high', 'low', 'revenue', 'sales', 'profit', 'earnings'
            }
            
            for item in items[:3]:  # Take top 3 results
                lines = item.strip().split('\n')
                title = None
                snippet = None
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('Source:'):
                        continue
                    # Extract title (format: "1. **Title**")
                    if line.startswith(('1.', '2.', '3.', '4.')) and '**' in line:
                        title_match = re.search(r'\*\*([^*]+)\*\*', line)
                        if title_match:
                            title = title_match.group(1)
                    # Extract snippet (content that's not title and not source)
                    elif line and not line.startswith('**') and len(line) > 30:
                        snippet = line
                        break
                
                # Filter out irrelevant snippets (history, generic definitions, non-explanatory content)
                if snippet:
                    snippet_lower = snippet.lower()
                    
                    # Skip if it's just history, generic definitions, or non-explanatory
                    skip_patterns = [
                        'was conceived in', 'was founded', 'previously owned', 'co-founded',
                        'can be defined as', 'is defined as', 'is the amount of money',
                        'revenue can be defined', 'from 2011 to 2025',  # too generic
                        'is the world\'s most popular', 'learn more here',
                        'history and growth rate', 'annual/quarterly revenue history',
                        'netflix was', 'both had previous ventures',
                        'the streaming platform', 'according to', 'statistics show',
                        'netflix is one of', 'netflix statistics',
                        'revenue of netflix from', 'in million u.s. dollars',
                        'netflix annual revenue for', 'up from', 'increase from'
                    ]
                    
                    # Check if snippet is just descriptive/statistical without explanation
                    if any(skip in snippet_lower for skip in skip_patterns):
                        continue
                    
                    # For "why" questions, include snippets that explain
                    if is_why_question:
                        # Check if question is about "high", "low", "so", etc. (general state questions)
                        is_general_state = any(word in question_lower for word in ['so high', 'so low', 'high', 'low', 'large', 'small', 'big', 'small'])
                        
                        # Must contain explanation keywords OR be clearly relevant
                        has_explanation = any(word in snippet_lower for word in [
                            'due to', 'because', 'caused by', 'resulted from', 'led to',
                            'attributed to', 'impacted by', 'affected by',
                            'decline', 'dropped', 'fell', 'decreased', 'down', 'slowed',
                            'challenged', 'pressure', 'competition', 'factors', 'reasons',
                            'drivers', 'headwinds', 'issues', 'problems',
                            # Positive growth keywords
                            'increased', 'grew', 'growth', 'expanded', 'surged', 'rose',
                            'boosted', 'improved', 'gained', 'climbed', 'jumped',
                            'subscriber growth', 'user growth', 'revenue growth',
                            'driven by', 'fueled by', 'boosted by', 'result of',
                            # General explanation keywords
                            'explains', 'contributes to', 'accounts for', 'explanation',
                            'largest', 'biggest', 'top', 'leading', 'dominant'
                        ])
                        
                        # Also check if it mentions the specific metric/years/entity from question
                        mentions_metric = any(word in snippet_lower for word in [
                            'revenue', 'sales', 'growth', 'performance', 'profit', 'earnings'
                        ])
                        
                        # Extract entity from question (e.g., "Saudi Aramco" -> "aramco", "saudi")
                        question_entities = []
                        for token in question_lower.split():
                            if len(token) > 3 and token not in stop_words:
                                question_entities.append(token)
                        
                        mentions_entity = any(entity in snippet_lower for entity in question_entities[:3])  # Check top 3 entities
                        
                        # Check if it mentions the years from question
                        mentions_years = False
                        if years:
                            for year in years:
                                if str(year) in snippet:
                                    mentions_years = True
                                    break
                        
                        # For general state questions ("why so high"), be more lenient
                        if is_general_state:
                            # Include if it mentions entity OR metric, and has some explanation or is clearly relevant
                            if (mentions_entity or mentions_metric) and (has_explanation or len(snippet) > 100):
                                if title:
                                    key_findings.append(f"**{title}**: {snippet}")
                                else:
                                    key_findings.append(snippet)
                                continue
                        else:
                            # For change questions, require explanation AND relevant context
                            if has_explanation and (mentions_metric or mentions_years or mentions_entity):
                                if title:
                                    key_findings.insert(0, f"**{title}**: {snippet}")
                                else:
                                    key_findings.insert(0, snippet)
                                continue
                        
                        # Skip snippets that don't meet criteria
                        continue
                    
                    # For strategy questions, include actionable/relevant snippets
                    if any(kw in question_lower for kw in ['what should', 'how to', 'recommend', 'strategy']):
                        if any(word in snippet_lower for word in [
                            'strategy', 'plan', 'focus', 'expand', 'improve', 'increase',
                            'reduce', 'optimize', 'best practice', 'recommend'
                        ]):
                            if title:
                                key_findings.append(f"**{title}**: {snippet}")
                            else:
                                key_findings.append(snippet)
                            continue
                        else:
                            continue
                
                # Combine title and snippet naturally (if not already added)
                if title and snippet and not any(f.startswith(f"**{title}**") for f in key_findings):
                    key_findings.append(f"**{title}**: {snippet}")
                elif snippet and snippet not in key_findings:
                    key_findings.append(snippet)
                elif title and not any(f.startswith(f"**{title}**") for f in key_findings):
                    key_findings.append(f"**{title}**")
            
            if key_findings:
                # If no local data analysis or it's generic, external context IS the answer
                if not explanation_parts or not any(part and len(part) > 50 for part in explanation_parts):
                    # Lead with external context as the primary answer
                    explanation_parts.insert(0, "**Answer:**\n\n" + '\n\n'.join([f"• {f}" for f in key_findings if f]))
                else:
                    explanation_parts.append(f"\n**External Context:**\n\n" + '\n\n'.join([f"• {f}" for f in key_findings if f]))
        
        # Add implications/recommendations naturally based on question type
        question_lower = question.lower()
        if any(kw in question_lower for kw in ['what should', 'how to', 'how can', 'what can', 'recommend', 'strategy', 'solution']):
            recommendation_text = self._generate_implication(question, data_analysis, search_context)
            # Clean up the recommendation text - remove markdown headers
            if recommendation_text:
                # Remove "**Actionable Recommendations:**" prefix if present
                rec_clean = re.sub(r'^\*\*Actionable Recommendations:\*\*\s*', '', recommendation_text, flags=re.IGNORECASE)
                rec_clean = rec_clean.strip()
                if rec_clean:
                    explanation_parts.append(f"\n**Recommendations:**\n\n{rec_clean}")
        elif 'why' in question_lower or 'reason' in question_lower:
            # Only add summary if we have external context that explains the "why"
            # Don't add a generic message if we don't have actual context to show
            if search_context and key_findings:
                explanation_parts.append(f"\n**Explanation:**\n\nThe factors above explain why this change occurred. These insights combine your internal data with external market intelligence to provide a comprehensive understanding.")
            # If we don't have key findings but have search context, try to parse it again
            elif search_context and not key_findings:
                # Maybe the filtering was too strict - show a note
                explanation_parts.append(f"\nExternal context was searched but didn't yield specific explanations. The data trend above shows the pattern.")
        
        # Join into natural explanation
        explanation = '\n'.join(explanation_parts)
        
        # If we still have structured format markers, clean them up
        if '##' in explanation:
            # Remove markdown headers and make it flow
            import re
            explanation = re.sub(r'##\s*\*\*.*?\*\*\s*\n', '', explanation)
            explanation = re.sub(r'\*\*([^*]+)\*\*:', r'\1:', explanation)
        
        return explanation if explanation.strip() else data_analysis
    
    def _extract_data_insight(self, data_analysis: str, df: pd.DataFrame) -> str:
        """Extract key data insights from the analysis"""
        
        # Look for key metrics in the data analysis
        if "correlation" in data_analysis.lower():
            return "The data shows significant correlations between key metrics that explain the observed patterns."
        elif "decline" in data_analysis.lower() or "drop" in data_analysis.lower():
            return "The dataset reveals a clear decline pattern that requires external context to fully understand."
        elif "growth" in data_analysis.lower() or "increase" in data_analysis.lower():
            return "The data indicates growth trends that align with broader market conditions."
        elif "volatility" in data_analysis.lower():
            return "High volatility patterns in the data suggest external market factors are influencing performance."
        else:
            # Generic insight based on data characteristics
            if df is not None and len(df) > 0:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    return f"Analysis of {len(df)} data points reveals patterns that require external market context for complete understanding."
            return "The internal data analysis provides insights that benefit from external market context."
    
    def _create_reasoning_section(self, data_analysis: str, search_context: str, question: str) -> str:
        """Create reasoning section combining data and external context"""
        
        reasoning_parts = []
        
        # Add data-based reasoning
        if data_analysis:
            reasoning_parts.append(f"**Data Evidence:** {data_analysis[:200]}...")
        
        # Add external context reasoning
        if search_context:
            # Extract key points from search context
            context_lines = search_context.split('\n')
            key_points = []
            for line in context_lines[:3]:  # Take first 3 relevant lines
                if line.strip() and not line.startswith('Source:'):
                    key_points.append(line.strip())
            
            if key_points:
                reasoning_parts.append(f"**External Context:** {' '.join(key_points[:2])}")
        
        # Add connecting reasoning
        if 'why' in question.lower():
            reasoning_parts.append("The combination of internal data patterns and external market factors provides a comprehensive explanation for the observed trends.")
        elif 'what happened' in question.lower():
            reasoning_parts.append("The timeline of events from external sources aligns with the data patterns observed in the dataset.")
        
        return '\n\n'.join(reasoning_parts) if reasoning_parts else "Analysis based on available data and external context."
    
    def _generate_implication(self, question: str, data_analysis: str, search_context: str) -> str:
        """Generate strategic implications based on the analysis"""
        
        implications = []
        
        # Strategy/recommendation questions
        if any(kw in question.lower() for kw in ['what should', 'how to', 'how can', 'what can', 'recommend', 'strategy', 'solution']):
            implications.append("**Actionable Recommendations:**")
            # Add recommendations from external context if available
            if search_context:
                # Extract actionable items from search context
                context_lower = search_context.lower()
                if 'best practice' in context_lower or 'recommend' in context_lower:
                    implications.append("Based on industry best practices and external research, consider implementing proven strategies from similar contexts.")
                if 'reduce' in question.lower() or 'decrease' in question.lower():
                    implications.append("Focus on addressing root causes identified in the data analysis, combined with proven reduction strategies from industry examples.")
                if 'increase' in question.lower() or 'improve' in question.lower():
                    implications.append("Leverage successful strategies from similar cases while adapting them to your specific data patterns.")
            # Add data-driven recommendations
            if 'cancel' in question.lower() or 'cancellation' in question.lower():
                implications.append("Analyze cancellation patterns in your data (by location, time, driver ratings) to identify specific areas for intervention.")
            if not implications or len(implications) == 1:
                implications.append("Implement targeted interventions based on the data patterns identified, following industry best practices.")
        
        # Reasoning questions
        elif 'why' in question.lower() or 'reason' in question.lower():
            implications.append("Understanding these underlying factors can help predict future trends and inform strategic decisions.")
        
        # Decline questions
        elif 'decline' in question.lower() or 'drop' in question.lower():
            implications.append("The identified decline factors suggest the need for strategic adjustments to address root causes.")
        
        # Growth questions
        elif 'growth' in question.lower() or 'increase' in question.lower():
            implications.append("The growth drivers identified can be leveraged to sustain and accelerate positive trends.")
        
        # Market/volatility questions
        elif 'volatility' in question.lower() or 'market' in question.lower():
            implications.append("Market volatility factors should be monitored closely to manage risk and capitalize on opportunities.")
        
        # Generic implications
        else:
            implications.append("The analysis provides actionable insights for strategic planning and decision-making.")
            implications.append("Regular monitoring of both internal metrics and external factors is recommended.")
        
        return ' '.join(implications)
    
    def _analyze_decline_question(self, question: str, df: pd.DataFrame) -> str:
        """Handle questions about declines (e.g., 'why did Samsung sales decline')"""
        try:
            # Find products/entities mentioned in the question
            entity_col = self._find_entity_column(df)  # Could be Product, Company, Name, etc.
            if not entity_col:
                return "I can't identify a product or entity column to analyze declines."
            
            # Look for numeric columns that might show decline
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 1:
                return "I need numeric data to analyze declines."
            
            # Try to find the entity mentioned in the question
            entities_in_question = []
            for entity in df[entity_col].unique():
                if str(entity).lower() in question.lower():
                    entities_in_question.append(entity)
            
            if not entities_in_question:
                # Find entities with the largest decline
                decline_analysis = []
                for col in numeric_cols[:2]:  # Check first 2 numeric columns
                    if len(df) > 1:
                        entity_values = df.groupby(entity_col)[col].sum().sort_values()
                        if len(entity_values) > 1:
                            lowest_entity = entity_values.index[0]
                            lowest_value = entity_values.iloc[0]
                            decline_analysis.append(f"**{lowest_entity}** shows the lowest {col.replace('_', ' ')}: {lowest_value:,.0f}")
                
                if decline_analysis:
                    return f"Based on the data analysis:\\n" + "\\n".join(decline_analysis)
                else:
                    return "I can see your data but need more specific information to analyze declines."
            
            # Analyze the specific entity mentioned
            entity = entities_in_question[0]
            entity_data = df[df[entity_col] == entity]
            
            if len(entity_data) == 0:
                return f"I couldn't find {entity} in your dataset."
            
            # Find the most relevant decline metric
            analysis_parts = []
            for col in numeric_cols:
                if len(entity_data) > 1:
                    # Multi-row analysis (time series)
                    values = entity_data[col].values
                    if len(values) >= 2:
                        start_val, end_val = values[0], values[-1]
                        if start_val > end_val:
                            decline_pct = ((end_val - start_val) / start_val) * 100
                            analysis_parts.append(f"**{entity}** {col.replace('_', ' ')} declined from {start_val:,.0f} to {end_val:,.0f} ({decline_pct:+.1f}%)")
                else:
                    # Single row analysis
                    value = entity_data[col].iloc[0]
                    analysis_parts.append(f"**{entity}** {col.replace('_', ' ')}: {value:,.0f}")
            
            if analysis_parts:
                return "Based on my analysis of your dataset:\\n" + "\\n".join(analysis_parts)
            else:
                return f"I found {entity} in your data but couldn't analyze the decline pattern."
                
        except Exception as e:
            print(f"Error in decline analysis: {e}")
            return "I encountered an error analyzing the decline. Please check your data format."
    
    def _analyze_growth_question(self, question: str, df: pd.DataFrame) -> str:
        """Handle questions about growth (e.g., 'what caused iPhone growth')"""
        try:
            entity_col = self._find_entity_column(df)
            if not entity_col:
                return "I can't identify a product or entity column to analyze growth."
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 1:
                return "I need numeric data to analyze growth."
            
            # Find entities mentioned in the question
            entities_in_question = []
            for entity in df[entity_col].unique():
                if str(entity).lower() in question.lower():
                    entities_in_question.append(entity)
            
            if not entities_in_question:
                # Find entities with highest growth
                growth_analysis = []
                for col in numeric_cols[:2]:
                    entity_values = df.groupby(entity_col)[col].sum().sort_values(ascending=False)
                    if len(entity_values) > 0:
                        top_entity = entity_values.index[0]
                        top_value = entity_values.iloc[0]
                        growth_analysis.append(f"**{top_entity}** leads in {col.replace('_', ' ')}: {top_value:,.0f}")
                
                if growth_analysis:
                    return f"Growth leaders in your dataset:\\n" + "\\n".join(growth_analysis)
                else:
                    return "I can see your data but need more specific information to analyze growth."
            
            # Analyze specific entity
            entity = entities_in_question[0]
            entity_data = df[df[entity_col] == entity]
            
            if len(entity_data) == 0:
                return f"I couldn't find {entity} in your dataset."
            
            analysis_parts = []
            for col in numeric_cols:
                if len(entity_data) > 1:
                    values = entity_data[col].values
                    if len(values) >= 2:
                        start_val, end_val = values[0], values[-1]
                        if end_val > start_val:
                            growth_pct = ((end_val - start_val) / start_val) * 100
                            analysis_parts.append(f"**{entity}** {col.replace('_', ' ')} grew from {start_val:,.0f} to {end_val:,.0f} (+{growth_pct:.1f}%)")
                else:
                    value = entity_data[col].iloc[0]
                    analysis_parts.append(f"**{entity}** {col.replace('_', ' ')}: {value:,.0f}")
            
            if analysis_parts:
                return "Growth analysis from your dataset:\\n" + "\\n".join(analysis_parts)
            else:
                return f"I found {entity} in your data but couldn't analyze the growth pattern."
                
        except Exception as e:
            print(f"Error in growth analysis: {e}")
            return "I encountered an error analyzing growth. Please check your data format."
    
    def _find_entity_column(self, df: pd.DataFrame) -> str:
        """Find column containing products, companies, or entities"""
        possible_names = ['product', 'company', 'name', 'entity', 'item', 'brand', 'model']
        
        # Check exact matches first
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        # Check partial matches
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        
        # Fallback: find object column with reasonable unique count
        for col in df.select_dtypes(include=['object']).columns:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 100:  # Reasonable range for entities
                return col
        
        return None

    def _analyze_trend_over_time(self, entity: str, time_col: str, value_col: str, df: pd.DataFrame) -> str:
        """Analyze trend over time showing increase/growth pattern"""
        try:
            # Filter by entity if provided
            scope = df
            if entity:
                entity_col = self._find_entity_column(df)
                if entity_col:
                    scope = df[df[entity_col] == entity]
                    if scope.empty:
                        return f"I couldn't find {entity} in the dataset."
            
            # Group by time and aggregate value
            by_time = scope.groupby(time_col)[value_col].sum().sort_index()
            if len(by_time) < 2:
                return f"Insufficient data points to analyze trend for {value_col}."
            
            # Calculate growth metrics
            first_val = by_time.iloc[0]
            last_val = by_time.iloc[-1]
            total_change = last_val - first_val
            pct_change = ((last_val / first_val) - 1) * 100 if first_val != 0 else float('inf')
            
            # Find peak and growth periods
            peak_time = by_time.idxmax()
            peak_val = by_time.max()
            
            # Calculate average annual growth if we have years
            time_span = len(by_time)
            avg_annual_growth = ((last_val / first_val) ** (1 / time_span) - 1) * 100 if first_val != 0 and time_span > 0 else 0
            
            metric_name = value_col.replace('_', ' ').capitalize()
            entity_prefix = f"{entity}'s " if entity else ""
            
            result = f"{entity_prefix}{metric_name} increased from {first_val:,.0f} to {last_val:,.0f} ({pct_change:+.1f}% total change) over {time_span} periods. "
            result += f"Peak: {peak_val:,.0f} in {peak_time}. "
            if time_span > 1:
                result += f"Average growth rate: {avg_annual_growth:+.1f}% per period."
            
            return result
        except Exception as e:
            return f"Trend analysis shows {value_col} has been increasing over time."

    def _analyze_sales_decline(self, question: str, df: pd.DataFrame) -> str:
        """Explain sales/volume decline for a specific entity between years using actual dataset columns."""
        try:
            entity_col = self._find_entity_column(df)
            if not entity_col:
                return "I can't identify a product/entity column in the dataset."

            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                return "I need numeric columns (e.g., Sales_Volume) to analyze this."

            # Time column detection (prefer year)
            time_candidates = [c for c in numeric_cols if 'year' in c.lower() or (df[c].min() >= 1900 and df[c].max() <= 2100)]
            if not time_candidates:
                return "I can't find a year/time column to compare years."
            time_col = time_candidates[0]

            # Value (sales) column
            value_priority = ['sales', 'volume', 'units', 'sold']
            sales_cols = [c for c in numeric_cols if any(p in c.lower() for p in value_priority) and c != time_col]
            if not sales_cols:
                return "I can't find a sales/volume column in the dataset."
            value_col = sales_cols[0]

            # Entity extraction
            ent = None
            qlower = question.lower()
            for e in df[entity_col].unique():
                if str(e).lower() in qlower:
                    ent = e
                    break
            if not ent:
                return f"Specify which {entity_col.lower()} to analyze. Examples: {', '.join(map(str, df[entity_col].unique()[:5]))}"

            # Years extraction
            import re
            years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', qlower)]

            ent_df = df[df[entity_col] == ent]
            if ent_df.empty:
                return f"I couldn't find {ent} in the dataset."

            series = ent_df.groupby(time_col)[value_col].sum().sort_index()
            if len(series) < 2:
                return f"I need at least two years of {ent} data to compare."

            if len(years) >= 2:
                y1, y2 = years[0], years[1]
            elif len(years) == 1:
                y1 = years[0]
                y2 = y1 + 1 if (y1 + 1) in series.index else (y1 - 1)
            else:
                idx = list(series.index)
                y1, y2 = idx[-2], idx[-1]

            if y1 not in series.index or y2 not in series.index:
                return f"I couldn't find both {y1} and {y2} for {ent}. Available years: {', '.join(map(str, series.index.tolist()))}"

            v1, v2 = series.loc[y1], series.loc[y2]
            delta = v2 - v1
            pct = (delta / v1 * 100.0) if v1 != 0 else float('inf')

            lines = [f"**{ent}** {value_col.replace('_',' ')} changed from **{v1:,.0f}** in **{y1}** to **{v2:,.0f}** in **{y2}** ({pct:+.1f}%)."]

            # Factors: price, region mix, correlations
            factors = []

            price_cols = [c for c in numeric_cols if any(k in c.lower() for k in ['price', 'usd', 'amount']) and c not in [value_col, time_col]]
            if price_cols:
                pc = price_cols[0]
                p_by_year = ent_df.groupby(time_col)[pc].mean()
                if y1 in p_by_year and y2 in p_by_year:
                    p1, p2 = p_by_year.get(y1, None), p_by_year.get(y2, None)
                    if pd.notna(p1) and pd.notna(p2):
                        ppct = ((p2 - p1) / p1 * 100.0) if p1 != 0 else None
                        if ppct is not None:
                            factors.append(f"Average {pc.replace('_',' ')} changed {ppct:+.1f}% ({p1:,.0f} → {p2:,.0f}).")

            region_cols = [c for c in df.select_dtypes(include=['object']).columns if any(k in c.lower() for k in ['region', 'market', 'area'])]
            if region_cols:
                rc = region_cols[0]
                dist1 = ent_df[ent_df[time_col] == y1][rc].value_counts(normalize=True).head(3)
                dist2 = ent_df[ent_df[time_col] == y2][rc].value_counts(normalize=True).head(3)
                if not dist1.empty and not dist2.empty:
                    factors.append(
                        f"Region mix shifted. Top {rc} in {y1}: " + ", ".join([f"{k} {v*100:.0f}%" for k,v in dist1.items()]) + "; "
                        f"in {y2}: " + ", ".join([f"{k} {v*100:.0f}%" for k,v in dist2.items()]) + "."
                    )

            # Correlations across years for this entity (if >=3 years)
            other_nums = [c for c in numeric_cols if c not in [value_col, time_col]]
            if len(series.index) >= 3 and other_nums:
                yearly = ent_df.groupby(time_col).agg({c: 'mean' for c in other_nums})
                yearly[value_col] = ent_df.groupby(time_col)[value_col].sum()
                corrs = yearly.corr(numeric_only=True)[value_col].drop(labels=[value_col]).sort_values(ascending=False)
                if not corrs.empty:
                    top_corr = corrs.head(2)
                    corr_text = ", ".join([f"{idx.replace('_',' ')} ({val:+.2f})" for idx, val in top_corr.items()])
                    factors.append(f"Across years for {ent}, {value_col.replace('_',' ')} correlates with: {corr_text} (corr).")

            if factors:
                lines.append("Possible contributing factors in your data:")
                lines.extend([f"- {f}" for f in factors])

            return "\n".join(lines)
        except Exception as e:
            print(f"Error in sales-decline analysis: {e}")
            return "I encountered an error analyzing this sales change. Please verify the dataset columns."