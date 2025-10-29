# agents/smart_chat_agent.py
from services.you_smart import call_smart
from services.you_search import search_web
import pandas as pd
import asyncio

class SmartChatAgent:
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
            else:
                print(f"❌ No search results returned")
        
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
        
        try:
            # Try You.com API with enhanced context
            out = call_smart(prompt, use_tools=True)  # Enable tools for better analysis
            if out and "Request failed" not in out:
                return {
                    "answer": out,
                    "visualizations": None,  # Simplified for now
                    "suggested_actions": None,  # Simplified for now
                    "citations": self._extract_citations(out)
                }
        except Exception as e:
            print(f"🔄 You.com API failed, using local analysis: {e}")
        
        # Intelligent local fallback with search context
        local_answer = self._create_local_analysis_with_context(question, context, df, search_context)
        return {
            "answer": local_answer,
            "visualizations": None,  # Simplified for now
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
        years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', q)]

        # time/value columns
        time_col = self._find_time_column(df)
        value_col = self._pick_value_column(df)

        # mode inference (no phrase hardcoding beyond "why" for causal)
        mode = 'non_time_trend'
        if entity and time_col and value_col:
            if len(years) >= 1:
                mode = 'compare_years'
            elif 'why' in q or 'reason' in q or 'cause' in q:
                mode = 'factor_explanation'
            else:
                mode = 'over_time_summary'
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

        if mode == 'compare_years' and entity and time_col and value_col:
            series = df[df[entity_col] == entity].groupby(time_col)[value_col].sum().sort_index()
            if len(series) < 2:
                return f"I need at least two years of {entity} data to compare."
            y1 = years[0]
            y2 = years[1] if len(years) > 1 else (y1 + 1 if (y1 + 1) in series.index else (y1 - 1))
            if y1 not in series.index or y2 not in series.index:
                return f"I couldn't find both {y1} and {y2} for {entity}. Available years: {', '.join(map(str, series.index.tolist()))}"
            v1, v2 = series.loc[y1], series.loc[y2]
            delta = v2 - v1
            pct = (delta / v1 * 100.0) if v1 != 0 else float('inf')
            return f"**{entity}** {value_col.replace('_',' ')} changed from **{v1:,.0f}** in **{y1}** to **{v2:,.0f}** in **{y2}** ({pct:+.1f}%)."

        if mode == 'over_time_summary' and entity and time_col and value_col:
            sub = df[df[entity_col] == entity]
            if sub.empty:
                return f"I couldn't find {entity} in the dataset."
            by_year = sub.groupby(time_col)[value_col].sum().sort_index()
            peak_year = by_year.idxmax()
            peak_val = by_year.max()
            return f"**{entity}** {value_col.replace('_',' ')} peaks at **{peak_year}** with **{peak_val:,.0f}**. Range: {by_year.index.min()}–{by_year.index.max()} ({len(by_year)} periods)."

        if mode == 'factor_explanation' and entity and time_col and value_col:
            return self._analyze_sales_decline("why " + entity, df)

        if mode == 'strategy':
            return self._handle_strategy(df, entity_col, entity, time_col, value_col)

        if mode == 'non_time_trend':
            return self._dataset_trends_and_correlations(df)

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
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        priority = ['sales', 'volume', 'units', 'sold', 'revenue', 'amount', 'count']
        for c in numeric_cols:
            cl = c.lower()
            if any(p in cl for p in priority):
                return c
        # fallback: most variable numeric col
        if numeric_cols:
            variances = [(c, df[c].std()) for c in numeric_cols]
            variances.sort(key=lambda x: (0 if pd.isna(x[1]) else -x[1]))
            return variances[0][0] if variances else None
        return None

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
        
        # Find potential entity columns (categorical with reasonable unique count)
        entity_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 50]
        
        # Find potential value columns (numeric)
        value_cols = [col for col in numeric_cols if col.lower() not in ['year', 'id', 'index']]
        
        # Find time columns
        time_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['year', 'date', 'time'])]
        
        # Generate visualizations based on available data
        if entity_cols and value_cols:
            # Top entities by value
            entity_col = entity_cols[0]
            value_col = value_cols[0]
            
            entity_values = df.groupby(entity_col)[value_col].sum().nlargest(5)
            if not entity_values.empty:
                visualizations.append({
                    "type": "bar",
                    "title": f"Top 5 {entity_col} by {value_col}",
                    "data": {
                        "labels": entity_values.index.tolist(),
                        "datasets": [{
                            "label": value_col,
                            "data": entity_values.values.tolist(),
                            "backgroundColor": "#3B82F6"
                        }]
                    }
                })
        
        if time_cols and value_cols:
            # Time series
            time_col = time_cols[0]
            value_col = value_cols[0]
            
            time_values = df.groupby(time_col)[value_col].sum().sort_index()
            if len(time_values) > 1:
                visualizations.append({
                    "type": "line",
                    "title": f"{value_col} Over Time",
                    "data": {
                        "labels": time_values.index.astype(str).tolist(),
                        "datasets": [{
                            "label": value_col,
                            "data": time_values.values.tolist(),
                            "borderColor": "#10B981",
                            "backgroundColor": "rgba(16, 185, 129, 0.1)",
                            "fill": True
                        }]
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
        """Detect if a question needs external context search"""
        
        question_lower = question.lower()
        
        # Contextual question indicators
        contextual_keywords = [
            'why', 'because', 'reason', 'cause', 'due to', 'behind',
            'compared to', 'vs', 'versus', 'difference between',
            'what happened', 'what caused', 'how come', 'explain',
            'background', 'context', 'history', 'trend', 'market',
            'economic', 'political', 'global', 'worldwide', 'international'
        ]
        
        # Time-based comparisons that need context
        time_comparisons = [
            '2023', '2024', '2022', '2025', 'last year', 'this year',
            'recent', 'lately', 'currently', 'now', 'today'
        ]
        
        # Check for contextual keywords
        has_contextual = any(keyword in question_lower for keyword in contextual_keywords)
        
        # Check for time-based questions
        has_temporal = any(time_word in question_lower for time_word in time_comparisons)
        
        # Questions asking for explanations about patterns
        has_explanation = any(word in question_lower for word in ['why', 'explain', 'reason', 'cause'])
        
        return has_contextual or (has_temporal and has_explanation)
    
    async def _get_contextual_search(self, question: str) -> str:
        """Get contextual information from web search"""
        
        try:
            # Generate smart search queries based on the question
            search_queries = self._generate_search_queries(question)
            
            # Perform searches
            loop = asyncio.get_event_loop()
            search_tasks = [
                loop.run_in_executor(None, search_web, query, 2) 
                for query in search_queries
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and format results
            combined_results = []
            for result_group in results:
                if isinstance(result_group, list):
                    combined_results.extend(result_group)
            
            if not combined_results:
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
        """Generate smart search queries based on the user's question - GENERIC approach"""
        
        # Extract key terms from question without hardcoding domains
        question_words = question.lower().split()
        
        # Remove common words
        stop_words = {'what', 'why', 'how', 'when', 'where', 'is', 'are', 'was', 'were', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        meaningful_words = [word for word in question_words if word not in stop_words and len(word) > 2]
        
        # Extract years mentioned
        years = [word for word in question_words if word.isdigit() and 1900 <= int(word) <= 2100]
        
        queries = []
        
        if len(meaningful_words) >= 2:
            # Create search queries using the actual terms from the question
            main_terms = ' '.join(meaningful_words[:3])  # Use first 3 meaningful words
            
            if years:
                # Time-based query with extracted years
                year_terms = ' '.join(years)
                queries.append(f"{main_terms} {year_terms} trends analysis")
                queries.append(f"{main_terms} {year_terms} reasons factors")
            else:
                # General contextual query
                queries.append(f"{main_terms} trends analysis")
                queries.append(f"{main_terms} factors explanation")
        else:
            # Fallback: use the whole question
            queries.append(f"{question} trends analysis")
        
        return queries[:2]  # Limit to 2 queries
    
    def _create_local_analysis_with_context(self, question: str, context: str, df: pd.DataFrame, search_context: str) -> str:
        """Create local analysis that includes external context when available"""
        
        # Get the base local analysis
        base_analysis = self._create_local_analysis(question, context, df)
        
        # If we have search context, enhance the response
        if search_context:
            # Limit search context size to prevent response issues
            limited_context = search_context[:800] + "..." if len(search_context) > 800 else search_context
            enhanced_response = f"{base_analysis}\n\n**External Context:**\n{limited_context}"
            return enhanced_response
        
        return base_analysis
    
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
            years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', qlower)]

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