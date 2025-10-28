import pandas as pd
import numpy as np
from typing import Dict, Any
from services.you_smart import call_smart

class SmartChatAgent:
    """Enhanced Smart Chat Agent using You.com Smart API with analytical reasoning"""
    
    def __init__(self):
        pass

    async def process_query(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Process user question with context-rich prompt and reasoning instructions"""
        
        # 1️⃣ Build analytical dataset summary
        context = self._create_rich_context(df)
        
        # 2️⃣ Build reasoning-oriented prompt
        prompt = f"""
You are InsightForge AI — an analytical business and data intelligence assistant.
Use the dataset context below to reason through the user's question.

If the question is about:
- **Patterns or trends** → discuss increases/decreases, seasonal changes, or anomalies.
- **Comparisons** → compute logical differences and possible reasons.
- **Causes** → infer plausible drivers based on numeric patterns and dataset context.
- **Predictions or strategy** → base reasoning on data direction and global context.

DATA CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Analyze the numeric and categorical trends logically.
2. Provide concise, factual reasoning (2–3 short paragraphs).
3. Mention supporting variables or evidence.
4. If relevant, end with 1–2 actionable recommendations.
5. If uncertain, say what further analysis you'd perform.
"""

        response = call_smart(prompt)

        # Intelligent fallback with local analysis if Smart API fails
        if not response or any(err in response for err in ["Error", "Failed", "decode", "Request failed"]):
            print("⚠️ Smart API failed, trying local analysis...")
            try:
                # Use the rule-based query agent for local analysis
                from agents.query_agent import QueryAgent
                query_agent = QueryAgent()
                local_result = query_agent.analyze_query(question, df)
                
                if local_result and "answer" in local_result:
                    response = local_result["answer"] + "\n\n*Note: Analyzed locally due to API connectivity issues.*"
                else:
                    response = "⚡ I'm having trouble with both cloud and local analysis. Could you try rephrasing your question with more specific column names?"
            except Exception as e:
                print(f"⚠️ Local analysis also failed: {e}")
                response = "⚡ Analysis services are temporarily slow. Could you be more specific about which columns you want me to analyze?"

        return {
            "answer": response.strip(),
            "visualizations": None,
            "suggested_actions": [
                {"label": "📊 Visualize trend", "action": "chart"},
                {"label": "💡 Get recommendations", "action": "advice"},
                {"label": "🔎 Compare categories", "action": "compare"}
            ],
            "citations": []
        }

    # --- Context builders ---
    def _create_rich_context(self, df: pd.DataFrame) -> str:
        """Create a richer dataset summary that includes key metrics and trends"""
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        summary_parts = [f"Rows: {n_rows:,}, Columns: {n_cols}"]
        if numeric_cols:
            desc = df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']].round(2)
            top_metrics = desc.head(3).to_dict('index')
            summary_parts.append(f"Numeric overview (mean/std/min/max of top 3 metrics): {top_metrics}")
        
        if categorical_cols:
            cat_summary = {}
            for c in categorical_cols[:3]:
                top_val = df[c].value_counts().idxmax()
                cat_summary[c] = str(top_val)
            summary_parts.append(f"Most common categorical values: {cat_summary}")

        # Detect time-like columns and trends
        time_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'year', 'month', 'time'])]
        if time_cols:
            summary_parts.append(f"Time-related columns detected: {', '.join(time_cols)}")

        # Include sample data for structure understanding
        sample = df.head(2).to_dict('records')
        summary_parts.append(f"Sample records: {sample}")

        return "\n".join(summary_parts)