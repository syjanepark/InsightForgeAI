import pandas as pd
import io, uuid
from typing import Dict, Any
from core.schemas import AnalyzeResponse, KPI, Chart, Insight, Evidence

class DataAgent:
    def analyze(self, file_bytes: bytes) -> AnalyzeResponse:
        df = pd.read_csv(io.BytesIO(file_bytes))

        # --- basic profiling ---
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        kpis = []
        for col in numeric_cols[:3]:
            val = df[col].mean()
            kpis.append(KPI(name=col, value=round(val, 2)))

        # --- intelligent chart based on data structure ---
        chart_spec = self._create_intelligent_chart(df, numeric_cols)
        chart = Chart(
            type=chart_spec["type"],
            spec=chart_spec["spec"]
        )

        # --- keywords from column names ---
        keywords = [c for c in df.columns if c not in numeric_cols][:5]

        # --- mock insight (placeholder) ---
        insights = [
            Insight(
                title="Preliminary data summary",
                why="Shows average metrics for key columns.",
                recommendations=["Upload full dataset", "Check data consistency"],
                evidence=[Evidence(source="data_agent", title="local analysis", url="N/A")]
            )
        ]

        return AnalyzeResponse(
            run_id=str(uuid.uuid4()),
            kpis=kpis,
            charts=[chart],
            keywords=keywords,
            insights=insights
        )
    
    def _create_intelligent_chart(self, df: pd.DataFrame, numeric_cols: list) -> dict:
        """Create meaningful charts based on data structure analysis"""
        
        # Find columns that appear to be related (similar patterns/names)
        related_columns = self._find_related_columns(df, numeric_cols)
        
        if len(related_columns) > 1:
            # Create comparison chart for related metrics
            totals = {}
            for col in related_columns:
                total = df[col].sum()
                # Clean up column names for display (generic)
                clean_name = col.replace('_', ' ').title()
                totals[clean_name] = round(total, 2)
            
            # Determine chart title based on data patterns
            chart_title = self._generate_chart_title(related_columns)
            
            return {
                "type": "bar",
                "spec": {
                    "labels": list(totals.keys()),
                    "datasets": [{
                        "label": f"Total by Category",
                        "data": list(totals.values()),
                        "backgroundColor": ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444"][:len(totals)]
                    }],
                    "title": chart_title
                }
            }
        
        # Fallback: Show distribution of top numeric column
        elif numeric_cols:
            primary_metric = numeric_cols[0]
            
            # Find a good categorical column for grouping
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
            grouping_col = None
            
            for col in categorical_cols:
                unique_count = df[col].nunique()
                if 3 <= unique_count <= 10:  # Good for visualization
                    grouping_col = col
                    break
            
            if grouping_col:
                # Group by category and show averages
                grouped = df.groupby(grouping_col)[primary_metric].mean().round(2)
                top_groups = grouped.nlargest(5)  # Top 5 only
                
                return {
                    "type": "bar", 
                    "spec": {
                        "labels": list(top_groups.index),
                        "datasets": [{
                            "label": f"Average {primary_metric}",
                            "data": list(top_groups.values),
                            "backgroundColor": "#3B82F6"
                        }],
                        "title": f"Top 5 {grouping_col} by {primary_metric}"
                    }
                }
        
        # Ultimate fallback: Simple metrics summary
        return {
            "type": "bar",
            "spec": {
                "labels": [col.replace('_', ' ').title() for col in numeric_cols[:4]],
                "datasets": [{
                    "label": "Average Values",
                    "data": [round(df[col].mean(), 2) for col in numeric_cols[:4]],
                    "backgroundColor": "#3B82F6"
                }],
                "title": "Key Metrics Overview"
            }
        }
    
    def _find_related_columns(self, df: pd.DataFrame, numeric_cols: list) -> list:
        """Find columns that appear to be related based on naming patterns"""
        
        # Look for columns with similar suffixes or common patterns
        pattern_groups = {}
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            # Extract potential pattern (suffix after underscore)
            if '_' in col:
                suffix = col.split('_')[-1]
                prefix = '_'.join(col.split('_')[:-1])
                
                if prefix not in pattern_groups:
                    pattern_groups[prefix] = []
                pattern_groups[prefix].append(col)
        
        # Find the largest group of related columns
        largest_group = []
        for group in pattern_groups.values():
            if len(group) > len(largest_group):
                largest_group = group
        
        # If we found a good group, return it
        if len(largest_group) >= 2:
            return largest_group
        
        # Fallback: look for columns with similar value ranges (might be related)
        similar_range_cols = []
        for col in numeric_cols[:5]:  # Check top 5 numeric columns
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            # Look for columns with similar statistical properties
            for other_col in numeric_cols:
                if col != other_col:
                    other_mean = df[other_col].mean()
                    other_std = df[other_col].std()
                    
                    # Similar if means are within 50% and both have data
                    if (col_mean > 0 and other_mean > 0 and 
                        abs(col_mean - other_mean) / max(col_mean, other_mean) < 0.5):
                        if col not in similar_range_cols:
                            similar_range_cols.append(col)
                        if other_col not in similar_range_cols:
                            similar_range_cols.append(other_col)
        
        return similar_range_cols[:4] if len(similar_range_cols) >= 2 else numeric_cols[:4]
    
    def _generate_chart_title(self, columns: list) -> str:
        """Generate appropriate chart title based on column analysis"""
        
        if not columns:
            return "Data Overview"
        
        # Analyze column patterns to infer chart type
        col_sample = columns[0].lower()
        
        # Common business patterns (generic detection)
        if any(term in col_sample for term in ['sales', 'revenue', 'income']):
            return "Performance Comparison"
        elif any(term in col_sample for term in ['cost', 'expense', 'spend']):
            return "Cost Analysis"
        elif any(term in col_sample for term in ['count', 'volume', 'quantity']):
            return "Volume Metrics"
        elif any(term in col_sample for term in ['rate', 'percent', '%']):
            return "Rate Comparison"
        elif '_' in col_sample:
            # Try to infer from prefix
            prefix = col_sample.split('_')[0]
            return f"{prefix.title()} Breakdown"
        else:
            return "Key Metrics Comparison"