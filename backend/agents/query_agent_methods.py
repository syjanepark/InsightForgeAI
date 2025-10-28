# This file contains all the handler methods for the QueryAgent class
# Split for better organization

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import re
from core.schemas import Chart

class QueryAgentMethods:
    """Universal methods for handling any type of data query"""
    
    def _handle_ranking(self, question: str, intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Handle ranking questions like 'which has the most/least'"""
        
        question_lower = question.lower()
        
        # Check for gender-based questions
        if any(term in question_lower for term in ['female', 'women', 'woman', 'male', 'men', 'man']):
            return self._handle_gender_ranking(question, intent, df, dataset_info)
        
        # Find columns mentioned in the question or suitable for ranking
        relevant_cols = self._find_relevant_columns(intent['keywords'], df, dataset_info)
        
        # If no specific columns mentioned, try to infer from question context
        if not relevant_cols:
            # Look for department/role questions
            if any(term in question_lower for term in ['department', 'role', 'job', 'position', 'team']):
                role_cols = [col for col in df.columns if any(term in col.lower() for term in ['department', 'role', 'job', 'position', 'team'])]
                if role_cols:
                    return self._handle_department_ranking(question, df, role_cols[0])
            
            return {
                "answer": f"I couldn't identify which columns to rank. Your dataset has: {', '.join(df.columns[:10])}. Could you specify which column you'd like me to rank by?",
                "visualizations": None
            }
        
        # Find categorical column for grouping
        grouping_col = self._find_grouping_column(df, dataset_info)
        if not grouping_col:
            return {
                "answer": "I need a categorical column to group the data for ranking, but couldn't identify one in your dataset.",
                "visualizations": None
            }
        
        # Perform ranking
        results = []
        charts = []
        
        for col in relevant_cols[:2]:  # Top 2 relevant columns
            try:
                # Group and aggregate
                grouped = df.groupby(grouping_col)[col].agg(['mean', 'sum', 'count']).reset_index()
                grouped = grouped.dropna()
                
                # Determine ranking direction
                ascending = any(word in question.lower() for word in ['least', 'lowest', 'minimum', 'smallest', 'bottom'])
                
                # Choose aggregation method
                agg_method = 'sum' if 'total' in question.lower() else 'mean'
                top_entries = grouped.nsmallest(5, agg_method) if ascending else grouped.nlargest(5, agg_method)
                
                if len(top_entries) > 0:
                    results.append({
                        'column': col,
                        'method': agg_method,
                        'entries': top_entries,
                        'direction': 'lowest' if ascending else 'highest'
                    })
                    
                    # Create visualization
                    chart_data = {
                        "type": "bar",
                        "spec": {
                            "x": top_entries[grouping_col].tolist(),
                            "y": top_entries[agg_method].tolist(),
                            "title": f"{'Lowest' if ascending else 'Highest'} {col} by {grouping_col}",
                            "xAxisLabel": grouping_col,
                            "yAxisLabel": col
                        }
                    }
                    charts.append(Chart(**chart_data))
            except:
                continue
        
        if results:
            answer_parts = []
            for result in results:
                answer_parts.append(f"**{result['direction'].title()} {result['column']} ({'Total' if result['method'] == 'sum' else 'Average'}):**\n")
                
                for i, row in result['entries'].head(5).iterrows():
                    group_name = row[grouping_col]
                    value = row[result['method']]
                    answer_parts.append(f"{len([x for x in answer_parts if x.startswith(' ')])+1}. {group_name}: {value:,.2f}\n")
                answer_parts.append("\n")
            
            return {
                "answer": "".join(answer_parts).strip(),
                "visualizations": charts
            }
        
        return {
            "answer": f"I couldn't generate meaningful rankings. Try asking about specific columns like: {', '.join(df.columns[:5])}",
            "visualizations": None
        }
    
    def _handle_trend_analysis(self, question: str, intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Handle trend analysis questions"""
        
        # Find time column
        time_col = self._find_time_column(df, dataset_info)
        if not time_col:
            return {
                "answer": "I need a time column to analyze trends, but couldn't identify one in your dataset. Do you have year, date, or time data?",
                "visualizations": None
            }
        
        # Find measurement columns
        measure_cols = self._find_relevant_columns(intent['keywords'], df, dataset_info)
        if not measure_cols:
            measure_cols = [col for col in dataset_info['numeric_columns'] if col != time_col][:3]
        
        if not measure_cols:
            return {
                "answer": "I need numeric columns to analyze trends, but couldn't find suitable ones in your dataset.",
                "visualizations": None
            }
        
        # Apply time filtering if specified
        filtered_df = df.copy()
        if intent['time_range']:
            start_year, end_year = intent['time_range']
            if end_year:
                filtered_df = df[(df[time_col] >= start_year) & (df[time_col] <= end_year)]
            else:
                filtered_df = df[df[time_col] >= start_year]
        
        # Analyze trends
        trends = []
        for col in measure_cols:
            time_series = filtered_df[[time_col, col]].dropna().sort_values(time_col)
            if len(time_series) > 1:
                first_val = time_series[col].iloc[0]
                last_val = time_series[col].iloc[-1]
                change = last_val - first_val
                pct_change = (change / first_val * 100) if first_val != 0 else 0
                
                trends.append({
                    'column': col,
                    'change': change,
                    'pct_change': pct_change,
                    'direction': 'increased' if change > 0 else 'decreased' if change < 0 else 'remained stable'
                })
        
        if trends:
            time_desc = ""
            if intent['time_range']:
                start, end = intent['time_range']
                time_desc = f" from {start}" + (f" to {end}" if end else " onwards")
            
            answer = f"Here are the trends in your data{time_desc}:\n\n"
            for trend in trends:
                answer += f"**{trend['column']}**: {trend['direction']} by {abs(trend['change']):.2f} ({abs(trend['pct_change']):.1f}%)\n"
            
            return {
                "answer": answer,
                "visualizations": None
            }
        
        return {
            "answer": "I couldn't detect clear trends in your data. Try specifying time periods or different columns to analyze.",
            "visualizations": None
        }
    
    def _handle_comparison(self, question: str, intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Handle comparison questions"""
        
        # Find grouping column
        grouping_col = self._find_grouping_column(df, dataset_info)
        if not grouping_col:
            return {
                "answer": "I need categorical data to make comparisons, but couldn't identify suitable columns in your dataset.",
                "visualizations": None
            }
        
        # Find columns to compare
        compare_cols = self._find_relevant_columns(intent['keywords'], df, dataset_info)
        if not compare_cols:
            compare_cols = dataset_info['numeric_columns'][:3]
        
        if not compare_cols:
            return {
                "answer": "I need numeric columns to compare, but couldn't find suitable ones.",
                "visualizations": None
            }
        
        comparisons = []
        for col in compare_cols[:2]:
            comparison = df.groupby(grouping_col)[col].agg(['mean', 'std', 'count']).reset_index()
            comparison = comparison.dropna().head(8)
            if len(comparison) > 1:
                comparisons.append({
                    'column': col,
                    'data': comparison
                })
        
        if comparisons:
            answer = f"Here's a comparison of your data by {grouping_col}:\n\n"
            for comp in comparisons:
                answer += f"**{comp['column']}:**\n"
                for _, row in comp['data'].iterrows():
                    answer += f"- {row[grouping_col]}: {row['mean']:.2f} (±{row['std']:.2f})\n"
                answer += "\n"
            
            return {
                "answer": answer,
                "visualizations": None
            }
        
        return {
            "answer": f"I couldn't generate meaningful comparisons. Your dataset has these columns: {', '.join(df.columns[:8])}",
            "visualizations": None
        }
    
    def _handle_aggregation(self, question: str, intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Handle aggregate calculations"""
        
        # Find relevant numeric columns
        relevant_cols = self._find_relevant_columns(intent['keywords'], df, dataset_info)
        if not relevant_cols:
            relevant_cols = dataset_info['numeric_columns'][:5]
        
        if not relevant_cols:
            return {
                "answer": "I need numeric columns to calculate statistics, but couldn't find any in your dataset.",
                "visualizations": None
            }
        
        aggregates = []
        for col in relevant_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                aggregates.append({
                    'column': col,
                    'mean': col_data.mean(),
                    'total': col_data.sum(),
                    'median': col_data.median(),
                    'count': len(col_data),
                    'std': col_data.std()
                })
        
        if aggregates:
            answer = "Here are the key statistics:\n\n"
            for agg in aggregates:
                answer += f"**{agg['column']}:**\n"
                answer += f"- Average: {agg['mean']:,.2f}\n"
                answer += f"- Total: {agg['total']:,.2f}\n"
                answer += f"- Median: {agg['median']:,.2f}\n"
                answer += f"- Count: {agg['count']:,}\n"
                if agg['std'] > 0:
                    answer += f"- Standard Deviation: {agg['std']:,.2f}\n"
                answer += "\n"
            
            return {
                "answer": answer,
                "visualizations": None
            }
        
        return {
            "answer": f"I couldn't calculate meaningful statistics. Your dataset columns are: {', '.join(df.columns[:10])}",
            "visualizations": None
        }
    
    def _handle_explanation(self, question: str, intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Explain data and statistics contextually"""
        
        rows, cols = dataset_info['shape']
        explanation = f"Let me explain your dataset:\n\n"
        explanation += f"📊 **Dataset Overview**: {rows:,} rows and {cols} columns\n\n"
        
        # Explain column types
        type_counts = {}
        for col, col_type in dataset_info['column_types'].items():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        if type_counts:
            explanation += "🏗️ **Data Structure**:\n"
            for col_type, count in type_counts.items():
                explanation += f"- {count} {col_type} column{'s' if count > 1 else ''}\n"
            explanation += "\n"
        
        # Explain key columns
        if dataset_info['numeric_columns']:
            explanation += "📈 **Key Measurements**:\n"
            for col in dataset_info['numeric_columns'][:3]:
                if col in dataset_info['data_ranges']:
                    ranges = dataset_info['data_ranges'][col]
                    explanation += f"- {col}: ranges from {ranges['min']:,.2f} to {ranges['max']:,.2f} (avg: {ranges['mean']:,.2f})\n"
            explanation += "\n"
        
        # Explain what questions can be asked
        explanation += "🔍 **What you can ask me**:\n"
        if any(dataset_info['column_types'][col] == 'geographic' for col in df.columns):
            explanation += "• Ranking: 'Which [location] has the highest [measurement]?'\n"
        if any(dataset_info['column_types'][col] == 'temporal' for col in df.columns):
            explanation += "• Trends: 'How did [measurement] change over time?'\n"
        explanation += "• Comparisons: 'Compare [measurement] between different [categories]'\n"
        explanation += "• Statistics: 'What's the average [measurement]?'\n"
        
        return {
            "answer": explanation,
            "visualizations": None
        }
    
    def _handle_display(self, question: str, intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Handle display/show requests"""
        
        # Find relevant columns based on keywords
        relevant_cols = self._find_relevant_columns(intent['keywords'], df, dataset_info)
        
        if not relevant_cols:
            # Show general overview
            sample_data = df.head(3).to_dict('records')
            answer = f"Here's a sample of your data ({len(df):,} total rows):\n\n"
            
            for i, row in enumerate(sample_data, 1):
                answer += f"**Row {i}:**\n"
                for col, value in list(row.items())[:5]:  # Show first 5 columns
                    answer += f"- {col}: {value}\n"
                answer += "\n"
            
            answer += f"**All columns**: {', '.join(df.columns)}"
            
            return {
                "answer": answer,
                "visualizations": None
            }
        
        # Show specific columns requested
        display_df = df[relevant_cols].head(10)
        answer = f"Here are the {len(relevant_cols)} column{'s' if len(relevant_cols) > 1 else ''} you requested:\n\n"
        
        for col in relevant_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                if col in dataset_info['numeric_columns']:
                    answer += f"**{col}**: {col_data.min():.2f} to {col_data.max():.2f} (avg: {col_data.mean():.2f})\n"
                else:
                    unique_vals = col_data.unique()[:5]
                    answer += f"**{col}**: {len(col_data.unique())} unique values including {', '.join(map(str, unique_vals))}\n"
        
        return {
            "answer": answer,
            "visualizations": None
        }
    
    def _handle_general_inquiry(self, question: str, intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Handle general questions about the dataset"""
        
        rows, cols = dataset_info['shape']
        answer_parts = [f"Your dataset contains {rows:,} rows and {cols} columns.\n"]
        
        # Describe the data types found
        temporal_cols = [col for col, col_type in dataset_info['column_types'].items() if col_type == 'temporal']
        geographic_cols = [col for col, col_type in dataset_info['column_types'].items() if col_type == 'geographic']
        measurement_cols = [col for col, col_type in dataset_info['column_types'].items() if col_type == 'measurement']
        
        if geographic_cols:
            answer_parts.append(f"📍 Geographic data: {', '.join(geographic_cols)}")
        if temporal_cols:
            answer_parts.append(f"📅 Time data: {', '.join(temporal_cols)}")
        if measurement_cols:
            answer_parts.append(f"📊 Measurements: {', '.join(measurement_cols[:5])}")
        
        answer_parts.append("\n🤔 **Try asking me questions like:**")
        answer_parts.append("• 'What are the highest values in [column]?'")
        answer_parts.append("• 'Show me trends over time'")
        answer_parts.append("• 'Compare different categories'")
        answer_parts.append("• 'What does this data mean?'")
        
        return {
            "answer": "\n".join(answer_parts),
            "visualizations": None
        }
    
    # Helper methods
    def _find_relevant_columns(self, keywords: List[str], df: pd.DataFrame, dataset_info: Dict) -> List[str]:
        """Find columns relevant to the query keywords"""
        relevant = []
        
        # Look for exact matches first
        for keyword in keywords:
            for col in df.columns:
                if keyword.lower() in col.lower() and col not in relevant:
                    relevant.append(col)
        
        # If no exact matches, look for partial matches
        if not relevant:
            for keyword in keywords:
                for col in df.columns:
                    col_words = re.findall(r'\w+', col.lower())
                    if any(keyword in word for word in col_words) and col not in relevant:
                        relevant.append(col)
        
        return relevant
    
    def _find_grouping_column(self, df: pd.DataFrame, dataset_info: Dict) -> str:
        """Find the best column for grouping data"""
        # Prefer geographic, then categorical, then any text column
        for col, col_type in dataset_info['column_types'].items():
            if col_type == 'geographic':
                return col
        
        for col, col_type in dataset_info['column_types'].items():
            if col_type == 'categorical':
                return col
        
        # Fallback to any text column with reasonable number of unique values
        for col in dataset_info['text_columns']:
            if len(df[col].dropna().unique()) < 50:
                return col
        
        return None
    
    def _find_time_column(self, df: pd.DataFrame, dataset_info: Dict) -> str:
        """Find the time/date column in the dataset"""
        for col, col_type in dataset_info['column_types'].items():
            if col_type == 'temporal':
                return col
        return None
    
    def _handle_gender_ranking(self, question: str, intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Handle gender/demographic filtering questions universally"""
        
        question_lower = question.lower()
        
        # Find ANY column that might contain the filtering criteria
        filter_col, filter_value, filter_term = self._find_filter_column_and_value(question_lower, df)
        
        if not filter_col or not filter_value:
            # Try generic approach - look for any column with binary/categorical values
            potential_cols = []
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                if 2 <= len(unique_vals) <= 10:  # Reasonable range for categorical data
                    potential_cols.append(col)
            
            if potential_cols:
                return {
                    "answer": f"I found some categorical columns that might contain the data you're looking for: {', '.join(potential_cols)}. Could you be more specific about which column and value you want to filter by?",
                    "visualizations": None
                }
            else:
                return {
                    "answer": "I couldn't identify columns suitable for filtering. Your dataset appears to be primarily numeric or have too many unique values per column.",
                    "visualizations": None
                }
        
        # Find grouping column from question context
        grouping_col = self._find_grouping_column_from_question(question_lower, df, dataset_info)
        
        if not grouping_col:
            return {
                "answer": f"I found data to filter by ({filter_col}: {filter_value}) but couldn't identify what to group by. Try asking 'which [category] has most [filter_term]' where [category] is a column name.",
                "visualizations": None
            }
        
        try:
            # Filter the data
            filtered_data = self._filter_dataframe(df, filter_col, filter_value)
            
            if len(filtered_data) == 0:
                available_values = df[filter_col].dropna().unique()[:10]
                return {
                    "answer": f"No data found matching '{filter_value}' in column '{filter_col}'. Available values: {', '.join(map(str, available_values))}",
                    "visualizations": None
                }
            
            # Count by grouping column
            counts = filtered_data[grouping_col].value_counts().head(10)
            
            if len(counts) == 0:
                return {
                    "answer": f"No data found for {filter_term} in the {grouping_col} column.",
                    "visualizations": None
                }
            
            # Create response
            answer_parts = [f"Here are the {grouping_col.lower()}s with the most entries matching '{filter_value}':\n\n"]
            
            for i, (group, count) in enumerate(counts.head(5).items(), 1):
                answer_parts.append(f"{i}. **{group}**: {count:,} entries")
            
            # Calculate percentage for top result
            total_in_top_group = len(df[df[grouping_col] == counts.index[0]])
            pct = (counts.iloc[0] / total_in_top_group * 100) if total_in_top_group > 0 else 0
            answer_parts.append(f"\n💡 **{counts.index[0]}** has the most matching entries ({counts.iloc[0]:,} out of {total_in_top_group:,} total = {pct:.1f}%)")
            
            # Create visualization
            chart_data = {
                "type": "bar",
                "spec": {
                    "x": counts.head(8).index.tolist(),
                    "y": counts.head(8).values.tolist(),
                    "title": f"Count of '{filter_value}' by {grouping_col}",
                    "xAxisLabel": grouping_col,
                    "yAxisLabel": "Count"
                }
            }
            visualizations = [Chart(**chart_data)]
            
            return {
                "answer": "\n".join(answer_parts),
                "visualizations": visualizations
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an issue analyzing the data: {str(e)}. The dataset structure might not support this type of filtering analysis.",
                "visualizations": None
            }
    
    def _find_filter_column_and_value(self, question_lower: str, df: pd.DataFrame):
        """Universally find what column and value to filter by from the question"""
        
        # Extract potential filter terms from question
        filter_terms = []
        if any(term in question_lower for term in ['female', 'women', 'woman']):
            filter_terms.extend(['female', 'f', 'woman', 'women'])
        if any(term in question_lower for term in ['male', 'men', 'man']) and 'female' not in question_lower:
            filter_terms.extend(['male', 'm', 'man', 'men'])
        
        # Look for these terms in actual data
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                unique_vals = df[col].dropna().unique()
                
                # Check if any filter terms match values in this column
                for val in unique_vals:
                    val_str = str(val).lower()
                    for term in filter_terms:
                        if term in val_str:
                            return col, val, term
                
                # Also check for partial matches
                for val in unique_vals:
                    val_str = str(val).lower()
                    for term in filter_terms:
                        if any(t in val_str for t in ['fem', 'mal'] if t in term):
                            return col, val, term
        
        return None, None, None
    
    def _find_grouping_column_from_question(self, question_lower: str, df: pd.DataFrame, dataset_info: Dict):
        """Find grouping column from question context"""
        
        # Look for explicit mentions first
        for col in df.columns:
            col_words = col.lower().split()
            if any(word in question_lower for word in col_words):
                # Make sure it's a good grouping column
                unique_count = len(df[col].dropna().unique())
                if 2 <= unique_count <= 50:  # Reasonable for grouping
                    return col
        
        # Look for question patterns
        question_patterns = {
            'department': ['department', 'dept'],
            'role': ['role', 'job', 'position', 'title'],
            'team': ['team', 'group'],
            'category': ['category', 'type', 'class'],
            'division': ['division', 'unit'],
        }
        
        for pattern_type, patterns in question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                # Find columns that might match this pattern
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in patterns):
                        return col
        
        # Fallback to best categorical column
        return self._find_grouping_column(df, dataset_info)
    
    def _filter_dataframe(self, df: pd.DataFrame, filter_col: str, filter_value):
        """Universally filter dataframe by column and value"""
        
        try:
            # Try string contains first (case insensitive)
            if df[filter_col].dtype == 'object':
                filtered = df[df[filter_col].str.contains(str(filter_value), case=False, na=False)]
                if len(filtered) > 0:
                    return filtered
            
            # Try exact match
            filtered = df[df[filter_col] == filter_value]
            if len(filtered) > 0:
                return filtered
            
            # Try case insensitive exact match for strings
            if df[filter_col].dtype == 'object':
                filtered = df[df[filter_col].str.lower() == str(filter_value).lower()]
                return filtered
                
        except:
            pass
        
        return pd.DataFrame()  # Return empty dataframe if filtering fails

