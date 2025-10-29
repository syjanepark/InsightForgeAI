import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import re
from core.schemas import Chart
from .query_agent_methods import QueryAgentMethods

class QueryAgent(QueryAgentMethods):
    """Universal query processor that adapts to ANY dataset structure and context"""
    
    def analyze_query(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Intelligently analyze any question against any dataset"""
        
        # First, understand the dataset structure
        dataset_info = self._analyze_dataset_structure(df)
        
        # Extract intent from the question
        query_intent = self._understand_question_intent(question)
        
        # Generate response based on data and intent
        return self._generate_contextual_response(question, query_intent, df, dataset_info)
    
    def _analyze_dataset_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ANY dataset structure intelligently"""
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_columns': df.select_dtypes(include=['object', 'string']).columns.tolist(),
            'categorical_columns': {},
            'column_types': {},
            'data_ranges': {}
        }
        
        # Analyze each column intelligently
        for col in df.columns:
            col_data = df[col].dropna()
            info['column_types'][col] = self._identify_column_type(col, col_data)
            
            if col in info['numeric_columns'] and len(col_data) > 0:
                info['data_ranges'][col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'count': len(col_data)
                }
            
            if col in info['text_columns']:
                unique_vals = col_data.unique()
                if len(unique_vals) < 100:  # Reasonable for categorical data
                    info['categorical_columns'][col] = list(unique_vals[:20])  # Store sample values
        
        return info
    
    def _identify_column_type(self, col_name: str, data: pd.Series) -> str:
        """Identify what type of data a column contains"""
        col_lower = col_name.lower()
        
        # Check for time/date columns
        if any(word in col_lower for word in ['year', 'date', 'time', 'month', 'day']):
            return 'temporal'
        
        # Check for geographic columns
        if len(data) > 0 and data.dtype == 'object':
            unique_vals = data.unique()
            if len(unique_vals) < 200:
                sample_vals = [str(v).lower() for v in unique_vals[:10]]
                # Look for geographic indicators
                geo_indicators = ['united', 'republic', 'kingdom', 'island', 'state', 'country', 'city']
                if any(any(indicator in val for indicator in geo_indicators) for val in sample_vals):
                    return 'geographic'
                elif any(word in col_lower for word in ['country', 'nation', 'region', 'state', 'city', 'location']):
                    return 'geographic'
        
        # Check for identifier columns
        if any(word in col_lower for word in ['id', 'code', 'key', 'index']):
            return 'identifier'
        
        # Check for measurement columns
        if data.dtype in ['int64', 'float64'] and len(data) > 0:
            if data.std() > 0:  # Has variation
                return 'measurement'
        
        # Check for categorical
        if data.dtype == 'object' and len(data.unique()) < len(data) * 0.5:
            return 'categorical'
        
        return 'general'
    
    def _understand_question_intent(self, question: str) -> Dict[str, Any]:
        """Understand the intent of ANY question"""
        
        question_lower = question.lower()
        
        # Determine query type based on question structure
        intent = {
            'type': 'general',
            'action': 'describe',
            'filters': [],
            'grouping': None,
            'time_range': None,
            'keywords': []
        }
        
        # Identify the type of question
        if any(word in question_lower for word in ['most', 'least', 'top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'maximum', 'minimum']):
            intent['type'] = 'ranking'
            intent['action'] = 'rank'
        elif any(word in question_lower for word in ['trend', 'change', 'increase', 'decrease', 'grow', 'decline', 'over time', 'from', 'to']):
            intent['type'] = 'trend'
            intent['action'] = 'analyze_trend'
        elif any(word in question_lower for word in ['compare', 'comparison', 'versus', 'vs', 'difference', 'between']):
            intent['type'] = 'comparison'
            intent['action'] = 'compare'
        elif any(word in question_lower for word in ['average', 'mean', 'total', 'sum', 'count', 'statistics']):
            intent['type'] = 'aggregate'
            intent['action'] = 'calculate'
        elif any(word in question_lower for word in ['should', 'how to', 'what to do', 'improve', 'increase', 'strategy', 'recommend', 'boost', 'grow']):
            intent['type'] = 'strategy'
            intent['action'] = 'recommend'
        elif any(phrase in question_lower for phrase in ['what does', 'meaning', 'explain', 'interpret']):
            intent['type'] = 'explanation'
            intent['action'] = 'explain'
        elif any(word in question_lower for word in ['show', 'display', 'list']):
            intent['type'] = 'display'
            intent['action'] = 'show'
        
        # Extract time references
        time_matches = re.findall(r'\b(19|20)\d{2}\b', question)
        if len(time_matches) >= 2:
            intent['time_range'] = (int(time_matches[0]), int(time_matches[-1]))
        elif len(time_matches) == 1:
            intent['time_range'] = (int(time_matches[0]), None)
        
        # Extract all potential keywords from the question
        words = re.findall(r'\b\w+\b', question_lower)
        intent['keywords'] = [word for word in words if len(word) > 2 and word not in 
                             ['the', 'and', 'for', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'this', 'that', 'with', 'from', 'they', 'what', 'does', 'mean']]
        
        return intent
    
    def _generate_contextual_response(self, question: str, query_intent: Dict, df: pd.DataFrame, dataset_info: Dict) -> Dict[str, Any]:
        """Generate contextual responses for ANY type of data and question"""
        
        try:
            # Route to appropriate handler based on intent
            if query_intent['type'] == 'ranking':
                return self._handle_ranking(question, query_intent, df, dataset_info)
            elif query_intent['type'] == 'trend':
                return self._handle_trend_analysis(question, query_intent, df, dataset_info)
            elif query_intent['type'] == 'comparison':
                return self._handle_comparison(question, query_intent, df, dataset_info)
            elif query_intent['type'] == 'aggregate':
                return self._handle_aggregation(question, query_intent, df, dataset_info)
            elif query_intent['type'] == 'strategy':
                return self._handle_strategy_question(question, query_intent, df, dataset_info)
            elif query_intent['type'] == 'explanation':
                return self._handle_explanation(question, query_intent, df, dataset_info)
            elif query_intent['type'] == 'display':
                return self._handle_display(question, query_intent, df, dataset_info)
            else:
                return self._handle_general_inquiry(question, query_intent, df, dataset_info)
                
        except Exception as e:
            return {
                "answer": f"I encountered an issue analyzing your data: {str(e)}. Let me provide a general overview instead.\n\nYour dataset has {dataset_info['shape'][0]:,} rows and {dataset_info['shape'][1]} columns. The columns are: {', '.join(dataset_info['columns'][:10])}{'...' if len(dataset_info['columns']) > 10 else ''}",
                "visualizations": None
            }
