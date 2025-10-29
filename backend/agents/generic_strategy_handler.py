"""
Generic Strategy Handler - NO HARDCODING
Pure statistical analysis for business strategy questions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class GenericStrategyHandler:
    """Handles strategy questions using pure statistical analysis"""
    
    def handle_strategy_question(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Handle strategy questions with generic data analysis"""
        
        question_lower = question.lower()
        
        # Extract target entity from question using generic pattern matching
        target_entity = self._extract_target_entity(question_lower, df)
        
        # Find performance metrics generically
        performance_metrics = self._identify_performance_metrics(df)
        
        if not performance_metrics:
            return {
                "answer": "I need performance metrics in your data to provide strategic recommendations. Could you specify which metrics you'd like to improve?",
                "visualizations": None
            }
        
        # Analyze performance patterns
        analysis = self._analyze_performance_patterns(df, target_entity, performance_metrics)
        
        # Generate generic strategic recommendations
        recommendations = self._generate_generic_recommendations(analysis)
        
        # Format response
        answer = self._format_strategy_response(question, analysis, recommendations)
        
        return {
            "answer": answer,
            "visualizations": None
        }
    
    def _extract_target_entity(self, question_lower: str, df: pd.DataFrame) -> str:
        """Extract target entity from question using data-driven approach"""
        
        # Look for words in the question that match column values or names
        potential_targets = []
        
        # Check categorical columns for matches
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        
        for col in categorical_cols:
            unique_values = df[col].dropna().unique()
            for value in unique_values:
                if str(value).lower() in question_lower:
                    potential_targets.append(str(value))
        
        # Check column names for matches
        for col in df.columns:
            if col.lower() in question_lower:
                potential_targets.append(col)
        
        return potential_targets[0] if potential_targets else "overall performance"
    
    def _identify_performance_metrics(self, df: pd.DataFrame) -> List[str]:
        """Identify performance metrics using statistical analysis"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Score columns based on business relevance indicators
        performance_scores = {}
        
        for col in numeric_cols:
            score = 0
            col_lower = col.lower()
            
            # Statistical indicators of importance
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # High variance suggests important business metric
                cv = col_data.std() / abs(col_data.mean()) if col_data.mean() != 0 else 0
                score += min(cv * 10, 50)  # Cap at 50 points
                
                # Positive values suggest performance metrics
                if col_data.min() >= 0:
                    score += 20
                
                # Range suggests measurement scale
                if col_data.max() > col_data.min():
                    score += 15
            
            # Common business metric patterns (generic)
            business_indicators = ['sales', 'revenue', 'profit', 'income', 'performance', 
                                 'growth', 'rate', 'volume', 'count', 'total', 'amount']
            
            for indicator in business_indicators:
                if indicator in col_lower:
                    score += 30
                    break
            
            performance_scores[col] = score
        
        # Return top scoring columns
        sorted_metrics = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        return [col for col, score in sorted_metrics if score > 20][:5]  # Top 5 relevant metrics
    
    def _analyze_performance_patterns(self, df: pd.DataFrame, target_entity: str, metrics: List[str]) -> Dict[str, Any]:
        """Analyze performance patterns using statistical methods"""
        
        analysis = {
            'target': target_entity,
            'metrics_analyzed': len(metrics),
            'patterns': {},
            'performance_level': 'unknown',
            'improvement_areas': []
        }
        
        if not metrics:
            return analysis
        
        # Analyze each metric
        for metric in metrics:
            metric_data = df[metric].dropna()
            
            if len(metric_data) == 0:
                continue
            
            # Statistical analysis
            metric_stats = {
                'mean': metric_data.mean(),
                'median': metric_data.median(),
                'std': metric_data.std(),
                'min': metric_data.min(),
                'max': metric_data.max(),
                'distribution': 'normal' if abs(metric_data.mean() - metric_data.median()) / metric_data.std() < 0.5 else 'skewed'
            }
            
            # Performance assessment based on distribution
            q25 = metric_data.quantile(0.25)
            q75 = metric_data.quantile(0.75)
            
            if target_entity != "overall performance":
                # Try to find target-specific performance
                target_performance = self._find_target_performance(df, target_entity, metric)
                if target_performance is not None:
                    if target_performance < q25:
                        metric_stats['target_performance'] = 'below_average'
                    elif target_performance > q75:
                        metric_stats['target_performance'] = 'above_average'
                    else:
                        metric_stats['target_performance'] = 'average'
            
            analysis['patterns'][metric] = metric_stats
        
        # Overall performance assessment
        below_avg_count = sum(1 for p in analysis['patterns'].values() 
                             if p.get('target_performance') == 'below_average')
        
        if below_avg_count > len(metrics) / 2:
            analysis['performance_level'] = 'underperforming'
        elif below_avg_count == 0:
            analysis['performance_level'] = 'strong'
        else:
            analysis['performance_level'] = 'mixed'
        
        return analysis
    
    def _find_target_performance(self, df: pd.DataFrame, target_entity: str, metric: str) -> float:
        """Find performance for specific target using generic matching"""
        
        # Try to match target with categorical data
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        
        for col in categorical_cols:
            matching_rows = df[df[col].astype(str).str.lower().str.contains(target_entity.lower(), na=False)]
            if len(matching_rows) > 0 and metric in matching_rows.columns:
                return matching_rows[metric].mean()
        
        return None
    
    def _generate_generic_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on statistical analysis"""
        
        recommendations = []
        
        # Performance-based recommendations
        if analysis['performance_level'] == 'underperforming':
            recommendations.extend([
                f"📊 **Performance Analysis**: {analysis['target']} shows below-average performance across multiple metrics",
                f"🎯 **Focus Areas**: Concentrate resources on improving the {len(analysis['improvement_areas'])} weakest performing areas",
                f"📈 **Benchmarking**: Study top performers in your dataset to identify success patterns"
            ])
        
        elif analysis['performance_level'] == 'strong':
            recommendations.extend([
                f"✅ **Maintain Excellence**: {analysis['target']} shows strong performance - focus on sustaining current strategies",
                f"🚀 **Scale Success**: Replicate successful approaches in other areas of your business",
                f"💡 **Innovation**: Use strong position to test new strategies and approaches"
            ])
        
        else:  # mixed performance
            recommendations.extend([
                f"⚖️ **Mixed Performance**: {analysis['target']} shows varied results across different metrics",
                f"🔍 **Identify Patterns**: Analyze which specific areas are performing well vs. poorly",
                f"🎯 **Targeted Approach**: Develop specific strategies for each performance tier"
            ])
        
        # Metric-specific recommendations
        if analysis['patterns']:
            high_variance_metrics = [metric for metric, data in analysis['patterns'].items() 
                                   if data['std'] / data['mean'] > 0.5 if data['mean'] != 0]
            
            if high_variance_metrics:
                recommendations.append(f"📊 **Consistency Focus**: High variability in {', '.join(high_variance_metrics[:2])} suggests process optimization opportunities")
        
        # Data-driven recommendations
        recommendations.extend([
            f"📈 **Data-Driven Decisions**: Continue monitoring {analysis['metrics_analyzed']} key metrics for ongoing optimization",
            f"🔄 **Regular Review**: Set up periodic analysis to track improvement progress"
        ])
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _format_strategy_response(self, question: str, analysis: Dict[str, Any], recommendations: List[str]) -> str:
        """Format the strategic response"""
        
        response = f"**Strategic Analysis for: {analysis['target'].title()}**\n\n"
        
        # Performance summary
        response += f"📊 **Current Status**: {analysis['performance_level'].replace('_', ' ').title()} performance "
        response += f"based on analysis of {analysis['metrics_analyzed']} key metrics\n\n"
        
        # Recommendations
        response += "🎯 **Strategic Recommendations**:\n"
        for i, rec in enumerate(recommendations, 1):
            response += f"{i}. {rec}\n"
        
        # Statistical insight
        if analysis['patterns']:
            response += f"\n💡 **Key Insight**: Analysis covers {len(analysis['patterns'])} performance indicators "
            response += f"for data-driven decision making\n"
        
        response += f"\n*Analysis based on statistical patterns in your dataset - no assumptions made about business domain*"
        
        return response
