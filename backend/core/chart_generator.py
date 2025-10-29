"""
InsightForge AI - Professional Chart Generator
Generates accurate, business-focused visualizations for executive dashboards
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class ChartType(Enum):
    """Professional chart types for business intelligence"""
    KPI_CARD = "kpi_card"
    TREND_LINE = "trend_line"
    PERFORMANCE_BAR = "performance_bar"
    DISTRIBUTION_HISTOGRAM = "distribution_histogram"
    CORRELATION_SCATTER = "correlation_scatter"
    SEGMENT_PIE = "segment_pie"
    COMPARISON_COLUMN = "comparison_column"
    HEATMAP_MATRIX = "heatmap_matrix"
    WATERFALL = "waterfall"
    GAUGE_METER = "gauge_meter"

@dataclass
class ChartData:
    """Structured chart data for frontend consumption"""
    chart_type: str
    title: str
    subtitle: str
    data: Dict[str, Any]
    config: Dict[str, Any]
    insights: List[str]
    business_context: str

class BusinessChartGenerator:
    """Professional chart generator for business intelligence dashboards"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#3B82F6',      # Blue
            'secondary': '#8B5CF6',    # Purple  
            'success': '#10B981',      # Green
            'warning': '#F59E0B',      # Amber
            'danger': '#EF4444',       # Red
            'info': '#06B6D4',         # Cyan
            'neutral': '#6B7280'       # Gray
        }
    
    def generate_executive_dashboard(self, df: pd.DataFrame, business_metrics: Any) -> List[ChartData]:
        """Generate a complete executive dashboard with key business charts"""
        
        charts = []
        
        # 1. KPI Overview Cards
        kpi_cards = self._create_kpi_cards(df, business_metrics)
        charts.extend(kpi_cards)
        
        # 2. Performance Trends
        if business_metrics.time_series_capable:
            trend_chart = self._create_trend_analysis(df, business_metrics)
            if trend_chart:
                charts.append(trend_chart)
        
        # 3. Segment Performance
        segment_charts = self._create_segment_analysis(df, business_metrics)
        charts.extend(segment_charts)
        
        # 4. Distribution Analysis
        distribution_chart = self._create_distribution_analysis(df, business_metrics)
        if distribution_chart:
            charts.append(distribution_chart)
        
        # 5. Correlation Insights
        correlation_chart = self._create_correlation_analysis(df, business_metrics)
        if correlation_chart:
            charts.append(correlation_chart)
        
        return charts
    
    def _create_kpi_cards(self, df: pd.DataFrame, business_metrics: Any) -> List[ChartData]:
        """Create KPI overview cards for executive dashboard"""
        
        cards = []
        
        for i, kpi_col in enumerate(business_metrics.kpi_columns[:4]):  # Top 4 KPIs
            if kpi_col in business_metrics.performance_metrics:
                metrics = business_metrics.performance_metrics[kpi_col]
                
                # Calculate trend (if possible)
                trend_direction = self._calculate_trend_direction(df, kpi_col)
                
                # Format value professionally
                current_value = metrics['mean']
                formatted_value = self._format_business_value(current_value, kpi_col)
                
                # Generate insights
                insights = []
                if metrics['std'] / metrics['mean'] > 0.5:
                    insights.append("High variability indicates opportunity for process optimization")
                
                if trend_direction == "up":
                    insights.append("Positive growth trend observed")
                elif trend_direction == "down":
                    insights.append("Declining trend requires attention")
                
                card = ChartData(
                    chart_type=ChartType.KPI_CARD.value,
                    title=kpi_col,
                    subtitle=f"Average: {formatted_value}",
                    data={
                        'value': current_value,
                        'formatted_value': formatted_value,
                        'change_percentage': self._calculate_change_percentage(df, kpi_col),
                        'trend': trend_direction,
                        'min': metrics['min'],
                        'max': metrics['max'],
                        'median': metrics['median']
                    },
                    config={
                        'color': list(self.color_palette.values())[i % len(self.color_palette)],
                        'show_sparkline': True,
                        'format_type': self._detect_value_format(kpi_col)
                    },
                    insights=insights,
                    business_context=f"Key performance indicator tracking {kpi_col.lower()} across all business segments"
                )
                cards.append(card)
        
        return cards
    
    def _create_trend_analysis(self, df: pd.DataFrame, business_metrics: Any) -> Optional[ChartData]:
        """Create time-based trend analysis chart"""
        
        # Find time column
        time_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['date', 'year', 'month', 'time', 'period'])]
        
        if not time_cols or not business_metrics.kpi_columns:
            return None
        
        time_col = time_cols[0]
        primary_kpi = business_metrics.kpi_columns[0]
        
        # Group by time period and calculate trends
        try:
            trend_data = df.groupby(time_col)[primary_kpi].agg(['mean', 'sum', 'count']).reset_index()
            trend_data = trend_data.sort_values(time_col)
            
            # Prepare chart data
            chart_data = {
                'labels': trend_data[time_col].tolist(),
                'datasets': [
                    {
                        'label': f'{primary_kpi} (Average)',
                        'data': trend_data['mean'].tolist(),
                        'borderColor': self.color_palette['primary'],
                        'backgroundColor': f"{self.color_palette['primary']}20",
                        'fill': True
                    }
                ]
            }
            
            # Generate insights
            insights = []
            if len(trend_data) >= 3:
                recent_avg = trend_data['mean'].tail(3).mean()
                earlier_avg = trend_data['mean'].head(3).mean()
                
                if recent_avg > earlier_avg * 1.1:
                    insights.append("Strong upward trend in recent periods")
                elif recent_avg < earlier_avg * 0.9:
                    insights.append("Declining trend requires strategic attention")
                else:
                    insights.append("Stable performance with minor fluctuations")
            
            return ChartData(
                chart_type=ChartType.TREND_LINE.value,
                title=f"{primary_kpi} Trend Analysis",
                subtitle=f"Performance over {time_col.lower()}",
                data=chart_data,
                config={
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'scales': {
                        'y': {'beginAtZero': True}
                    }
                },
                insights=insights,
                business_context=f"Time-series analysis showing {primary_kpi.lower()} performance trends for strategic planning"
            )
            
        except Exception as e:
            print(f"⚠️  Could not create trend analysis: {e}")
            return None
    
    def _create_segment_analysis(self, df: pd.DataFrame, business_metrics: Any) -> List[ChartData]:
        """Create business segment performance charts"""
        
        charts = []
        
        if not business_metrics.segmentation_columns or not business_metrics.kpi_columns:
            return charts
        
        # Primary segmentation analysis
        segment_col = business_metrics.segmentation_columns[0]
        primary_kpi = business_metrics.kpi_columns[0]
        
        try:
            # Group by segment and calculate performance
            segment_performance = df.groupby(segment_col)[primary_kpi].agg([
                'mean', 'sum', 'count', 'std'
            ]).reset_index()
            segment_performance = segment_performance.sort_values('sum', ascending=False)
            
            # Create bar chart for segment comparison
            chart_data = {
                'labels': segment_performance[segment_col].tolist(),
                'datasets': [
                    {
                        'label': f'{primary_kpi} by {segment_col}',
                        'data': segment_performance['sum'].tolist(),
                        'backgroundColor': [
                            self.color_palette['primary'],
                            self.color_palette['secondary'],
                            self.color_palette['success'],
                            self.color_palette['warning'],
                            self.color_palette['info']
                        ][:len(segment_performance)]
                    }
                ]
            }
            
            # Generate segment insights
            insights = []
            top_segment = segment_performance.iloc[0]
            total_value = segment_performance['sum'].sum()
            top_percentage = (top_segment['sum'] / total_value * 100) if total_value > 0 else 0
            
            insights.append(f"{top_segment[segment_col]} leads with {top_percentage:.1f}% of total {primary_kpi.lower()}")
            
            if len(segment_performance) > 1:
                bottom_segment = segment_performance.iloc[-1]
                performance_gap = top_segment['sum'] / bottom_segment['sum'] if bottom_segment['sum'] > 0 else 0
                if performance_gap > 3:
                    insights.append(f"Significant performance gap detected - top segment outperforms by {performance_gap:.1f}x")
            
            bar_chart = ChartData(
                chart_type=ChartType.PERFORMANCE_BAR.value,
                title=f"{primary_kpi} by {segment_col}",
                subtitle="Segment performance comparison",
                data=chart_data,
                config={
                    'responsive': True,
                    'indexAxis': 'y' if len(segment_performance) > 6 else 'x'
                },
                insights=insights,
                business_context=f"Segment analysis identifying top performers in {segment_col.lower()} for strategic resource allocation"
            )
            charts.append(bar_chart)
            
        except Exception as e:
            print(f"⚠️  Could not create segment analysis: {e}")
        
        return charts
    
    def _create_distribution_analysis(self, df: pd.DataFrame, business_metrics: Any) -> Optional[ChartData]:
        """Create distribution analysis for key metrics"""
        
        if not business_metrics.kpi_columns:
            return None
        
        primary_kpi = business_metrics.kpi_columns[0]
        
        try:
            # Create histogram data
            kpi_data = df[primary_kpi].dropna()
            
            # Calculate bins intelligently
            num_bins = min(20, max(5, int(np.sqrt(len(kpi_data)))))
            counts, bin_edges = np.histogram(kpi_data, bins=num_bins)
            
            # Create labels for bins
            bin_labels = []
            for i in range(len(bin_edges) - 1):
                label = f"{self._format_business_value(bin_edges[i], primary_kpi)} - {self._format_business_value(bin_edges[i+1], primary_kpi)}"
                bin_labels.append(label)
            
            chart_data = {
                'labels': bin_labels,
                'datasets': [
                    {
                        'label': f'{primary_kpi} Distribution',
                        'data': counts.tolist(),
                        'backgroundColor': f"{self.color_palette['primary']}80",
                        'borderColor': self.color_palette['primary'],
                        'borderWidth': 1
                    }
                ]
            }
            
            # Generate distribution insights
            insights = []
            mean_val = kpi_data.mean()
            median_val = kpi_data.median()
            
            if abs(mean_val - median_val) / mean_val > 0.2:
                if mean_val > median_val:
                    insights.append("Right-skewed distribution indicates few high performers driving averages")
                else:
                    insights.append("Left-skewed distribution suggests consistent performance with few outliers")
            else:
                insights.append("Normal distribution indicates balanced performance across segments")
            
            # Check for outliers
            Q1 = kpi_data.quantile(0.25)
            Q3 = kpi_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = kpi_data[(kpi_data < Q1 - 1.5 * IQR) | (kpi_data > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(kpi_data) * 100
                insights.append(f"{outlier_pct:.1f}% outliers detected - investigate for process improvements")
            
            return ChartData(
                chart_type=ChartType.DISTRIBUTION_HISTOGRAM.value,
                title=f"{primary_kpi} Distribution Analysis",
                subtitle="Understanding performance spread across business",
                data=chart_data,
                config={
                    'responsive': True,
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Frequency'}}
                    }
                },
                insights=insights,
                business_context=f"Distribution analysis of {primary_kpi.lower()} to identify performance patterns and optimization opportunities"
            )
            
        except Exception as e:
            print(f"⚠️  Could not create distribution analysis: {e}")
            return None
    
    def _create_correlation_analysis(self, df: pd.DataFrame, business_metrics: Any) -> Optional[ChartData]:
        """Create correlation analysis between key metrics"""
        
        if len(business_metrics.kpi_columns) < 2:
            return None
        
        try:
            # Calculate correlation matrix for top KPIs
            kpi_data = df[business_metrics.kpi_columns[:5]].corr()
            
            # Convert to format suitable for heatmap
            correlation_data = []
            labels = business_metrics.kpi_columns[:5]
            
            for i, row_label in enumerate(labels):
                for j, col_label in enumerate(labels):
                    correlation_data.append({
                        'x': col_label,
                        'y': row_label,
                        'value': float(kpi_data.iloc[i, j])
                    })
            
            chart_data = {
                'datasets': [{
                    'label': 'Correlation',
                    'data': correlation_data,
                    'backgroundColor': lambda ctx: self._get_correlation_color(ctx.parsed.value)
                }]
            }
            
            # Generate correlation insights
            insights = []
            
            # Find strongest correlations (excluding diagonal)
            strong_correlations = []
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    corr_value = kpi_data.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        relationship = "strong positive" if corr_value > 0 else "strong negative"
                        strong_correlations.append(f"{labels[i]} has {relationship} correlation with {labels[j]} ({corr_value:.2f})")
            
            if strong_correlations:
                insights.extend(strong_correlations[:3])  # Top 3 correlations
            else:
                insights.append("No strong correlations found - metrics operate independently")
            
            return ChartData(
                chart_type=ChartType.HEATMAP_MATRIX.value,
                title="KPI Correlation Analysis",
                subtitle="Understanding relationships between key metrics",
                data=chart_data,
                config={
                    'responsive': True,
                    'plugins': {
                        'legend': {'display': False}
                    }
                },
                insights=insights,
                business_context="Correlation analysis revealing metric interdependencies for strategic decision-making"
            )
            
        except Exception as e:
            print(f"⚠️  Could not create correlation analysis: {e}")
            return None
    
    def _calculate_trend_direction(self, df: pd.DataFrame, column: str) -> str:
        """Calculate trend direction for a metric"""
        try:
            # If there's a time column, calculate actual trend
            time_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['date', 'year', 'month', 'time', 'period'])]
            
            if time_cols:
                time_col = time_cols[0]
                trend_data = df.groupby(time_col)[column].mean()
                if len(trend_data) >= 3:
                    recent = trend_data.tail(3).mean()
                    earlier = trend_data.head(3).mean()
                    
                    if recent > earlier * 1.05:
                        return "up"
                    elif recent < earlier * 0.95:
                        return "down"
            
            return "stable"
        except:
            return "stable"
    
    def _calculate_change_percentage(self, df: pd.DataFrame, column: str) -> float:
        """Calculate percentage change for a metric"""
        try:
            time_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['date', 'year', 'month', 'time', 'period'])]
            
            if time_cols:
                time_col = time_cols[0]
                trend_data = df.groupby(time_col)[column].mean()
                if len(trend_data) >= 2:
                    current = trend_data.iloc[-1]
                    previous = trend_data.iloc[-2]
                    
                    if previous != 0:
                        return ((current - previous) / previous * 100)
            
            return 0.0
        except:
            return 0.0
    
    def _format_business_value(self, value: float, column_name: str) -> str:
        """Format values for business presentation"""
        
        column_lower = column_name.lower()
        
        # Currency formatting
        if any(indicator in column_lower for indicator in ['revenue', 'sales', 'cost', 'price', 'amount']):
            if value >= 1_000_000:
                return f"${value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"${value/1_000:.1f}K"
            else:
                return f"${value:.2f}"
        
        # Percentage formatting
        elif any(indicator in column_lower for indicator in ['percent', 'rate', '%']):
            return f"{value:.1f}%"
        
        # Unit formatting
        elif any(indicator in column_lower for indicator in ['units', 'count', 'quantity']):
            if value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:,.0f}"
        
        # Default formatting
        else:
            if value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:.2f}"
    
    def _detect_value_format(self, column_name: str) -> str:
        """Detect the appropriate format type for a column"""
        column_lower = column_name.lower()
        
        if any(indicator in column_lower for indicator in ['revenue', 'sales', 'cost', 'price', 'amount']):
            return 'currency'
        elif any(indicator in column_lower for indicator in ['percent', 'rate', '%']):
            return 'percentage'
        elif any(indicator in column_lower for indicator in ['units', 'count', 'quantity']):
            return 'units'
        else:
            return 'number'
    
    def _get_correlation_color(self, value: float) -> str:
        """Get color for correlation value"""
        if value >= 0.7:
            return self.color_palette['success']
        elif value >= 0.3:
            return self.color_palette['info']
        elif value >= -0.3:
            return self.color_palette['neutral']
        elif value >= -0.7:
            return self.color_palette['warning']
        else:
            return self.color_palette['danger']

def generate_business_charts(df: pd.DataFrame, business_metrics: Any) -> List[Dict[str, Any]]:
    """Main function to generate professional business charts"""
    
    generator = BusinessChartGenerator()
    charts = generator.generate_executive_dashboard(df, business_metrics)
    
    # Convert to JSON-serializable format
    chart_data = []
    for chart in charts:
        chart_dict = {
            'chart_type': chart.chart_type,
            'title': chart.title,
            'subtitle': chart.subtitle,
            'data': chart.data,
            'config': chart.config,
            'insights': chart.insights,
            'business_context': chart.business_context
        }
        chart_data.append(chart_dict)
    
    return chart_data
