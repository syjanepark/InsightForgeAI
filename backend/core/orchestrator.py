import uuid
import pandas as pd
import io
from typing import List, Dict, Any
from agents.data_agent import DataAgent
from agents.insight_agent import InsightAgent
from core.schemas import AnalyzeResponse
from core.cache import cache_get, cache_set
from core.data_store import data_store
from core.etl_pipeline import process_business_data, DataQualityReport, BusinessMetrics
from core.chart_generator import generate_business_charts

class Orchestrator:
    def __init__(self):
        self.data_agent = DataAgent()
        self.research_agent = None
        self.insight_agent = InsightAgent()

    async def run(self, file_bytes: bytes, filename: str = "dataset.csv") -> AnalyzeResponse:
        """Professional business intelligence analysis pipeline"""
        run_id = str(uuid.uuid4())
        
        print(f"🚀 Starting InsightForge AI analysis pipeline for {filename}")
        
        # Step 1: Professional ETL Pipeline
        print("📊 Running ETL pipeline...")
        clean_df, quality_report, business_metrics = process_business_data(file_bytes, filename)
        
        # Step 2: Generate Professional Business Charts
        print("📈 Generating executive dashboard charts...")
        professional_charts = generate_business_charts(clean_df, business_metrics)
        
        # Step 3: Extract keywords for web intelligence (from business metrics)
        keywords = []
        keywords.extend(business_metrics.kpi_columns[:3])  # Top 3 KPIs
        keywords.extend(business_metrics.segmentation_columns[:2])  # Top 2 segments
        
        # Add business context keywords
        if quality_report.quality_score < 70:
            keywords.append("data quality improvement")
        keywords.append("business intelligence best practices")
        
        # Step 4: Web Intelligence Research (cached by keyword)
        print("🌐 Enriching with web intelligence...")
        evidence = []
        for kw in keywords:
            cached = cache_get(kw)
            if cached:
                evidence.extend(cached)
                continue
            # Research agent disabled
        
        # Step 5: Legacy data analysis (for compatibility)
        data_result = self.data_agent.analyze(file_bytes)
        
        # Step 6: Professional insight generation
        print("🧠 Generating strategic insights...")
        professional_insights = await self._generate_professional_insights(
            clean_df, quality_report, business_metrics, evidence
        )
        
        # Step 7: Create professional response
        response = AnalyzeResponse(
            run_id=run_id,
            kpis=self._create_professional_kpis(business_metrics),
            charts=professional_charts,
            keywords=keywords,
            insights=professional_insights,
            summary=self._create_executive_summary(quality_report, business_metrics)
        )
        
        # Store processed data and analysis for chat queries
        data_store.store_data(run_id, clean_df, {
            'kpis': response.kpis,
            'charts': response.charts,
            'keywords': response.keywords,
            'insights': response.insights,
            'quality_report': quality_report.__dict__,
            'business_metrics': business_metrics.__dict__
        })
        
        print(f"✅ Analysis complete - Quality Score: {quality_report.quality_score:.1f}/100")
        return response
    
    async def _generate_professional_insights(self, df: pd.DataFrame, quality_report: DataQualityReport, 
                                           business_metrics: BusinessMetrics, evidence: List[str]) -> List[str]:
        """Generate professional business insights"""
        
        insights = []
        
        # Data quality insights
        if quality_report.quality_score >= 90:
            insights.append("✅ Excellent data quality enables reliable business intelligence and strategic decision-making")
        elif quality_report.quality_score >= 70:
            insights.append("⚠️  Good data quality with minor improvements needed for optimal business analysis")
        else:
            insights.append("🔴 Data quality issues detected - recommend ETL process improvements before strategic decisions")
        
        # Business performance insights
        if business_metrics.kpi_columns:
            primary_kpi = business_metrics.kpi_columns[0]
            kpi_metrics = business_metrics.performance_metrics.get(primary_kpi, {})
            
            if kpi_metrics:
                cv = kpi_metrics['std'] / kpi_metrics['mean'] if kpi_metrics['mean'] != 0 else 0
                if cv > 0.5:
                    insights.append(f"📈 High variability in {primary_kpi} indicates significant optimization opportunities")
                elif cv < 0.1:
                    insights.append(f"📊 Consistent {primary_kpi} performance shows stable business operations")
        
        # Segmentation insights
        if business_metrics.segmentation_columns:
            segment_col = business_metrics.segmentation_columns[0]
            unique_segments = df[segment_col].nunique()
            insights.append(f"🎯 {unique_segments} distinct segments in {segment_col} provide clear targeting opportunities")
        
        # Time series insights
        if business_metrics.time_series_capable:
            insights.append("📅 Time series analysis capabilities enable trend forecasting and seasonal planning")
        
        # Web intelligence integration
        if evidence:
            insights.append("🌐 Analysis enriched with real-time market intelligence for strategic context")
        
        return insights
    
    def _create_professional_kpis(self, business_metrics: BusinessMetrics) -> List[Dict[str, Any]]:
        """Create professional KPI summary"""
        
        kpis = []
        for kpi_col in business_metrics.kpi_columns[:5]:  # Top 5 KPIs
            if kpi_col in business_metrics.performance_metrics:
                metrics = business_metrics.performance_metrics[kpi_col]
                kpis.append({
                    'name': kpi_col,
                    'value': metrics['mean'],
                    'trend': 'stable',  # Will be calculated with time series
                    'format': self._detect_kpi_format(kpi_col),
                    'business_impact': 'high' if kpi_col in business_metrics.high_variance_metrics else 'medium'
                })
        
        return kpis
    
    def _detect_kpi_format(self, kpi_name: str) -> str:
        """Detect appropriate format for KPI display"""
        kpi_lower = kpi_name.lower()
        
        if any(term in kpi_lower for term in ['revenue', 'sales', 'cost', 'price']):
            return 'currency'
        elif any(term in kpi_lower for term in ['percent', 'rate', '%']):
            return 'percentage'
        elif any(term in kpi_lower for term in ['units', 'count', 'quantity']):
            return 'units'
        else:
            return 'number'
    
    def _create_executive_summary(self, quality_report: DataQualityReport, business_metrics: BusinessMetrics) -> str:
        """Create executive summary for business stakeholders"""
        
        summary_parts = []
        
        # Dataset overview
        summary_parts.append(f"Analyzed {quality_report.total_rows:,} business records across {quality_report.total_columns} dimensions")
        
        # Quality assessment
        if quality_report.quality_score >= 85:
            summary_parts.append(f"High-quality dataset (Score: {quality_report.quality_score:.0f}/100) suitable for strategic decision-making")
        else:
            summary_parts.append(f"Dataset quality score: {quality_report.quality_score:.0f}/100 - data improvements recommended")
        
        # Business metrics summary
        if business_metrics.kpi_columns:
            summary_parts.append(f"Identified {len(business_metrics.kpi_columns)} key performance indicators for business monitoring")
        
        if business_metrics.segmentation_columns:
            summary_parts.append(f"Found {len(business_metrics.segmentation_columns)} segmentation opportunities for targeted strategies")
        
        # Time series capability
        if business_metrics.time_series_capable:
            summary_parts.append("Time-based analysis enabled for trend forecasting and seasonal planning")
        
        return ". ".join(summary_parts) + "."