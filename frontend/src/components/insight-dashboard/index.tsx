"use client";

import { useState, useEffect } from "react";
import { Chart } from "@/components/charts";

export interface InsightCard {
  id: string;
  title: string;
  summary: string;
  sources: string[];
  category: "trend" | "anomaly" | "recommendation" | "correlation";
  confidence: number;
  chartData?: any;
}

export interface DashboardData {
  insights: InsightCard[];
  charts: {
    id: string;
    type: "bar" | "line" | "pie";
    title: string;
    data: any[];
    xKey?: string;
    yKey?: string;
  }[];
  summary: {
    totalRows: number;
    totalColumns: number;
    keyMetrics: { name: string; value: string }[];
    columnTypes?: { [key: string]: 'numeric' | 'text' | 'date' };
    numericColumns?: string[];
    textColumns?: string[];
  };
}

interface InsightDashboardProps {
  data: DashboardData;
  isVisible: boolean;
}

export function InsightDashboard({ data, isVisible }: InsightDashboardProps) {
  const [visibleCards, setVisibleCards] = useState<number>(0);

  useEffect(() => {
    if (isVisible && data.insights.length > 0) {
      // Animate cards in one by one
      const timer = setInterval(() => {
        setVisibleCards(prev => {
          if (prev < data.insights.length) {
            return prev + 1;
          }
          clearInterval(timer);
          return prev;
        });
      }, 200);

      return () => clearInterval(timer);
    }
  }, [isVisible, data.insights.length]);

  if (!isVisible) return null;

  return (
    <div className="space-y-8 p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-deep-indigo mb-4">
          Your Data Insights
        </h1>
        <p className="text-xl text-deep-indigo/70">
          AI-powered analysis with actionable recommendations
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass-card p-6 text-center hover-lift">
          <div className="text-3xl font-bold text-primary mb-2">
            {data.summary.totalRows.toLocaleString()}
          </div>
          <div className="text-sm text-deep-indigo/70">Total Rows</div>
        </div>
        <div className="glass-card p-6 text-center hover-lift">
          <div className="text-3xl font-bold text-secondary mb-2">
            {data.summary.totalColumns}
          </div>
          <div className="text-sm text-deep-indigo/70">Columns Analyzed</div>
        </div>
        <div className="glass-card p-6 text-center hover-lift">
          <div className="text-3xl font-bold text-accent mb-2">
            {data.insights.length}
          </div>
          <div className="text-sm text-deep-indigo/70">Insights Generated</div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {data.charts.map((chart, index) => (
          <ChartCard key={chart.id} chart={chart} delay={index * 100} />
        ))}
      </div>

      {/* Insights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {data.insights.map((insight, index) => (
          <InsightCardComponent
            key={insight.id}
            insight={insight}
            isVisible={index < visibleCards}
            delay={index * 200}
          />
        ))}
      </div>
    </div>
  );
}

function ChartCard({ chart, delay }: { chart: DashboardData['charts'][0]; delay: number }) {
  return (
    <Chart
      title={chart.title}
      data={chart.data}
      type={chart.type}
      xKey={chart.xKey}
      yKey={chart.yKey}
      delay={delay}
    />
  );
}

function InsightCardComponent({ 
  insight, 
  isVisible, 
  delay 
}: { 
  insight: InsightCard; 
  isVisible: boolean; 
  delay: number;
}) {
  const getCategoryIcon = (category: string) => {
    switch (category) {
      case "trend":
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
        );
      case "anomaly":
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      case "recommendation":
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        );
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "trend":
        return "from-primary to-secondary";
      case "anomaly":
        return "from-red-400 to-orange-400";
      case "recommendation":
        return "from-accent to-primary";
      case "correlation":
        return "from-secondary to-accent";
      default:
        return "from-primary to-secondary";
    }
  };

  if (!isVisible) return null;

  return (
    <div 
      className="glass-card p-6 hover-lift slide-up-fade group cursor-pointer"
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${getCategoryColor(insight.category)} flex items-center justify-center text-white group-hover:scale-110 transition-transform duration-300`}>
          {getCategoryIcon(insight.category)}
        </div>
        <div className="text-right">
          <div className="text-xs text-deep-indigo/60 uppercase tracking-wide font-medium">
            {insight.category}
          </div>
          <div className="text-sm font-bold text-primary">
            {insight.confidence}% confidence
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="mb-4">
        <h3 className="text-lg font-bold text-deep-indigo mb-2 group-hover:text-primary transition-colors duration-300">
          {insight.title}
        </h3>
        <p className="text-sm text-deep-indigo/70 leading-relaxed">
          {insight.summary}
        </p>
      </div>

      {/* Sources */}
      <div className="border-t border-white/20 pt-4">
        <div className="text-xs text-deep-indigo/60 mb-2">Sources:</div>
        <div className="flex flex-wrap gap-1">
          {insight.sources.map((source, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-gradient-to-r from-primary/10 to-secondary/10 text-primary text-xs rounded-full font-medium border border-white/20"
            >
              {source}
            </span>
          ))}
        </div>
      </div>

      {/* Hover Effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-secondary/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
    </div>
  );
}
