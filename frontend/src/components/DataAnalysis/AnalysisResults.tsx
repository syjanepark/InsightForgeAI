"use client";

import { AnalysisResult } from "@/lib/api";

interface AnalysisResultsProps {
  data: AnalysisResult;
}

export function AnalysisResults({ data }: AnalysisResultsProps) {
  return (
    <div className="space-y-6">
      {/* Success Header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 bg-green-100 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-full px-6 py-3 mb-6">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-green-700 dark:text-green-300 font-medium">Analysis Complete</span>
        </div>
        <h2 className="text-3xl md:text-4xl font-bold text-black dark:text-white mb-4">
          Your Insights Are Ready
        </h2>
        <p className="text-gray-600 dark:text-gray-300 text-lg max-w-2xl mx-auto">
          AI-powered analysis complete. Here are your strategic insights and recommendations.
        </p>
      </div>

      {/* Charts Section */}
      {data.charts && data.charts.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-1 p-6">
          <h3 className="text-2xl font-bold text-black dark:text-white mb-6 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-primary to-secondary rounded-lg flex items-center justify-center mr-4">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            Data Visualizations
          </h3>
          <div className="grid gap-4">
            {data.charts.map((chart, i) => (
              <div key={i} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <h4 className="text-lg font-semibold text-black dark:text-white mb-2">{chart.spec.title}</h4>
                <div className="inline-block bg-primary/10 text-primary px-3 py-1 rounded-full text-sm font-medium">
                  {chart.type}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Insights Section */}
      {data.insights && data.insights.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-1 p-6">
          <h3 className="text-2xl font-bold text-black dark:text-white mb-6 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-primary to-secondary rounded-lg flex items-center justify-center mr-4">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            AI Insights
          </h3>
          <div className="space-y-6">
            {data.insights.map((insight, i) => (
              <div key={i} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                <div className="flex items-start space-x-4">
                  <div className="w-8 h-8 bg-gradient-to-r from-primary to-secondary rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                    <span className="text-white font-bold text-sm">{i + 1}</span>
                  </div>
                  <div className="flex-1">
                    <h4 className="text-xl font-bold text-black dark:text-white mb-3">{insight.title}</h4>
                    <p className="text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">{insight.why}</p>
                    {insight.recommendations && insight.recommendations.length > 0 && (
                      <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
                        <p className="text-lg font-semibold text-primary mb-3 flex items-center">
                          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Recommendations
                        </p>
                        <ul className="space-y-2">
                          {insight.recommendations.map((rec, j) => (
                            <li key={j} className="text-gray-600 dark:text-gray-300 flex items-start">
                              <span className="text-primary mr-3 text-lg font-bold">•</span>
                              {rec}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* KPIs Section */}
      {data.kpis && data.kpis.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-1 p-6">
          <h3 className="text-2xl font-bold text-black dark:text-white mb-6 flex items-center">
            <div className="w-8 h-8 bg-gradient-to-r from-primary to-secondary rounded-lg flex items-center justify-center mr-4">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            Key Performance Indicators
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {data.kpis.map((kpi, i) => (
              <div key={i} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <h4 className="text-lg font-semibold text-black dark:text-white mb-2">{kpi.name}</h4>
                <p className="text-2xl font-bold text-primary mb-1">{kpi.value.toLocaleString()}</p>
                <p className={`text-sm ${kpi.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {kpi.change >= 0 ? '+' : ''}{kpi.change}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
