"use client";

import { useState, useEffect } from "react";
import { AgentSidebar, Agent, AgentStatus } from "@/components/Layouts/agent-sidebar";
import { EnhancedUpload } from "@/components/enhanced-upload";
import { InsightDashboard, DashboardData } from "@/components/insight-dashboard";
import { ChatPanel, ChatMessage } from "@/components/chat-panel";
import { analyzeCSV as apiAnalyzeCSV, AnalysisResult, sendChatMessage } from "@/lib/api";
import { analyzeCSV } from "@/lib/csv-analyzer";

export default function Home() {
  const [currentView, setCurrentView] = useState<"upload" | "dashboard">("upload");
  const [agents, setAgents] = useState<Agent[]>([
    {
      id: "data-analyst",
      name: "Data Analyst",
      icon: "📊",
      status: "idle" as AgentStatus,
      description: "Ready to analyze your CSV data",
    },
    {
      id: "researcher",
      name: "Researcher",
      icon: "🔍",
      status: "idle" as AgentStatus,
      description: "Waiting to gather market insights",
    },
    {
      id: "insight-generator",
      name: "Insight Generator",
      icon: "💡",
      status: "idle" as AgentStatus,
      description: "Standing by to generate insights",
    },
    {
      id: "advisor",
      name: "Advisor",
      icon: "🎯",
      status: "idle" as AgentStatus,
      description: "Ready to provide recommendations",
    },
  ]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);

  // Chat state
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isAiTyping, setIsAiTyping] = useState(false);
  const [hasNewMessages, setHasNewMessages] = useState(false);

  // Simulate AI agents working
  async function simulateAgentWork() {
    const agentOrder = ["data-analyst", "researcher", "insight-generator", "advisor"];
    
    for (let i = 0; i < agentOrder.length; i++) {
      const agentId = agentOrder[i];
      
      // Set agent to active
      setAgents(prev => prev.map(agent => 
        agent.id === agentId 
          ? { ...agent, status: "active" as AgentStatus, progress: 0 }
          : agent
      ));

      // Simulate progress
      for (let progress = 0; progress <= 100; progress += 20) {
        await new Promise(resolve => setTimeout(resolve, 300));
        setAgents(prev => prev.map(agent => 
          agent.id === agentId 
            ? { ...agent, progress }
            : agent
        ));
      }

      // Mark as completed
      setAgents(prev => prev.map(agent => 
        agent.id === agentId 
          ? { ...agent, status: "completed" as AgentStatus, progress: 100 }
          : agent
      ));

      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }

  async function handleUpload(file: File) {
    setLoading(true);
    setError(null);
    
    try {
      console.log('🚀 Starting upload process for file:', file.name);
      
      // Update agents with more specific status messages
      setAgents(prev => prev.map(agent => 
        agent.id === "data-analyst" 
          ? { ...agent, description: "🔍 Parsing CSV data and identifying patterns..." }
          : agent.id === "researcher"
          ? { ...agent, description: "🌐 Gathering market context and insights..." }
          : agent.id === "insight-generator"
          ? { ...agent, description: "💡 Processing data with advanced AI tools..." }
          : agent.id === "advisor"
          ? { ...agent, description: "🎯 Formulating strategic recommendations..." }
          : agent
      ));
      
      // Start agent simulation
      await simulateAgentWork();
      console.log('✅ Agent simulation completed');
      
      // Analyze the actual CSV data using backend API
      console.log('📡 Calling backend API...');
      const analysisResult = await apiAnalyzeCSV(file);
      console.log('✅ Backend API response:', analysisResult);
      
      // Store the run ID for chat queries
      setCurrentRunId(analysisResult.run_id);
      
      // Transform to dashboard format - use local analysis as fallback for compatibility
      console.log('📊 Processing local CSV analysis...');
      const localAnalysis = await analyzeCSV(file);
      console.log('✅ Local analysis completed:', localAnalysis);
      
      const dashboardData: DashboardData = {
        insights: localAnalysis.insights,
        charts: localAnalysis.charts,
        summary: localAnalysis.summary,
      };
      
      console.log('📈 Dashboard data prepared:', dashboardData);
      console.log('🎯 Setting view to dashboard...');

      setDashboardData(dashboardData);
      setCurrentView("dashboard");

      // Reset agent descriptions to default
      setAgents(prev => prev.map(agent => ({
        ...agent, 
        description: agent.id === "data-analyst" ? "Ready to analyze your CSV data"
          : agent.id === "researcher" ? "Waiting to gather market insights"
          : agent.id === "insight-generator" ? "Standing by to generate insights"
          : agent.id === "advisor" ? "Ready to provide recommendations"
          : agent.description
      })));

      // Add welcome message to chat with specific insights
      const welcomeMessage: ChatMessage = {
        id: Date.now().toString(),
        type: "ai",
        content: `Great! I've analyzed your CSV file with ${localAnalysis.summary.totalRows} rows and ${localAnalysis.summary.totalColumns} columns. I found ${analysisResult.insights.length} key insights and generated ${analysisResult.charts.length} visualizations. Feel free to ask me any questions about your data!`,
        timestamp: new Date(),
      };
      setChatMessages([welcomeMessage]);
      setHasNewMessages(true);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      // Reset agents to idle on error and restore descriptions
      setAgents(prev => prev.map(agent => ({ 
        ...agent, 
        status: "idle" as AgentStatus,
        description: agent.id === "data-analyst" ? "Ready to analyze your CSV data"
          : agent.id === "researcher" ? "Waiting to gather market insights"
          : agent.id === "insight-generator" ? "Standing by to generate insights"
          : agent.id === "advisor" ? "Ready to provide recommendations"
          : agent.description
      })));
    } finally {
      setLoading(false);
    }
  }

  async function handleChatMessage(message: string) {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: "user",
      content: message,
      timestamp: new Date(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setIsAiTyping(true);

    // Add loading message with context about processing type
    const isComplexQuery = /\b(most|least|average|calculate|count|sum|compare|correlation|trend|pattern|why|what caused|reason|because|explain)\b/i.test(message);
    const loadingMessage: ChatMessage = {
      id: (Date.now() + 0.5).toString(),
      type: "ai",
      content: isComplexQuery 
        ? "🔍 Analyzing your data with advanced AI tools (40s timeout, will retry faster if needed)..." 
        : "💭 Processing your question quickly...",
      timestamp: new Date(),
      isTyping: true
    };
    setChatMessages(prev => [...prev, loadingMessage]);

    try {
      // Send message to backend API
      const chatResponse = await sendChatMessage({
        question: message,
        run_id: currentRunId || undefined
      });

      // Remove loading message and add real response
      setChatMessages(prev => prev.filter(msg => !msg.isTyping));
      
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: chatResponse.answer,
        timestamp: new Date(),
        suggested_actions: chatResponse.suggested_actions,
        citations: chatResponse.citations,
      };
      
      setChatMessages(prev => [...prev, aiMessage]);
      
      if (!chatOpen) {
        setHasNewMessages(true);
      }
    } catch (error) {
      // Remove loading message and add error message
      setChatMessages(prev => prev.filter(msg => !msg.isTyping));
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: error instanceof Error && error.message.includes('timed out') 
          ? "⏰ Analysis timed out - your query was too complex for real-time processing. Try asking a simpler question or break it into smaller parts."
          : "I apologize, but I'm having trouble processing your question right now. Please try again in a moment.",
        timestamp: new Date(),
      };
      
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsAiTyping(false);
    }
  }

  function toggleChat() {
    setChatOpen(prev => !prev);
    if (!chatOpen) {
      setHasNewMessages(false);
    }
  }

  function resetAnalysis() {
    setCurrentView("upload");
    setDashboardData(null);
    setError(null);
    setChatMessages([]);
    setCurrentRunId(null);
    setAgents(prev => prev.map(agent => ({ 
      ...agent, 
      status: "idle" as AgentStatus, 
      progress: undefined 
    })));
  }

  return (
    <div className="min-h-screen flex">
      {/* Agent Sidebar */}
      <AgentSidebar agents={agents} />

      {/* Main Content */}
      <div className="flex-1 ml-80">
        {/* Header */}
        <div className="glass-card mx-6 mt-6 mb-8 p-6 border-b border-white/20">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-deep-indigo">
                InsightForge AI
              </h1>
              <p className="text-deep-indigo/70">Enterprise Data Intelligence</p>
            </div>
            {currentView === "dashboard" && (
              <button
                onClick={resetAnalysis}
                className="px-4 py-2 glass-card border border-white/30 text-deep-indigo hover:text-primary transition-colors rounded-lg hover-lift"
              >
                Analyze New File
              </button>
            )}
          </div>
        </div>

        {/* Content Area */}
        <div className="px-6 pb-6">
          {error && (
            <div className="mb-8 glass-card p-6 border-2 border-red-300 bg-red-50/50">
              <div className="flex items-center">
                <svg className="w-6 h-6 text-red-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <div>
                  <h3 className="text-lg font-semibold text-red-800">Analysis Failed</h3>
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
              <button
                onClick={resetAnalysis}
                className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Try Again
              </button>
            </div>
          )}

          {currentView === "upload" && !error && (
            <EnhancedUpload 
              onUpload={handleUpload} 
              loading={loading}
              onTransform={() => setCurrentView("dashboard")}
            />
          )}

          {currentView === "dashboard" && dashboardData && (
            <InsightDashboard 
              data={dashboardData} 
              isVisible={true}
            />
          )}
        </div>
      </div>

      {/* Chat Panel */}
      <ChatPanel
        isOpen={chatOpen}
        onToggle={toggleChat}
        messages={chatMessages}
        onSendMessage={handleChatMessage}
        isAiTyping={isAiTyping}
        hasNewMessages={hasNewMessages}
      />
    </div>
  );
}
