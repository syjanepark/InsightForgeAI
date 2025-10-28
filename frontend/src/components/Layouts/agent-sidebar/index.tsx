"use client";

import { useState } from "react";
import { Logo } from "@/components/logo";

export type AgentStatus = "idle" | "active" | "completed" | "error";

export interface Agent {
  id: string;
  name: string;
  icon: React.ReactNode;
  status: AgentStatus;
  description: string;
  progress?: number;
}

interface AgentSidebarProps {
  agents: Agent[];
  onAgentClick?: (agentId: string) => void;
}

export function AgentSidebar({ agents, onAgentClick }: AgentSidebarProps) {
  return (
    <aside className="fixed left-0 top-0 h-screen w-80 glass-card border-r border-white/30 backdrop-blur-xl z-50">
      <div className="flex h-full flex-col p-6">
        {/* Logo */}
        <div className="mb-8">
          <Logo />
          <div className="mt-2 text-sm text-deep-indigo/70 font-medium">
            AI Agent Timeline
          </div>
        </div>

        {/* Agents List */}
        <div className="flex-1 space-y-4">
          <h2 className="text-lg font-semibold text-deep-indigo mb-6">
            AI Agents
          </h2>
          
          {agents.map((agent, index) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              index={index}
              onClick={() => onAgentClick?.(agent.id)}
            />
          ))}
        </div>

        {/* Footer */}
        <div className="mt-6 pt-6 border-t border-white/20">
          <div className="text-xs text-deep-indigo/60 text-center">
            InsightForge AI • Enterprise Data Intelligence
          </div>
        </div>
      </div>
    </aside>
  );
}

function AgentCard({ agent, index, onClick }: { 
  agent: Agent; 
  index: number;
  onClick: () => void;
}) {
  const getStatusColor = (status: AgentStatus) => {
    switch (status) {
      case "idle":
        return "bg-gray-100 border-gray-200";
      case "active":
        return "bg-primary/10 border-primary glow-violet";
      case "completed":
        return "bg-green-100 border-green-300 glow-sky";
      case "error":
        return "bg-red-100 border-red-300";
      default:
        return "bg-gray-100 border-gray-200";
    }
  };

  const getIconGlow = (status: AgentStatus) => {
    return status === "active" ? "agent-icon-glow active" : "agent-icon-glow";
  };

  return (
    <div
      className={`relative group cursor-pointer transition-all duration-300 hover:scale-105 ${
        agent.status === "active" ? "slide-up-fade" : ""
      }`}
      onClick={onClick}
    >
      {/* Connection Line */}
      {index < 3 && (
        <div className="absolute left-6 top-16 w-px h-8 bg-gradient-to-b from-primary/30 to-secondary/30"></div>
      )}

      <div className={`glass-card p-4 border-2 transition-all duration-300 hover-lift ${getStatusColor(agent.status)}`}>
        <div className="flex items-start space-x-4">
          {/* Agent Icon */}
          <div className={`w-12 h-12 rounded-full flex items-center justify-center ${getIconGlow(agent.status)}`}
               style={{
                 background: agent.status === "active" 
                   ? "linear-gradient(135deg, #8B7CF6, #A78BFA)" 
                   : agent.status === "completed"
                   ? "linear-gradient(135deg, #60A5FA, #34D399)"
                   : "linear-gradient(135deg, #E5E7EB, #F3F4F6)"
               }}>
            <div className="text-white text-lg">
              {agent.icon}
            </div>
          </div>

          {/* Agent Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-semibold text-deep-indigo truncate">
                {agent.name}
              </h3>
              <StatusIndicator status={agent.status} />
            </div>
            
            <p className="text-sm text-deep-indigo/70 leading-relaxed">
              {agent.description}
            </p>

            {/* Progress Bar */}
            {agent.status === "active" && agent.progress !== undefined && (
              <div className="mt-3">
                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div 
                    className="bg-gradient-to-r from-primary to-secondary h-1.5 rounded-full transition-all duration-500"
                    style={{ width: `${agent.progress}%` }}
                  ></div>
                </div>
                <div className="text-xs text-primary mt-1 font-medium">
                  {agent.progress}% Complete
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusIndicator({ status }: { status: AgentStatus }) {
  switch (status) {
    case "idle":
      return (
        <div className="w-2 h-2 rounded-full bg-gray-400"></div>
      );
    case "active":
      return (
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
      );
    case "completed":
      return (
        <div className="w-4 h-4 text-green-600">
          <svg fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        </div>
      );
    case "error":
      return (
        <div className="w-4 h-4 text-red-600">
          <svg fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        </div>
      );
  }
}
