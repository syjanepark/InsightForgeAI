"use client";

import { useState, useRef, useEffect } from "react";

export interface ChatMessage {
  id: string;
  type: "user" | "ai";
  content: string;
  timestamp: Date;
  isTyping?: boolean;
  suggested_actions?: Array<{
    label: string;
    action: string;
  }>;
  citations?: string[];
}

interface ChatPanelProps {
  isOpen: boolean;
  onToggle: () => void;
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isAiTyping?: boolean;
  hasNewMessages?: boolean;
}

export function ChatPanel({ 
  isOpen, 
  onToggle, 
  messages, 
  onSendMessage, 
  isAiTyping = false,
  hasNewMessages = false 
}: ChatPanelProps) {
  const [inputValue, setInputValue] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen, messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !isAiTyping) {
      onSendMessage(inputValue.trim());
      setInputValue("");
    }
  };

  return (
    <>
      {/* Chat Toggle Button */}
      <button
        onClick={onToggle}
        className={`fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full flex items-center justify-center transition-all duration-300 hover:scale-110 ${
          isOpen 
            ? "bg-red-500 glow-violet" 
            : hasNewMessages 
            ? "bg-gradient-to-br from-primary to-secondary glow-violet animate-pulse" 
            : "bg-gradient-to-br from-primary to-secondary glow-lilac hover:glow-violet"
        }`}
      >
        {isOpen ? (
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <>
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            {hasNewMessages && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
            )}
          </>
        )}
      </button>

      {/* Chat Panel */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 w-96 h-[500px] z-40 glass-card border border-white/30 flex flex-col slide-up-fade">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-white/20">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-deep-indigo">Assistant</h3>
                <p className="text-xs text-deep-indigo/60">Data analysis assistant</p>
              </div>
            </div>
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
            {messages.length === 0 && (
              <div className="text-center py-8">
                <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
                  <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <p className="text-sm text-deep-indigo/60">
                  Start a conversation! I&apos;m here to help analyze your data.
                </p>
              </div>
            )}

            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            
            {isAiTyping && (
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center flex-shrink-0">
                  <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </div>
                <div className="glass-card px-4 py-3 max-w-xs">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <form onSubmit={handleSubmit} className="p-4 border-t border-white/20">
            <div className="flex space-x-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about your data..."
                className="flex-1 px-4 py-2 rounded-full glass-card border border-white/20 text-deep-indigo placeholder-deep-indigo/50 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
                disabled={isAiTyping}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isAiTyping}
                className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-white hover:scale-110 transition-transform duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}

function ChatMessage({ message }: { message: ChatMessage }) {
  if (message.type === "user") {
    return (
      <div className="flex justify-end">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white px-4 py-3 rounded-2xl rounded-br-md max-w-xs shadow-lg">
          <p className="text-sm text-white">{message.content}</p>
          <p className="text-xs text-white/70 mt-1">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start space-x-3">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center flex-shrink-0">
        {message.isTyping ? (
          <div className="flex space-x-1">
            <div className="w-1 h-1 bg-white rounded-full animate-bounce"></div>
            <div className="w-1 h-1 bg-white rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
            <div className="w-1 h-1 bg-white rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
          </div>
        ) : (
          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
        )}
      </div>
      <div className="max-w-xs">
        <div className={`glass-card px-4 py-3 border border-white/20 ${message.isTyping ? 'animate-pulse' : ''}`}>
          <p className="text-sm text-deep-indigo">{message.content}</p>
          <p className="text-xs text-deep-indigo/60 mt-1">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
        
        {/* Suggested Actions */}
        {message.suggested_actions && message.suggested_actions.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {message.suggested_actions.map((action, index) => (
              <button
                key={index}
                className="px-2 py-1 text-xs bg-gradient-to-r from-primary/10 to-secondary/10 text-primary rounded-full border border-primary/20 hover:bg-gradient-to-r hover:from-primary/20 hover:to-secondary/20 transition-all duration-200 hover:scale-105"
                onClick={() => {
                  // TODO: Handle action click
                  console.log('Action clicked:', action.action);
                }}
              >
                {action.label}
              </button>
            ))}
          </div>
        )}
        
        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="mt-2">
            <p className="text-xs text-deep-indigo/50 mb-1">Sources:</p>
            <div className="flex flex-wrap gap-1">
              {message.citations.map((citation, index) => (
                <span
                  key={index}
                  className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full border"
                >
                  {citation}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
