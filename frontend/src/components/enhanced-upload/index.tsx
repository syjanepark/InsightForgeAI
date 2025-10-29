"use client";

import { useState } from "react";

interface EnhancedUploadProps {
  onUpload: (file: File) => void;
  loading?: boolean;
  onTransform?: () => void;
}

export function EnhancedUpload({ onUpload, loading = false, onTransform }: EnhancedUploadProps) {
  const [dragging, setDragging] = useState(false);
  const [uploaded, setUploaded] = useState(false);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type === "text/csv") {
      setUploaded(true);
      onUpload(file);
      setTimeout(() => {
        onTransform?.();
      }, 1000);
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      setUploaded(true);
      onUpload(file);
      setTimeout(() => {
        onTransform?.();
      }, 1000);
    }
  }

  if (uploaded && loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="glass-card p-8 text-center max-w-md">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
          <div className="animate-spin rounded-full h-10 w-10 border-2 border-white border-t-transparent"></div>
        </div>
        <h3 className="text-xl font-semibold text-deep-indigo mb-2 flex items-center justify-center gap-2">
          <span className="animate-pulse">🔄</span>
          Running Business Intelligence Analysis
        </h3>
        <div className="space-y-2 text-deep-indigo/70">
          <div className="flex items-center justify-center gap-2">
            <span className="animate-bounce">1️⃣</span>
            <span>ETL: Clean, type-cast, normalize</span>
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className="animate-bounce" style={{animationDelay: "0.2s"}}>2️⃣</span>
            <span>Analyze: KPIs, trends, deltas, correlations</span>
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className="animate-bounce" style={{animationDelay: "0.4s"}}>3️⃣</span>
            <span>Generate: Precomputed chart specs</span>
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className="animate-bounce" style={{animationDelay: "0.6s"}}>4️⃣</span>
            <span>Prepare: Chat analysis context</span>
          </div>
        </div>
          <div className="mt-4 w-full bg-gray-200 rounded-full h-2">
            <div className="bg-gradient-to-r from-primary to-accent h-2 rounded-full animate-pulse" style={{ width: "60%" }}></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-[500px] px-4">
      <div
        className={`relative group max-w-2xl w-full transition-all duration-500 transform hover:scale-105 ${
          dragging ? "scale-105" : ""
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !loading && document.getElementById("enhanced-csv-input")?.click()}
      >
        {/* Main Upload Area */}
        <div className={`glass-card p-12 text-center cursor-pointer border-2 border-dashed transition-all duration-300 ${
          dragging 
            ? "border-primary bg-primary/5 glow-violet scale-105" 
            : "border-white/30 hover:border-primary/50 hover:glow-lilac"
        }`}>
          
          {/* Animated Background Gradient */}
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-secondary/5 to-accent/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          
          <div className="relative z-10">
            {/* Large Icon */}
            <div className="mx-auto w-24 h-24 mb-8 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-2xl glow-violet group-hover:scale-110 transition-transform duration-300">
              <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>

            {/* Main Heading */}
            <h1 className="text-4xl font-bold text-deep-indigo mb-4 group-hover:text-primary transition-colors duration-300">
              {dragging ? "Drop your CSV here" : "Drop your CSV or click to upload"}
            </h1>
            
            {/* Subtitle */}
            <p className="text-xl text-deep-indigo/70 mb-8 group-hover:text-primary/80 transition-colors duration-300">
              Let our AI team analyze it
            </p>

            {/* Features Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="glass-card p-4 border border-white/20 hover-lift">
                <div className="w-8 h-8 mx-auto mb-2 text-primary">
                  <svg fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </div>
                <div className="text-sm font-medium text-deep-indigo">Instant Analysis</div>
                <div className="text-xs text-deep-indigo/60 mt-1">Real-time processing</div>
              </div>
              
              <div className="glass-card p-4 border border-white/20 hover-lift">
                <div className="w-8 h-8 mx-auto mb-2 text-secondary">
                  <svg fill="currentColor" viewBox="0 0 20 20">
                    <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"/>
                  </svg>
                </div>
                <div className="text-sm font-medium text-deep-indigo">Smart Insights</div>
                <div className="text-xs text-deep-indigo/60 mt-1">AI-powered reports</div>
              </div>
              
              <div className="glass-card p-4 border border-white/20 hover-lift">
                <div className="w-8 h-8 mx-auto mb-2 text-accent">
                  <svg fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M6 6V5a3 3 0 013-3h2a3 3 0 013 3v1h2a2 2 0 012 2v3.57A22.952 22.952 0 0110 13a22.95 22.95 0 01-8-1.43V8a2 2 0 012-2h2zm2-1a1 1 0 011-1h2a1 1 0 011 1v1H8V5zm1 5a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z" clipRule="evenodd"/>
                  </svg>
                </div>
                <div className="text-sm font-medium text-deep-indigo">Secure Processing</div>
                <div className="text-xs text-deep-indigo/60 mt-1">Enterprise grade</div>
              </div>
            </div>
            
            {/* File Requirements */}
            <div className="flex items-center justify-center space-x-8 text-deep-indigo/60 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                <span className="font-medium">CSV format only</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-secondary rounded-full animate-pulse" style={{ animationDelay: "0.2s" }}></div>
                <span className="font-medium">Up to 50MB</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-accent rounded-full animate-pulse" style={{ animationDelay: "0.4s" }}></div>
                <span className="font-medium">Instant results</span>
              </div>
            </div>
          </div>
        </div>
        
        <input
          id="enhanced-csv-input"
          type="file"
          accept=".csv"
          onChange={handleFileSelect}
          className="hidden"
          disabled={loading}
        />
      </div>
    </div>
  );
}
