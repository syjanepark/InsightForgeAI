"use client";

import { useState } from "react";

interface CSVUploadProps {
  onUpload: (file: File) => void;
  loading?: boolean;
}

export function CSVUpload({ onUpload, loading = false }: CSVUploadProps) {
  const [dragging, setDragging] = useState(false);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    document.getElementById("csv-file-input")?.setAttribute("aria-pressed", "true");
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type === "text/csv") {
      onUpload(file);
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    document.getElementById("csv-file-input")?.setAttribute("aria-pressed", "true");
    const file = e.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  }

  return (
    <div
      className={`relative group bg-white dark:bg-gray-800 rounded-lg border-2 border-dashed transition-all duration-300 hover:shadow-lg hover:-translate-y-1 ${
        dragging 
          ? "border-primary bg-primary/5 scale-105" 
          : "border-gray-300 hover:border-primary"
      } ${loading ? "opacity-50 pointer-events-none" : "cursor-pointer"}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !loading && document.getElementById("csv-file-input")?.click()}
    >
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-secondary/5 to-primary/5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
      
      <div className="relative z-10 p-8 text-center">
        {/* Icon */}
        <div className="mx-auto w-16 h-16 bg-gradient-to-br from-primary to-secondary rounded-xl flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform duration-300">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>

        {/* Badge */}
        <div className="inline-block bg-gradient-to-r from-primary to-secondary text-white px-4 py-1.5 rounded-full text-xs font-bold mb-4 shadow-lg">
          UPLOAD AREA
        </div>

        {/* Title */}
        <h3 className="text-2xl font-bold text-black dark:text-white mb-4 group-hover:text-primary transition-colors duration-300">
          {dragging ? "Drop your CSV file here" : "Upload Your CSV File"}
        </h3>
        
        {/* Description */}
        <p className="text-gray-600 dark:text-gray-300 mb-6 text-lg group-hover:text-primary transition-colors duration-300">
          {dragging ? "Release to upload" : "Click to browse or drag and drop"}
        </p>
        
        {/* File Requirements */}
        <div className="flex items-center justify-center space-x-6 text-gray-600 dark:text-gray-300 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-primary rounded-full"></div>
            <span className="font-medium">CSV format only</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-secondary rounded-full"></div>
            <span className="font-medium">Up to 10MB</span>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="absolute inset-0 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg flex items-center justify-center">
            <div className="flex items-center space-x-3">
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
              <span className="text-primary font-medium">Processing...</span>
            </div>
          </div>
        )}
      </div>
      
      <input
        aria-label="Upload CSV file"
        aria-hidden="true"
        id="csv-file-input"
        type="file"
        accept=".csv"
        onChange={handleFileSelect}
        className="hidden"
        disabled={loading}
      />
    </div>
  );
}
