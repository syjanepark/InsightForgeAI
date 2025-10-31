export function Logo() {
  return (
    <div className="flex items-center space-x-3">
      {/* Icon */}
      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-lg glow-violet bg-cololr-violet">
        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      </div>
      
      {/* Text */}
      <div className="flex flex-col">
        <div className="text-xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text">
          InsightForge
        </div>
        <div className="text-xs text-deep-indigo/60 font-medium tracking-wide">
          AI
        </div>
      </div>
    </div>
  );
}
