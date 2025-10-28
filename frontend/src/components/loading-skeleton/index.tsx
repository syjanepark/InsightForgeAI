export function DashboardLoadingSkeleton() {
  return (
    <div className="space-y-8 p-6 animate-pulse">
      {/* Header Skeleton */}
      <div className="text-center mb-8">
        <div className="h-10 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-lg w-80 mx-auto mb-4"></div>
        <div className="h-6 bg-primary/10 rounded-lg w-96 mx-auto"></div>
      </div>

      {/* Summary Stats Skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {[1, 2, 3].map((i) => (
          <div key={i} className="glass-card p-6 text-center">
            <div className="h-8 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-lg mb-2"></div>
            <div className="h-4 bg-primary/10 rounded-lg w-20 mx-auto"></div>
          </div>
        ))}
      </div>

      {/* Charts Section Skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {[1, 2].map((i) => (
          <div key={i} className="glass-card p-6">
            <div className="h-6 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-lg w-40 mb-4"></div>
            <div className="h-64 bg-gradient-to-br from-primary/5 to-secondary/5 rounded-lg flex items-center justify-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary/20 to-secondary/20 animate-pulse"></div>
            </div>
          </div>
        ))}
      </div>

      {/* Insights Grid Skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <InsightCardSkeleton key={i} delay={i * 100} />
        ))}
      </div>
    </div>
  );
}

function InsightCardSkeleton({ delay }: { delay: number }) {
  return (
    <div 
      className="glass-card p-6 slide-up-fade"
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Header Skeleton */}
      <div className="flex items-start justify-between mb-4">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary/20 to-secondary/20 animate-pulse"></div>
        <div className="text-right">
          <div className="h-3 bg-primary/10 rounded w-16 mb-1"></div>
          <div className="h-4 bg-primary/20 rounded w-20"></div>
        </div>
      </div>

      {/* Content Skeleton */}
      <div className="mb-4">
        <div className="h-6 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-lg mb-2"></div>
        <div className="space-y-2">
          <div className="h-4 bg-primary/10 rounded-lg"></div>
          <div className="h-4 bg-primary/10 rounded-lg w-4/5"></div>
          <div className="h-4 bg-primary/10 rounded-lg w-3/5"></div>
        </div>
      </div>

      {/* Sources Skeleton */}
      <div className="border-t border-white/20 pt-4">
        <div className="h-3 bg-primary/10 rounded w-12 mb-2"></div>
        <div className="flex flex-wrap gap-1">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-6 bg-primary/10 rounded-full w-16"></div>
          ))}
        </div>
      </div>
    </div>
  );
}
