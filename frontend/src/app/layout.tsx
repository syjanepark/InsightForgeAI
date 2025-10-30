import "@/css/style.css";

import type { Metadata } from "next";
import NextTopLoader from "nextjs-toploader";
import type { PropsWithChildren } from "react";
import { Providers } from "./providers";
import { ToastManager } from "@/components/ui/toast";
import { ScreenLoaderProvider } from "@/components/ui/ScreenLoader";

export const metadata: Metadata = {
  title: {
    template: "%s | InsightForge AI - Enterprise Data Intelligence",
    default: "InsightForge AI - Enterprise Data Intelligence",
  },
  description:
    "Transform your data into actionable insights with AI-powered analysis. Upload CSV files and get comprehensive reports with strategic recommendations.",
};

export default function RootLayout({ children }: PropsWithChildren) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Providers>
          <ToastManager />
          <NextTopLoader color="#8B7CF6" showSpinner={false} />
          
          {/* Main Content Area */}
          <main className="min-h-screen">
            <ScreenLoaderProvider>{children}</ScreenLoaderProvider>
          </main>
        </Providers>
      </body>
    </html>
  );
}
