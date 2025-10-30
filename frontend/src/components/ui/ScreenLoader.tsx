"use client";
import { useState, createContext, useContext, ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

// Context setup for global access
const LoaderContext = createContext({
  showLoader: () => {},
  hideLoader: () => {},
});

export const useScreenLoader = () => useContext(LoaderContext);

// ✅ Provider wraps your app in layout.tsx or _app.tsx
export const ScreenLoaderProvider = ({ children }: { children: ReactNode }) => {
  const [visible, setVisible] = useState(false);
  const [loadingText, setLoadingText] = useState("Processing...");

  const showLoader = () => {
    setVisible(true);
  };
  if(visible){
    // setTimeout(function(){
    //     setLoadingText('Usually done within 5 seconds. Taking longer than expected !');
    //     setTimeout(function(){
    //         setLoadingText('Usually done within 30 seconds. If this is taking longer, please check your internet connection or try re-uploading the file.');
    //     }, 30000);
    // }, 5000);
    }

  const hideLoader = () => setVisible(false);

  return (
    <LoaderContext.Provider value={{ showLoader, hideLoader }}>
      {children}
      <AnimatePresence>
        {visible && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.8 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col items-center justify-center bg-gradient-to-r from-violet-500 to-indigo-500 p-6 rounded-2xl shadow-2xl"
            >
              <Loader2 className="h-24 w-10rem text-white animate-spin" />
              <p
              role="alert"
              aria-live="assertive" 
              className="text-white mt-3 text-sm opacity-90">{loadingText}</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </LoaderContext.Provider>
  );
};