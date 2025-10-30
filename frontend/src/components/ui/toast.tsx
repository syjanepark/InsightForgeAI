"use client";
import { Toaster, toast } from "react-hot-toast";
import { CheckCircle, AlertTriangle, XCircle } from "lucide-react";

export const ToastManager = () => (
  <Toaster
    position="top-center"
    toastOptions={{
      duration: 5000,
      className: "toast-glass",
      style: {
        fontFamily: "Inter, sans-serif",
        marginTop: "1rem",
      },
    }}
    containerStyle={{
      top: "1rem",
    }}
  />
);

export const showSuccess = (msg: string, duration = 5000) =>
  toast.custom(
    <div 
    role="status"
    aria-live="polite"
    aria-label={`Success notification of ${msg}`}
    className="toast-glass toast-success flex items-center gap-3">
      <CheckCircle className="h-5 w-5 text-emerald-500" />
      <span>{msg}</span>
    </div>,
    { duration }
  );

export const showWarning = (msg: string, duration = 5000) =>
  toast.custom(
    <div 
    role="alert"
    aria-live="assertive"
    aria-label={`Warning notification of ${msg}`}

    className="toast-glass toast-warning flex items-center gap-3">
      <AlertTriangle className="h-5 w-5 text-amber-500" />
      <span>{msg}</span>
    </div>,
    { duration }
  );

export const showError = (msg: string, duration = 5000) =>
  toast.custom(
    <div 
      role="alert"
      aria-live="assertive"
      aria-label={`Error notification of ${msg}`}

    className="toast-glass toast-error flex items-center gap-3">
      <XCircle className="h-5 w-5 text-rose-500" />
      <span>{msg}</span>
    </div>,
    { duration }
  );

export const showInfo = (msg: string, duration = 5000) =>
  toast.custom(
    <div
    role="status"
    aria-live="polite"
    aria-label={`Info notification of ${msg}`}

    className="toast-glass toast-info flex items-center gap-3">
      <XCircle className="h-5 w-5 text-sky-500" />
      <span>{msg}</span>
    </div>,
    { duration }
  );
