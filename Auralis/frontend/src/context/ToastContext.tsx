import React, { createContext, useContext, useState, useCallback } from 'react';
import { X, CheckCircle, AlertTriangle, Info, type LucideIcon } from 'lucide-react';

type ToastType = 'success' | 'error' | 'warning' | 'info';

interface Toast {
    id: number;
    message: string;
    type: ToastType;
}

interface ToastContextType {
    addToast: (message: string, type?: ToastType, duration?: number) => void;
}

const ToastContext = createContext<ToastContextType | null>(null);

const icons: Record<ToastType, LucideIcon> = {
    success: CheckCircle,
    error: AlertTriangle,
    warning: AlertTriangle,
    info: Info,
};

const colors: Record<ToastType, string> = {
    success: 'bg-emerald-500',
    error: 'bg-red-500',
    warning: 'bg-amber-500',
    info: 'bg-blue-500',
};

export function ToastProvider({ children }: { children: React.ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const removeToast = useCallback((id: number) => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
    }, []);

    const addToast = useCallback((message: string, type: ToastType = 'success', duration: number = 3500) => {
        const id = Date.now();
        setToasts((prev) => [...prev, { id, message, type }]);
        setTimeout(() => removeToast(id), duration);
    }, [removeToast]);

    return (
        <ToastContext.Provider value={{ addToast }}>
            {children}
            <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full px-4">
                {toasts.map((toast) => {
                    const Icon = icons[toast.type] || Info;
                    return (
                        <div
                            key={toast.id}
                            className={`flex items-start gap-3 ${colors[toast.type]} text-white rounded-xl px-4 py-3 shadow-lg text-sm font-medium animate-in slide-in-from-right`}
                            role="alert"
                        >
                            <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
                            <span className="flex-1">{toast.message}</span>
                            <button
                                onClick={() => removeToast(toast.id)}
                                className="flex-shrink-0 opacity-80 hover:opacity-100"
                                aria-label="Dismiss notification"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                    );
                })}
            </div>
        </ToastContext.Provider>
    );
}

export function useToast() {
    const ctx = useContext(ToastContext);
    if (!ctx) throw new Error('useToast must be used within ToastProvider');
    return ctx;
}