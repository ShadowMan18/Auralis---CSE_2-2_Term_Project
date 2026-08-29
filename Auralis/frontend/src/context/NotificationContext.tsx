import React, { createContext, useContext } from 'react';

const NotificationContext = createContext<null>(null);

export function NotificationProvider({ children }: { children: React.ReactNode }) {
    return (
        <NotificationContext.Provider value={null}>
            {children}
        </NotificationContext.Provider>
    )
}

export function useTheme() {
    const ctx = useContext(NotificationContext);
    if (!ctx) throw new Error('useTheme must be used within NotificationProvider');
    return ctx;
}