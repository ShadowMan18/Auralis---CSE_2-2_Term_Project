import React, { createContext, useContext } from 'react';

const LanguageContext = createContext<null>(null);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
    return (
        <LanguageContext.Provider value={null}>
            {children}
        </LanguageContext.Provider>
    )
}

export function useLanguage() {
    const ctx = useContext(LanguageContext);
    if (!ctx) throw new Error('useTheme must be used within LanguageProvider');
    return ctx;
}