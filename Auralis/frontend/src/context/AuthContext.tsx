import React, { createContext, useContext, useEffect, useState } from 'react';

interface UserInfo {
    user_type: 'client' | 'admin';
    user_id: number;
    name: string;
    email: string;
    profile_picture: string | null;
}

interface AuthContextType {
    user: UserInfo | null;
    login: (user: UserInfo, token: string) => void;
    logout: () => void;
    updateUser: (user_updates: Partial<UserInfo>) => void;
    loading: boolean
}

const authContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<UserInfo | null>(null);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {
        const stored = localStorage.getItem('user');
        if (stored) {
            try {
                setUser(JSON.parse(stored));
            }
            catch {
                localStorage.removeItem('user')
            }
        }
        setLoading(false);
    }, []);

    const login = (user: UserInfo, token: string) => {
        setUser(user);
        localStorage.setItem('user', JSON.stringify(user));
        if (token) localStorage.setItem('token', token);
    }

    const logout = () => {
        setUser(null);
        localStorage.clear();
    }

    const updateUser = (user_updates: Partial<UserInfo>) => {
        if (user) {
            const updated_user = { ...user, ...user_updates };
            setUser(updated_user);
            localStorage.setItem('user', JSON.stringify('user'))
        }
    }

    return (
        <authContext.Provider value={{ user, login, logout, updateUser, loading }}>
            {children}
        </authContext.Provider>
    )
}

export function useAuth() {
    const ctx = useContext(authContext);
    if (!ctx) throw new Error('useTheme must be used within AuthProvider');
    return ctx;
}