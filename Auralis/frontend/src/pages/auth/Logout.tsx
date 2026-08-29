import { useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';
import * as authAPI from '@/services/auth_api';

export function Logout() {
    const { logout } = useAuth();
    
    useEffect(() => {
        const handleLogout = async () => {
            try {
                await authAPI.logout();
            }
            finally {
                logout();
            }
        };
        handleLogout();
    }, [logout]);

    return null;
}