import { useAuth } from "@/context/AuthContext";
import { Navigate, Outlet } from "react-router-dom";

export function RequireRole({ allowed }: { allowed: string[] }) {
    const { user } = useAuth();
    const hasAccess = user && allowed.includes(user.user_type);

    if (!hasAccess) return <Navigate to='/unauthorized' replace />;
    return <Outlet />;
}