import { useAuth } from "@/context/AuthContext";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { Spinner } from "../ui/spinner";

export function RequireAuth() {
    const { user, loading } = useAuth();
    const location = useLocation();

    if (loading) return <Spinner />;
    if (!user) return <Navigate to='/auth/login' state={{ from: location }} replace />;
    return <Outlet />;
}