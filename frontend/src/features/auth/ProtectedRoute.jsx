import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from './AuthProvider';

export function ProtectedRoute({children}) {
    const location = useLocation();
    const {isAuthenticated, isAuthLoading} = useAuth();

    if (isAuthLoading) return null;

    if (!isAuthenticated){
        return <Navigate to="/login" replace state={{ from: location }} />;
    }

    return children;
}
