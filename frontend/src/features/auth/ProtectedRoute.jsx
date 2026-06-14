import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from './UseAuth';

export function ProtectedRoute({children}) {
    const location = useLocation();
    const {isAuthenticated, isAuthLoading} = useAuth();

    if (isAuthLoading) return "Loading...";

    if (!isAuthenticated){
        return <Navigate to="/login" replace state={{ from: location }} />;
    }

    return children;
}
