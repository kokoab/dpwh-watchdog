import { Navigate, useLocation } from "react-router-dom";
import { supabase } from '../../lib/supabase';
import { useAuth } from './UseAuth';

export function LoginPage () {
    const location = useLocation();
    const {isAuthenticated, isAuthLoading} = useAuth();

    const redirectTo = location.state?.from?.pathname  || "/chat";

    async function signInWithGoogle() {
        await supabase.auth.signInWithOAuth({
            provider: "google",

            options: {
                redirectTo: `${window.location.origin}${redirectTo}`,
            },
        });
    }

    if (isAuthLoading) return null;
    if (isAuthenticated) return <Navigate to={redirectTo} replace />

    return(
        <main className="auth-page"> 
            <section className="auth-panel">
                <h1>Sign in</h1>
                <button type="button" onClick={signInWithGoogle}> Continue with Google </button>
            </section>
        </main>
    )

}