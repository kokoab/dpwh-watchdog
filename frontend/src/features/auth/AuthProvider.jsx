import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { supabase } from '../../lib/supabase';

const AuthContext = createContext(null);

export function AuthProvider({children}) {
    const [session, setSession] = useState(null);
    const [user, setUser] = useState(null);
    const [isAuthLoading, setIsAuthLoading] = useState(true);

    useEffect(() => {
        let mounted = true;

        supabase.auth.getSession().then(({ data }) => {
            if (!mounted) return;
            setSession(data.session ?? null);
            setUser(data.session?.user ?? null);
            setIsAuthLoading(false);
        });

        const { data: subscription } = supabase.auth.onAuthStateChange((_event, nextSession) => {
            setSession(nextSession ?? null);
            setUser(nextSession?.user ?? null);
            setIsAuthLoading(false);
        })
        
        return () => {
            mounted = false;
            subscription.subscription.unsubscribe();
        };

    }, []);

    const value = useMemo(
        () => ({
            session,
            user,
            accessToken: session?.access_token ?? null,
            isAuthenticated: Boolean(user),
            isAuthLoading,
        }), [session, user, isAuthLoading]
    );
    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

