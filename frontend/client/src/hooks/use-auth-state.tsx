import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Define the shape of the authentication state
interface AuthState {
  isLoggedIn: boolean;
  userId: string | null;
  phoneNumber: string | null;
}

// Define the shape of the context object
type AuthContextType = {
  authState: AuthState;
  login: (userId: string, phoneNumber: string) => void;
  logout: () => void;
};

// Create a context with a default value
const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [authState, setAuthState] = useState<AuthState>(() => {
    // Initialize state from localStorage or default to logged out
    try {
      const storedAuth = localStorage.getItem('auth');
      if (storedAuth) {
        return JSON.parse(storedAuth);
      }
    } catch (error) {
      console.error("Failed to parse auth state from localStorage:", error);
    }
    return {
      isLoggedIn: false,
      userId: null,
      phoneNumber: null,
    };
  });

  // Effect to save authState to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('auth', JSON.stringify(authState));
  }, [authState]);

  // Login function
  const login = (userId: string, phoneNumber: string) => {
    setAuthState({
      isLoggedIn: true,
      userId,
      phoneNumber,
    });
  };

  // Logout function
  const logout = async () => {
    // Call backend logout endpoint
    try {
      await fetch('http://localhost:6900/logout', { method: 'POST', credentials: 'include' });
    } catch (error) {
      console.error('Error during backend logout:', error);
    }
    // Reset frontend auth state and clear localStorage
    setAuthState({
      isLoggedIn: false,
      userId: null,
      phoneNumber: null,
    });
    localStorage.removeItem('auth');
  };

  return (
    <AuthContext.Provider value={{ authState, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook to use the auth context
export function useAuthState() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuthState must be used within an AuthProvider');
  }
  return context;
}