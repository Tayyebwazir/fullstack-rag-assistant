import React, { useState } from 'react';
import {
  auth,
  googleProvider,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
} from '../firebase';
import { Lock, Mail, AlertCircle, LogIn, UserPlus, Sparkles } from 'lucide-react';

export const AuthPage: React.FC = () => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please fill in both email and password.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
    } catch (err: any) {
      const msg = err.message || 'Authentication failed.';
      if (msg.includes('api-key-not-valid')) {
        setError(
          'Firebase API Key is invalid or missing. Please add your real Firebase credentials in frontend/.env file.'
        );
      } else if (msg.includes('auth/invalid-credential') || msg.includes('auth/wrong-password')) {
        setError('Incorrect email or password.');
      } else if (msg.includes('auth/email-already-in-use')) {
        setError('An account with this email already exists.');
      } else if (msg.includes('auth/weak-password')) {
        setError('Password should be at least 6 characters.');
      } else if (msg.includes('operation-not-allowed')) {
        setError(
          'Email/Password provider is disabled. Enable it in Firebase Console -> Authentication -> Sign-in method.'
        );
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setLoading(true);
    setError(null);
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err: any) {
      const msg = err.message || 'Google sign-in failed.';
      if (msg.includes('api-key-not-valid')) {
        setError(
          'Firebase API Key is invalid or missing. Please add your real Firebase credentials in frontend/.env file.'
        );
      } else if (msg.includes('operation-not-allowed')) {
        setError(
          'Google sign-in is disabled. Enable Google in Firebase Console -> Authentication -> Sign-in method.'
        );
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        width: '100vw',
        backgroundColor: '#0B1120',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Background Ambient Glows */}
      <div
        style={{
          position: 'absolute',
          top: '15%',
          left: '20%',
          width: '350px',
          height: '350px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: '15%',
          right: '20%',
          width: '350px',
          height: '350px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          backgroundColor: '#0F172A',
          border: '1px solid #1E293B',
          borderRadius: '16px',
          width: '100%',
          maxWidth: '440px',
          padding: '36px 32px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
          position: 'relative',
          zIndex: 10,
        }}
      >
        {/* Brand Header */}
        <div style={{ textAlign: 'center', marginBottom: '28px' }}>
          <div
            style={{
              width: '56px',
              height: '56px',
              borderRadius: '14px',
              background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px',
              boxShadow: '0 10px 15px -3px rgba(59, 130, 246, 0.4)',
              color: '#FFFFFF',
              fontSize: '24px',
            }}
          >
            📚
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: 700, color: '#F8FAFC', margin: '0 0 6px' }}>
            PDF RAG Assistant
          </h1>
          <p style={{ fontSize: '12px', color: '#64748B', margin: 0 }}>
            Sign in to access your Weaviate Cloud Vector Knowledge Base
          </p>
        </div>

        {error && (
          <div
            style={{
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid #DC2626',
              borderRadius: '8px',
              padding: '12px',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '10px',
              color: '#F87171',
              fontSize: '12px',
              lineHeight: 1.4,
            }}
          >
            <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '1px' }} />
            <span>{error}</span>
          </div>
        )}

        {/* 1-Click Google Sign In */}
        <button
          type="button"
          onClick={handleGoogleSignIn}
          disabled={loading}
          style={{
            width: '100%',
            padding: '12px',
            borderRadius: '10px',
            border: '1px solid #334155',
            backgroundColor: '#1E293B',
            color: '#F8FAFC',
            fontSize: '13px',
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            marginBottom: '20px',
            transition: 'all 0.15s',
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24">
            <path
              fill="#4285F4"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="#34A853"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="#FBBC05"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
            />
            <path
              fill="#EA4335"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
            />
          </svg>
          Continue with Google
        </button>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            margin: '20px 0',
            color: '#64748B',
            fontSize: '11px',
          }}
        >
          <div style={{ flex: 1, height: '1px', backgroundColor: '#334155' }} />
          <span style={{ padding: '0 10px', letterSpacing: '0.5px' }}>OR WITH EMAIL</span>
          <div style={{ flex: 1, height: '1px', backgroundColor: '#334155' }} />
        </div>

        {/* Email & Password Form */}
        <form onSubmit={handleEmailAuth} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '12px', color: '#94A3B8', marginBottom: '6px' }}>
              Email address
            </label>
            <div
              className="auth-input-wrapper"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                borderRadius: '8px',
                padding: '10px 14px',
              }}
            >
              <Mail size={16} color="#64748B" />
              <input
                type="email"
                className="auth-input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@domain.com"
                required
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', color: '#94A3B8', marginBottom: '6px' }}>
              Password
            </label>
            <div
              className="auth-input-wrapper"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                borderRadius: '8px',
                padding: '10px 14px',
              }}
            >
              <Lock size={16} color="#64748B" />
              <input
                type="password"
                className="auth-input"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              marginTop: '10px',
              width: '100%',
              padding: '12px',
              borderRadius: '8px',
              border: 'none',
              backgroundColor: '#3B82F6',
              color: '#FFFFFF',
              fontWeight: 600,
              fontSize: '13px',
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              boxShadow: '0 4px 6px -1px rgba(59, 130, 246, 0.4)',
            }}
          >
            {isSignUp ? <UserPlus size={16} /> : <LogIn size={16} />}
            {loading ? 'Please wait...' : isSignUp ? 'Create Account' : 'Sign In to Access RAG'}
          </button>
        </form>

        {/* Toggle between Sign In and Sign Up */}
        <div style={{ marginTop: '20px', textAlign: 'center', fontSize: '12px', color: '#94A3B8' }}>
          {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button
            type="button"
            onClick={() => {
              setIsSignUp(!isSignUp);
              setError(null);
            }}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#38BDF8',
              fontWeight: 600,
              cursor: 'pointer',
              textDecoration: 'underline',
              marginLeft: '4px',
            }}
          >
            {isSignUp ? 'Sign In' : 'Create One'}
          </button>
        </div>
      </div>
    </div>
  );
};

