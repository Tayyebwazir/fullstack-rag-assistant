import React, { useState } from 'react';
import {
  auth,
  googleProvider,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
} from '../firebase';
import { X, Lock, Mail, AlertCircle, LogIn, UserPlus } from 'lucide-react';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose }) => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

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
      onClose();
    } catch (err: any) {
      const msg = err.message || 'Authentication failed.';
      if (msg.includes('auth/invalid-credential') || msg.includes('auth/wrong-password')) {
        setError('Incorrect email or password.');
      } else if (msg.includes('auth/email-already-in-use')) {
        setError('An account with this email already exists.');
      } else if (msg.includes('auth/weak-password')) {
        setError('Password should be at least 6 characters.');
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
      onClose();
    } catch (err: any) {
      setError(err.message || 'Google sign-in failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.75)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        backdropFilter: 'blur(4px)',
      }}
    >
      <div
        style={{
          backgroundColor: '#1E293B',
          border: '1px solid #334155',
          borderRadius: '12px',
          width: '100%',
          maxWidth: '420px',
          padding: '24px',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
          position: 'relative',
        }}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            background: 'transparent',
            border: 'none',
            color: '#94A3B8',
            cursor: 'pointer',
          }}
        >
          <X size={18} />
        </button>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '20px' }}>
          <div
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '10px',
              backgroundColor: '#3B82F6',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 12px',
              color: '#FFFFFF',
            }}
          >
            <Lock size={22} />
          </div>
          <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#F8FAFC', margin: '0 0 6px' }}>
            {isSignUp ? 'Create your Account' : 'Sign in to RAG Assistant'}
          </h2>
          <p style={{ fontSize: '12px', color: '#94A3B8', margin: 0 }}>
            Secure Firebase JWT Authentication
          </p>
        </div>

        {error && (
          <div
            style={{
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid #DC2626',
              borderRadius: '8px',
              padding: '10px',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              color: '#F87171',
              fontSize: '12px',
            }}
          >
            <AlertCircle size={16} />
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
            padding: '10px',
            borderRadius: '8px',
            border: '1px solid #334155',
            backgroundColor: '#0F172A',
            color: '#F8FAFC',
            fontSize: '13px',
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            marginBottom: '16px',
            transition: 'background-color 0.15s',
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
            margin: '16px 0',
            color: '#64748B',
            fontSize: '11px',
          }}
        >
          <div style={{ flex: 1, height: '1px', backgroundColor: '#334155' }} />
          <span style={{ padding: '0 10px' }}>OR WITH EMAIL</span>
          <div style={{ flex: 1, height: '1px', backgroundColor: '#334155' }} />
        </div>

        {/* Email & Password Form */}
        <form onSubmit={handleEmailAuth} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '11px', color: '#94A3B8', marginBottom: '4px' }}>
              Email address
            </label>
            <div
              className="auth-input-wrapper"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                borderRadius: '8px',
                padding: '8px 12px',
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
            <label style={{ display: 'block', fontSize: '11px', color: '#94A3B8', marginBottom: '4px' }}>
              Password
            </label>
            <div
              className="auth-input-wrapper"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                borderRadius: '8px',
                padding: '8px 12px',
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
              marginTop: '8px',
              width: '100%',
              padding: '10px',
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
            }}
          >
            {isSignUp ? <UserPlus size={16} /> : <LogIn size={16} />}
            {loading ? 'Processing...' : isSignUp ? 'Create Account' : 'Sign In'}
          </button>
        </form>

        {/* Toggle between Sign In and Sign Up */}
        <div style={{ marginTop: '16px', textAlign: 'center', fontSize: '12px', color: '#94A3B8' }}>
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

