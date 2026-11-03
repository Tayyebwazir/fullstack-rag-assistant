import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { auth, onAuthStateChanged, signOut, User } from './firebase';
import { AuthModal } from './components/AuthModal';
import { AuthPage } from './components/AuthPage';
import {
  FileText,
  Database,
  Send,
  Zap,
  CheckCircle,
  AlertCircle,
  Copy,
  Check,
  RefreshCw,
  Layers,
  Sparkles,
  Award,
  Hash,
  BookOpen,
  LogIn,
  LogOut,
  User as UserIcon,
  Building2,
  UploadCloud,
  Trash2,
  Edit3,
  ShieldCheck,
  PlusCircle,
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

interface Chunk {
  content: string;
  source: string;
  page: number;
  chunk_index: number;
  tenant?: string;
}

interface RerankedChunk extends Chunk {
  original_rank?: number;
  relevance_score?: number;
  rerank_position?: number;
}

interface TenantMetadata {
  id: string;
  name: string;
  document: string;
  description: string;
}

interface UserProfile {
  uid: string;
  email: string;
  tenant_id: string;
  permissions: string[];
}

interface WeaviateDocSummary {
  source: string;
  chunk_count: number;
  tenant: string;
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [streamedAnswer, setStreamedAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [tokenCount, setTokenCount] = useState(0);
  const [streamStartTime, setStreamStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0);

  const [retrievedChunks, setRetrievedChunks] = useState<Chunk[]>([]);
  const [rerankedChunks, setRerankedChunks] = useState<RerankedChunk[]>([]);

  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Multi-Tenancy & Authorization States
  const [tenants, setTenants] = useState<TenantMetadata[]>([]);
  const [selectedTenant, setSelectedTenant] = useState<string>('alpha');
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);

  // Document Operations States
  const [availableSources, setAvailableSources] = useState<string[]>([]);
  const [selectedSource, setSelectedSource] = useState<string>('all');
  const [tenantDocs, setTenantDocs] = useState<WeaviateDocSummary[]>([]);
  const [opLoading, setOpLoading] = useState(false);
  const [opStatus, setOpStatus] = useState<{ success: boolean; msg: string } | null>(null);

  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const updateInputRef = useRef<HTMLInputElement>(null);
  const [targetUpdateDoc, setTargetUpdateDoc] = useState<string>('');

  const answerRef = useRef<HTMLDivElement>(null);

  // Listen to Firebase Auth state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (usr) => {
      setCurrentUser(usr);
      setAuthLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // Check Backend health on load
  useEffect(() => {
    fetch(`${API_BASE}/`)
      .then((res) => {
        if (res.ok) setBackendOnline(true);
        else setBackendOnline(false);
      })
      .catch(() => setBackendOnline(false));

    fetch(`${API_BASE}/tenants`)
      .then((res) => res.json())
      .then((data) => {
        if (data.tenants) setTenants(data.tenants);
      })
      .catch(() => {});
  }, []);

  // Fetch User Profile from Firebase Database when authenticated
  const fetchUserProfile = async () => {
    if (!currentUser) return;
    try {
      const token = await currentUser.getIdToken();
      const res = await fetch(`${API_BASE}/user/profile`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok && data.profile) {
        setUserProfile(data.profile);
        setSelectedTenant(data.profile.tenant_id || 'alpha');
      }
    } catch {}
  };

  useEffect(() => {
    if (currentUser) {
      fetchUserProfile();
    }
  }, [currentUser]);

  // Fetch Documents for the active tenant
  const fetchTenantDocuments = async (tId: string) => {
    if (!currentUser) return;
    try {
      const token = await currentUser.getIdToken();
      const res = await fetch(`${API_BASE}/documents/list?tenant_id=${encodeURIComponent(tId)}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok) {
        if (data.weaviate_documents) setTenantDocs(data.weaviate_documents);
        if (data.available_files) setAvailableSources(data.available_files);
      }
    } catch {}
  };

  useEffect(() => {
    if (selectedTenant && currentUser) {
      fetchTenantDocuments(selectedTenant);
    }
  }, [selectedTenant, currentUser]);

  // Update timer while streaming
  useEffect(() => {
    let interval: any;
    if (isStreaming && streamStartTime) {
      interval = setInterval(() => {
        setElapsedTime(Math.round((Date.now() - streamStartTime) / 100) / 10);
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isStreaming, streamStartTime]);

  // Scroll to bottom of answer while streaming
  useEffect(() => {
    if (answerRef.current && isStreaming) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [streamedAnswer, isStreaming]);

  // 1. Switch User Tenant Assignment in Firebase Database
  const handleSelectTenant = async (newTenant: string) => {
    setSelectedTenant(newTenant);
    setOpStatus(null);
    try {
      const token = await currentUser?.getIdToken();
      const res = await fetch(`${API_BASE}/user/select-tenant`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ tenant_id: newTenant }),
      });
      const data = await res.json();
      if (res.ok && data.profile) {
        setUserProfile(data.profile);
        fetchTenantDocuments(newTenant);
      }
    } catch {}
  };

  // 2. Upload Document
  const handleUploadFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setOpLoading(true);
    setOpStatus(null);

    try {
      const token = await currentUser?.getIdToken();
      const formData = new FormData();
      formData.append('file', file);
      formData.append('tenant_id', selectedTenant);

      const res = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        setOpStatus({ success: true, msg: `Uploaded "${file.name}". Ready for ingestion!` });
        fetchTenantDocuments(selectedTenant);
      } else {
        setOpStatus({ success: false, msg: data.detail || 'Upload failed.' });
      }
    } catch (err: any) {
      setOpStatus({ success: false, msg: err.message || 'Upload error.' });
    } finally {
      setOpLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // 3. Ingest Document
  const handleIngestDocument = async (sourceName?: string) => {
    const targetSource = sourceName || (selectedSource !== 'all' ? selectedSource : availableSources[0]);
    if (!targetSource) {
      setOpStatus({ success: false, msg: 'Please select a document to ingest.' });
      return;
    }

    setOpLoading(true);
    setOpStatus(null);

    try {
      const token = await currentUser?.getIdToken();
      const res = await fetch(`${API_BASE}/documents/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ source: targetSource, tenant_id: selectedTenant }),
      });
      const data = await res.json();

      if (res.ok) {
        setOpStatus({
          success: true,
          msg: `Ingested ${data.chunks_ingested} chunks of "${targetSource}" into tenant [${selectedTenant.toUpperCase()}]!`,
        });
        fetchTenantDocuments(selectedTenant);
      } else {
        setOpStatus({ success: false, msg: data.detail || 'Ingestion failed.' });
      }
    } catch (err: any) {
      setOpStatus({ success: false, msg: err.message || 'Ingestion error.' });
    } finally {
      setOpLoading(false);
    }
  };

  // 4. Update Document
  const handleUpdateFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !targetUpdateDoc) return;

    setOpLoading(true);
    setOpStatus(null);

    try {
      const token = await currentUser?.getIdToken();
      const formData = new FormData();
      formData.append('file', file);
      formData.append('tenant_id', selectedTenant);

      const res = await fetch(`${API_BASE}/documents/update`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        setOpStatus({
          success: true,
          msg: `Updated and re-embedded "${file.name}" with ${data.chunks_ingested} chunks in tenant [${selectedTenant.toUpperCase()}].`,
        });
        fetchTenantDocuments(selectedTenant);
      } else {
        setOpStatus({ success: false, msg: data.detail || 'Update failed.' });
      }
    } catch (err: any) {
      setOpStatus({ success: false, msg: err.message || 'Update error.' });
    } finally {
      setOpLoading(false);
      setTargetUpdateDoc('');
      if (updateInputRef.current) updateInputRef.current.value = '';
    }
  };

  // 5. Delete Document
  const handleDeleteDocument = async (sourceName: string) => {
    if (!window.confirm(`Are you sure you want to delete "${sourceName}" from tenant [${selectedTenant}]?`)) {
      return;
    }

    setOpLoading(true);
    setOpStatus(null);

    try {
      const token = await currentUser?.getIdToken();
      const res = await fetch(`${API_BASE}/documents/delete`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ source: sourceName, tenant_id: selectedTenant }),
      });
      const data = await res.json();

      if (res.ok) {
        setOpStatus({
          success: true,
          msg: `Deleted "${sourceName}" chunks from tenant [${selectedTenant.toUpperCase()}].`,
        });
        fetchTenantDocuments(selectedTenant);
      } else {
        setOpStatus({ success: false, msg: data.detail || 'Delete failed.' });
      }
    } catch (err: any) {
      setOpStatus({ success: false, msg: err.message || 'Delete error.' });
    } finally {
      setOpLoading(false);
    }
  };

  // Handle Ask with SSE Streaming & Tenant Authorisation
  const handleAsk = async (queryText?: string) => {
    const q = (queryText || question).trim();
    if (!q || isStreaming) return;

    setErrorMsg(null);
    setStreamedAnswer('');
    setTokenCount(0);
    setRetrievedChunks([]);
    setRerankedChunks([]);
    setIsStreaming(true);
    setStreamStartTime(Date.now());
    setElapsedTime(0);

    try {
      const token = await currentUser?.getIdToken();
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = `Bearer ${token}`;

      const response = await fetch(`${API_BASE}/ask-stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          question: q,
          source_filter: selectedSource === 'all' ? null : selectedSource,
          tenant_id: selectedTenant,
        }),
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => null);
        throw new Error(errJson?.detail || `Server returned status ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');

      if (!reader) throw new Error('ReadableStream not supported.');

      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data:')) {
            const jsonStr = trimmed.replace(/^data:\s*/, '');
            try {
              const event = JSON.parse(jsonStr);

              if (event.type === 'metadata') {
                if (event.chunks) setRetrievedChunks(event.chunks);
                if (event.reranked) setRerankedChunks(event.reranked);
              } else if (event.type === 'token') {
                setStreamedAnswer((prev) => prev + event.token);
                setTokenCount((prev) => prev + 1);
              } else if (event.type === 'error') {
                setErrorMsg(event.error);
                setIsStreaming(false);
              } else if (event.type === 'done') {
                setIsStreaming(false);
              }
            } catch {
              // Ignore fragmented SSE
            }
          }
        }
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Query failed. Please check tenant authorisation.');
    } finally {
      setIsStreaming(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(streamedAnswer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (authLoading) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          backgroundColor: '#0B1120',
          color: '#94A3B8',
          gap: '12px',
        }}
      >
        <RefreshCw size={26} className="pulsing" color="#3B82F6" />
        <span style={{ fontSize: '13px' }}>Initializing Authentication...</span>
      </div>
    );
  }

  if (!currentUser) {
    return <AuthPage />;
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        backgroundColor: '#0B1120',
        color: '#F8FAFC',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      }}
    >
      {/* Hidden File Inputs for Document CRUD */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleUploadFile}
        accept=".pdf"
        style={{ display: 'none' }}
      />
      <input
        type="file"
        ref={updateInputRef}
        onChange={handleUpdateFile}
        accept=".pdf"
        style={{ display: 'none' }}
      />

      {/* Top Header Navbar */}
      <header
        style={{
          height: '56px',
          borderBottom: '1px solid #1E293B',
          backgroundColor: '#0F172A',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 20px',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              backgroundColor: '#3B82F6',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#FFFFFF',
              fontWeight: 700,
            }}
          >
            📚
          </div>
          <div>
            <h1 style={{ fontSize: '15px', fontWeight: 700, color: '#F8FAFC', margin: 0 }}>
              Multi-Tenant PDF RAG Assistant
            </h1>
            <p style={{ fontSize: '11px', color: '#64748B', margin: 0 }}>
              FastAPI Auth • Firebase Database (Users & Permissions) • Weaviate Cloud Shards
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '3px 9px',
              borderRadius: '20px',
              fontSize: '11.5px',
              backgroundColor: backendOnline ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              color: backendOnline ? '#34D399' : '#F87171',
              border: `1px solid ${backendOnline ? '#059669' : '#DC2626'}`,
            }}
          >
            <div
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: backendOnline ? '#10B981' : '#EF4444',
              }}
            />
            {backendOnline ? 'Backend Online' : 'Backend Offline'}
          </div>

          <div
            style={{
              padding: '3px 9px',
              borderRadius: '20px',
              fontSize: '11px',
              backgroundColor: '#1E293B',
              color: '#94A3B8',
              border: '1px solid #334155',
            }}
          >
            Model: <strong style={{ color: '#E2E8F0' }}>gemini-3.5-flash-lite</strong>
          </div>

          {/* User Auth Profile Bar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '3px 10px',
              borderRadius: '20px',
              backgroundColor: '#1E293B',
              border: '1px solid #334155',
            }}
          >
            <div
              style={{
                width: '22px',
                height: '22px',
                borderRadius: '50%',
                backgroundColor: '#3B82F6',
                color: '#FFFFFF',
                fontSize: '11px',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {currentUser.email ? currentUser.email.charAt(0).toUpperCase() : 'U'}
            </div>
            <span style={{ fontSize: '11.5px', color: '#F1F5F9', maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {currentUser.email || currentUser.uid}
            </span>
            <button
              onClick={() => signOut(auth)}
              title="Sign Out"
              style={{
                background: 'transparent',
                border: 'none',
                color: '#94A3B8',
                cursor: 'pointer',
                padding: '2px',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <LogOut size={13} />
            </button>
          </div>
        </div>
      </header>

      {/* Main 3-Column Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr 340px', flex: 1, overflow: 'hidden' }}>
        
        {/* ======================================================== */}
        {/* COLUMN 1: FIREBASE AUTHORIZATION, CRUD & INGESTION */}
        {/* ======================================================== */}
        <aside
          style={{
            borderRight: '1px solid #1E293B',
            backgroundColor: '#0F172A',
            display: 'flex',
            flexDirection: 'column',
            overflowY: 'auto',
            padding: '16px',
            gap: '14px',
          }}
        >
          {/* 1. Firebase Database Tenant Authorisation Card */}
          <div
            style={{
              backgroundColor: '#1E293B',
              borderRadius: '10px',
              padding: '14px',
              border: '1px solid #334155',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <ShieldCheck size={16} color="#10B981" />
              <h2 style={{ fontSize: '13px', fontWeight: 600, color: '#F1F5F9', margin: 0 }}>
                Firebase Authorisation
              </h2>
            </div>
            <p style={{ fontSize: '11px', color: '#94A3B8', margin: '0 0 10px 0' }}>
              User ID, Tenant assignment, and permissions verified via Firebase Database.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11.5px' }}>
                <span style={{ color: '#94A3B8' }}>Active Tenant:</span>
                <select
                  value={selectedTenant}
                  onChange={(e) => handleSelectTenant(e.target.value)}
                  style={{
                    backgroundColor: '#0B1120',
                    color: '#38BDF8',
                    border: '1px solid #38BDF8',
                    borderRadius: '6px',
                    padding: '3px 8px',
                    fontSize: '11.5px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    outline: 'none',
                  }}
                >
                  <option value="alpha">Tenant Alpha</option>
                  <option value="beta">Tenant Beta</option>
                  <option value="tri">Tenant Tri</option>
                </select>
              </div>

              {/* Granted Permissions Pills */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
                {(userProfile?.permissions || ['read', 'write', 'update', 'delete', 'ingest']).map((perm) => (
                  <span
                    key={perm}
                    style={{
                      fontSize: '9.5px',
                      textTransform: 'uppercase',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      backgroundColor: 'rgba(56, 189, 248, 0.1)',
                      color: '#38BDF8',
                      border: '1px solid rgba(56, 189, 248, 0.3)',
                      fontWeight: 600,
                    }}
                  >
                    ✓ {perm}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* 2. Document Operations (Upload, Ingest, Update, Delete) */}
          <div
            style={{
              backgroundColor: '#1E293B',
              borderRadius: '10px',
              padding: '14px',
              border: '1px solid #334155',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Database size={16} color="#3B82F6" />
                <h2 style={{ fontSize: '13px', fontWeight: 600, color: '#F1F5F9', margin: 0 }}>
                  Document Operations
                </h2>
              </div>
              <span style={{ fontSize: '10px', color: '#94A3B8', backgroundColor: '#0B1120', padding: '2px 6px', borderRadius: '4px' }}>
                Shard: [{selectedTenant.toUpperCase()}]
              </span>
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '10px' }}>
              {/* Add / Upload Document */}
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={opLoading}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  backgroundColor: '#0F172A',
                  color: '#38BDF8',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  padding: '8px 10px',
                  fontSize: '11.5px',
                  fontWeight: 600,
                  cursor: opLoading ? 'not-allowed' : 'pointer',
                }}
              >
                <PlusCircle size={14} /> Add PDF
              </button>

              {/* Ingest Selected Document */}
              <button
                onClick={() => handleIngestDocument()}
                disabled={opLoading}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  backgroundColor: '#2563EB',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '8px 10px',
                  fontSize: '11.5px',
                  fontWeight: 600,
                  cursor: opLoading ? 'not-allowed' : 'pointer',
                }}
              >
                {opLoading ? <RefreshCw size={13} className="pulsing" /> : <Zap size={13} />} Ingest
              </button>
            </div>

            {/* Operation Feedback Message */}
            {opStatus && (
              <div
                style={{
                  marginBottom: '10px',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  fontSize: '11px',
                  backgroundColor: opStatus.success ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                  color: opStatus.success ? '#34D399' : '#F87171',
                  border: `1px solid ${opStatus.success ? '#059669' : '#DC2626'}`,
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '6px',
                }}
              >
                {opStatus.success ? <CheckCircle size={14} style={{ flexShrink: 0, marginTop: '1px' }} /> : <AlertCircle size={14} style={{ flexShrink: 0, marginTop: '1px' }} />}
                <span>{opStatus.msg}</span>
              </div>
            )}

            {/* Shard Stored Documents List */}
            <div style={{ marginTop: '8px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>
                Documents in Tenant Shard:
              </div>

              {tenantDocs.length === 0 ? (
                <div
                  style={{
                    padding: '12px',
                    textAlign: 'center',
                    fontSize: '11px',
                    color: '#64748B',
                    border: '1px dashed #334155',
                    borderRadius: '6px',
                  }}
                >
                  No documents ingested yet in tenant [{selectedTenant}]. Click "Add PDF" or "Ingest" above.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {tenantDocs.map((doc, idx) => (
                    <div
                      key={idx}
                      style={{
                        backgroundColor: '#0F172A',
                        padding: '8px 10px',
                        borderRadius: '6px',
                        border: '1px solid #334155',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div style={{ overflow: 'hidden', marginRight: '6px' }}>
                        <div style={{ fontSize: '11.5px', color: '#F1F5F9', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          📄 {doc.source}
                        </div>
                        <div style={{ fontSize: '10px', color: '#64748B' }}>
                          {doc.chunk_count} chunks • Shard {doc.tenant}
                        </div>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        {/* Update Button */}
                        <button
                          onClick={() => {
                            setTargetUpdateDoc(doc.source);
                            updateInputRef.current?.click();
                          }}
                          title="Update & Re-embed document"
                          style={{
                            backgroundColor: 'transparent',
                            border: '1px solid #334155',
                            color: '#38BDF8',
                            padding: '4px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                          }}
                        >
                          <Edit3 size={11} />
                        </button>

                        {/* Delete Button */}
                        <button
                          onClick={() => handleDeleteDocument(doc.source)}
                          title="Delete from Weaviate Shard"
                          style={{
                            backgroundColor: 'transparent',
                            border: '1px solid #334155',
                            color: '#F87171',
                            padding: '4px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                          }}
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 3. Retrieved Chunks Header & List */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Layers size={14} color="#8B5CF6" />
              <h2 style={{ fontSize: '13px', fontWeight: 600, color: '#F1F5F9', margin: 0 }}>
                Retrieved Chunks
              </h2>
            </div>
            <span
              style={{
                fontSize: '11px',
                backgroundColor: '#1E293B',
                color: '#94A3B8',
                padding: '2px 8px',
                borderRadius: '12px',
                border: '1px solid #334155',
              }}
            >
              {retrievedChunks.length} chunks
            </span>
          </div>

          {/* Chunks Cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', flex: 1 }}>
            {retrievedChunks.length === 0 ? (
              <div
                style={{
                  padding: '16px 12px',
                  textAlign: 'center',
                  color: '#64748B',
                  fontSize: '11.5px',
                  border: '1px dashed #334155',
                  borderRadius: '8px',
                }}
              >
                Ask a question to see raw chunks retrieved from tenant [{selectedTenant}].
              </div>
            ) : (
              retrievedChunks.map((chunk, idx) => (
                <div
                  key={idx}
                  style={{
                    backgroundColor: '#1E293B',
                    borderRadius: '8px',
                    padding: '10px 12px',
                    border: '1px solid #334155',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{ fontSize: '11px', fontWeight: 600, color: '#38BDF8' }}>
                      Chunk #{idx + 1}
                    </span>
                    <span style={{ fontSize: '10px', backgroundColor: '#312E81', color: '#A5B4FC', padding: '1px 5px', borderRadius: '4px' }}>
                      Tenant: {chunk.tenant || selectedTenant}
                    </span>
                    <span style={{ fontSize: '10px', color: '#94A3B8' }}>
                      Page {chunk.page}
                    </span>
                  </div>

                  <div style={{ fontSize: '10px', color: '#64748B', marginBottom: '6px' }}>
                    📄 {chunk.source}
                  </div>

                  <p style={{ fontSize: '11.5px', color: '#CBD5E1', lineHeight: 1.5, margin: 0 }}>
                    {chunk.content.length > 220 ? `${chunk.content.substring(0, 220)}...` : chunk.content}
                  </p>
                </div>
              ))
            )}
          </div>
        </aside>

        {/* ======================================================== */}
        {/* COLUMN 2: STREAMING QUESTION & ANSWER (MIDDLE) */}
        {/* ======================================================== */}
        <main
          style={{
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: '#0B1120',
            overflow: 'hidden',
            padding: '20px 24px',
            gap: '14px',
          }}
        >
          {/* Multi-Tenancy Workspace & Document Scope Selectors */}
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            {/* Multi-Tenancy Workspace Selector */}
            <div
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                backgroundColor: '#1E293B',
                border: '1px solid #334155',
                borderRadius: '8px',
                padding: '8px 12px',
              }}
            >
              <Building2 size={16} color="#A78BFA" style={{ flexShrink: 0 }} />
              <span style={{ fontSize: '12px', color: '#94A3B8', whiteSpace: 'nowrap' }}>
                Tenant:
              </span>
              <select
                value={selectedTenant}
                onChange={(e) => handleSelectTenant(e.target.value)}
                disabled={isStreaming}
                style={{
                  flex: 1,
                  backgroundColor: '#0F172A',
                  border: '1px solid #334155',
                  color: '#F8FAFC',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  fontSize: '12px',
                  outline: 'none',
                  cursor: 'pointer',
                }}
              >
                <option value="alpha">🏢 Tenant Alpha (Attention is all you need)</option>
                <option value="beta">🏢 Tenant Beta (LLM Autonomous Agents)</option>
                <option value="tri">🏢 Tenant Tri (OOP 15-Page Notes)</option>
              </select>
            </div>

            {/* Document Filter Dropdown */}
            <div
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                backgroundColor: '#1E293B',
                border: '1px solid #334155',
                borderRadius: '8px',
                padding: '8px 12px',
              }}
            >
              <BookOpen size={16} color="#38BDF8" style={{ flexShrink: 0 }} />
              <span style={{ fontSize: '12px', color: '#94A3B8', whiteSpace: 'nowrap' }}>
                Query Scope:
              </span>
              <select
                value={selectedSource}
                onChange={(e) => setSelectedSource(e.target.value)}
                disabled={isStreaming}
                style={{
                  flex: 1,
                  backgroundColor: '#0F172A',
                  border: '1px solid #334155',
                  color: '#F8FAFC',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  fontSize: '12px',
                  outline: 'none',
                  cursor: 'pointer',
                }}
              >
                <option value="all">📚 All Tenant Documents</option>
                {availableSources.map((src, i) => (
                  <option key={i} value={src}>
                    📄 {src}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Question Input Box */}
          <div
            style={{
              backgroundColor: '#1E293B',
              borderRadius: '10px',
              padding: '12px',
              border: '1px solid #334155',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}
          >
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  handleAsk();
                }
              }}
              placeholder={`Ask anything from tenant [${selectedTenant.toUpperCase()}]... (Press Ctrl + Enter)`}
              style={{
                width: '100%',
                height: '70px',
                backgroundColor: 'transparent',
                border: 'none',
                color: '#F8FAFC',
                fontSize: '13px',
                resize: 'none',
                outline: 'none',
                fontFamily: 'inherit',
              }}
            />

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '11px', color: '#64748B' }}>
                Tenant Shard: <strong style={{ color: '#38BDF8' }}>{selectedTenant}</strong>
              </span>

              <button
                onClick={() => handleAsk()}
                disabled={isStreaming || !question.trim()}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  backgroundColor: isStreaming || !question.trim() ? '#334155' : '#3B82F6',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '8px 16px',
                  fontSize: '12.5px',
                  fontWeight: 600,
                  cursor: isStreaming || !question.trim() ? 'not-allowed' : 'pointer',
                }}
              >
                {isStreaming ? (
                  <>
                    <RefreshCw size={14} className="pulsing" /> Generating...
                  </>
                ) : (
                  <>
                    <Send size={14} /> Ask Question
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Streaming Answer Box */}
          <div
            style={{
              flex: 1,
              backgroundColor: '#1E293B',
              borderRadius: '10px',
              border: '1px solid #334155',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Answer Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '10px 16px',
                borderBottom: '1px solid #334155',
                backgroundColor: '#0F172A',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sparkles size={16} color="#38BDF8" />
                <h3 style={{ fontSize: '13px', fontWeight: 600, color: '#F1F5F9', margin: 0 }}>
                  Streaming Answer
                </h3>
                {isStreaming && (
                  <span
                    style={{
                      fontSize: '11px',
                      color: '#34D399',
                      backgroundColor: 'rgba(16, 185, 129, 0.1)',
                      padding: '2px 8px',
                      borderRadius: '12px',
                    }}
                  >
                    Live Token Stream...
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                {tokenCount > 0 && (
                  <span style={{ fontSize: '11px', color: '#94A3B8' }}>
                    {tokenCount} tokens • {elapsedTime}s
                  </span>
                )}

                {streamedAnswer && (
                  <button
                    onClick={copyToClipboard}
                    style={{
                      background: 'transparent',
                      border: '1px solid #334155',
                      color: '#94A3B8',
                      borderRadius: '4px',
                      padding: '4px 8px',
                      fontSize: '11px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    {copied ? <Check size={12} color="#10B981" /> : <Copy size={12} />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                )}
              </div>
            </div>

            {/* Answer Content */}
            <div
              ref={answerRef}
              style={{
                flex: 1,
                padding: '18px',
                overflowY: 'auto',
                fontSize: '13.5px',
                lineHeight: 1.65,
                color: '#E2E8F0',
                fontFamily: 'inherit',
              }}
            >
              {errorMsg ? (
                <div style={{ color: '#F87171', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <AlertCircle size={16} /> {errorMsg}
                </div>
              ) : streamedAnswer ? (
                <div className="markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {streamedAnswer.replace(/\[source:[^\]]*\]/gi, '').trim()}
                  </ReactMarkdown>
                  {isStreaming && <span className="cursor-blink" />}
                </div>
              ) : (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                    color: '#64748B',
                    gap: '8px',
                  }}
                >
                  <Sparkles size={28} color="#334155" />
                  <p style={{ margin: 0, fontSize: '13px' }}>
                    Answers scoped to tenant [{selectedTenant}] will stream here in real-time.
                  </p>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* ======================================================== */}
        {/* COLUMN 3: RERANKER (RIGHT) */}
        {/* ======================================================== */}
        <aside
          style={{
            borderLeft: '1px solid #1E293B',
            backgroundColor: '#0F172A',
            display: 'flex',
            flexDirection: 'column',
            overflowY: 'auto',
            padding: '16px',
            gap: '14px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Award size={15} color="#10B981" />
              <h2 style={{ fontSize: '13px', fontWeight: 600, color: '#F1F5F9', margin: 0 }}>
                Reranker
              </h2>
            </div>
            <span
              style={{
                fontSize: '10px',
                padding: '2px 8px',
                borderRadius: '12px',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                color: '#34D399',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                fontWeight: 600,
              }}
            >
              Gemini Reranker
            </span>
          </div>

          <p style={{ fontSize: '11px', color: '#94A3B8', margin: 0 }}>
            Re-scores retrieved chunks by semantic relevance to ensure top-quality context.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', flex: 1 }}>
            {rerankedChunks.length === 0 ? (
              <div
                style={{
                  padding: '16px 12px',
                  textAlign: 'center',
                  color: '#64748B',
                  fontSize: '11.5px',
                  border: '1px dashed #334155',
                  borderRadius: '8px',
                }}
              >
                Reranked results will appear here after vector search in tenant [{selectedTenant}].
              </div>
            ) : (
              rerankedChunks.map((chunk, idx) => (
                <div
                  key={idx}
                  style={{
                    backgroundColor: '#1E293B',
                    borderRadius: '8px',
                    padding: '10px 12px',
                    border: '1px solid #334155',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{ fontSize: '11.5px', fontWeight: 700, color: '#10B981' }}>
                      Rank #{chunk.rerank_position || idx + 1}
                    </span>
                    <span style={{ fontSize: '10px', color: '#94A3B8' }}>
                      Orig: #{chunk.original_rank}
                    </span>
                    <span style={{ fontSize: '10px', color: '#38BDF8', fontWeight: 600 }}>
                      Score: {chunk.relevance_score?.toFixed(1) || 'N/A'}/10
                    </span>
                  </div>

                  <p style={{ fontSize: '11px', color: '#CBD5E1', lineHeight: 1.45, margin: 0 }}>
                    {chunk.content.length > 180 ? `${chunk.content.substring(0, 180)}...` : chunk.content}
                  </p>
                </div>
              ))
            )}
          </div>
        </aside>
      </div>

      <AuthModal isOpen={authModalOpen} onClose={() => setAuthModalOpen(false)} />
    </div>
  );
}
