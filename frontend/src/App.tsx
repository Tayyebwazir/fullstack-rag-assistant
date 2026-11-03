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
  LogOut,
  User as UserIcon,
  Building2,
  UploadCloud,
  Trash2,
  ShieldCheck,
  ChevronDown,
  X,
  FileUp,
  SlidersHorizontal,
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

interface Chunk {
  content: string;
  source: string;
  page: number;
  chunk_index: number;
  doc_id?: string;
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
  description: string;
}

interface UserProfile {
  uid: string;
  user_uuid?: string;
  email: string;
  tenant_id: string;
  is_active?: boolean;
  permissions: string[];
}

interface TenantDocument {
  doc_id: string;
  tenant_id: string;
  owner_uid: string;
  filename: string;
  file_type: string;
  file_size: number;
  upload_timestamp: string;
  updated_timestamp?: string;
  version: number;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  chunk_count: number;
  error_message?: string;
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
  const [tenants, setTenants] = useState<TenantMetadata[]>([
    { id: 'alpha', name: 'Tenant Alpha', description: 'Tenant Alpha Knowledge Base' },
    { id: 'beta', name: 'Tenant Beta', description: 'Tenant Beta Knowledge Base' },
    { id: 'tri', name: 'Tenant Tri', description: 'Tenant Tri Knowledge Base' },
  ]);
  const [selectedTenant, setSelectedTenant] = useState<string>('alpha');
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);

  // Tenant Documents
  const [documents, setDocuments] = useState<TenantDocument[]>([]);
  const [selectedSourceFilter, setSelectedSourceFilter] = useState<string>('all');
  const [docsLoading, setDocsLoading] = useState(false);

  // Manage Documents Dropdown & Modals State
  const [isManageMenuOpen, setIsManageMenuOpen] = useState(false);
  const [activeModal, setActiveModal] = useState<'update' | 'delete' | 'ingest' | null>(null);
  const [selectedDocForAction, setSelectedDocForAction] = useState<TenantDocument | null>(null);

  // Operation Feedback
  const [opLoading, setOpLoading] = useState(false);
  const [opStatus, setOpStatus] = useState<{ success: boolean; msg: string } | null>(null);

  // Auth States
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const updateFileInputRef = useRef<HTMLInputElement>(null);
  const manageMenuRef = useRef<HTMLDivElement>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  // Listen to Firebase Auth state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (usr) => {
      setCurrentUser(usr);
      setAuthLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // Close Manage Menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (manageMenuRef.current && !manageMenuRef.current.contains(event.target as Node)) {
        setIsManageMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Check Backend health on load
  useEffect(() => {
    fetch(`${API_BASE}/`)
      .then((res) => setBackendOnline(res.ok))
      .catch(() => setBackendOnline(false));
  }, []);

  // Fetch User Profile from Firebase Database
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
        if (data.available_tenants) setTenants(data.available_tenants);
      }
    } catch {}
  };

  useEffect(() => {
    if (currentUser) {
      fetchUserProfile();
    }
  }, [currentUser]);

  // Fetch Documents for the active tenant
  const fetchTenantDocuments = async () => {
    if (!currentUser) return;
    setDocsLoading(true);
    try {
      const token = await currentUser.getIdToken();
      const res = await fetch(`${API_BASE}/documents/list`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok && data.documents) {
        setDocuments(data.documents);
      } else {
        setDocuments([]);
      }
    } catch {
      setDocuments([]);
    } finally {
      setDocsLoading(false);
    }
  };

  useEffect(() => {
    if (currentUser && selectedTenant) {
      fetchTenantDocuments();
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
    setSelectedSourceFilter('all');
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
        fetchTenantDocuments();
      }
    } catch {}
  };

  // 2. Action: Add Document (Upload PDF, DOCX, DOC)
  const handleUploadFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsManageMenuOpen(false);
    setOpLoading(true);
    setOpStatus(null);

    try {
      const token = await currentUser?.getIdToken();
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        setOpStatus({
          success: true,
          msg: `Uploaded "${file.name}". You can now ingest it through the pipeline.`,
        });
        fetchTenantDocuments();
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

  // 3. Action: Ingest Document
  const handleIngestDocument = async (docId: string) => {
    setActiveModal(null);
    setOpLoading(true);
    setOpStatus(null);

    // Optimistically update document status to processing
    setDocuments((prev) =>
      prev.map((d) => (d.doc_id === docId ? { ...d, status: 'processing' } : d))
    );

    try {
      const token = await currentUser?.getIdToken();
      const res = await fetch(`${API_BASE}/documents/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ doc_id: docId }),
      });
      const data = await res.json();

      if (res.ok) {
        setOpStatus({
          success: true,
          msg: `Ingested ${data.chunks_ingested} chunks into Tenant [${selectedTenant.toUpperCase()}]!`,
        });
        fetchTenantDocuments();
      } else {
        setOpStatus({ success: false, msg: data.detail || 'Ingestion failed.' });
        fetchTenantDocuments();
      }
    } catch (err: any) {
      setOpStatus({ success: false, msg: err.message || 'Ingestion error.' });
      fetchTenantDocuments();
    } finally {
      setOpLoading(false);
    }
  };

  // 4. Action: Update Document
  const handleUpdateDocumentSubmit = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !selectedDocForAction) return;

    setActiveModal(null);
    setOpLoading(true);
    setOpStatus(null);

    try {
      const token = await currentUser?.getIdToken();
      const formData = new FormData();
      formData.append('doc_id', selectedDocForAction.doc_id);
      formData.append('file', file);

      const res = await fetch(`${API_BASE}/documents/update`, {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        setOpStatus({
          success: true,
          msg: `Updated "${file.name}" to version ${data.document?.version || 2} with ${data.document?.chunk_count || 0} chunks!`,
        });
        fetchTenantDocuments();
      } else {
        setOpStatus({ success: false, msg: data.detail || 'Update failed.' });
      }
    } catch (err: any) {
      setOpStatus({ success: false, msg: err.message || 'Update error.' });
    } finally {
      setOpLoading(false);
      setSelectedDocForAction(null);
      if (updateFileInputRef.current) updateFileInputRef.current.value = '';
    }
  };

  // 5. Action: Delete Document
  const handleDeleteDocument = async (docId: string) => {
    setActiveModal(null);
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
        body: JSON.stringify({ doc_id: docId }),
      });
      const data = await res.json();

      if (res.ok) {
        setOpStatus({
          success: true,
          msg: `Permanently deleted document and its vector chunks from Tenant [${selectedTenant.toUpperCase()}].`,
        });
        fetchTenantDocuments();
      } else {
        setOpStatus({ success: false, msg: data.detail || 'Delete failed.' });
      }
    } catch (err: any) {
      setOpStatus({ success: false, msg: err.message || 'Delete error.' });
    } finally {
      setOpLoading(false);
      setSelectedDocForAction(null);
    }
  };

  // Handle Ask with SSE Streaming & Tenant Scoping
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
          source_filter: selectedSourceFilter === 'all' ? null : selectedSourceFilter,
          tenant_id: selectedTenant,
        }),
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => null);
        throw new Error(errJson?.detail || `Server returned status ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error('No readable stream available');

      let accumulated = '';
      let tokens = 0;
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;
            try {
              const parsed = JSON.parse(jsonStr);
              if (parsed.type === 'metadata') {
                if (parsed.chunks) setRetrievedChunks(parsed.chunks);
                if (parsed.reranked) setRerankedChunks(parsed.reranked);
              } else if (parsed.type === 'token') {
                accumulated += parsed.token;
                tokens += 1;
                setStreamedAnswer(accumulated);
                setTokenCount(tokens);
              } else if (parsed.type === 'error') {
                setErrorMsg(parsed.error);
              } else if (parsed.type === 'done') {
                setIsStreaming(false);
              }
            } catch {}
          }
        }
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error fetching answer');
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleAsk();
    }
  };

  const handleCopy = () => {
    if (!streamedAnswer) return;
    navigator.clipboard.writeText(streamedAnswer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatDate = (isoString: string) => {
    if (!isoString) return '';
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  if (authLoading) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0B1120', color: '#38BDF8' }}>
        <RefreshCw size={28} className="spin" />
        <span style={{ marginLeft: '12px', fontSize: '15px' }}>Loading Multi-Tenant Workspace...</span>
      </div>
    );
  }

  if (!currentUser) {
    return <AuthPage />;
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#0B1120', color: '#F8FAFC', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Hidden file inputs for document operations */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleUploadFile}
        accept=".pdf,.docx,.doc"
        style={{ display: 'none' }}
      />
      <input
        type="file"
        ref={updateFileInputRef}
        onChange={handleUpdateDocumentSubmit}
        accept=".pdf,.docx,.doc"
        style={{ display: 'none' }}
      />

      {/* TOP HEADER */}
      <header style={{ height: '62px', borderBottom: '1px solid #1E293B', backgroundColor: '#0F172A', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '38px', height: '38px', borderRadius: '10px', background: 'linear-gradient(135deg, #0284C7, #2563EB)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 12px rgba(2, 132, 199, 0.3)' }}>
            <Database size={20} color="#FFFFFF" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontWeight: 700, fontSize: '16px', letterSpacing: '-0.3px', color: '#F8FAFC' }}>
                Multi-Tenant RAG Assistant
              </span>
              <span style={{ fontSize: '10px', backgroundColor: '#1E293B', color: '#38BDF8', padding: '2px 8px', borderRadius: '12px', border: '1px solid #334155', fontWeight: 600 }}>
                Enterprise
              </span>
            </div>
            <div style={{ fontSize: '11px', color: '#64748B' }}>
              Weaviate Sharded Isolation • Firebase Auth • Gemini 2.5
            </div>
          </div>
        </div>

        {/* Header Right: Status, Tenant & User */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          {/* Backend Status Pill */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', padding: '5px 12px', borderRadius: '20px', backgroundColor: backendOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', border: `1px solid ${backendOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`, color: backendOnline ? '#34D399' : '#F87171' }}>
            <div style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: backendOnline ? '#10B981' : '#EF4444' }} />
            <span>{backendOnline ? 'Backend Online (Port 8000)' : 'Backend Offline'}</span>
          </div>

          {/* Active Tenant Badge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', padding: '5px 12px', borderRadius: '8px', backgroundColor: '#1E293B', border: '1px solid #334155', color: '#38BDF8' }}>
            <Building2 size={13} />
            <span style={{ fontWeight: 600 }}>Tenant: {selectedTenant.toUpperCase()}</span>
          </div>

          {/* User Account Button */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 10px', borderRadius: '8px', backgroundColor: '#1E293B', border: '1px solid #334155' }}>
            <div style={{ width: '24px', height: '24px', borderRadius: '50%', backgroundColor: '#0284C7', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: 700, color: '#FFFFFF' }}>
              {currentUser.email ? currentUser.email.charAt(0).toUpperCase() : 'U'}
            </div>
            <span style={{ fontSize: '12px', color: '#F1F5F9', maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {currentUser.email || 'User'}
            </span>
            <button
              onClick={() => signOut(auth)}
              title="Sign Out"
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94A3B8', display: 'flex', alignItems: 'center', padding: '2px' }}
            >
              <LogOut size={14} />
            </button>
          </div>
        </div>
      </header>

      {/* MAIN 3-COLUMN LAYOUT */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '320px 1fr 340px', height: 'calc(100vh - 62px)', overflow: 'hidden' }}>

        {/* LEFT COLUMN: TENANT & DOCUMENT MANAGEMENT */}
        <aside style={{ backgroundColor: '#0F172A', borderRight: '1px solid #1E293B', display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

          {/* User & Tenant Info */}
          <div style={{ padding: '16px', borderBottom: '1px solid #1E293B' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11.5px', color: '#94A3B8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                <ShieldCheck size={14} color="#10B981" />
                Firebase Authorisation
              </div>
              <span style={{ fontSize: '10.5px', color: '#10B981', backgroundColor: 'rgba(16, 185, 129, 0.1)', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                Active
              </span>
            </div>

            {/* Tenant Selector */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <label style={{ fontSize: '11px', color: '#64748B', fontWeight: 600 }}>Active Tenant Shard</label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px' }}>
                {tenants.map((t) => {
                  const isActive = selectedTenant === t.id;
                  return (
                    <button
                      key={t.id}
                      onClick={() => handleSelectTenant(t.id)}
                      style={{
                        padding: '6px 4px',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: 600,
                        cursor: 'pointer',
                        border: isActive ? '1px solid #38BDF8' : '1px solid #334155',
                        backgroundColor: isActive ? 'rgba(56, 189, 248, 0.15)' : '#1E293B',
                        color: isActive ? '#38BDF8' : '#94A3B8',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      {t.name.replace('Tenant ', '')}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Permissions Badges */}
            <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
              {(userProfile?.permissions || ['READ', 'WRITE', 'UPDATE', 'DELETE', 'INGEST']).map((p) => (
                <span
                  key={p}
                  style={{
                    fontSize: '9.5px',
                    backgroundColor: '#1E293B',
                    color: '#94A3B8',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    border: '1px solid #334155',
                    fontFamily: 'monospace',
                  }}
                >
                  {p}
                </span>
              ))}
            </div>
          </div>

          {/* MANAGE DOCUMENTS SECTION */}
          <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', borderBottom: '1px solid #1E293B' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '12px', fontWeight: 700, color: '#F1F5F9', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Layers size={14} color="#38BDF8" />
                Document Operations
              </span>
              <span style={{ fontSize: '11px', color: '#64748B' }}>
                {documents.length} {documents.length === 1 ? 'doc' : 'docs'}
              </span>
            </div>

            {/* Single Main Button: Manage Documents with Dropdown */}
            <div style={{ position: 'relative' }} ref={manageMenuRef}>
              <button
                onClick={() => setIsManageMenuOpen(!isManageMenuOpen)}
                disabled={opLoading}
                style={{
                  width: '100%',
                  padding: '9px 14px',
                  borderRadius: '8px',
                  backgroundColor: '#0284C7',
                  color: '#FFFFFF',
                  border: 'none',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: opLoading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  boxShadow: '0 4px 14px rgba(2, 132, 199, 0.3)',
                  opacity: opLoading ? 0.7 : 1,
                  transition: 'background 0.2s',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <SlidersHorizontal size={15} />
                  <span>Manage Documents</span>
                </div>
                <ChevronDown size={15} style={{ transform: isManageMenuOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
              </button>

              {/* Dropdown Menu */}
              {isManageMenuOpen && (
                <div
                  style={{
                    position: 'absolute',
                    top: 'calc(100% + 6px)',
                    left: 0,
                    right: 0,
                    backgroundColor: '#1E293B',
                    borderRadius: '8px',
                    border: '1px solid #334155',
                    boxShadow: '0 10px 25px rgba(0, 0, 0, 0.5)',
                    zIndex: 50,
                    overflow: 'hidden',
                  }}
                >
                  {/* Option 1: Add Document */}
                  <button
                    onClick={() => {
                      setIsManageMenuOpen(false);
                      fileInputRef.current?.click();
                    }}
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '10px',
                      background: 'none',
                      border: 'none',
                      borderBottom: '1px solid #334155',
                      cursor: 'pointer',
                      textAlign: 'left',
                      color: '#F8FAFC',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#273549')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    <UploadCloud size={16} color="#38BDF8" style={{ marginTop: '2px', flexShrink: 0 }} />
                    <div>
                      <div style={{ fontSize: '12.5px', fontWeight: 600 }}>Add Document</div>
                      <div style={{ fontSize: '11px', color: '#94A3B8' }}>Upload a new PDF, DOCX, or DOC file</div>
                    </div>
                  </button>

                  {/* Option 2: Update Document */}
                  <button
                    onClick={() => {
                      setIsManageMenuOpen(false);
                      if (documents.length === 0) {
                        setOpStatus({ success: false, msg: 'No documents available to update.' });
                        return;
                      }
                      setSelectedDocForAction(documents[0]);
                      setActiveModal('update');
                    }}
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '10px',
                      background: 'none',
                      border: 'none',
                      borderBottom: '1px solid #334155',
                      cursor: 'pointer',
                      textAlign: 'left',
                      color: '#F8FAFC',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#273549')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    <RefreshCw size={15} color="#F59E0B" style={{ marginTop: '2px', flexShrink: 0 }} />
                    <div>
                      <div style={{ fontSize: '12.5px', fontWeight: 600 }}>Update Document</div>
                      <div style={{ fontSize: '11px', color: '#94A3B8' }}>Replace an existing document with a new version</div>
                    </div>
                  </button>

                  {/* Option 3: Delete Document */}
                  <button
                    onClick={() => {
                      setIsManageMenuOpen(false);
                      if (documents.length === 0) {
                        setOpStatus({ success: false, msg: 'No documents available to delete.' });
                        return;
                      }
                      setSelectedDocForAction(documents[0]);
                      setActiveModal('delete');
                    }}
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '10px',
                      background: 'none',
                      border: 'none',
                      borderBottom: '1px solid #334155',
                      cursor: 'pointer',
                      textAlign: 'left',
                      color: '#F8FAFC',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#273549')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    <Trash2 size={15} color="#EF4444" style={{ marginTop: '2px', flexShrink: 0 }} />
                    <div>
                      <div style={{ fontSize: '12.5px', fontWeight: 600 }}>Delete Document</div>
                      <div style={{ fontSize: '11px', color: '#94A3B8' }}>Permanently remove document & vector chunks</div>
                    </div>
                  </button>

                  {/* Option 4: Ingest Document */}
                  <button
                    onClick={() => {
                      setIsManageMenuOpen(false);
                      if (documents.length === 0) {
                        setOpStatus({ success: false, msg: 'No documents uploaded yet to ingest.' });
                        return;
                      }
                      setSelectedDocForAction(documents[0]);
                      setActiveModal('ingest');
                    }}
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '10px',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      textAlign: 'left',
                      color: '#F8FAFC',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#273549')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    <Zap size={15} color="#10B981" style={{ marginTop: '2px', flexShrink: 0 }} />
                    <div>
                      <div style={{ fontSize: '12.5px', fontWeight: 600 }}>Ingest Document</div>
                      <div style={{ fontSize: '11px', color: '#94A3B8' }}>Chunk, embed & index in Weaviate shard</div>
                    </div>
                  </button>
                </div>
              )}
            </div>

            {/* Status Alert Banner */}
            {opStatus && (
              <div
                style={{
                  padding: '8px 10px',
                  borderRadius: '6px',
                  fontSize: '11.5px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '6px',
                  backgroundColor: opStatus.success ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                  border: `1px solid ${opStatus.success ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                  color: opStatus.success ? '#34D399' : '#F87171',
                }}
              >
                {opStatus.success ? <CheckCircle size={14} style={{ marginTop: '1px', flexShrink: 0 }} /> : <AlertCircle size={14} style={{ marginTop: '1px', flexShrink: 0 }} />}
                <span>{opStatus.msg}</span>
              </div>
            )}
          </div>

          {/* TENANT DOCUMENTS LIST */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '2px' }}>
              Authorized Tenant Documents
            </div>

            {docsLoading ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '32px 0', color: '#94A3B8', fontSize: '12px', gap: '8px' }}>
                <RefreshCw size={14} className="spin" />
                <span>Loading documents...</span>
              </div>
            ) : documents.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '36px 12px', border: '1px dashed #334155', borderRadius: '8px', color: '#94A3B8' }}>
                <FileText size={32} style={{ margin: '0 auto 10px', opacity: 0.35, color: '#38BDF8' }} />
                <p style={{ fontSize: '12.5px', fontWeight: 500, margin: 0, color: '#CBD5E1', lineHeight: '1.4' }}>
                  No documents available. Click Manage Documents to add your first document.
                </p>
              </div>
            ) : (
              documents.map((doc) => {
                const isSelected = selectedSourceFilter === doc.filename;
                return (
                  <div
                    key={doc.doc_id}
                    style={{
                      padding: '10px',
                      borderRadius: '8px',
                      backgroundColor: isSelected ? 'rgba(56, 189, 248, 0.08)' : '#1E293B',
                      border: isSelected ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid #334155',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px',
                      transition: 'border 0.15s ease',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '6px' }}>
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', minWidth: 0 }}>
                        <FileText size={15} color="#38BDF8" style={{ marginTop: '2px', flexShrink: 0 }} />
                        <div style={{ minWidth: 0 }}>
                          <div
                            style={{
                              fontSize: '12px',
                              fontWeight: 600,
                              color: '#F1F5F9',
                              whiteSpace: 'nowrap',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              maxWidth: '180px',
                            }}
                            title={doc.filename}
                          >
                            {doc.filename}
                          </div>
                          <div style={{ fontSize: '10px', color: '#64748B' }}>
                            {formatDate(doc.upload_timestamp)} • v{doc.version}
                          </div>
                        </div>
                      </div>

                      {/* File type pill */}
                      <span
                        style={{
                          fontSize: '9px',
                          fontWeight: 700,
                          backgroundColor: '#0F172A',
                          color: '#38BDF8',
                          padding: '1px 5px',
                          borderRadius: '4px',
                          border: '1px solid #334155',
                          textTransform: 'uppercase',
                        }}
                      >
                        {doc.file_type}
                      </span>
                    </div>

                    {/* Status & Action Row */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '2px' }}>
                      {/* Ingestion Status Badge */}
                      {doc.status === 'completed' ? (
                        <span style={{ fontSize: '10.5px', color: '#34D399', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <CheckCircle size={11} />
                          {doc.chunk_count} chunks indexed
                        </span>
                      ) : doc.status === 'processing' ? (
                        <span style={{ fontSize: '10.5px', color: '#FBBF24', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <RefreshCw size={11} className="spin" />
                          Processing...
                        </span>
                      ) : doc.status === 'failed' ? (
                        <span style={{ fontSize: '10.5px', color: '#F87171', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <AlertCircle size={11} />
                          Failed
                        </span>
                      ) : (
                        <span style={{ fontSize: '10.5px', color: '#94A3B8', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <FileUp size={11} />
                          Uploaded (Not Ingested)
                        </span>
                      )}

                      {/* Card Quick Actions */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        {doc.status !== 'completed' && doc.status !== 'processing' && (
                          <button
                            onClick={() => handleIngestDocument(doc.doc_id)}
                            title="Ingest document into Weaviate"
                            style={{
                              background: 'none',
                              border: '1px solid #0284C7',
                              borderRadius: '4px',
                              padding: '2px 6px',
                              fontSize: '10px',
                              fontWeight: 600,
                              color: '#38BDF8',
                              cursor: 'pointer',
                            }}
                          >
                            Ingest
                          </button>
                        )}
                        <button
                          onClick={() => {
                            setSelectedDocForAction(doc);
                            setActiveModal('update');
                          }}
                          title="Update document"
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#94A3B8',
                            cursor: 'pointer',
                            padding: '3px',
                          }}
                        >
                          <RefreshCw size={12} />
                        </button>
                        <button
                          onClick={() => {
                            setSelectedDocForAction(doc);
                            setActiveModal('delete');
                          }}
                          title="Delete document"
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#F87171',
                            cursor: 'pointer',
                            padding: '3px',
                          }}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        {/* MIDDLE COLUMN: QUERY & STREAMING ANSWER */}
        <main style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', backgroundColor: '#0B1120' }}>

          {/* Query Filter Bar */}
          <div style={{ padding: '12px 20px', borderBottom: '1px solid #1E293B', backgroundColor: '#0F172A', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BookOpen size={15} color="#38BDF8" />
              <span style={{ fontSize: '12.5px', color: '#94A3B8', fontWeight: 600 }}>Query Scope:</span>
              <select
                value={selectedSourceFilter}
                onChange={(e) => setSelectedSourceFilter(e.target.value)}
                style={{
                  backgroundColor: '#1E293B',
                  color: '#F1F5F9',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  padding: '4px 10px',
                  fontSize: '12px',
                  outline: 'none',
                  cursor: 'pointer',
                  maxWidth: '280px',
                }}
              >
                <option value="all">All Documents in Tenant [{selectedTenant.toUpperCase()}]</option>
                {documents.map((d) => (
                  <option key={d.doc_id} value={d.filename}>
                    {d.filename}
                  </option>
                ))}
              </select>
            </div>
            <span style={{ fontSize: '11px', color: '#64748B' }}>
              Isolated Shard: <strong>{selectedTenant}</strong>
            </span>
          </div>

          {/* Ask Question Box */}
          <div style={{ padding: '16px 20px', borderBottom: '1px solid #1E293B', backgroundColor: '#0F172A' }}>
            <div style={{ position: 'relative' }}>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={`Ask a question about ${selectedTenant.toUpperCase()}'s documents... (Ctrl+Enter to ask)`}
                rows={3}
                style={{
                  width: '100%',
                  backgroundColor: '#1E293B',
                  color: '#F8FAFC',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  padding: '10px 14px',
                  fontSize: '13px',
                  resize: 'none',
                  outline: 'none',
                  boxSizing: 'border-box',
                  fontFamily: 'inherit',
                }}
              />
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '8px' }}>
                <span style={{ fontSize: '11px', color: '#64748B' }}>
                  Press <kbd style={{ padding: '2px 5px', borderRadius: '4px', backgroundColor: '#334155', color: '#94A3B8' }}>Ctrl+Enter</kbd> to submit
                </span>
                <button
                  onClick={() => handleAsk()}
                  disabled={isStreaming || !question.trim()}
                  style={{
                    backgroundColor: isStreaming || !question.trim() ? '#1E293B' : '#0284C7',
                    color: '#FFFFFF',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '7px 16px',
                    fontSize: '12.5px',
                    fontWeight: 600,
                    cursor: isStreaming || !question.trim() ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.15s',
                  }}
                >
                  {isStreaming ? (
                    <>
                      <RefreshCw size={13} className="spin" />
                      <span>Thinking...</span>
                    </>
                  ) : (
                    <>
                      <Send size={13} />
                      <span>Ask Question</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Streaming Answer Container */}
          <div style={{ flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column' }} ref={answerRef}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sparkles size={16} color="#38BDF8" />
                <span style={{ fontSize: '13.5px', fontWeight: 600, color: '#F1F5F9' }}>Answer</span>
                {isStreaming && (
                  <span style={{ fontSize: '11px', color: '#38BDF8', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <span className="pulsing-dot" /> Streaming from Gemini
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                {streamedAnswer && (
                  <>
                    <span style={{ fontSize: '11px', color: '#64748B' }}>
                      {tokenCount} tokens • {elapsedTime}s
                    </span>
                    <button
                      onClick={handleCopy}
                      style={{
                        background: 'none',
                        border: '1px solid #334155',
                        borderRadius: '4px',
                        padding: '3px 8px',
                        fontSize: '11px',
                        color: copied ? '#34D399' : '#94A3B8',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                    >
                      {copied ? <Check size={11} /> : <Copy size={11} />}
                      {copied ? 'Copied' : 'Copy'}
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Error Message if any */}
            {errorMsg && (
              <div style={{ padding: '12px 14px', borderRadius: '8px', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#F87171', fontSize: '12.5px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <AlertCircle size={16} style={{ flexShrink: 0 }} />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Markdown Answer or Placeholder */}
            {!streamedAnswer && !isStreaming ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#64748B' }}>
                <BookOpen size={36} style={{ opacity: 0.3, marginBottom: '12px' }} />
                <div style={{ fontSize: '14px', fontWeight: 500, color: '#94A3B8' }}>
                  Ready to answer your queries
                </div>
                <div style={{ fontSize: '12px', marginTop: '4px' }}>
                  Ask any question about documents in Tenant [{selectedTenant.toUpperCase()}]
                </div>
              </div>
            ) : (
              <div style={{ fontSize: '13.5px', lineHeight: '1.65', color: '#E2E8F0' }} className="markdown-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamedAnswer}</ReactMarkdown>
              </div>
            )}
          </div>
        </main>

        {/* RIGHT COLUMN: RETRIEVED CHUNKS & RERANKER */}
        <aside style={{ backgroundColor: '#0F172A', borderLeft: '1px solid #1E293B', display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

          {/* Header */}
          <div style={{ padding: '14px 16px', borderBottom: '1px solid #1E293B', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Award size={15} color="#10B981" />
              <span style={{ fontSize: '13px', fontWeight: 700, color: '#F1F5F9' }}>
                Reranker & Retrieved
              </span>
            </div>
            <span style={{ fontSize: '11px', color: '#64748B' }}>
              {rerankedChunks.length || retrievedChunks.length} chunks
            </span>
          </div>

          {/* Chunks List */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {(rerankedChunks.length > 0 ? rerankedChunks : retrievedChunks).length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 12px', color: '#64748B' }}>
                <Hash size={30} style={{ margin: '0 auto 10px', opacity: 0.3 }} />
                <div style={{ fontSize: '12.5px', color: '#94A3B8' }}>No chunks retrieved yet</div>
                <div style={{ fontSize: '11px', marginTop: '4px' }}>
                  Submit a query to view semantic chunks from Weaviate
                </div>
              </div>
            ) : (
              (rerankedChunks.length > 0 ? rerankedChunks : retrievedChunks).map((chunk, idx) => {
                const score = (chunk as RerankedChunk).relevance_score;
                return (
                  <div
                    key={idx}
                    style={{
                      padding: '10px 12px',
                      borderRadius: '8px',
                      backgroundColor: '#1E293B',
                      border: '1px solid #334155',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: '11px', fontWeight: 600, color: '#38BDF8', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <FileText size={11} />
                        {chunk.source}
                      </span>
                      <span style={{ fontSize: '10px', color: '#94A3B8', backgroundColor: '#0F172A', padding: '1px 5px', borderRadius: '4px' }}>
                        Page {chunk.page}
                      </span>
                    </div>

                    <div style={{ fontSize: '11.5px', color: '#CBD5E1', lineHeight: '1.45', maxHeight: '110px', overflowY: 'auto' }}>
                      {chunk.content}
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '2px', borderTop: '1px solid #273549', paddingTop: '4px' }}>
                      <span style={{ fontSize: '10px', color: '#64748B' }}>
                        Tenant: <strong>{chunk.tenant || selectedTenant}</strong>
                      </span>
                      {score !== undefined && (
                        <span style={{ fontSize: '10px', color: score >= 7 ? '#34D399' : '#FBBF24', fontWeight: 600 }}>
                          Score: {score}/10
                        </span>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>
      </div>

      {/* ========================================================= */}
      {/* MODAL 1: UPDATE DOCUMENT                                 */}
      {/* ========================================================= */}
      {activeModal === 'update' && (
        <div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0, 0, 0, 0.75)', backdropFilter: 'blur(3px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div style={{ width: '440px', backgroundColor: '#1E293B', borderRadius: '12px', border: '1px solid #334155', padding: '24px', boxShadow: '0 20px 40px rgba(0, 0, 0, 0.6)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <RefreshCw size={18} color="#F59E0B" />
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: '#F8FAFC' }}>Update Document</h3>
              </div>
              <button
                onClick={() => setActiveModal(null)}
                style={{ background: 'none', border: 'none', color: '#94A3B8', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>

            <p style={{ fontSize: '12.5px', color: '#94A3B8', marginTop: 0, marginBottom: '16px', lineHeight: '1.4' }}>
              Select an existing document belonging to Tenant [<strong>{selectedTenant.toUpperCase()}</strong>] and replace it with a new PDF, DOCX, or DOC file.
            </p>

            {/* Document Selection */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{ fontSize: '11.5px', color: '#64748B', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                Document to Replace
              </label>
              <select
                value={selectedDocForAction?.doc_id || ''}
                onChange={(e) => {
                  const found = documents.find((d) => d.doc_id === e.target.value);
                  if (found) setSelectedDocForAction(found);
                }}
                style={{
                  width: '100%',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  backgroundColor: '#0F172A',
                  color: '#F1F5F9',
                  border: '1px solid #334155',
                  fontSize: '12.5px',
                  outline: 'none',
                }}
              >
                {documents.map((d) => (
                  <option key={d.doc_id} value={d.doc_id}>
                    {d.filename} (v{d.version} • {d.chunk_count} chunks)
                  </option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                onClick={() => setActiveModal(null)}
                style={{
                  padding: '8px 14px',
                  borderRadius: '6px',
                  border: '1px solid #334155',
                  backgroundColor: '#0F172A',
                  color: '#94A3B8',
                  fontSize: '12.5px',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => updateFileInputRef.current?.click()}
                style={{
                  padding: '8px 16px',
                  borderRadius: '6px',
                  border: 'none',
                  backgroundColor: '#0284C7',
                  color: '#FFFFFF',
                  fontSize: '12.5px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <UploadCloud size={14} />
                <span>Select New File & Update</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* MODAL 2: DELETE DOCUMENT CONFIRMATION                    */}
      {/* ========================================================= */}
      {activeModal === 'delete' && selectedDocForAction && (
        <div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0, 0, 0, 0.75)', backdropFilter: 'blur(3px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div style={{ width: '420px', backgroundColor: '#1E293B', borderRadius: '12px', border: '1px solid #334155', padding: '24px', boxShadow: '0 20px 40px rgba(0, 0, 0, 0.6)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
              <div style={{ width: '36px', height: '36px', borderRadius: '8px', backgroundColor: 'rgba(239, 68, 68, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Trash2 size={20} color="#EF4444" />
              </div>
              <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: '#F8FAFC' }}>
                Delete Document?
              </h3>
            </div>

            <p style={{ fontSize: '12.5px', color: '#94A3B8', marginTop: 0, marginBottom: '14px', lineHeight: '1.45' }}>
              Are you sure you want to permanently delete{' '}
              <strong style={{ color: '#F1F5F9' }}>"{selectedDocForAction.filename}"</strong> from Tenant [<strong>{selectedTenant.toUpperCase()}</strong>]?
            </p>

            <div style={{ padding: '10px 12px', borderRadius: '6px', backgroundColor: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.2)', marginBottom: '18px', fontSize: '11.5px', color: '#FCA5A5' }}>
              This will permanently delete the source file and remove all {selectedDocForAction.chunk_count} vector chunks from Weaviate. This action cannot be undone.
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                onClick={() => setActiveModal(null)}
                style={{
                  padding: '8px 14px',
                  borderRadius: '6px',
                  border: '1px solid #334155',
                  backgroundColor: '#0F172A',
                  color: '#94A3B8',
                  fontSize: '12.5px',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteDocument(selectedDocForAction.doc_id)}
                style={{
                  padding: '8px 16px',
                  borderRadius: '6px',
                  border: 'none',
                  backgroundColor: '#DC2626',
                  color: '#FFFFFF',
                  fontSize: '12.5px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <Trash2 size={14} />
                <span>Delete Permanently</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* MODAL 3: INGEST DOCUMENT SELECTOR                        */}
      {/* ========================================================= */}
      {activeModal === 'ingest' && (
        <div style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0, 0, 0, 0.75)', backdropFilter: 'blur(3px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div style={{ width: '440px', backgroundColor: '#1E293B', borderRadius: '12px', border: '1px solid #334155', padding: '24px', boxShadow: '0 20px 40px rgba(0, 0, 0, 0.6)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Zap size={18} color="#10B981" />
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: '#F8FAFC' }}>Ingest Document</h3>
              </div>
              <button
                onClick={() => setActiveModal(null)}
                style={{ background: 'none', border: 'none', color: '#94A3B8', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>

            <p style={{ fontSize: '12.5px', color: '#94A3B8', marginTop: 0, marginBottom: '16px', lineHeight: '1.4' }}>
              Select an uploaded document to extract text, generate vector embeddings, and index into Tenant [<strong>{selectedTenant.toUpperCase()}</strong>]'s Weaviate shard.
            </p>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ fontSize: '11.5px', color: '#64748B', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                Select Document
              </label>
              <select
                value={selectedDocForAction?.doc_id || ''}
                onChange={(e) => {
                  const found = documents.find((d) => d.doc_id === e.target.value);
                  if (found) setSelectedDocForAction(found);
                }}
                style={{
                  width: '100%',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  backgroundColor: '#0F172A',
                  color: '#F1F5F9',
                  border: '1px solid #334155',
                  fontSize: '12.5px',
                  outline: 'none',
                }}
              >
                {documents.map((d) => (
                  <option key={d.doc_id} value={d.doc_id}>
                    {d.filename} ({d.status === 'completed' ? `${d.chunk_count} chunks indexed` : d.status})
                  </option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                onClick={() => setActiveModal(null)}
                style={{
                  padding: '8px 14px',
                  borderRadius: '6px',
                  border: '1px solid #334155',
                  backgroundColor: '#0F172A',
                  color: '#94A3B8',
                  fontSize: '12.5px',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => selectedDocForAction && handleIngestDocument(selectedDocForAction.doc_id)}
                disabled={!selectedDocForAction}
                style={{
                  padding: '8px 16px',
                  borderRadius: '6px',
                  border: 'none',
                  backgroundColor: '#10B981',
                  color: '#FFFFFF',
                  fontSize: '12.5px',
                  fontWeight: 600,
                  cursor: !selectedDocForAction ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <Zap size={14} />
                <span>Start Ingestion</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Auth Modal if triggered */}
      {authModalOpen && <AuthModal isOpen={authModalOpen} onClose={() => setAuthModalOpen(false)} />}
    </div>
  );
}
