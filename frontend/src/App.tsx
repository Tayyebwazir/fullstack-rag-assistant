import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { auth, onAuthStateChanged, signOut, User } from './firebase';
import { AuthPage } from './components/AuthPage';
import {
  GraduationCap,
  MessageSquare,
  FileText,
  UploadCloud,
  Send,
  Zap,
  CheckCircle2,
  Copy,
  Check,
  RefreshCw,
  Plus,
  Trash2,
  ShieldCheck,
  HelpCircle,
  FileCheck,
  Sparkles,
  BookOpen,
  Image as ImageIcon,
  Paperclip,
  CheckCircle,
  Clock,
  Layers,
  ChevronRight,
  ExternalLink,
  Lock,
  Printer,
  Eye,
  X,
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

interface ChatMessage {
  id: string;
  sender: 'student' | 'ai';
  text: string;
  sources?: Chunk[];
  timestamp: string;
}

export default function App() {
  // Authentication & Profile
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [selectedTenant, setSelectedTenant] = useState<string>('');

  // Navigation & View
  const [activeTab, setActiveTab] = useState<'chat' | 'documents'>('chat');

  // Chat & Q&A
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [activeSelectedChunk, setActiveSelectedChunk] = useState<Chunk | null>(null);

  // Documents
  const [documents, setDocuments] = useState<TenantDocument[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [selectedSourceFilter, setSelectedSourceFilter] = useState<string>('all');

  // Extracted Text Inspector Modal & Print
  const [extractedTextModal, setExtractedTextModal] = useState<{
    isOpen: boolean;
    filename: string;
    text: string;
    loading?: boolean;
  }>({ isOpen: false, filename: '', text: '' });
  const [lastExtractedDoc, setLastExtractedDoc] = useState<{ filename: string; text: string } | null>(null);
  const [copiedExtracted, setCopiedExtracted] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Listen to Auth State
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (usr) => {
      setCurrentUser(usr);
      setAuthLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // Fetch User Profile on Login
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
        setSelectedTenant(data.profile.tenant_id || data.assigned_tenant || '');
      }
    } catch (err) {
      console.error('Error fetching profile:', err);
    }
  };

  useEffect(() => {
    if (currentUser) {
      fetchUserProfile();
    }
  }, [currentUser]);

  // Fetch Documents belonging to this student's tenant
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
  }, [currentUser, selectedTenant]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isAsking]);

  // Upload and Ingest Notes / Images / PDFs
  const handleFileUpload = async (file: File) => {
    if (!currentUser) return;
    setUploadLoading(true);
    setUploadStatus(`Uploading ${file.name}...`);

    try {
      const token = await currentUser.getIdToken();
      const formData = new FormData();
      formData.append('file', file);

      const uploadRes = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      const uploadData = await uploadRes.json();
      if (!uploadRes.ok) {
        throw new Error(uploadData.detail || 'Upload failed');
      }

      const docId = uploadData.document?.doc_id;
      setUploadStatus(`Running OCR & Embedding into Tenant Shard...`);

      // Trigger automatic chunking, PaddleOCR & vector ingestion
      const ingestRes = await fetch(`${API_BASE}/documents/ingest`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ doc_id: docId }),
      });

      const ingestData = await ingestRes.json();
      if (!ingestRes.ok) {
        throw new Error(ingestData.detail || 'Ingestion failed');
      }

      setUploadStatus(`✓ Ready! Ingested ${ingestData.chunks_ingested} chunks.`);
      if (ingestData.extracted_text) {
        const docInfo = { filename: file.name, text: ingestData.extracted_text };
        setLastExtractedDoc(docInfo);
        setExtractedTextModal({
          isOpen: true,
          filename: file.name,
          text: ingestData.extracted_text,
          loading: false,
        });
      }
      await fetchTenantDocuments();
      setTimeout(() => setUploadStatus(null), 4000);
    } catch (err: any) {
      setUploadStatus(`❌ ${err.message || 'Error processing file'}`);
      setTimeout(() => setUploadStatus(null), 5000);
    } finally {
      setUploadLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Fetch and Show Extracted Text on Demand
  const handleFetchAndShowExtractedText = async (docId: string, filename: string) => {
    if (!currentUser) return;
    setExtractedTextModal({ isOpen: true, filename, text: '', loading: true });
    try {
      const token = await currentUser.getIdToken();
      const res = await fetch(`${API_BASE}/documents/${docId}/extracted-text`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok && data.extracted_text) {
        setExtractedTextModal({ isOpen: true, filename, text: data.extracted_text, loading: false });
        setLastExtractedDoc({ filename, text: data.extracted_text });
      } else {
        setExtractedTextModal({
          isOpen: true,
          filename,
          text: data.detail || 'No text could be extracted from this document.',
          loading: false,
        });
      }
    } catch (err: any) {
      setExtractedTextModal({
        isOpen: true,
        filename,
        text: `Error fetching extracted text: ${err.message}`,
        loading: false,
      });
    }
  };

  // Print Extracted Text in Clean Printable Window
  const handlePrintText = (filename: string, text: string) => {
    if (!text) return;
    const printWindow = window.open('', '_blank', 'width=800,height=900');
    if (!printWindow) {
      window.print();
      return;
    }
    const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    printWindow.document.write(`<!DOCTYPE html>
<html>
  <head>
    <title>Extracted Text - ${filename}</title>
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace, sans-serif;
        padding: 40px;
        color: #111827;
        background: #FFFFFF;
        line-height: 1.6;
        font-size: 14px;
      }
      .header {
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 16px;
        margin-bottom: 24px;
      }
      .title { font-size: 20px; font-weight: 700; color: #1F2937; margin: 0 0 8px 0; }
      .meta { display: flex; gap: 10px; align-items: center; }
      .badge {
        display: inline-block;
        background: #EEF2FF;
        color: #4F46E5;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
      }
      .content {
        white-space: pre-wrap;
        word-wrap: break-word;
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 24px;
        font-family: inherit;
        font-size: 13.5px;
      }
      @media print {
        body { padding: 0; }
        .content { border: none; background: transparent; padding: 0; }
      }
    </style>
  </head>
  <body>
    <div class="header">
      <h1 class="title">${filename}</h1>
      <div class="meta">
        <span class="badge">Paddle PP-OCRv5 Extraction</span>
      </div>
    </div>
    <div class="content">${escaped}</div>
  </body>
</html>`);
    printWindow.document.close();
    printWindow.focus();
    setTimeout(() => {
      printWindow.print();
    }, 400);
  };

  // Copy Extracted Text
  const handleCopyExtractedText = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedExtracted(true);
    setTimeout(() => setCopiedExtracted(false), 2000);
  };

  // Delete Document
  const handleDeleteDocument = async (docId: string, filename: string) => {
    if (!currentUser || !window.confirm(`Delete "${filename}" and its vector embeddings?`)) return;
    try {
      const token = await currentUser.getIdToken();
      const res = await fetch(`${API_BASE}/documents/delete`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ doc_id: docId }),
      });
      if (res.ok) {
        await fetchTenantDocuments();
      }
    } catch (err) {
      console.error('Delete error:', err);
    }
  };

  // Submit Question to RAG Pipeline
  const handleAsk = async (customPrompt?: string) => {
    const q = customPrompt || question;
    if (!q.trim() || !currentUser || isAsking) return;

    const studentMsg: ChatMessage = {
      id: String(Date.now()),
      sender: 'student',
      text: q.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, studentMsg]);
    setQuestion('');
    setIsAsking(true);

    try {
      const token = await currentUser.getIdToken();
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: q.trim(),
          source_filter: selectedSourceFilter === 'all' ? null : selectedSourceFilter,
        }),
      });

      const data = await res.json();
      if (res.ok) {
        const retrievedSources = data.sources || data.retrieved_chunks || [];
        const aiMsg: ChatMessage = {
          id: String(Date.now() + 1),
          sender: 'ai',
          text: data.answer || 'No response generated.',
          sources: retrievedSources,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setMessages((prev) => [...prev, aiMsg]);
        if (retrievedSources.length > 0) {
          setActiveSelectedChunk(retrievedSources[0]);
        }
      } else {
        const errorMsgText: ChatMessage = {
          id: String(Date.now() + 1),
          sender: 'ai',
          text: `⚠️ **Error:** ${data.detail || 'Could not retrieve answer. Please make sure documents are uploaded and indexed.'}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setMessages((prev) => [...prev, errorMsgText]);
      }
    } catch (err: any) {
      const errorMsgText: ChatMessage = {
        id: String(Date.now() + 1),
        sender: 'ai',
        text: `⚠️ **Connection Error:** Could not connect to backend server.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsgText]);
    } finally {
      setIsAsking(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Auth Loading Screen
  if (authLoading) {
    return (
      <div style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center', background: '#0F172A', color: '#F8FAFC' }}>
        <RefreshCw className="animate-spin" size={32} style={{ color: '#8B5CF6' }} />
      </div>
    );
  }

  // Not logged in -> Show Auth Page
  if (!currentUser) {
    return <AuthPage />;
  }

  const studentName = currentUser.displayName || currentUser.email?.split('@')[0] || 'Student';
  const tenantDisplay = selectedTenant ? `${selectedTenant.slice(0, 8)}...` : 'Assigning...';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0B1120', color: '#F8FAFC', overflow: 'hidden' }}>
      {/* ==========================================
          TOP HEADER BAR
      ========================================== */}
      <header
        style={{
          height: '60px',
          borderBottom: '1px solid #1E293B',
          background: '#0F172A',
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
              width: '38px',
              height: '38px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#FFFFFF',
              boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)',
            }}
          >
            <GraduationCap size={22} />
          </div>
          <div>
            <h1 style={{ fontSize: '16px', fontWeight: 700, letterSpacing: '-0.3px', margin: 0, color: '#F8FAFC' }}>
              Student AI Assistant
            </h1>
            <span style={{ fontSize: '11px', color: '#94A3B8' }}>OCR + Weaviate Multi-Tenant RAG</span>
          </div>
        </div>

        {/* Center Navigation Tabs */}
        <div style={{ display: 'flex', gap: '6px', background: '#1E293B', padding: '4px', borderRadius: '10px', border: '1px solid #334155' }}>
          <button
            onClick={() => setActiveTab('chat')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              fontSize: '13px',
              fontWeight: 500,
              borderRadius: '7px',
              border: 'none',
              cursor: 'pointer',
              background: activeTab === 'chat' ? '#6366F1' : 'transparent',
              color: activeTab === 'chat' ? '#FFFFFF' : '#94A3B8',
              transition: 'all 0.15s ease',
            }}
          >
            <MessageSquare size={15} /> Chat with Notes
          </button>
          <button
            onClick={() => setActiveTab('documents')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              fontSize: '13px',
              fontWeight: 500,
              borderRadius: '7px',
              border: 'none',
              cursor: 'pointer',
              background: activeTab === 'documents' ? '#6366F1' : 'transparent',
              color: activeTab === 'documents' ? '#FFFFFF' : '#94A3B8',
              transition: 'all 0.15s ease',
            }}
          >
            <BookOpen size={15} /> My Notes ({documents.length})
          </button>
        </div>

        {/* Right User & Tenant Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          {/* Tenant Shard Pill */}
          <div
            title={`Weaviate Tenant Shard: ${selectedTenant}`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'rgba(99, 102, 241, 0.12)',
              border: '1px solid rgba(99, 102, 241, 0.3)',
              padding: '5px 12px',
              borderRadius: '20px',
              fontSize: '12px',
              color: '#A5B4FC',
              fontWeight: 500,
            }}
          >
            <Lock size={12} style={{ color: '#818CF8' }} />
            <span>Shard: <strong style={{ color: '#FFFFFF' }}>{tenantDisplay}</strong></span>
          </div>

          {/* Student Profile & Sign Out */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: '#6366F1',
                color: '#FFFFFF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '13px',
                fontWeight: 600,
              }}
            >
              {studentName.charAt(0).toUpperCase()}
            </div>
            <button
              onClick={() => signOut(auth)}
              style={{
                background: 'transparent',
                border: '1px solid #334155',
                color: '#94A3B8',
                padding: '5px 10px',
                borderRadius: '7px',
                fontSize: '12px',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = '#EF4444')}
              onMouseLeave={(e) => (e.currentTarget.style.color = '#94A3B8')}
            >
              Sign Out
            </button>
          </div>
        </div>
      </header>

      {/* ==========================================
          MAIN 3-COLUMN WORKSPACE BODY
      ========================================== */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* ==========================================
            LEFT SIDEBAR: Notes Navigation & Quick Docs
        ========================================== */}
        <aside
          style={{
            width: '260px',
            borderRight: '1px solid #1E293B',
            background: '#0F172A',
            display: 'flex',
            flexDirection: 'column',
            flexShrink: 0,
            padding: '16px',
          }}
        >
          {/* New Chat Button */}
          <button
            onClick={() => setMessages([])}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '10px 14px',
              background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
              border: 'none',
              borderRadius: '9px',
              color: '#FFFFFF',
              fontWeight: 600,
              fontSize: '13px',
              cursor: 'pointer',
              boxShadow: '0 4px 12px rgba(99, 102, 241, 0.25)',
              marginBottom: '18px',
            }}
          >
            <Plus size={16} /> New Chat
          </button>

          {/* Upload Button Shortcut */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            style={{ display: 'none' }}
            accept=".pdf,.docx,.doc,.png,.jpg,.jpeg,.webp"
          />

          {/* Notes List Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Study Materials ({documents.length})
            </span>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadLoading}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#818CF8',
                fontSize: '12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <Plus size={13} /> Add
            </button>
          </div>

          {/* Document list */}
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {documents.length === 0 ? (
              <div style={{ padding: '24px 12px', textAlign: 'center', color: '#64748B', fontSize: '12px' }}>
                <FileText size={28} style={{ opacity: 0.3, marginBottom: '8px' }} />
                <p>No notes uploaded yet.</p>
                <p style={{ fontSize: '11px', marginTop: '4px' }}>Upload PDF slides or handwritten photos.</p>
              </div>
            ) : (
              documents.map((doc) => {
                const isImage = ['PNG', 'JPG', 'JPEG', 'WEBP'].includes(doc.file_type);
                return (
                  <div
                    key={doc.doc_id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 10px',
                      background: '#1E293B',
                      borderRadius: '8px',
                      border: '1px solid #334155',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                      {isImage ? (
                        <ImageIcon size={16} style={{ color: '#F472B6', flexShrink: 0 }} />
                      ) : (
                        <FileText size={16} style={{ color: '#818CF8', flexShrink: 0 }} />
                      )}
                      <div style={{ minWidth: 0 }}>
                        <p style={{ fontSize: '12px', fontWeight: 500, color: '#F8FAFC', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', margin: 0 }}>
                          {doc.filename}
                        </p>
                        <span style={{ fontSize: '10px', color: doc.status === 'completed' ? '#10B981' : '#F59E0B' }}>
                          {doc.status === 'completed' ? `${doc.chunk_count} chunks` : doc.status}
                        </span>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexShrink: 0 }}>
                      <button
                        onClick={() => handleFetchAndShowExtractedText(doc.doc_id, doc.filename)}
                        title="Print / View Extracted Text"
                        style={{
                          background: 'rgba(99, 102, 241, 0.12)',
                          border: '1px solid rgba(99, 102, 241, 0.3)',
                          borderRadius: '5px',
                          color: '#A5B4FC',
                          cursor: 'pointer',
                          padding: '3px 6px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '3px',
                          fontSize: '10px',
                          fontWeight: 600,
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'rgba(99, 102, 241, 0.3)';
                          e.currentTarget.style.color = '#FFFFFF';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'rgba(99, 102, 241, 0.12)';
                          e.currentTarget.style.color = '#A5B4FC';
                        }}
                      >
                        <Eye size={11} /> OCR
                      </button>
                      <button
                        onClick={() => handleDeleteDocument(doc.doc_id, doc.filename)}
                        title="Delete Note"
                        style={{ background: 'transparent', border: 'none', color: '#64748B', cursor: 'pointer', padding: '4px' }}
                        onMouseEnter={(e) => (e.currentTarget.style.color = '#EF4444')}
                        onMouseLeave={(e) => (e.currentTarget.style.color = '#64748B')}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Privacy & Tenant Footer Card */}
          <div
            style={{
              marginTop: '12px',
              padding: '12px',
              background: 'rgba(99, 102, 241, 0.08)',
              border: '1px solid rgba(99, 102, 241, 0.2)',
              borderRadius: '9px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#A5B4FC', fontSize: '11px', fontWeight: 600 }}>
              <ShieldCheck size={14} style={{ color: '#10B981' }} />
              1-to-1 Tenant Isolation
            </div>
            <p style={{ fontSize: '10px', color: '#94A3B8', marginTop: '4px', lineHeight: 1.4 }}>
              Your notes are stored strictly in your dedicated Weaviate shard. No other student can query your data.
            </p>
          </div>
        </aside>

        {/* ==========================================
            CENTER PANEL: Main Study Chat & Q&A
        ========================================== */}
        <main
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            background: '#0B1120',
            overflow: 'hidden',
          }}
        >
          {/* Scrollable Chat Area */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '24px 32px' }}>
            {/* Top Welcome Banner */}
            <div
              style={{
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.08))',
                border: '1px solid rgba(99, 102, 241, 0.25)',
                borderRadius: '14px',
                padding: '20px 24px',
                marginBottom: '20px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <h2 style={{ fontSize: '20px', fontWeight: 700, color: '#FFFFFF', margin: '0 0 6px 0' }}>
                    👋 Hi, {studentName}!
                  </h2>
                  <p style={{ fontSize: '13px', color: '#94A3B8', margin: 0 }}>
                    Your personal AI study assistant. Upload handwritten notes, PDFs, or photos and ask anything about them.
                  </p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {lastExtractedDoc && (
                    <button
                      onClick={() => handlePrintText(lastExtractedDoc.filename, lastExtractedDoc.text)}
                      title="Print extracted text from uploaded notes"
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        background: 'rgba(16, 185, 129, 0.15)',
                        border: '1px solid #10B981',
                        color: '#6EE7B7',
                        padding: '9px 14px',
                        borderRadius: '9px',
                        fontWeight: 600,
                        fontSize: '12px',
                        cursor: 'pointer',
                      }}
                    >
                      <Printer size={15} /> Print Extracted Text
                    </button>
                  )}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploadLoading}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '7px',
                      background: '#6366F1',
                      border: 'none',
                      color: '#FFFFFF',
                      padding: '9px 18px',
                      borderRadius: '9px',
                      fontWeight: 600,
                      fontSize: '13px',
                      cursor: 'pointer',
                      boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)',
                    }}
                  >
                    <UploadCloud size={16} /> Upload Notes (OCR)
                  </button>
                </div>
              </div>

              {/* Upload Status Notification Banner */}
              {uploadStatus && (
                <div
                  style={{
                    marginTop: '12px',
                    padding: '8px 14px',
                    borderRadius: '8px',
                    background: uploadStatus.startsWith('❌') ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.15)',
                    border: uploadStatus.startsWith('❌') ? '1px solid #EF4444' : '1px solid #10B981',
                    fontSize: '12px',
                    color: uploadStatus.startsWith('❌') ? '#FCA5A5' : '#6EE7B7',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '6px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {uploadLoading && <RefreshCw size={13} className="animate-spin" />}
                    <span>{uploadStatus}</span>
                  </div>
                  {lastExtractedDoc && !uploadLoading && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <button
                        onClick={() => setExtractedTextModal({ isOpen: true, filename: lastExtractedDoc.filename, text: lastExtractedDoc.text })}
                        style={{
                          background: 'rgba(255, 255, 255, 0.15)',
                          border: 'none',
                          borderRadius: '6px',
                          color: '#FFFFFF',
                          fontSize: '11px',
                          fontWeight: 600,
                          padding: '4px 8px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                        }}
                      >
                        <Eye size={12} /> View Text
                      </button>
                      <button
                        onClick={() => handlePrintText(lastExtractedDoc.filename, lastExtractedDoc.text)}
                        style={{
                          background: '#10B981',
                          border: 'none',
                          borderRadius: '6px',
                          color: '#FFFFFF',
                          fontSize: '11px',
                          fontWeight: 600,
                          padding: '4px 8px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                        }}
                      >
                        <Printer size={12} /> Print Text
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 4 Quick Action Study Pills */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '24px' }}>
              <button
                onClick={() => handleAsk('Explain the core concepts and key ideas from my uploaded notes in detail.')}
                style={{
                  background: '#1E293B',
                  border: '1px solid #334155',
                  borderRadius: '10px',
                  padding: '12px',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#6366F1')}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#334155')}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <HelpCircle size={15} style={{ color: '#818CF8' }} />
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>Ask Questions</span>
                </div>
                <p style={{ fontSize: '11px', color: '#94A3B8', margin: 0 }}>Get clear answers from your notes</p>
              </button>

              <button
                onClick={() => handleAsk('Provide a concise, comprehensive summary of the uploaded document with bullet points.')}
                style={{
                  background: '#1E293B',
                  border: '1px solid #334155',
                  borderRadius: '10px',
                  padding: '12px',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#6366F1')}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#334155')}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <FileText size={15} style={{ color: '#A78BFA' }} />
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>Summarize</span>
                </div>
                <p style={{ fontSize: '11px', color: '#94A3B8', margin: 0 }}>Create quick high-yield summaries</p>
              </button>

              <button
                onClick={() => handleAsk('Extract all important definitions, formulas, and key points from my notes.')}
                style={{
                  background: '#1E293B',
                  border: '1px solid #334155',
                  borderRadius: '10px',
                  padding: '12px',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#6366F1')}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#334155')}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <FileCheck size={15} style={{ color: '#34D399' }} />
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>Extract Notes</span>
                </div>
                <p style={{ fontSize: '11px', color: '#94A3B8', margin: 0 }}>Get key points and definitions</p>
              </button>

              <button
                onClick={() => handleAsk('Generate a 3-question practice quiz based on my notes with answers and explanations.')}
                style={{
                  background: '#1E293B',
                  border: '1px solid #334155',
                  borderRadius: '10px',
                  padding: '12px',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#6366F1')}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#334155')}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <Sparkles size={15} style={{ color: '#FBBF24' }} />
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>Create Quiz</span>
                </div>
                <p style={{ fontSize: '11px', color: '#94A3B8', margin: 0 }}>Generate practice exam questions</p>
              </button>
            </div>

            {/* Chat Messages Stream */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              {messages.length === 0 && (
                <div style={{ padding: '40px', textAlign: 'center', color: '#64748B' }}>
                  <MessageSquare size={36} style={{ opacity: 0.25, marginBottom: '12px' }} />
                  <p style={{ fontSize: '14px', margin: 0 }}>Ask any question about your uploaded notes or choose a quick study prompt above.</p>
                </div>
              )}

              {messages.map((msg) => (
                <div
                  key={msg.id}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: msg.sender === 'student' ? 'flex-end' : 'flex-start',
                  }}
                >
                  {/* Sender indicator */}
                  <span style={{ fontSize: '11px', color: '#64748B', marginBottom: '4px', marginLeft: msg.sender === 'ai' ? '40px' : 0 }}>
                    {msg.sender === 'student' ? 'You' : 'AI Study Assistant'} • {msg.timestamp}
                  </span>

                  <div style={{ display: 'flex', gap: '10px', maxWidth: '85%' }}>
                    {msg.sender === 'ai' && (
                      <div
                        style={{
                          width: '32px',
                          height: '32px',
                          borderRadius: '8px',
                          background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: '#FFFFFF',
                          flexShrink: 0,
                          marginTop: '2px',
                        }}
                      >
                        <Sparkles size={16} />
                      </div>
                    )}

                    <div
                      style={{
                        background: msg.sender === 'student' ? '#6366F1' : '#1E293B',
                        color: '#F8FAFC',
                        padding: '14px 18px',
                        borderRadius: msg.sender === 'student' ? '14px 14px 2px 14px' : '2px 14px 14px 14px',
                        border: msg.sender === 'student' ? 'none' : '1px solid #334155',
                        fontSize: '14px',
                        lineHeight: 1.6,
                      }}
                    >
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>

                      {/* Source Citations Badges */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid #334155' }}>
                          <span style={{ fontSize: '11px', fontWeight: 600, color: '#94A3B8', display: 'block', marginBottom: '6px' }}>
                            Sources ({msg.sources.length}):
                          </span>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                            {msg.sources.map((s, idx) => (
                              <button
                                key={idx}
                                onClick={() => setActiveSelectedChunk(s)}
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '4px',
                                  background: 'rgba(99, 102, 241, 0.15)',
                                  border: '1px solid rgba(99, 102, 241, 0.3)',
                                  padding: '3px 8px',
                                  borderRadius: '5px',
                                  fontSize: '11px',
                                  color: '#C7D2FE',
                                  cursor: 'pointer',
                                }}
                              >
                                <FileText size={11} /> {s.source} (p. {s.page})
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Copy Button */}
                      {msg.sender === 'ai' && (
                        <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'flex-end' }}>
                          <button
                            onClick={() => copyToClipboard(msg.text, msg.id)}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: '#64748B',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px',
                              fontSize: '11px',
                            }}
                          >
                            {copiedId === msg.id ? <Check size={12} style={{ color: '#10B981' }} /> : <Copy size={12} />}
                            {copiedId === msg.id ? 'Copied' : 'Copy'}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {/* Streaming / Loading Indicator */}
              {isAsking && (
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '8px',
                      background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#FFFFFF',
                    }}
                  >
                    <RefreshCw size={16} className="animate-spin" />
                  </div>
                  <div style={{ background: '#1E293B', padding: '12px 16px', borderRadius: '10px', border: '1px solid #334155' }}>
                    <span style={{ fontSize: '13px', color: '#94A3B8' }}>Searching Weaviate shard and synthesizing response...</span>
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>
          </div>

          {/* Bottom Chat Input Bar */}
          <div style={{ padding: '16px 32px', borderTop: '1px solid #1E293B', background: '#0F172A' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                background: '#1E293B',
                borderRadius: '12px',
                border: '1px solid #334155',
                padding: '6px 12px',
              }}
            >
              <button
                onClick={() => fileInputRef.current?.click()}
                title="Attach Note or Image (OCR)"
                style={{ background: 'transparent', border: 'none', color: '#94A3B8', cursor: 'pointer', padding: '6px' }}
                onMouseEnter={(e) => (e.currentTarget.style.color = '#FFFFFF')}
                onMouseLeave={(e) => (e.currentTarget.style.color = '#94A3B8')}
              >
                <Paperclip size={18} />
              </button>

              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleAsk()}
                placeholder="Ask a question about your uploaded notes or photos... (Press Enter)"
                style={{
                  flex: 1,
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  color: '#F8FAFC',
                  fontSize: '14px',
                }}
              />

              <button
                onClick={() => handleAsk()}
                disabled={!question.trim() || isAsking}
                style={{
                  background: question.trim() ? '#6366F1' : '#334155',
                  color: '#FFFFFF',
                  border: 'none',
                  padding: '8px 14px',
                  borderRadius: '8px',
                  fontWeight: 600,
                  fontSize: '13px',
                  cursor: question.trim() ? 'pointer' : 'not-allowed',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  transition: 'all 0.15s',
                }}
              >
                <Send size={15} /> Ask
              </button>
            </div>
          </div>
        </main>

        {/* ==========================================
            RIGHT PANEL: Document & OCR Inspector
        ========================================== */}
        <aside
          style={{
            width: '320px',
            borderLeft: '1px solid #1E293B',
            background: '#0F172A',
            display: 'flex',
            flexDirection: 'column',
            flexShrink: 0,
            padding: '16px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
            <Layers size={17} style={{ color: '#818CF8' }} />
            <h3 style={{ fontSize: '13px', fontWeight: 700, color: '#F8FAFC', margin: 0, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
              Study Context & Preview
            </h3>
          </div>

          {/* Active Retrieved Chunk Preview */}
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {activeSelectedChunk ? (
              <div
                style={{
                  background: '#1E293B',
                  border: '1px solid #334155',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: '#F8FAFC', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {activeSelectedChunk.source}
                  </span>
                  <span style={{ fontSize: '11px', background: '#334155', padding: '2px 6px', borderRadius: '4px', color: '#94A3B8' }}>
                    Page {activeSelectedChunk.page}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: '12px',
                    color: '#CBD5E1',
                    lineHeight: 1.6,
                    background: '#0F172A',
                    padding: '10px',
                    borderRadius: '6px',
                    maxHeight: '260px',
                    overflowY: 'auto',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {activeSelectedChunk.content}
                </div>
              </div>
            ) : (
              <div style={{ padding: '20px 10px', textAlign: 'center', color: '#64748B', fontSize: '12px' }}>
                <BookOpen size={24} style={{ opacity: 0.3, marginBottom: '6px' }} />
                <p>When you ask a question, the retrieved chunk from your notes will appear here.</p>
              </div>
            )}

            {/* OCR + RAG Live Pipeline Card */}
            <div
              style={{
                marginTop: 'auto',
                background: 'rgba(30, 41, 59, 0.7)',
                border: '1px solid #334155',
                borderRadius: '10px',
                padding: '14px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#F8FAFC', fontSize: '12px', fontWeight: 600, marginBottom: '10px' }}>
                <Zap size={14} style={{ color: '#F59E0B' }} />
                OCR + RAG Pipeline Status
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '11px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#10B981' }}>
                  <CheckCircle2 size={13} />
                  <span>PaddleOCR: Image & Scan Extraction</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#10B981' }}>
                  <CheckCircle2 size={13} />
                  <span>Weaviate: Isolated Tenant Shard</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#10B981' }}>
                  <CheckCircle2 size={13} />
                  <span>Gemini: Citations & Answer Synthesis</span>
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* ==========================================
          EXTRACTED TEXT & PRINT INSPECTION MODAL
      ========================================== */}
      {extractedTextModal.isOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '20px',
          }}
          onClick={() => setExtractedTextModal({ isOpen: false, filename: '', text: '' })}
        >
          <div
            style={{
              background: '#0F172A',
              border: '1px solid #334155',
              borderRadius: '16px',
              width: '100%',
              maxWidth: '750px',
              maxHeight: '85vh',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
              overflow: 'hidden',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '16px 20px',
                borderBottom: '1px solid #1E293B',
                background: '#1E293B',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <FileText size={18} style={{ color: '#818CF8' }} />
                <div>
                  <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#F8FAFC', margin: 0 }}>
                    Extracted Text Preview
                  </h3>
                  <span style={{ fontSize: '12px', color: '#94A3B8' }}>{extractedTextModal.filename}</span>
                </div>
                <span
                  style={{
                    background: 'rgba(99, 102, 241, 0.15)',
                    color: '#A5B4FC',
                    border: '1px solid rgba(99, 102, 241, 0.3)',
                    fontSize: '11px',
                    fontWeight: 600,
                    padding: '2px 8px',
                    borderRadius: '6px',
                  }}
                >
                  PP-OCRv5
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  onClick={() => handleCopyExtractedText(extractedTextModal.text)}
                  disabled={!extractedTextModal.text || extractedTextModal.loading}
                  style={{
                    background: '#334155',
                    border: 'none',
                    color: copiedExtracted ? '#10B981' : '#F8FAFC',
                    padding: '6px 12px',
                    borderRadius: '7px',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}
                >
                  {copiedExtracted ? <Check size={14} /> : <Copy size={14} />}
                  {copiedExtracted ? 'Copied' : 'Copy'}
                </button>

                <button
                  onClick={() => handlePrintText(extractedTextModal.filename, extractedTextModal.text)}
                  disabled={!extractedTextModal.text || extractedTextModal.loading}
                  style={{
                    background: '#10B981',
                    border: 'none',
                    color: '#FFFFFF',
                    padding: '6px 14px',
                    borderRadius: '7px',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}
                >
                  <Printer size={14} /> Print Text
                </button>

                <button
                  onClick={() => setExtractedTextModal({ isOpen: false, filename: '', text: '' })}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: '#94A3B8',
                    cursor: 'pointer',
                    padding: '6px',
                    borderRadius: '6px',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = '#FFFFFF')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = '#94A3B8')}
                >
                  <X size={18} />
                </button>
              </div>
            </div>

            {/* Modal Body */}
            <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
              {extractedTextModal.loading ? (
                <div style={{ padding: '40px', textAlign: 'center', color: '#94A3B8', fontSize: '13px' }}>
                  <RefreshCw size={24} className="animate-spin" style={{ margin: '0 auto 12px', color: '#818CF8' }} />
                  <p>Running Paddle PP-OCRv5 extraction on note...</p>
                </div>
              ) : (
                <pre
                  style={{
                    background: '#030712',
                    border: '1px solid #1F2937',
                    borderRadius: '10px',
                    padding: '16px',
                    color: '#E2E8F0',
                    fontSize: '13px',
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                    lineHeight: 1.65,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    margin: 0,
                  }}
                >
                  {extractedTextModal.text || 'No text detected in this document.'}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
