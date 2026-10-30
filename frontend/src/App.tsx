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
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

interface Chunk {
  content: string;
  source: string;
  page: number;
  chunk_index: number;
}

interface RerankedChunk extends Chunk {
  original_rank?: number;
  relevance_score?: number;
  rerank_position?: number;
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

  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<{ success: boolean; msg: string } | null>(null);

  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [availableSources, setAvailableSources] = useState<string[]>([]);
  const [selectedSource, setSelectedSource] = useState<string>('all');

  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);

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

    // Fetch available PDF source documents
    fetch(`${API_BASE}/sources`)
      .then((res) => res.json())
      .then((data) => {
        if (data.sources) setAvailableSources(data.sources);
      })
      .catch(() => {});
  }, []);

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

  // Handle Ingest
  const handleIngest = async () => {
    setIsIngesting(true);
    setIngestStatus(null);
    setErrorMsg(null);

    try {
      const token = await currentUser?.getIdToken();
      const headers: Record<string, string> = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;

      const res = await fetch(`${API_BASE}/ingest`, { method: 'POST', headers });
      const data = await res.json();

      if (res.ok) {
        setIngestStatus({
          success: true,
          msg: `Ingested ${data.chunks_ingested || 58} chunks from ${data.pdf || 'document'}`,
        });
      } else {
        setIngestStatus({
          success: false,
          msg: data.detail || 'Failed to ingest document.',
        });
      }
    } catch (err: any) {
      setIngestStatus({
        success: false,
        msg: err.message || 'Cannot connect to backend server.',
      });
    } finally {
      setIsIngesting(false);
    }
  };

  // Handle Ask with SSE Streaming
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
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
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
              // Ignore non-json or fragmented SSE lines
            }
          }
        }
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Query failed. Please check the backend connection.');
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

  // 1. Show Authentication Page first if not logged in
  if (!currentUser) {
    return <AuthPage />;
  }

  // 2. Once authenticated, open the RAG system
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', backgroundColor: '#0B1120' }}>
      {/* Top Navbar */}
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 24px',
          borderBottom: '1px solid #1E293B',
          backgroundColor: '#0F172A',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '36px',
              height: '36px',
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
            <h1 style={{ fontSize: '16px', fontWeight: 700, color: '#F8FAFC', margin: 0 }}>
              PDF RAG Assistant
            </h1>
            <p style={{ fontSize: '11px', color: '#64748B', margin: 0 }}>
              TypeScript + Node.js Frontend • Gemini + Weaviate Cloud
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '20px',
              fontSize: '12px',
              backgroundColor: backendOnline ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              color: backendOnline ? '#34D399' : '#F87171',
              border: `1px solid ${backendOnline ? '#059669' : '#DC2626'}`,
            }}
          >
            <div
              style={{
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                backgroundColor: backendOnline ? '#10B981' : '#EF4444',
              }}
            />
            {backendOnline ? 'Backend Online (Port 8000)' : 'Backend Offline'}
          </div>

          <div
            style={{
              padding: '4px 10px',
              borderRadius: '20px',
              fontSize: '11px',
              backgroundColor: '#1E293B',
              color: '#94A3B8',
              border: '1px solid #334155',
            }}
          >
            Model: <strong style={{ color: '#E2E8F0' }}>gemini-3.5-flash-lite</strong>
          </div>

          {/* User Auth Profile / Login Button */}
          {currentUser ? (
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
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: '#3B82F6',
                  color: '#FFFFFF',
                  fontSize: '11px',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                }}
              >
                {currentUser.photoURL ? (
                  <img src={currentUser.photoURL} alt="User" style={{ width: '100%', height: '100%' }} />
                ) : (
                  (currentUser.email?.[0] || 'U').toUpperCase()
                )}
              </div>
              <span style={{ fontSize: '11px', color: '#E2E8F0', maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {currentUser.displayName || currentUser.email}
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
                <LogOut size={14} />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setAuthModalOpen(true)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '5px 12px',
                borderRadius: '20px',
                fontSize: '12px',
                fontWeight: 600,
                backgroundColor: '#3B82F6',
                color: '#FFFFFF',
                border: 'none',
                cursor: 'pointer',
                transition: 'background-color 0.15s',
              }}
            >
              <LogIn size={13} /> Sign In
            </button>
          )}
        </div>
      </header>

      {/* Main 3-Column Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr 340px', flex: 1, overflow: 'hidden' }}>
        {/* ======================================================== */}
        {/* COLUMN 1: INGESTION BUTTON & RETRIEVED CHUNKS */}
        {/* ======================================================== */}
        <aside
          style={{
            borderRight: '1px solid #1E293B',
            backgroundColor: '#0F172A',
            display: 'flex',
            flexDirection: 'column',
            overflowY: 'auto',
            padding: '16px',
            gap: '16px',
          }}
        >
          {/* Ingestion Card */}
          <div
            style={{
              backgroundColor: '#1E293B',
              borderRadius: '10px',
              padding: '14px',
              border: '1px solid #334155',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <Database size={16} color="#3B82F6" />
              <h2 style={{ fontSize: '13px', fontWeight: 600, color: '#F1F5F9', margin: 0 }}>
                Document Ingestion
              </h2>
            </div>
            <p style={{ fontSize: '11px', color: '#94A3B8', marginBottom: '12px' }}>
              Extract, chunk, embed, and upload the notes PDF to Weaviate Cloud.
            </p>

            <button
              onClick={handleIngest}
              disabled={isIngesting}
              style={{
                width: '100%',
                padding: '10px 14px',
                borderRadius: '8px',
                border: 'none',
                backgroundColor: isIngesting ? '#475569' : '#3B82F6',
                color: '#FFFFFF',
                fontWeight: 600,
                fontSize: '13px',
                cursor: isIngesting ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'background-color 0.2s',
              }}
            >
              {isIngesting ? (
                <>
                  <RefreshCw size={15} className="pulsing" /> Ingesting Chunks...
                </>
              ) : (
                <>
                  <Zap size={15} /> Ingest to Weaviate
                </>
              )}
            </button>

            {ingestStatus && (
              <div
                style={{
                  marginTop: '10px',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  fontSize: '11px',
                  backgroundColor: ingestStatus.success ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                  color: ingestStatus.success ? '#34D399' : '#F87171',
                  border: `1px solid ${ingestStatus.success ? '#059669' : '#DC2626'}`,
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '6px',
                }}
              >
                {ingestStatus.success ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                <span>{ingestStatus.msg}</span>
              </div>
            )}
          </div>

          {/* Retrieved Chunks Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Layers size={15} color="#8B5CF6" />
              <h2 style={{ fontSize: '13px', fontWeight: 600, color: '#F1F5F9', margin: 0 }}>
                Retrieved Chunks
              </h2>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              {retrievedChunks.length > 0 && (
                <button
                  onClick={() => {
                    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(retrievedChunks, null, 2));
                    const downloadAnchor = document.createElement('a');
                    downloadAnchor.setAttribute('href', dataStr);
                    downloadAnchor.setAttribute('download', 'retrieved_chunks.json');
                    document.body.appendChild(downloadAnchor);
                    downloadAnchor.click();
                    downloadAnchor.remove();
                  }}
                  title="Download chunks as JSON"
                  style={{
                    backgroundColor: '#1E293B',
                    border: '1px solid #334155',
                    color: '#38BDF8',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    fontSize: '10px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '3px',
                  }}
                >
                  <Copy size={10} /> JSON
                </button>
              )}
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
          </div>

          {/* Chunks List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', flex: 1 }}>
            {retrievedChunks.length === 0 ? (
              <div
                style={{
                  padding: '20px 12px',
                  textAlign: 'center',
                  color: '#64748B',
                  fontSize: '12px',
                  border: '1px dashed #334155',
                  borderRadius: '8px',
                }}
              >
                Ask a question to see raw chunks retrieved by Weaviate vector search.
              </div>
            ) : (
              retrievedChunks.map((chunk, idx) => (
                <div
                  key={idx}
                  style={{
                    backgroundColor: '#1E293B',
                    borderRadius: '8px',
                    padding: '12px',
                    border: '1px solid #334155',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                  }}
                >
                  {/* Header: Chunk #, Page #, Chunk Index */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      marginBottom: '8px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ fontSize: '12px', fontWeight: 700, color: '#38BDF8' }}>
                        Chunk #{idx + 1}
                      </span>
                      <span
                        style={{
                          fontSize: '10px',
                          color: '#64748B',
                        }}
                      >
                        (index: {chunk.chunk_index})
                      </span>
                    </div>

                    <span
                      style={{
                        fontSize: '10px',
                        backgroundColor: '#0F172A',
                        color: '#E2E8F0',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        border: '1px solid #334155',
                        fontWeight: 600,
                      }}
                    >
                      Page {chunk.page}
                    </span>
                  </div>

                  {/* Document Source Metadata Badge */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '11px',
                      color: '#93C5FD',
                      backgroundColor: 'rgba(59, 130, 246, 0.15)',
                      border: '1px solid rgba(59, 130, 246, 0.35)',
                      borderRadius: '6px',
                      padding: '4px 8px',
                      marginBottom: '8px',
                    }}
                  >
                    <FileText size={13} style={{ flexShrink: 0, color: '#3B82F6' }} />
                    <span
                      style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        fontWeight: 500,
                      }}
                      title={chunk.source}
                    >
                      {chunk.source}
                    </span>
                  </div>

                  {/* Chunk Content */}
                  <div
                    style={{
                      fontSize: '11px',
                      color: '#CBD5E1',
                      backgroundColor: '#0F172A',
                      padding: '10px',
                      borderRadius: '6px',
                      maxHeight: '130px',
                      overflowY: 'auto',
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'var(--font-mono)',
                      lineHeight: 1.5,
                      border: '1px solid #1E293B',
                    }}
                  >
                    {chunk.content}
                  </div>
                </div>
              ))
            )}
          </div>
        </aside>

        {/* ======================================================== */}
        {/* COLUMN 2: STREAMING TOKEN QUESTION & ANSWER (MIDDLE) */}
        {/* ======================================================== */}
        <main
          style={{
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: '#0B1120',
            overflow: 'hidden',
            padding: '20px 24px',
            gap: '16px',
          }}
        >
          {/* Single Document Filter Dropdown */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
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
              <option value="all">📚 All Ingested Documents (Full Knowledge Base)</option>
              {availableSources.map((src, i) => (
                <option key={i} value={src}>
                  📄 {src}
                </option>
              ))}
            </select>
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
              placeholder="Ask anything "
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
                Press <strong>Ctrl + Enter</strong> to ask
              </span>

              <button
                onClick={() => handleAsk()}
                disabled={isStreaming || !question.trim()}
                style={{
                  padding: '8px 18px',
                  borderRadius: '6px',
                  border: 'none',
                  backgroundColor: isStreaming || !question.trim() ? '#475569' : '#3B82F6',
                  color: '#FFFFFF',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: isStreaming || !question.trim() ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                {isStreaming ? (
                  <>
                    <RefreshCw size={13} className="pulsing" /> Streaming...
                  </>
                ) : (
                  <>
                    <Send size={13} /> Ask Question
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
            {/* Answer Header with Streaming Stats */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '10px 16px',
                borderBottom: '1px solid #334155',
                backgroundColor: '#182234',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sparkles size={16} color="#3B82F6" />
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>
                  Streaming Answer
                </span>
                {isStreaming && (
                  <span
                    style={{
                      fontSize: '11px',
                      backgroundColor: 'rgba(59, 130, 246, 0.2)',
                      color: '#60A5FA',
                      padding: '2px 8px',
                      borderRadius: '10px',
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
                    {streamedAnswer}
                  </ReactMarkdown>
                  {isStreaming && <span className="cursor-blink" />}
                </div>
              ) : (
                <div style={{ color: '#64748B', textAlign: 'center', marginTop: '60px' }}>
                  <BookOpen size={28} style={{ margin: '0 auto 8px', opacity: 0.5 }} />
                  <p>Type your question above to see live token generation.</p>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* ======================================================== */}
        {/* COLUMN 3: RERANKER (RIGHT SIDE) */}
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
              <Award size={16} color="#10B981" />
              <h2 style={{ fontSize: '13px', fontWeight: 600, color: '#F1F5F9', margin: 0 }}>
                Reranker
              </h2>
            </div>
            <span
              style={{
                fontSize: '10px',
                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                color: '#34D399',
                padding: '2px 8px',
                borderRadius: '12px',
                border: '1px solid #059669',
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
                  padding: '20px 12px',
                  textAlign: 'center',
                  color: '#64748B',
                  fontSize: '12px',
                  border: '1px dashed #334155',
                  borderRadius: '8px',
                }}
              >
                Reranked results with relevance scores will appear here after asking a question.
              </div>
            ) : (
              rerankedChunks.map((item, idx) => {
                const score = item.relevance_score || 0;
                const scorePct = Math.min(100, (score / 10) * 100);

                return (
                  <div
                    key={idx}
                    style={{
                      backgroundColor: '#1E293B',
                      borderRadius: '8px',
                      padding: '10px',
                      border: '1px solid #334155',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '6px',
                      }}
                    >
                      <span
                        style={{
                          fontSize: '11px',
                          fontWeight: 700,
                          backgroundColor: '#10B981',
                          color: '#FFFFFF',
                          padding: '1px 6px',
                          borderRadius: '4px',
                        }}
                      >
                        Rank #{item.rerank_position || idx + 1}
                      </span>

                      <div style={{ display: 'flex', gap: '4px' }}>
                        <span
                          style={{
                            fontSize: '10px',
                            backgroundColor: '#0F172A',
                            color: '#94A3B8',
                            padding: '1px 6px',
                            borderRadius: '4px',
                            border: '1px solid #334155',
                          }}
                        >
                          Orig: #{item.original_rank || idx + 1}
                        </span>
                        <span
                          style={{
                            fontSize: '10px',
                            backgroundColor: '#0F172A',
                            color: '#34D399',
                            padding: '1px 6px',
                            borderRadius: '4px',
                            border: '1px solid #059669',
                          }}
                        >
                          Score: {score.toFixed(1)}/10
                        </span>
                      </div>
                    </div>

                    {/* Score Bar */}
                    <div
                      style={{
                        width: '100%',
                        height: '4px',
                        backgroundColor: '#0F172A',
                        borderRadius: '2px',
                        overflow: 'hidden',
                        marginBottom: '6px',
                      }}
                    >
                      <div
                        style={{
                          width: `${scorePct}%`,
                          height: '100%',
                          backgroundColor: '#10B981',
                        }}
                      />
                    </div>

                    <div
                      style={{
                        fontSize: '11px',
                        color: '#CBD5E1',
                        backgroundColor: '#0F172A',
                        padding: '8px',
                        borderRadius: '6px',
                        maxHeight: '110px',
                        overflowY: 'auto',
                        whiteSpace: 'pre-wrap',
                        fontFamily: 'var(--font-mono)',
                        lineHeight: 1.4,
                      }}
                    >
                      {item.content}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>
      </div>

      {/* Firebase Authentication Modal */}
      <AuthModal isOpen={authModalOpen} onClose={() => setAuthModalOpen(false)} />
    </div>
  );
}

