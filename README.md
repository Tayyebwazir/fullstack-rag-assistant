# Full-Stack Multi-Tenant PDF RAG Assistant

A full-stack multi-tenant Retrieval-Augmented Generation (RAG) assistant application powered by **FastAPI**, **Weaviate Cloud**, **Google Gemini**, and a modern **React + TypeScript** frontend with **Firebase Authentication**.

---

## ✨ Features

- **Multi-Tenant Document Store**: Per-user tenant isolation in Weaviate Cloud based on authenticated Firebase UIDs.
- **Query Scope Selector**: Switch dynamically between querying the **Full Knowledge Base** or filtering search exclusively to a specific document (e.g., *Attention is all you need.pdf*, *LLM-based autonomous agents.pdf*, or *OOP_15_Page_Notes.pdf*).
- **Conversational Intent Router**: Intelligently classifies questions—bypasses vector retrieval and reranker for greetings (e.g., "Hello", "Hi") while reserving vector search for document-grounded inquiries.
- **Transparent Source Attribution**: Every retrieved chunk displays exact document metadata badges:
  - Source PDF name
  - Page number
  - Chunk sequence index
  - Chunk content snippet
  - JSON metadata export
- **Re-ranking Pipeline**: Scores and re-orders vector search results using Gemini before prompt compilation.
- **Real-Time Token Streaming**: Server-Sent Events (SSE) streaming answers word-by-word with immediate responsive feedback.
- **Secure Authentication**: Firebase Auth on the frontend with server-side ID token verification via Google's public key endpoints (no local service account key required).
- **Modern Responsive UI**: Dark-themed React + TypeScript UI built with Vite and Lucide icons.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, PyMuPDF (fitz), Google GenAI SDK (`google-genai`), `weaviate-client` v4.
- **Vector Database**: Weaviate Cloud (Multi-Tenancy enabled, Cosine distance).
- **LLM & Embeddings**: Google Gemini (`gemini-embedding-001`, `gemini-3.5-flash-lite`).
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide React, Firebase JS SDK.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Tayyebwazir/fullstack-rag-assistant.git
cd fullstack-rag-assistant
```

### 2. Backend Setup
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Provide your **Gemini API Key**, **Weaviate Cloud URL & API Key**, and **Firebase Project ID**.
4. Ingest sample documents into Weaviate:
   ```bash
   python -m app.main
   ```
5. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Configure frontend environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Add your Firebase web configuration keys.
4. Launch the Vite development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your browser.

---

## ⚠️ Known Limitations

- **Weaviate Cloud Sandbox Limits**: Free-tier Weaviate Cloud sandboxes restrict the number of active tenants (typically 3 active tenants) and expire after 14 days without an upgraded cluster plan.
- **Synchronous Ingestion Pipeline**: PDF text extraction, chunking, and embedding generation currently execute synchronously during the ingestion request. Very large files or batch uploads may experience high response latency without an asynchronous task queue (e.g., Celery, RQ, or Cloud Tasks).
- **Rate Limit Storage**: Ingestion rate limits are persisted using a file-backed JSON store (`rate_limits.json`). For horizontally scaled multi-worker or containerized deployments, a distributed database or memory store (such as Redis or Firestore) is recommended.
- **In-Memory Query Vector Caching**: Frequently searched query vectors are cached in-process to optimize latency and quota usage, but cache contents reset upon application process restarts.

---

## 🔒 Security & Secrets

All `.env` files, private keys, credentials, and virtual environment directories are strictly excluded in `.gitignore`. **Never commit actual API keys or credentials to version control.**

---

## 📄 License
MIT License.
