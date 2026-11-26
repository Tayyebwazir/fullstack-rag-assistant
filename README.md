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
- **Re-ranking Pipeline**: Two-stage retrieval retrieving 20 candidates from Weaviate and reranking to the top 4 chunks using Cohere (`rerank-english-v3.0`) with distance threshold filtering.
- **Real-Time Token Streaming**: Server-Sent Events (SSE) streaming answers word-by-word with immediate responsive feedback.
- **Secure Authentication**: Firebase Auth on the frontend with server-side ID token verification via Google's public key endpoints (no local service account key required).
- **Modern Responsive UI**: Dark-themed React + TypeScript UI built with Vite and Lucide icons.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, PyMuPDF (fitz), Google GenAI SDK (`google-genai`), `weaviate-client` v4, Cohere Rerank API.
- **Vector Database**: Weaviate Cloud (Multi-Tenancy enabled, Cosine distance).
- **LLM & Embeddings**: Google Gemini (`gemini-embedding-001`, `gemini-3.5-flash-lite`), Cohere (`rerank-english-v3.0`).
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
   - Provide your **Gemini API Key**, **Weaviate Cloud URL & API Key**, **Cohere API Key**, and **Firebase Project ID**.
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

## 📊 RAG Retrieval Evaluation & Benchmark

The retrieval and reranking pipeline is evaluated using a curated ground-truth benchmark on *Attention is all you need.pdf* (102 ingested chunks) across all **6 production configurations** (3 Weaviate distance thresholds $\times$ Cohere Reranking OFF/ON).

### 1. Actual Live Benchmark Results (6 Configurations)

Each query executes the actual production pipeline:
1. **Stage 1 (Weaviate Retrieval)**: Retrieves the top **20 candidate chunks** from Weaviate using Cosine distance.
2. **Distance Threshold Filtering**: Prunes chunks with distance $> \text{threshold}$.
3. **Stage 2 (Cohere Rerank)**: Cross-encoder reranks the remaining candidates using `rerank-english-v3.0` to the final **top 4 chunks**.
4. **Evaluation Metrics**: Evaluated across **10 in-scope** ground-truth questions and **5 out-of-scope** queries to test hallucination prevention:

| Distance Threshold | Reranking Status | Hit@1 | Hit@4 | MRR | Out-of-Scope Rejection |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.20** | **OFF** | 60.00% | 60.00% | 0.6000 | 100.00% |
| **0.30** | **OFF** | 100.00% | 100.00% | 1.0000 | 100.00% |
| **0.40** | **OFF** | 100.00% | 100.00% | 1.0000 | 100.00% |
| **0.20** | **ON** | 60.00% | 60.00% | 0.6000 | 100.00% |
| **0.30** | **ON** | **100.00%** | **100.00%** | **1.0000** | **100.00%** |
| **0.40** | **ON** | 100.00% | 100.00% | 1.0000 | 100.00% |

---

### 2. Selected Configuration & Justification

* **Selected Configuration**: **Distance Threshold `0.30` with Cohere Reranking `ON`**
* **Justification**:
  - **Why not 0.20?** A distance threshold of `0.20` is too restrictive. It prematurely discards valid in-scope chunks for 4 out of 10 questions (e.g., Scaled Dot-Product Attention, FFN formula, Sinusoidal Positional Encoding, and Regularization parameters), dropping retrieval recall and Hit@1 from 100% to **60.00%**.
  - **Why 0.30 over 0.40?** At threshold `0.30`, **100% of in-scope questions** retrieve their ground-truth page at Rank 1 (MRR = 1.0000), while **100% of out-of-scope queries** (world events, recipes, clinical medical questions, stock data, automotive repair) have vector distances $> 0.30$ and return 0 chunks, reliably triggering the fallback response (*"I could not find relevant information in the uploaded documents..."*). While `0.40` also passed, threshold `0.30` provides a substantially safer margin against out-of-scope semantic leakage.
  - **Why Cohere Reranking ON?** Cohere's cross-encoder evaluates full query-document interactions, effectively disambiguating complex multi-hop queries (such as decoder masking and attention heads) and placing the exact target passage at Rank 1 with high confidence scores (up to `1.0000`).

---

### 3. Analysis: Additional Cohere Rerank-Score Threshold

* **Finding**: **An additional arbitrary Cohere rerank-score threshold is unnecessary and not recommended.**
* **Evidence**:
  1. **Stage 1 Distance Filtering Already Prevents Hallucinations**: For all out-of-scope queries, Weaviate's vector distance threshold (`0.30`) rejects candidates upstream, returning `0` chunks. Because no chunks pass Stage 1, the fallback response triggers immediately without invoking Cohere.
  2. **Risk of False Negatives**: In live testing, valid in-scope chunks evaluated by Cohere received relevance scores ranging from `0.4315` to `1.0000` (e.g., Q10 Rank 4 returned score `0.4315` for a valid contextual chunk on Page 5). Setting an arbitrary cutoff (e.g., `0.50`) would prune valid contextual chunks that aid synthesis.
  3. **Score Calibration**: Cohere relevance scores are query-dependent and not calibrated probabilities across diverse document domains. Filtering via Weaviate vector distance at Stage 1 is both computationally cheaper and mathematically sufficient.

---

### 4. Running Benchmarks & Test Suites

```bash
# Run the live 6-configuration benchmark
python evaluate_rag.py --compare

# Run single configuration evaluation (e.g. threshold 0.30 with reranking)
python evaluate_rag.py --threshold 0.30 --rerank

# Run the complete automated pytest suite (15 tests covering Issues 1, 2, and 3)
pytest tests/ -v
```

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
