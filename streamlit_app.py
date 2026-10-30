import time
import uuid
import streamlit as st
from weaviate.classes.query import Filter
from app.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    COLLECTION_NAME,
    WEAVIATE_MODE,
    GENERATION_MODEL,
    EMBEDDING_MODEL,
)
from app.services.pdf_loader import load_pdf
from app.services.chunker import chunk_pages
from app.services.embeddings import embed_query, embed_document, embed_documents
from app.services.vector_store import search_chunks, connect_weaviate, ensure_collection
try:
    from app.services.reranker import rerank_chunks
except (ImportError, AttributeError):
    def rerank_chunks(question="", chunks=None, top_k=None):
        return []
from google import genai
from app.config import GEMINI_API_KEY


st.set_page_config(
    page_title="PDF RAG System - Weaviate & Gemini",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 5px;
        color: #1E293B;
    }
    .sub-title {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 12px;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .answer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px;
        border-radius: 6px;
        font-size: 14px;
        line-height: 1.6;
        color: #1E293B;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ingest_chunks_streamlit(chunks: list[dict], source: str, vectors: list[list[float]]) -> int:
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one vector")
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            ensure_collection(client)
        collection = client.collections.get(COLLECTION_NAME)
        if collection is None:
            raise RuntimeError(f"Failed to access collection: {COLLECTION_NAME}")

        collection.data.delete_many(where=Filter.by_property("source").equal(source))
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            object_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{index}")
            collection.data.insert(
                uuid=object_id,
                properties={
                    "content": chunk["content"],
                    "source": source,
                    "page": chunk["page"],
                    "chunk_index": index,
                },
                vector=vector,
            )
        return len(chunks)


def generate_answer_with_retry(prompt: str, model_name: str = GENERATION_MODEL) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    candidate_models = [model_name]
    for alt in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]:
        if alt not in candidate_models:
            candidate_models.append(alt)

    last_error = None
    for target_model in candidate_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    # Switch to next available model that has fresh quota
                    break
                if "503" in err_str or "high demand" in err_str.lower() or "UNAVAILABLE" in err_str:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    raise last_error or RuntimeError("Gemini model is currently unavailable. Please retry in a few moments.")


if "qa_result" not in st.session_state:
    st.session_state.qa_result = None
if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = None


with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**Document:** `{PDF_PATH.name}`")
    st.write(f"**Collection:** `{COLLECTION_NAME}`")
    st.write(f"**Mode:** `{WEAVIATE_MODE}`")
    st.write(f"**Embedding:** `{EMBEDDING_MODEL}`")

    selected_model = st.selectbox("Generation Model", options=["gemini-3.5-flash", "gemini-3.5-flash-lite"], index=0)

    top_k_val = st.slider("Top K Chunks", min_value=1, max_value=10, value=TOP_K)

    st.markdown("---")
    st.subheader("📥 Ingestion")
    if st.button("Ingest Document into Weaviate", use_container_width=True):
        with st.spinner("Extracting, chunking & ingesting..."):
            try:
                pdf_folder = PDF_PATH.parent
                pdf_files = sorted(list(pdf_folder.glob("*.pdf"))) if pdf_folder.exists() else [PDF_PATH]
                total_chunks = 0
                for pdf_file in pdf_files:
                    pages = load_pdf(pdf_file)
                    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                    contents = [c["content"] for c in chunks]
                    vectors = embed_documents(contents)
                    total = ingest_chunks_streamlit(chunks=chunks, source=pdf_file.name, vectors=vectors)
                    total_chunks += total
                st.session_state.ingest_status = f"✅ Ingested {total_chunks} chunks from {len(pdf_files)} PDFs successfully!"
                st.success(st.session_state.ingest_status)
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    if st.session_state.ingest_status:
        st.info(st.session_state.ingest_status)


st.markdown('<div class="main-title">📚 Document RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explore retrieval chunking, ask questions, and inspect reranked results side by side.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.1, 1.3, 0.9], gap="medium")

with col1:
    st.subheader("📄 Retrieved Chunks & Metadata")
    st.caption("Raw vector search results from Weaviate")
    chunks_container = st.container()

with col2:
    st.subheader("💬 Question & Answer")
    st.caption("Powered by Gemini Generation")

    sample_questions = [
        "What is Object-Oriented Programming (OOP)?",
        "What is polymorphism and what are its types?",
        "Explain encapsulation and data hiding with examples.",
        "What is the difference between abstraction and encapsulation?",
    ]
    selected_sample = st.selectbox(
        "Sample Questions:",
        options=["-- Select a sample or type below --"] + sample_questions,
    )

    default_q = ""
    if selected_sample != "-- Select a sample or type below --":
        default_q = selected_sample

    question_input = st.text_area(
        "Enter your question:",
        value=default_q,
        height=100,
        placeholder="e.g. What is method overriding in OOP?",
    )

    ask_button_clicked = st.button("🚀 Ask Question", use_container_width=True, type="primary")
    answer_container = st.container()

with col3:
    st.subheader("🎯 Reranker")
    st.caption("Backend Reranking Module")
    reranker_container = st.container()


if ask_button_clicked:
    if not question_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving chunks and generating answer..."):
            try:
                q_vector = embed_query(question_input)
                retrieved_chunks = search_chunks(query_vector=q_vector, limit=top_k_val)

                if not retrieved_chunks:
                    st.session_state.qa_result = {
                        "question": question_input,
                        "answer": "No relevant information found in the ingested document.",
                        "chunks": [],
                    }
                else:
                    context_parts = [
                        f"[source:{c['source']},page:{c['page']}]\n{c['content']}"
                        for c in retrieved_chunks
                    ]
                    context = "\n\n".join(context_parts)
                    prompt = f"""You are a document question answering assistant.
Context:
{context}

Question:
{question_input}
"""
                    answer_text = generate_answer_with_retry(prompt=prompt, model_name=selected_model)
                    reranked_results = rerank_chunks(question=question_input, chunks=retrieved_chunks, top_k=top_k_val)

                    st.session_state.qa_result = {
                        "question": question_input,
                        "answer": answer_text,
                        "chunks": retrieved_chunks,
                        "reranked_chunks": reranked_results,
                    }

            except Exception as e:
                st.error(f"Error during query: {e}")


if st.session_state.qa_result:
    res = st.session_state.qa_result
    chunks_list = res.get("chunks", [])

    with chunks_container:
        if chunks_list:
            st.success(f"Retrieved **{len(chunks_list)}** chunks for this question:")
            for idx, chunk in enumerate(chunks_list, start=1):
                p_num = chunk.get("page", "N/A")
                c_idx = chunk.get("chunk_index", idx - 1)
                text_content = chunk.get("content", "")

                with st.expander(f"📦 Chunk #{idx} (Page {p_num})", expanded=True):
                    st.markdown(
                        f"""
                        <span class="badge">📄 Page: {p_num}</span>
                        <span class="badge">🔢 Index: {c_idx}</span>
                        <span class="badge">📏 {len(text_content)} chars</span>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.text_area(
                        label=f"Chunk Content #{idx}",
                        value=text_content,
                        height=140,
                        disabled=True,
                        key=f"chunk_display_{idx}_{p_num}_{c_idx}",
                    )
        else:
            st.warning("No chunks were retrieved from Weaviate for this query.")

    with answer_container:
        st.markdown("### Answer")
        st.markdown(f'<div class="answer-box">{res.get("answer", "")}</div>', unsafe_allow_html=True)

    with reranker_container:
        reranked_list = res.get("reranked_chunks", [])
        if reranked_list:
            st.success(f"Reranked **{len(reranked_list)}** chunks:")
            for item in reranked_list:
                new_pos = item.get("rerank_position", 1)
                orig_pos = item.get("original_rank", 1)
                score_val = item.get("relevance_score", 0.0)
                p_num = item.get("page", "N/A")
                text_content = item.get("content", "")
                with st.expander(f"🎯 Rank #{new_pos} (Page {p_num})", expanded=True):
                    st.markdown(
                        f"""
                        <span class="badge">Rerank: #{new_pos}</span>
                        <span class="badge">Original: #{orig_pos}</span>
                        <span class="badge">Score: {score_val:.1f}/10</span>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.text_area(
                        label=f"Rerank Content #{new_pos}",
                        value=text_content,
                        height=140,
                        disabled=True,
                        key=f"rerank_display_{new_pos}_{p_num}",
                    )
        else:
            st.warning("No reranked results.")

else:
    with chunks_container:
        st.info("Ask a question in the middle panel to view retrieved chunks and metadata.")
    with reranker_container:
        st.info("Reranked chunks and relevance scores will appear here after asking a question.")
