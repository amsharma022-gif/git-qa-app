import os
import pickle
import tempfile
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
INDEX_FILE    = "index.faiss"
DATA_FILE     = "store.pkl"
PERSIST_DIR   = Path("./vector_store")
DOCS_DIR      = Path("./docs")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A App",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .source-box {
        background: #1e1e2e;
        border-left: 3px solid #6366f1;
        padding: 0.5rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: #a0a0b0;
        margin-top: 0.3rem;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ───────────────────────────────────────────────────────────
def load_documents(data_dir: Path):
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                reader = PdfReader(path)
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                continue
        elif suffix in {".txt", ".md", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
        else:
            continue
        if text.strip():
            documents.append({"source": path.name, "text": text})
    return documents


def split_text(text):
    text = text.replace("\r\n", "\n")
    if len(text) <= CHUNK_SIZE:
        return [text.strip()]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vector_store(documents, embeddings):
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    index_path = PERSIST_DIR / INDEX_FILE
    data_path  = PERSIST_DIR / DATA_FILE

    if index_path.exists() and data_path.exists():
        with open(data_path, "rb") as f:
            store = pickle.load(f)
        index = faiss.read_index(str(index_path))
        return index, store["chunks"], store["metadatas"]

    chunks, metadatas = [], []
    for doc in documents:
        for chunk in split_text(doc["text"]):
            chunks.append(chunk)
            metadatas.append({"source": doc["source"]})

    vectors      = embeddings.embed_documents(chunks)
    vector_array = np.array(vectors, dtype=np.float32)
    index        = faiss.IndexFlatL2(vector_array.shape[1])
    index.add(vector_array)

    with open(data_path, "wb") as f:
        pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)
    faiss.write_index(index, str(index_path))
    return index, chunks, metadatas


def retrieve_chunks(query, index, chunks, metadatas, embeddings, k=4):
    query_vector         = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    distances, indices   = index.search(query_vector, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append({"chunk": chunks[idx], "source": metadatas[idx]["source"], "distance": float(dist)})
    return results


def build_prompt(question, context_chunks, history):
    history_text = ""
    for msg in history[-4:]:  # last 4 exchanges for context
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['chunk']}" for c in context_chunks
    )
    return (
        "You are a helpful assistant. Answer using only the provided context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Conversation so far:\n{history_text}\n"
        f"User: {question}\nAssistant:"
    )


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", api_key=api_key)


def process_uploaded_file(uploaded_file):
    """Save uploaded file to docs/ and clear vector store cache."""
    DOCS_DIR.mkdir(exist_ok=True)
    save_path = DOCS_DIR / uploaded_file.name
    save_path.write_bytes(uploaded_file.getbuffer())
    if PERSIST_DIR.exists():
        import shutil
        shutil.rmtree(PERSIST_DIR)
    return save_path


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RAG Q&A App")
    st.markdown("Ask questions about your documents.")
    st.divider()

    # # API key input
    # groq_key = st.text_input("Groq API Key", type="password",
    #                           value=os.environ.get("GROQ_API_KEY", ""),
    #                           help="Get free key at console.groq.com")
    # if groq_key:
    #     os.environ["GROQ_API_KEY"] = groq_key

    st.divider()

    # File uploader
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for uf in uploaded_files:
            process_uploaded_file(uf)
        st.success(f"{len(uploaded_files)} file(s) uploaded!")

    st.divider()

    # Show loaded docs
    st.markdown("### Loaded Documents")
    if DOCS_DIR.exists():
        files = list(DOCS_DIR.rglob("*"))
        files = [f for f in files if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md", ".csv"}]
        if files:
            for f in files:
                st.markdown(f"- `{f.name}`")
        else:
            st.info("No documents yet. Upload files above.")
    else:
        st.info("No documents yet. Upload files above.")

    st.divider()

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header"> RAG Document Q&A</p>', unsafe_allow_html=True)
st.caption("Ask anything about your uploaded documents.")

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load embeddings and build index
embeddings = get_embeddings()

DOCS_DIR.mkdir(exist_ok=True)
documents = load_documents(DOCS_DIR)

# Re-save any files still in the uploader
if uploaded_files:
    for uf in uploaded_files:
        save_path = DOCS_DIR / uf.name
        if not save_path.exists():
            save_path.write_bytes(uf.getbuffer())
    documents = load_documents(DOCS_DIR)  # reload after re-saving

if not documents:
    st.warning("No documents found. Upload files using the sidebar.")
    st.stop()

with st.spinner("Indexing documents..."):
    index, chunks, metadatas = build_vector_store(documents, embeddings)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f'<div class="source-box"> {src}</div>', unsafe_allow_html=True)

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    llm = get_llm()
    if not llm:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant = retrieve_chunks(question, index, chunks, metadatas, embeddings)
            prompt   = build_prompt(question, relevant, st.session_state.messages)
            answer   = llm.invoke(prompt).content
            sources  = list(dict.fromkeys(c["source"] for c in relevant))

        st.markdown(answer)
        with st.expander("Sources"):
            for src in sources:
                st.markdown(f'<div class="source-box">{src}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })