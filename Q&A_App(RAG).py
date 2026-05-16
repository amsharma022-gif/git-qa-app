"""RAG Document Q&A App

Build a chatbot that answers questions over your own documents using
retrieval-augmented generation (RAG).


Install required packages:
    pip install langchain_huggingface faiss-cpu pypdf numpy


The app supports:
- loading PDF and TXT documents from a folder
- splitting documents into retrievable chunks
- creating a FAISS vector store from embeddings
- answering user questions with OpenAI over retrieved context
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from pypdf import PdfReader

from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
INDEX_FILE = "index.faiss"
DATA_FILE = "store.pkl"


def get_openai_api_key() -> str:
    """Require an OpenAI API key before attempting any OpenAI calls."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY")
    if not key:
        raise EnvironmentError(
            "Missing OpenAI credentials. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY "
            "in your environment before running this script."
        )
    return key


def load_documents(data_dir: Path) -> List[Dict[str, Any]]:
    """Load text and PDF files from a directory."""
    documents: List[Dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                reader = PdfReader(path)
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:
                print(f"Warning: could not read PDF {path}: {exc}")
                continue
        elif suffix in {".txt", ".md", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                print(f"Warning: could not read text file {path}: {exc}")
                continue
        else:
            continue

        if text.strip():
            documents.append({"source": str(path), "text": text})
    return documents


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    text = text.replace("\r\n", "\n")
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_vector_store(documents: List[Dict[str, Any]], persist_dir: Path):
    """Build or load a FAISS index for the documents."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    index_path = persist_dir / INDEX_FILE
    data_path = persist_dir / DATA_FILE
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if index_path.exists() and data_path.exists():
        with open(data_path, "rb") as f:
            store = pickle.load(f)
        index = faiss.read_index(str(index_path))
        return index, store["chunks"], store["metadatas"], embeddings

    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for document in documents:
        for chunk in split_text(document["text"]):
            chunks.append(chunk)
            metadatas.append({"source": document["source"]})

    if not chunks:
        raise ValueError("No text chunks were generated from your documents.")

    vectors = embeddings.embed_documents(chunks)
    vector_array = np.array(vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(vector_array.shape[1])
    index.add(vector_array)

    with open(data_path, "wb") as f:
        pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)
    faiss.write_index(index, str(index_path))

    return index, chunks, metadatas, embeddings


def retrieve_chunks(query: str, index, chunks: List[str], metadatas: List[Dict[str, Any]], embeddings, k: int = 4):
    """Return the top-k chunks most similar to the query."""
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    results: List[Dict[str, Any]] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({"chunk": chunks[idx], "metadata": metadatas[idx], "distance": float(distance)})
    return results


def build_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Create a prompt containing retrieved context and the user question."""
    context = "\n\n".join(
        f"Source: {item['metadata']['source']}\n{item['chunk']}"
        for item in context_chunks
    )
    return (
        "Answer the user question using only the information in the provided context. "
        "If the answer cannot be found in the context, say that you do not know.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )


def create_qa_function(index, chunks, metadatas, embeddings):
    """Return a function that answers questions using retrieved context."""
    import os
    llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", 
            api_key=os.environ.get("GROQ_API_KEY"))

    def answer(question: str) -> str:
        relevant = retrieve_chunks(question, index, chunks, metadatas, embeddings, k=4)
        if not relevant:
            return "No relevant context found."
        prompt = build_prompt(question, relevant)
        return llm.invoke(prompt).content

    return answer


def create_sample_document(data_dir: Path) -> None:
    """Create a sample document in the default docs folder."""
    sample_path = data_dir / "sample.txt"
    if sample_path.exists():
        return
    sample_path.write_text(
        "This is a sample document for the RAG Q&A app.\n\n"
        "You can ask questions about this text, and the app will retrieve the relevant "
        "information from it.\n\n"
        "Example question: What is this document for?\n",
        encoding="utf-8",
    )
    print(f"Created sample document: {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-based Document Q&A App")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./docs"),
        help="Directory containing PDF/TXT documents",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("./vector_store"),
        help="Directory to persist the FAISS index",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        if args.data_dir == Path("./docs"):
            args.data_dir.mkdir(parents=True, exist_ok=True)
            create_sample_document(args.data_dir)
        else:
            raise FileNotFoundError(f"Document folder not found: {args.data_dir}")

    documents = load_documents(args.data_dir)
    if not documents:
        if args.data_dir == Path("./docs"):
            create_sample_document(args.data_dir)
            documents = load_documents(args.data_dir)
            if not documents:
                raise ValueError(
                    f"Could not create a sample document in {args.data_dir}."
                )
        else:
            raise ValueError(
                f"No readable documents found in {args.data_dir}. "
                "Add PDF, TXT, MD, or CSV documents and rerun."
            )

    index, chunks, metadatas, embeddings = build_vector_store(documents, args.persist_dir)
    qa = create_qa_function(index, chunks, metadatas, embeddings)

    print("\nRAG Q&A is ready. Ask a question, or type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if not question or question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        answer = qa(question)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
