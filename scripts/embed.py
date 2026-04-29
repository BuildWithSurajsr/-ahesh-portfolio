"""
Local-only script: Process PDF and create ChromaDB embeddings.
Run this whenever the PDF changes, then commit data/chroma_db/ to git.

Usage: python scripts/embed.py <path-to-pdf>
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
import fitz  # PyMuPDF

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portfolio")


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)


def create_embeddings(chunks: list, pdf_name: str):
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name="models/gemini-embedding-001"
    )

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_name, "chunk_id": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Created {len(chunks)} embeddings in ChromaDB at {CHROMA_DB_PATH}")


if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python scripts/embed.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Processing: {pdf_path}")

    text = extract_text(pdf_path)
    print(f"Extracted {len(text)} characters")

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    create_embeddings(chunks, os.path.basename(pdf_path))
    print("Done! Commit data/chroma_db/ to git and deploy.")
