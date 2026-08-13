"""
rag.py — Retrieval-Augmented Generation pipeline (FREE / fully local version).

Responsibilities:
1. Load a PDF and extract text
2. Split text into overlapping chunks
3. Embed chunks with a LOCAL sentence-transformers model (no API key, no cost)
4. Store/query chunks in a persistent ChromaDB collection (also local)
"""

import os
import uuid
from typing import List

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Persistent local vector store (no external DB needed for the MVP)
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "vectorstore")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# Force cosine distance explicitly (0 = identical, 2 = opposite) so relevance
# scores are meaningful, instead of ChromaDB's raw default metric.
collection = chroma_client.get_or_create_collection(
    name="company_docs",
    metadata={"hnsw:space": "cosine"},
)

# Local embedding model — downloads once (~90MB) on first run, then runs
# fully offline on your CPU. Multilingual so it handles German text well.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model once and reuse it."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into chunks along natural boundaries (paragraphs, then
    sentences) instead of raw character cuts, so chunks don't get sliced
    mid-word or mid-sentence and lose important context (like a question
    header right before its answer).
    """
    text = text.strip()
    if not text:
        return []

    # First split on paragraph-like breaks (blank lines, or numbered
    # question markers such as "4." at the start of a line).
    import re
    paragraphs = re.split(r"\n\s*\n|(?=\n\s*\d+\.\s)", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph would blow past chunk_size, flush first.
        if current and len(current) + len(para) + 1 > chunk_size:
            chunks.append(current.strip())
            # Start the next chunk with a bit of overlap from the end of
            # the previous one, so context isn't lost across the boundary.
            current = current[-overlap:] + "\n" + para
        else:
            current = (current + "\n" + para) if current else para

        # A single paragraph longer than chunk_size on its own: split it
        # on sentence boundaries instead of paragraph boundaries.
        while len(current) > chunk_size * 1.5:
            sentences = re.split(r"(?<=[.!?])\s+", current)
            piece = ""
            remainder = []
            for i, sent in enumerate(sentences):
                if len(piece) + len(sent) + 1 <= chunk_size:
                    piece = (piece + " " + sent) if piece else sent
                else:
                    remainder = sentences[i:]
                    break
            if piece:
                chunks.append(piece.strip())
            current = " ".join(remainder)

    if current.strip():
        chunks.append(current.strip())

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using the local model."""
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def ingest_pdf(file_path: str, source_name: str, company_id: str = "default") -> int:
    """
    Full ingestion pipeline for one PDF:
    extract -> chunk -> embed -> store in ChromaDB.
    Returns number of chunks stored.
    """
    # Remove any previously-ingested chunks for this exact file first, so
    # re-uploading the same PDF (e.g. during testing) doesn't create
    # duplicate entries that crowd out other results in search.
    existing = collection.get(where={"source": source_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source_name, "company_id": company_id} for _ in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def search_company_docs(query: str, company_id: str = "default", top_k: int = 4) -> List[dict]:
    """
    Semantic search over the ingested company documents.
    Returns the top_k most relevant chunks with their source file.
    """
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"company_id": company_id},
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        # With cosine space, distance ranges roughly 0 (identical) to 2
        # (opposite). Convert to an intuitive 0-1 similarity score.
        similarity = round(max(0.0, 1 - (dist / 2)), 3)
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "relevance": similarity,
        })
    return hits


def list_ingested_sources(company_id: str = "default") -> List[str]:
    """Return the distinct source filenames ingested for a company."""
    if collection.count() == 0:
        return []
    all_meta = collection.get(where={"company_id": company_id})["metadatas"]
    return sorted({m["source"] for m in all_meta})
