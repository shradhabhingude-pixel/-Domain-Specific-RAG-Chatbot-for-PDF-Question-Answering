"""Text chunking, embeddings, and in-memory FAISS search."""

import re
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks near word boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        value = text[start:end].strip()
        if value:
            chunks.append(value)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def make_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = []
    for page in pages:
        for text in split_text(page["text"]):
            chunks.append(
                {"source": page["source"], "page": page["page"], "text": text}
            )
    return chunks


def build_index(chunks: list[dict[str, Any]], model: SentenceTransformer):
    if not chunks:
        raise ValueError("There is no PDF text to index.")
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    return index


def search(question: str, index, chunks: list[dict[str, Any]], model, top_k: int = 5):
    vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    scores, positions = index.search(vector, min(top_k, len(chunks)))
    results = []
    for score, position in zip(scores[0], positions[0]):
        if position >= 0:
            item = dict(chunks[int(position)])
            item["score"] = float(score)
            results.append(item)
    return results
