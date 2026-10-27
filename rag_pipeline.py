"""Coordinates PDF extraction, retrieval, and Ollama answer generation."""

import os

import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from document_loader import extract_pages
from prompt import build_messages
from vector_store import EMBEDDING_MODEL_NAME, build_index, make_chunks, search


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


@st.cache_resource(show_spinner="Loading the embedding model (first run may take a while)...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def process_documents(uploaded_files):
    pages = extract_pages(uploaded_files)
    chunks = make_chunks(pages)
    if not chunks:
        return [], None
    index = build_index(chunks, load_embedding_model())
    return chunks, index


def answer_question(question: str, index, chunks: list[dict]):
    model = load_embedding_model()
    sources = search(question, index, chunks, model, top_k=5)
    if not sources:
        return "I could not find this information in the uploaded documents.", []

    try:
        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": build_messages(question, sources),
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=180,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            "Cannot connect to Ollama. Start Ollama, then run "
            f"`ollama pull {OLLAMA_MODEL}` in PowerShell."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ConnectionError("Ollama took too long to respond. Try again.") from exc

    answer = response.json().get("message", {}).get("content", "").strip()
    return answer or "I could not find this information in the uploaded documents.", sources
