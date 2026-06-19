"""Bonus: unstructured -> chunk -> embedding ingestion (zero-key).

Mirrors the deck's RAG-ingestion pipeline. We use a deterministic hash-based
'embedder' so the lab runs with no API key and no model download — the point is
the *pipeline shape* (parse -> recursive chunk -> embed -> store), not embedding
quality. Swap `embed_text` for a real model in the extension exercise.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

EMBED_DIM = 16


def recursive_chunks(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Recursive-ish splitter on paragraph/sentence boundaries with overlap.
    Recursive ~fixed-size splitting is the strong 2026 default (deck §3)."""
    words = re.split(r"\s+", text.strip())
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if c]


def embed_text(text: str) -> list[float]:
    """Deterministic fake embedding: stable, no key. NOT semantically meaningful."""
    vec = [0.0] * EMBED_DIM
    for tok in re.findall(r"[a-zA-Z]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [round(v / norm, 4) for v in vec]


def ingest_docs(docs_dir: Path) -> list[dict]:
    """parse -> chunk -> embed for every doc; returns rows ready for a vector store."""
    rows = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(recursive_chunks(text)):
            rows.append(
                {
                    "doc": path.name,
                    "chunk_id": idx,
                    "text": chunk,
                    "embedding": embed_text(chunk),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Extension exercise 1 — Incremental re-embed with content hash (§3)
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """SHA-256 fingerprint of a chunk (first 16 hex chars is plenty for dedup)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ingest_docs_incremental(
    docs_dir: Path,
    cache_path: Path | None = None,
) -> tuple[list[dict], int]:
    """Only embed chunks whose content hash changed since the last run.

    Returns (new_rows, n_skipped).  Re-running on unchanged docs costs zero
    embedding calls — critical when embed_text is a paid API (e.g. OpenAI
    text-embedding-3-small).

    Swap `embed_text` for a real sentence-transformers / API model and this
    function becomes a production-grade incremental ingestion stage: only
    modified or new chunks are re-embedded, the rest are served from the cache.

    The cache is a JSON file mapping chunk_key -> content_hash. In production,
    store the hash column in the vector store itself (Pinecone, Qdrant, pgvector)
    so a single SELECT gives you the stale set — no external cache file needed.
    """
    cache_file = cache_path or Path(docs_dir).parent / ".embed_cache.json"
    old_cache: dict[str, str] = {}
    if cache_file.exists():
        try:
            old_cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    rows: list[dict] = []
    new_cache: dict[str, str] = {}
    n_skipped = 0

    for path in sorted(Path(docs_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(recursive_chunks(text)):
            key = f"{path.name}:{idx}"
            h = content_hash(chunk)
            new_cache[key] = h
            if old_cache.get(key) == h:
                n_skipped += 1
                continue
            rows.append(
                {
                    "doc": path.name,
                    "chunk_id": idx,
                    "text": chunk,
                    "embedding": embed_text(chunk),
                    "content_hash": h,
                }
            )

    cache_file.write_text(json.dumps(new_cache, indent=2), encoding="utf-8")
    return rows, n_skipped
