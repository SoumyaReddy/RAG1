"""Embeddings + ChromaDB index + retrieval (identical for Naive RAG and Hybrid RAG)."""
from __future__ import annotations

import numpy as np

from . import config as C


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 (384-d, unit-norm). Runs on CPU."""
    name = C.EMB_MODEL

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(C.EMB_MODEL, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


class TfidfEmbedder:
    """DRY-RUN ONLY stand-in so the Chroma code path can be exercised without downloading MiniLM."""
    name = "tfidf-dryrun"

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        m = self.vec.transform(texts).toarray().astype("float32")
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)


def make_embedder(dry_run: bool, corpus: list[str] | None = None):
    return TfidfEmbedder(corpus or []) if dry_run else SentenceTransformerEmbedder()


def build_index(chunks: list[dict], embedder, chroma_dir: str, collection_name: str = C.COLLECTION):
    """Create (or idempotently rebuild) a persistent cosine-space Chroma collection. Returns (collection, embeddings)."""
    import chromadb
    emb = embedder.encode([c["text"] for c in chunks])
    assert emb.shape[0] == len(chunks) and not np.isnan(emb).any(), "bad embeddings"
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["chunk_id"] for c in chunks], embeddings=emb.tolist(), documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "section": c["section"], "title": c["title"]} for c in chunks])
    assert col.count() == len(chunks)
    return col, emb


class Retriever:
    def __init__(self, embedder, chroma_dir: str, collection_name: str = C.COLLECTION):
        import chromadb
        self.embedder = embedder
        self.collection = chromadb.PersistentClient(path=chroma_dir).get_collection(collection_name)

    def __call__(self, query: str, k: int = C.K) -> list[dict]:
        qv = self.embedder.encode([query]).tolist()
        res = self.collection.query(query_embeddings=qv, n_results=k, include=["documents", "metadatas", "distances"])
        return [{"text": d, "source": m["source"], "section": m["section"], "distance": float(dist)}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
