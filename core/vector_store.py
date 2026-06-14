"""ChromaDB vector store backed by sentence-transformers embeddings."""

import hashlib
import os
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "sharp_rag_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"


class VectorStore:
    """Wraps a persistent ChromaDB collection with sentence-transformer embeddings."""

    def __init__(self, persist_dir: str = PERSIST_DIR) -> None:
        """Initialise the store; creates or loads the persistent collection."""
        self.persist_dir = persist_dir
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[dict]) -> None:
        """Embed and upsert a list of document dicts into the collection.

        Each dict must contain at minimum a ``text`` key. Optional keys are
        ``source`` and any additional metadata.
        """
        if not docs:
            return

        texts = [d["text"] for d in docs]
        embeddings = self._embed(texts)
        # Content-hash IDs make re-indexing idempotent: upsert is a no-op for
        # identical text, preventing collisions when the store is rebuilt.
        ids = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
        metadatas = [
            {k: v for k, v in d.items() if k != "text"} for d in docs
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query: str, n_results: int = 3) -> list[dict]:
        """Return the top-n most relevant document chunks for *query*.

        Each result dict has keys: ``id``, ``text``, ``source``,
        ``relevance_score``, plus any extra metadata fields.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self._embed([query])[0]
        # Note: "ids" is ALWAYS returned by ChromaDB query() regardless of include;
        # it must NOT be listed in include (only optional fields go there).
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs: list[dict] = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity: similarity = 1 - distance
            relevance_score = round(1.0 - dist, 4)
            docs.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "relevance_score": relevance_score,
                    **{k: v for k, v in meta.items() if k != "source"},
                }
            )

        return docs

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised embeddings for a list of texts."""
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


def initialize_store(persist_dir: str = PERSIST_DIR) -> VectorStore:
    """Convenience factory that creates and returns a ready-to-use VectorStore."""
    store = VectorStore(persist_dir=persist_dir)
    print(f"[VectorStore] Loaded collection '{COLLECTION_NAME}' "
          f"({store.count()} documents, persist_dir='{persist_dir}')")
    return store
