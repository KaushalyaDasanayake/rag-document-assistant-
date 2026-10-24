"""ChromaDB persistent vector store implementation for document chunks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI

from ragassistant.retrieval.embeddings import EmbeddingManager
from ragassistant.schemas.document import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document chunk indexing and semantic vector search in ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = "rag_docs",
        embedding_manager: EmbeddingManager | None = None,
        client: ClientAPI | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager

        if client is not None:
            self.client = client
        elif persist_directory is not None:
            db_path = Path(persist_directory)
            db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(db_path))
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return the total number of chunks indexed in the collection."""
        return self.collection.count()

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """Upsert document chunks into ChromaDB with metadata and embeddings."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        if embeddings is None:
            if self.embedding_manager is None:
                raise ValueError("Embedding manager is required when precomputed embeddings are not provided")
            embeddings = self.embedding_manager.embed_texts(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks into collection %s", len(chunks), self.collection_name)
        return len(chunks)

    def query(
        self,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform semantic similarity search over indexed chunks."""
        if top_k <= 0 or self.count() == 0:
            return []

        if query_embedding is None:
            if query is None:
                raise ValueError("Must provide either `query` text or `query_embedding`")
            if self.embedding_manager is None:
                raise ValueError("Embedding manager is required to encode raw query text")
            query_embedding = self.embedding_manager.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        for chunk_id, doc_text, meta, distance in zip(ids_list, docs_list, metas_list, distances_list, strict=False):
            # ChromaDB cosine distance: distance = 1 - cosine_sim
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            safe_meta = meta or {}

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_name=str(safe_meta.get("doc_name", "")),
                page_number=int(safe_meta.get("page_number", 1)),
                chunk_index=int(safe_meta.get("chunk_index", 0)),
                text=str(doc_text),
                token_count=int(safe_meta.get("token_count", 0)),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        return retrieved

    def clear(self) -> None:
        """Clear all indexed documents by recreating the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
