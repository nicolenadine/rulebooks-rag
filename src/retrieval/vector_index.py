"""Vector index for similarity search using FAISS."""

import pickle
from pathlib import Path
from typing import Optional
from uuid import UUID

import faiss
import numpy as np

from src.models.schema import Chunk


class VectorIndex:
    """FAISS-based vector index for chunk similarity search.

    Designed to be swappable with pgvector or other vector databases.
    """

    def __init__(self, dimension: int = 1536):
        """Initialize the vector index.

        Args:
            dimension: Dimension of embedding vectors.
        """
        self.dimension = dimension
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunk_ids: list[UUID] = []
        self._chunks: dict[UUID, Chunk] = {}

    @property
    def index(self) -> faiss.IndexFlatIP:
        """Get or create the FAISS index."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index.

        Args:
            chunks: Chunks with embeddings to index.

        Raises:
            ValueError: If any chunk is missing an embedding.
        """
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")

        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        for chunk in chunks:
            self._chunk_ids.append(chunk.chunk_id)
            self._chunks[chunk.chunk_id] = chunk

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score to include.

        Returns:
            List of (chunk, score) tuples, sorted by score descending.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < score_threshold:
                continue

            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]
            results.append((chunk, float(score)))

        return results

    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        """Get a chunk by its ID."""
        return self._chunks.get(chunk_id)

    def save(self, path: str | Path) -> None:
        """Save the index to disk.

        Args:
            path: Path to save the index (without extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path.with_suffix(".faiss")))

        metadata = {
            "dimension": self.dimension,
            "chunk_ids": self._chunk_ids,
            "chunks": self._chunks,
        }
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        """Load an index from disk.

        Args:
            path: Path to the saved index (without extension).

        Returns:
            Loaded VectorIndex instance.
        """
        path = Path(path)

        with open(path.with_suffix(".pkl"), "rb") as f:
            metadata = pickle.load(f)

        instance = cls(dimension=metadata["dimension"])
        instance._index = faiss.read_index(str(path.with_suffix(".faiss")))
        instance._chunk_ids = metadata["chunk_ids"]
        instance._chunks = metadata["chunks"]

        return instance

    def clear(self) -> None:
        """Clear all data from the index."""
        self._index = None
        self._chunk_ids = []
        self._chunks = {}

    @property
    def size(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._chunk_ids)
