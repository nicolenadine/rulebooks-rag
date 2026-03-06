"""Generate embeddings for text chunks."""

import os
from typing import Optional

import numpy as np
from openai import OpenAI

from src.models.schema import Chunk


class Embedder:
    """Generate embeddings using OpenAI's embedding models.

    This class is designed to be swappable with other embedding providers.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: int = 1536,
        batch_size: int = 100,
    ):
        """Initialize the embedder.

        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            dimensions: Embedding dimensions.
            batch_size: Maximum number of texts to embed in one API call.
        """
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embeddings for chunks and update them in place.

        Args:
            chunks: List of chunks to embed.

        Returns:
            The same chunks with embeddings populated.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        This is separate from embed_text to allow for query-specific
        preprocessing or model selection in the future.

        Args:
            query: The search query.

        Returns:
            Query embedding vector.
        """
        return self.embed_text(query)

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score (0-1).
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
