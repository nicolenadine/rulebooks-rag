"""Retrieval pipeline with context expansion."""

from typing import Optional
from uuid import UUID

from src.embeddings.embedder import Embedder
from src.graph.document_graph import DocumentGraphOps
from src.models.schema import Block, Chunk, DocumentGraph
from src.retrieval.vector_index import VectorIndex


class Retriever:
    """Retrieval pipeline with semantic search and context expansion."""

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: Embedder,
        graph: Optional[DocumentGraph] = None,
    ):
        """Initialize the retriever.

        Args:
            vector_index: The vector index for similarity search.
            embedder: The embedder for query encoding.
            graph: Document graph for context expansion (optional).
        """
        self.vector_index = vector_index
        self.embedder = embedder
        self._graph = graph
        self._graph_ops: Optional[DocumentGraphOps] = None

    @property
    def graph_ops(self) -> Optional[DocumentGraphOps]:
        """Get the graph operations helper."""
        if self._graph_ops is None and self._graph is not None:
            self._graph_ops = DocumentGraphOps(self._graph)
        return self._graph_ops

    def set_graph(self, graph: DocumentGraph) -> None:
        """Set or update the document graph.

        Args:
            graph: The document graph for context expansion.
        """
        self._graph = graph
        self._graph_ops = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expand_context: bool = True,
        context_before: int = 1,
        context_after: int = 1,
    ) -> list[tuple[Chunk, float, list[Block]]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The user's question.
            top_k: Number of chunks to retrieve.
            expand_context: Whether to expand context using the graph.
            context_before: Number of preceding blocks to include.
            context_after: Number of following blocks to include.

        Returns:
            List of (chunk, score, context_blocks) tuples.
        """
        query_embedding = self.embedder.embed_query(query)

        results = self.vector_index.search(query_embedding, top_k=top_k)

        expanded_results = []
        for chunk, score in results:
            if expand_context and self.graph_ops:
                context_blocks = self._expand_context(
                    chunk, before=context_before, after=context_after
                )
            else:
                context_blocks = []

            expanded_results.append((chunk, score, context_blocks))

        return expanded_results

    def _expand_context(
        self,
        chunk: Chunk,
        before: int = 1,
        after: int = 1,
    ) -> list[Block]:
        """Expand context by getting neighboring blocks.

        Args:
            chunk: The chunk to expand.
            before: Number of blocks before.
            after: Number of blocks after.

        Returns:
            List of context blocks not in the chunk.
        """
        if self.graph_ops is None:
            return []

        context_blocks = []
        chunk_block_ids = set(chunk.block_ids)

        if chunk.block_ids:
            first_block_id = chunk.block_ids[0]
            context_window = self.graph_ops.get_context_window(
                first_block_id, before=before, after=0
            )
            for block in context_window:
                if block.block_id not in chunk_block_ids:
                    context_blocks.append(block)

        if chunk.block_ids:
            last_block_id = chunk.block_ids[-1]
            context_window = self.graph_ops.get_context_window(
                last_block_id, before=0, after=after
            )
            for block in context_window:
                if block.block_id not in chunk_block_ids:
                    context_blocks.append(block)

        return context_blocks

    def retrieve_by_page(
        self,
        query: str,
        pdf_page: int,
        mini_page: Optional[int] = None,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve chunks from a specific page.

        Args:
            query: The search query.
            pdf_page: The PDF page to search within.
            mini_page: Optional mini-page filter.
            top_k: Number of results to return.

        Returns:
            List of (chunk, score) tuples from the specified page.
        """
        query_embedding = self.embedder.embed_query(query)

        all_results = self.vector_index.search(query_embedding, top_k=top_k * 5)

        filtered_results = []
        for chunk, score in all_results:
            if chunk.pdf_page != pdf_page:
                continue
            if mini_page is not None and chunk.mini_page != mini_page:
                continue
            filtered_results.append((chunk, score))
            if len(filtered_results) >= top_k:
                break

        return filtered_results

    def get_chunk_blocks(self, chunk: Chunk) -> list[Block]:
        """Get the actual Block objects for a chunk.

        Args:
            chunk: The chunk to get blocks for.

        Returns:
            List of Block objects in the chunk.
        """
        if self.graph_ops is None:
            return []

        blocks = []
        for block_id in chunk.block_ids:
            block = self.graph_ops.get_block(block_id)
            if block:
                blocks.append(block)

        return sorted(blocks, key=lambda b: b.reading_order)
