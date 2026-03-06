"""Tests for chunking module."""

import pytest

from src.chunking.chunk_builder import ChunkBuilder
from src.models.schema import Block, BlockType, BoundingBox, Document, DocumentGraph, Edge, EdgeType


class TestChunkBuilder:
    """Tests for ChunkBuilder."""

    @pytest.fixture
    def chunk_builder(self):
        return ChunkBuilder(max_tokens=100, min_tokens=20, overlap_tokens=10)

    def test_build_single_chunk(self, chunk_builder: ChunkBuilder, sample_document_graph: DocumentGraph):
        chunks = chunk_builder.build_chunks(sample_document_graph)

        assert len(chunks) >= 1

        for chunk in chunks:
            assert chunk.token_count > 0
            assert len(chunk.block_ids) > 0
            assert chunk.pdf_page >= 0

    def test_chunks_respect_token_limit(self, sample_document_graph: DocumentGraph):
        builder = ChunkBuilder(max_tokens=50, min_tokens=10, overlap_tokens=0)
        chunks = builder.build_chunks(sample_document_graph)

        for chunk in chunks:
            assert chunk.token_count <= 60

    def test_chunks_have_valid_bbox(self, chunk_builder: ChunkBuilder, sample_document_graph: DocumentGraph):
        chunks = chunk_builder.build_chunks(sample_document_graph)

        for chunk in chunks:
            assert chunk.bbox_union is not None
            assert chunk.bbox_union.x0 >= 0
            assert chunk.bbox_union.y0 >= 0
            assert chunk.bbox_union.x1 > chunk.bbox_union.x0
            assert chunk.bbox_union.y1 > chunk.bbox_union.y0

    def test_merge_small_chunks(self, chunk_builder: ChunkBuilder):
        from uuid import uuid4

        small_chunks = []
        for i in range(3):
            from src.models.schema import Chunk
            chunk = Chunk(
                chunk_id=uuid4(),
                text=f"Small chunk {i}",
                block_ids=[uuid4()],
                pdf_page=0,
                bbox_union=BoundingBox(x0=10, y0=10 + i * 30, x1=100, y1=40 + i * 30),
                token_count=5,
            )
            small_chunks.append(chunk)

        merged = chunk_builder.merge_small_chunks(small_chunks, min_tokens=15)

        assert len(merged) < len(small_chunks)

    def test_chunks_preserve_page_info(self, chunk_builder: ChunkBuilder):
        from uuid import uuid4
        from src.models.schema import Document

        document = Document(
            document_id=uuid4(),
            name="Multi-page test",
            source_path="test.json",
            total_pages=2,
        )

        blocks = []
        for page in range(2):
            for i in range(3):
                block = Block(
                    text=f"Page {page} block {i} content here.",
                    block_type=BlockType.TEXT,
                    bbox=BoundingBox(x0=10, y0=10 + i * 30, x1=200, y1=40 + i * 30),
                    pdf_page=page,
                    reading_order=page * 3 + i,
                )
                blocks.append(block)

        edges = []
        for i in range(len(blocks) - 1):
            edges.append(Edge(
                source_block_id=blocks[i].block_id,
                target_block_id=blocks[i + 1].block_id,
                edge_type=EdgeType.NEXT,
            ))

        graph = DocumentGraph(document=document, blocks=blocks, edges=edges)
        chunks = chunk_builder.build_chunks(graph)

        pages_in_chunks = {c.pdf_page for c in chunks}
        assert 0 in pages_in_chunks or 1 in pages_in_chunks
