"""Tests for graph modules."""

import pytest

from src.graph.document_graph import DocumentGraphOps
from src.graph.graph_builder import GraphBuilder
from src.models.schema import Block, BlockType, BoundingBox, Document, DocumentGraph, EdgeType


class TestGraphBuilder:
    """Tests for GraphBuilder."""

    def test_build_reading_order_edges(self, sample_document: Document, sample_blocks: list[Block]):
        builder = GraphBuilder(sample_document, sample_blocks)
        graph = builder.build()

        next_edges = [e for e in graph.edges if e.edge_type == EdgeType.NEXT]

        assert len(next_edges) == len(sample_blocks) - 1

        for i, edge in enumerate(next_edges):
            assert edge.source_block_id == sample_blocks[i].block_id
            assert edge.target_block_id == sample_blocks[i + 1].block_id

    def test_build_with_figures(self, sample_document: Document):
        blocks = [
            Block(
                text="Game Setup",
                block_type=BlockType.TITLE,
                bbox=BoundingBox(x0=10, y0=10, x1=200, y1=40),
                pdf_page=0,
                reading_order=0,
            ),
            Block(
                text="Place the board in the center.",
                block_type=BlockType.TEXT,
                bbox=BoundingBox(x0=10, y0=50, x1=200, y1=100),
                pdf_page=0,
                reading_order=1,
            ),
            Block(
                text="[Figure: Board setup diagram]",
                block_type=BlockType.FIGURE,
                bbox=BoundingBox(x0=220, y0=50, x1=400, y1=150),
                pdf_page=0,
                reading_order=2,
            ),
        ]

        builder = GraphBuilder(sample_document, blocks)
        graph = builder.build()

        illustrates_edges = [e for e in graph.edges if e.edge_type == EdgeType.ILLUSTRATES]

        assert len(illustrates_edges) > 0
        assert any(e.source_block_id == blocks[2].block_id for e in illustrates_edges)


class TestDocumentGraphOps:
    """Tests for DocumentGraphOps."""

    def test_get_block(self, sample_document_graph: DocumentGraph):
        ops = DocumentGraphOps(sample_document_graph)
        block = sample_document_graph.blocks[0]

        result = ops.get_block(block.block_id)

        assert result is not None
        assert result.block_id == block.block_id

    def test_get_next_blocks(self, sample_document_graph: DocumentGraph):
        ops = DocumentGraphOps(sample_document_graph)
        first_block = sample_document_graph.blocks[0]

        next_blocks = ops.get_next_blocks(first_block.block_id)

        assert len(next_blocks) == 1
        assert next_blocks[0].block_id == sample_document_graph.blocks[1].block_id

    def test_get_previous_blocks(self, sample_document_graph: DocumentGraph):
        ops = DocumentGraphOps(sample_document_graph)
        second_block = sample_document_graph.blocks[1]

        prev_blocks = ops.get_previous_blocks(second_block.block_id)

        assert len(prev_blocks) == 1
        assert prev_blocks[0].block_id == sample_document_graph.blocks[0].block_id

    def test_get_context_window(self, sample_document_graph: DocumentGraph):
        ops = DocumentGraphOps(sample_document_graph)
        middle_block = sample_document_graph.blocks[2]

        context = ops.get_context_window(middle_block.block_id, before=1, after=1)

        assert len(context) == 3
        assert context[0].block_id == sample_document_graph.blocks[1].block_id
        assert context[1].block_id == middle_block.block_id
        assert context[2].block_id == sample_document_graph.blocks[3].block_id

    def test_get_reading_order_sequence(self, sample_document_graph: DocumentGraph):
        ops = DocumentGraphOps(sample_document_graph)

        sequence = ops.get_reading_order_sequence(max_blocks=3)

        assert len(sequence) == 3
        assert sequence[0].reading_order == 0
        assert sequence[1].reading_order == 1
        assert sequence[2].reading_order == 2
