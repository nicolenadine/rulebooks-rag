"""Tests for Pydantic models."""

from uuid import uuid4

import pytest

from src.models.schema import Block, BlockType, BoundingBox, Chunk, Edge, EdgeType


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_create_bbox(self):
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=50.0)
        assert bbox.x0 == 10.0
        assert bbox.y0 == 20.0
        assert bbox.x1 == 100.0
        assert bbox.y1 == 50.0

    def test_bbox_properties(self):
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=50.0)
        assert bbox.width == 100.0
        assert bbox.height == 50.0
        assert bbox.area == 5000.0
        assert bbox.center == (50.0, 25.0)

    def test_bbox_union(self):
        bbox1 = BoundingBox(x0=0.0, y0=0.0, x1=50.0, y1=50.0)
        bbox2 = BoundingBox(x0=25.0, y0=25.0, x1=100.0, y1=100.0)
        union = bbox1.union(bbox2)

        assert union.x0 == 0.0
        assert union.y0 == 0.0
        assert union.x1 == 100.0
        assert union.y1 == 100.0

    def test_bbox_intersects(self):
        bbox1 = BoundingBox(x0=0.0, y0=0.0, x1=50.0, y1=50.0)
        bbox2 = BoundingBox(x0=25.0, y0=25.0, x1=100.0, y1=100.0)
        bbox3 = BoundingBox(x0=60.0, y0=60.0, x1=100.0, y1=100.0)

        assert bbox1.intersects(bbox2) is True
        assert bbox1.intersects(bbox3) is False


class TestBlock:
    """Tests for Block model."""

    def test_create_block(self, sample_bbox: BoundingBox):
        block = Block(
            text="Sample text",
            block_type=BlockType.TEXT,
            bbox=sample_bbox,
            pdf_page=0,
            reading_order=0,
        )

        assert block.text == "Sample text"
        assert block.block_type == BlockType.TEXT
        assert block.pdf_page == 0
        assert block.mini_page is None
        assert block.confidence == 1.0
        assert block.block_id is not None

    def test_block_with_mini_page(self, sample_bbox: BoundingBox):
        block = Block(
            text="Sample text",
            block_type=BlockType.TEXT,
            bbox=sample_bbox,
            pdf_page=0,
            mini_page=2,
            reading_order=0,
        )

        assert block.mini_page == 2


class TestEdge:
    """Tests for Edge model."""

    def test_create_edge(self):
        source_id = uuid4()
        target_id = uuid4()

        edge = Edge(
            source_block_id=source_id,
            target_block_id=target_id,
            edge_type=EdgeType.NEXT,
        )

        assert edge.source_block_id == source_id
        assert edge.target_block_id == target_id
        assert edge.edge_type == EdgeType.NEXT
        assert edge.weight == 1.0

    def test_edge_with_metadata(self):
        edge = Edge(
            source_block_id=uuid4(),
            target_block_id=uuid4(),
            edge_type=EdgeType.ILLUSTRATES,
            metadata={"proximity_score": 0.85},
        )

        assert edge.metadata["proximity_score"] == 0.85


class TestChunk:
    """Tests for Chunk model."""

    def test_create_chunk(self, sample_bbox: BoundingBox):
        block_ids = [uuid4() for _ in range(3)]

        chunk = Chunk(
            text="Combined chunk text",
            block_ids=block_ids,
            pdf_page=0,
            bbox_union=sample_bbox,
            token_count=25,
        )

        assert len(chunk.block_ids) == 3
        assert chunk.token_count == 25
        assert chunk.heading is None
        assert chunk.embedding is None
