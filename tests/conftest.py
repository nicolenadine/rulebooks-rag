"""Pytest configuration and fixtures."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

# Load .env from project root first so a real OPENAI_API_KEY is used when present.
# Only then set test defaults for keys that are still missing (e.g. in CI).
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except Exception:
    pass

# Set test environment variables only when not already set (e.g. by .env)
os.environ.setdefault("OPENAI_API_KEY", "test-api-key-for-testing")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "rulebook_rag_test")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")

from src.models.schema import Block, BlockType, BoundingBox, Chunk, Document, DocumentGraph, Edge, EdgeType


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """Create a sample bounding box."""
    return BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=50.0)


@pytest.fixture
def sample_block(sample_bbox: BoundingBox) -> Block:
    """Create a sample block."""
    return Block(
        block_id=uuid4(),
        text="This is sample text from a rulebook.",
        block_type=BlockType.TEXT,
        bbox=sample_bbox,
        pdf_page=0,
        mini_page=None,
        reading_order=0,
        confidence=1.0,
    )


@pytest.fixture
def sample_blocks() -> list[Block]:
    """Create a list of sample blocks."""
    blocks = []
    for i in range(5):
        block = Block(
            block_id=uuid4(),
            text=f"Block {i} text content.",
            block_type=BlockType.TEXT if i > 0 else BlockType.TITLE,
            bbox=BoundingBox(x0=10.0, y0=20.0 + i * 30, x1=200.0, y1=50.0 + i * 30),
            pdf_page=0,
            mini_page=None,
            reading_order=i,
            confidence=1.0,
        )
        blocks.append(block)
    return blocks


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document."""
    return Document(
        document_id=uuid4(),
        name="Test Rulebook",
        source_path="data/parsed/test.json",
        total_pages=5,
        metadata={"game": "Test Game"},
    )


@pytest.fixture
def sample_document_graph(sample_document: Document, sample_blocks: list[Block]) -> DocumentGraph:
    """Create a sample document graph."""
    edges = []
    for i in range(len(sample_blocks) - 1):
        edge = Edge(
            source_block_id=sample_blocks[i].block_id,
            target_block_id=sample_blocks[i + 1].block_id,
            edge_type=EdgeType.NEXT,
        )
        edges.append(edge)

    return DocumentGraph(
        document=sample_document,
        blocks=sample_blocks,
        edges=edges,
    )


@pytest.fixture
def sample_chunk(sample_blocks: list[Block]) -> Chunk:
    """Create a sample chunk."""
    text = "\n\n".join(b.text for b in sample_blocks[:3])
    block_ids = [b.block_id for b in sample_blocks[:3]]

    bbox_union = sample_blocks[0].bbox
    for block in sample_blocks[1:3]:
        bbox_union = bbox_union.union(block.bbox)

    return Chunk(
        chunk_id=uuid4(),
        text=text,
        block_ids=block_ids,
        pdf_page=0,
        mini_page=None,
        bbox_union=bbox_union,
        heading="Block 0 text content.",
        token_count=50,
    )
