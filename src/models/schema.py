"""Pydantic models for document graph and RAG pipeline."""

from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """Type of content block extracted from the PDF."""

    TEXT = "text"
    TITLE = "title"
    FIGURE = "figure"
    TABLE = "table"
    LIST = "list"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


class EdgeType(str, Enum):
    """Type of relationship between blocks."""

    NEXT = "next"  # Reading order connection
    IN_CHUNK = "in_chunk"  # Block belongs to chunk
    ILLUSTRATES = "illustrates"  # Figure illustrates text
    CAPTION_OF = "caption_of"  # Caption describes figure/table
    CHILD_OF = "child_of"  # Hierarchical relationship (e.g., bullet under heading)


class BoundingBox(BaseModel):
    """Bounding box coordinates for a block on a page."""

    x0: float = Field(..., description="Left edge coordinate")
    y0: float = Field(..., description="Top edge coordinate")
    x1: float = Field(..., description="Right edge coordinate")
    y1: float = Field(..., description="Bottom edge coordinate")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Return the bounding box that contains both boxes."""
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects with another."""
        return not (
            self.x1 < other.x0
            or self.x0 > other.x1
            or self.y1 < other.y0
            or self.y0 > other.y1
        )


class Block(BaseModel):
    """A content block extracted from the PDF."""

    block_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the block")
    text: str = Field(..., description="Text content of the block")
    block_type: BlockType = Field(..., description="Type of content block")
    bbox: BoundingBox = Field(..., description="Bounding box on the PDF page")
    pdf_page: int = Field(..., ge=0, description="Zero-indexed PDF page number")
    mini_page: Optional[int] = Field(
        default=None, description="Logical mini-page within the PDF page (for multi-page layouts)"
    )
    reading_order: int = Field(..., ge=0, description="Position in reading order")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="OCR/extraction confidence score"
    )
    raw_block_id: Optional[str] = Field(
        default=None, description="Original block ID from Reducto output"
    )

    class Config:
        frozen = False


class Edge(BaseModel):
    """A directed edge between two blocks in the document graph."""

    source_block_id: UUID = Field(..., description="Source block UUID")
    target_block_id: UUID = Field(..., description="Target block UUID")
    edge_type: EdgeType = Field(..., description="Type of relationship")
    weight: float = Field(default=1.0, description="Edge weight for traversal")
    metadata: dict = Field(default_factory=dict, description="Additional edge metadata")


class Chunk(BaseModel):
    """A semantic chunk of content for embedding and retrieval."""

    chunk_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the chunk")
    text: str = Field(..., description="Combined text content of the chunk")
    block_ids: list[UUID] = Field(..., description="List of block IDs in this chunk")
    pdf_page: int = Field(..., ge=0, description="Primary PDF page number")
    mini_page: Optional[int] = Field(default=None, description="Logical mini-page number")
    bbox_union: BoundingBox = Field(..., description="Bounding box containing all blocks")
    heading: Optional[str] = Field(default=None, description="Section heading for this chunk")
    token_count: int = Field(default=0, description="Number of tokens in the chunk")
    embedding: Optional[list[float]] = Field(default=None, description="Vector embedding")


class Document(BaseModel):
    """A rulebook document."""

    document_id: UUID = Field(default_factory=uuid4, description="Unique document identifier")
    name: str = Field(..., description="Document name/title")
    source_path: str = Field(..., description="Path to the source PDF or parsed JSON")
    total_pages: int = Field(..., ge=1, description="Total number of PDF pages")
    metadata: dict = Field(default_factory=dict, description="Additional document metadata")


class DocumentGraph(BaseModel):
    """Graph representation of a document with blocks and edges."""

    document: Document = Field(..., description="The source document")
    blocks: list[Block] = Field(default_factory=list, description="All blocks in the document")
    edges: list[Edge] = Field(default_factory=list, description="All edges between blocks")

    def get_block(self, block_id: UUID) -> Optional[Block]:
        """Get a block by its ID."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def get_neighbors(
        self, block_id: UUID, edge_type: Optional[EdgeType] = None
    ) -> list[tuple[Block, Edge]]:
        """Get all neighboring blocks connected by outgoing edges."""
        neighbors = []
        for edge in self.edges:
            if edge.source_block_id == block_id:
                if edge_type is None or edge.edge_type == edge_type:
                    block = self.get_block(edge.target_block_id)
                    if block:
                        neighbors.append((block, edge))
        return neighbors

    def get_blocks_by_page(self, pdf_page: int, mini_page: Optional[int] = None) -> list[Block]:
        """Get all blocks on a specific page."""
        blocks = [b for b in self.blocks if b.pdf_page == pdf_page]
        if mini_page is not None:
            blocks = [b for b in blocks if b.mini_page == mini_page]
        return sorted(blocks, key=lambda b: b.reading_order)


class Citation(BaseModel):
    """A citation referencing a specific location in the rulebook."""

    chunk_id: UUID = Field(..., description="ID of the cited chunk")
    block_ids: list[UUID] = Field(..., description="IDs of specific blocks cited")
    pdf_page: int = Field(..., description="PDF page number")
    mini_page: Optional[int] = Field(default=None, description="Logical mini-page number")
    bbox: BoundingBox = Field(..., description="Bounding box of the cited region")
    text_snippet: str = Field(..., description="Relevant text snippet from the citation")

    def format_reference(self) -> str:
        """Format the citation as a human-readable reference."""
        page_ref = f"p.{self.pdf_page + 1}"  # Convert to 1-indexed for display
        if self.mini_page is not None:
            page_ref += f" (section {self.mini_page + 1})"
        return page_ref


class QAResponse(BaseModel):
    """Response from the QA pipeline."""

    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer text")
    citations: list[Citation] = Field(default_factory=list, description="Supporting citations")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the answer"
    )
    retrieved_chunks: list[Chunk] = Field(
        default_factory=list, description="Chunks used for context"
    )
