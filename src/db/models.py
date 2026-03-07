"""SQLAlchemy models for the document graph database."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class DocumentModel(Base):
    """SQLAlchemy model for documents table."""

    __tablename__ = "documents"

    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()"
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    blocks: Mapped[list["BlockModel"]] = relationship(
        "BlockModel", back_populates="document", cascade="all, delete-orphan"
    )
    edges: Mapped[list["EdgeModel"]] = relationship(
        "EdgeModel", back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["ChunkModel"]] = relationship(
        "ChunkModel", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (CheckConstraint("total_pages >= 1", name="check_total_pages"),)


class BlockModel(Base):
    """SQLAlchemy model for blocks table."""

    __tablename__ = "blocks"

    block_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()"
    )
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    block_type: Mapped[str] = mapped_column(String(50), nullable=False)
    bbox_x0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)
    mini_page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reading_order: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    raw_block_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    document: Mapped["DocumentModel"] = relationship("DocumentModel", back_populates="blocks")
    source_edges: Mapped[list["EdgeModel"]] = relationship(
        "EdgeModel",
        foreign_keys="EdgeModel.source_block_id",
        back_populates="source_block",
        cascade="all, delete-orphan",
    )
    target_edges: Mapped[list["EdgeModel"]] = relationship(
        "EdgeModel",
        foreign_keys="EdgeModel.target_block_id",
        back_populates="target_block",
        cascade="all, delete-orphan",
    )
    chunk_associations: Mapped[list["ChunkBlockModel"]] = relationship(
        "ChunkBlockModel", back_populates="block", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("pdf_page >= 0", name="check_pdf_page"),
        CheckConstraint("reading_order >= 0", name="check_reading_order"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_confidence"),
        Index("idx_blocks_document", "document_id"),
        Index("idx_blocks_page", "document_id", "pdf_page"),
        Index("idx_blocks_mini_page", "document_id", "pdf_page", "mini_page"),
        Index("idx_blocks_reading_order", "document_id", "reading_order"),
    )


class EdgeModel(Base):
    """SQLAlchemy model for edges table."""

    __tablename__ = "edges"

    edge_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()"
    )
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    source_block_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("blocks.block_id", ondelete="CASCADE"), nullable=False
    )
    target_block_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("blocks.block_id", ondelete="CASCADE"), nullable=False
    )
    edge_type: Mapped[str] = mapped_column(String(50), nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    document: Mapped["DocumentModel"] = relationship("DocumentModel", back_populates="edges")
    source_block: Mapped["BlockModel"] = relationship(
        "BlockModel", foreign_keys=[source_block_id], back_populates="source_edges"
    )
    target_block: Mapped["BlockModel"] = relationship(
        "BlockModel", foreign_keys=[target_block_id], back_populates="target_edges"
    )

    __table_args__ = (
        UniqueConstraint("source_block_id", "target_block_id", "edge_type", name="uq_edge"),
        Index("idx_edges_document", "document_id"),
        Index("idx_edges_source", "source_block_id"),
        Index("idx_edges_target", "target_block_id"),
    )


class ChunkModel(Base):
    """SQLAlchemy model for chunks table."""

    __tablename__ = "chunks"

    chunk_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()"
    )
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)
    mini_page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bbox_x0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    heading: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Float), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    document: Mapped["DocumentModel"] = relationship("DocumentModel", back_populates="chunks")
    block_associations: Mapped[list["ChunkBlockModel"]] = relationship(
        "ChunkBlockModel", back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("pdf_page >= 0", name="check_chunk_pdf_page"),
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_page", "document_id", "pdf_page"),
    )


class ChunkBlockModel(Base):
    """SQLAlchemy model for chunk_blocks junction table."""

    __tablename__ = "chunk_blocks"

    chunk_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )
    block_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("blocks.block_id", ondelete="CASCADE"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    chunk: Mapped["ChunkModel"] = relationship("ChunkModel", back_populates="block_associations")
    block: Mapped["BlockModel"] = relationship("BlockModel", back_populates="chunk_associations")

    __table_args__ = (
        CheckConstraint("position >= 0", name="check_position"),
        Index("idx_chunk_blocks_chunk", "chunk_id"),
        Index("idx_chunk_blocks_block", "block_id"),
    )


class QueryTraceModel(Base):
    """SQLAlchemy model for query_traces table."""

    __tablename__ = "query_traces"

    trace_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default="uuid_generate_v4()"
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    document_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="SET NULL"), nullable=True
    )
    source_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="CURRENT_TIMESTAMP"
    )

    chunk_rows: Mapped[list["QueryTraceChunkModel"]] = relationship(
        "QueryTraceChunkModel", back_populates="trace", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("top_k >= 1", name="check_trace_top_k"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_trace_confidence"),
        Index("idx_query_traces_created_at", "created_at"),
        Index("idx_query_traces_document", "document_id"),
    )


class QueryTraceChunkModel(Base):
    """SQLAlchemy model for query_trace_chunks table."""

    __tablename__ = "query_trace_chunks"

    trace_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("query_traces.trace_id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)
    mini_page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    heading: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    trace: Mapped["QueryTraceModel"] = relationship(
        "QueryTraceModel", back_populates="chunk_rows"
    )

    __table_args__ = (
        CheckConstraint("rank >= 1", name="check_trace_chunk_rank"),
        CheckConstraint("pdf_page >= 0", name="check_trace_chunk_pdf_page"),
        Index("idx_query_trace_chunks_trace", "trace_id"),
    )
