"""Database connection and operations."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.db.models import (
    Base,
    BlockModel,
    ChunkBlockModel,
    ChunkModel,
    DocumentModel,
    EdgeModel,
)
from src.models.schema import Block, BoundingBox, Chunk, Document, DocumentGraph, Edge


class Database:
    """Database interface for the document graph.

    All configuration is loaded from settings.
    """

    def __init__(self):
        """Initialize database connection using settings."""
        self.connection_string = settings.postgres_url
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = create_engine(self.connection_string, echo=False, pool_pre_ping=True)
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        return self._session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def init_schema(self) -> None:
        """Initialize the database schema."""
        Base.metadata.create_all(self.engine)

    def init_schema_from_sql(self) -> None:
        """Initialize the database schema from the SQL file."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema_sql = f.read()

        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()

    def save_document(self, document: Document) -> UUID:
        """Save a document to the database (source_path stored in canonical form)."""
        canonical_source = str(Path(document.source_path).resolve())
        with self.session() as session:
            model = DocumentModel(
                document_id=document.document_id,
                name=document.name,
                source_path=canonical_source,
                total_pages=document.total_pages,
                metadata_=document.metadata,
            )
            session.add(model)
            return model.document_id

    def save_block(self, block: Block, document_id: UUID) -> UUID:
        """Save a block to the database."""
        with self.session() as session:
            model = BlockModel(
                block_id=block.block_id,
                document_id=document_id,
                text=block.text,
                block_type=block.block_type.value,
                bbox_x0=block.bbox.x0,
                bbox_y0=block.bbox.y0,
                bbox_x1=block.bbox.x1,
                bbox_y1=block.bbox.y1,
                pdf_page=block.pdf_page,
                mini_page=block.mini_page,
                reading_order=block.reading_order,
                confidence=block.confidence,
                raw_block_id=block.raw_block_id,
            )
            session.add(model)
            return model.block_id

    def save_edge(self, edge: Edge, document_id: UUID) -> None:
        """Save an edge to the database."""
        with self.session() as session:
            model = EdgeModel(
                document_id=document_id,
                source_block_id=edge.source_block_id,
                target_block_id=edge.target_block_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
                metadata_=edge.metadata,
            )
            session.add(model)

    def save_chunk(self, chunk: Chunk, document_id: UUID) -> UUID:
        """Save a chunk to the database."""
        with self.session() as session:
            model = ChunkModel(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                text=chunk.text,
                pdf_page=chunk.pdf_page,
                mini_page=chunk.mini_page,
                bbox_x0=chunk.bbox_union.x0,
                bbox_y0=chunk.bbox_union.y0,
                bbox_x1=chunk.bbox_union.x1,
                bbox_y1=chunk.bbox_union.y1,
                heading=chunk.heading,
                token_count=chunk.token_count,
                embedding=chunk.embedding,
            )
            session.add(model)

            for position, block_id in enumerate(chunk.block_ids):
                association = ChunkBlockModel(
                    chunk_id=chunk.chunk_id,
                    block_id=block_id,
                    position=position,
                )
                session.add(association)

            return model.chunk_id

    def save_document_graph(self, graph: DocumentGraph) -> UUID:
        """Save an entire document graph to the database.

        document.source_path is stored in canonical form (str(Path(...).resolve()))
        so lookups by source_path match across runs.
        """
        canonical_source = str(Path(graph.document.source_path).resolve())
        with self.session() as session:
            doc_model = DocumentModel(
                document_id=graph.document.document_id,
                name=graph.document.name,
                source_path=canonical_source,
                total_pages=graph.document.total_pages,
                metadata_=graph.document.metadata,
            )
            session.add(doc_model)

            for block in graph.blocks:
                block_model = BlockModel(
                    block_id=block.block_id,
                    document_id=graph.document.document_id,
                    text=block.text,
                    block_type=block.block_type.value,
                    bbox_x0=block.bbox.x0,
                    bbox_y0=block.bbox.y0,
                    bbox_x1=block.bbox.x1,
                    bbox_y1=block.bbox.y1,
                    pdf_page=block.pdf_page,
                    mini_page=block.mini_page,
                    reading_order=block.reading_order,
                    confidence=block.confidence,
                    raw_block_id=block.raw_block_id,
                )
                session.add(block_model)

            for edge in graph.edges:
                edge_model = EdgeModel(
                    document_id=graph.document.document_id,
                    source_block_id=edge.source_block_id,
                    target_block_id=edge.target_block_id,
                    edge_type=edge.edge_type.value,
                    weight=edge.weight,
                    metadata_=edge.metadata,
                )
                session.add(edge_model)

            return graph.document.document_id

    def load_document_graph(self, document_id: UUID) -> Optional[DocumentGraph]:
        """Load a document graph from the database."""
        with self.session() as session:
            doc_model = session.get(DocumentModel, document_id)
            if not doc_model:
                return None

            document = Document(
                document_id=doc_model.document_id,
                name=doc_model.name,
                source_path=doc_model.source_path,
                total_pages=doc_model.total_pages,
                metadata=doc_model.metadata_,
            )

            blocks = []
            for block_model in doc_model.blocks:
                block = Block(
                    block_id=block_model.block_id,
                    text=block_model.text,
                    block_type=block_model.block_type,
                    bbox=BoundingBox(
                        x0=block_model.bbox_x0,
                        y0=block_model.bbox_y0,
                        x1=block_model.bbox_x1,
                        y1=block_model.bbox_y1,
                    ),
                    pdf_page=block_model.pdf_page,
                    mini_page=block_model.mini_page,
                    reading_order=block_model.reading_order,
                    confidence=block_model.confidence,
                    raw_block_id=block_model.raw_block_id,
                )
                blocks.append(block)

            edges = []
            for edge_model in doc_model.edges:
                edge = Edge(
                    source_block_id=edge_model.source_block_id,
                    target_block_id=edge_model.target_block_id,
                    edge_type=edge_model.edge_type,
                    weight=edge_model.weight,
                    metadata=edge_model.metadata_,
                )
                edges.append(edge)

            return DocumentGraph(document=document, blocks=blocks, edges=edges)

    def get_document_by_source_path(self, source_path: str) -> Optional[UUID]:
        """Return document_id for a document with the given source_path, or None.

        source_path is normalized to a canonical absolute form so that relative vs
        absolute paths match (e.g. data/parsed/x.json vs /Users/.../data/parsed/x.json).
        """
        canonical = str(Path(source_path).resolve())
        with self.session() as session:
            model = (
                session.query(DocumentModel)
                .filter(DocumentModel.source_path == canonical)
                .first()
            )
            return model.document_id if model else None

    def update_chunk_embedding(self, chunk_id: UUID, embedding: list[float]) -> None:
        """Update the embedding for an existing chunk."""
        with self.session() as session:
            session.query(ChunkModel).filter(ChunkModel.chunk_id == chunk_id).update(
                {ChunkModel.embedding: embedding}
            )

    def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Get all chunks for a document."""
        with self.session() as session:
            chunk_models = (
                session.query(ChunkModel).filter(ChunkModel.document_id == document_id).all()
            )

            chunks = []
            for model in chunk_models:
                block_ids = [
                    assoc.block_id
                    for assoc in sorted(model.block_associations, key=lambda a: a.position)
                ]
                chunk = Chunk(
                    chunk_id=model.chunk_id,
                    text=model.text,
                    block_ids=block_ids,
                    pdf_page=model.pdf_page,
                    mini_page=model.mini_page,
                    bbox_union=BoundingBox(
                        x0=model.bbox_x0,
                        y0=model.bbox_y0,
                        x1=model.bbox_x1,
                        y1=model.bbox_y1,
                    ),
                    heading=model.heading,
                    token_count=model.token_count,
                    embedding=model.embedding,
                )
                chunks.append(chunk)

            return chunks


_database: Optional[Database] = None


def get_database() -> Database:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database
