"""Log and retrieve QA traces (question, answer, retrieved chunks) for debugging."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.config import settings
from src.models.schema import Block, Chunk


class ChunkRecord(BaseModel):
    """One retrieved chunk in a trace."""

    chunk_id: UUID
    rank: int
    similarity_score: float
    pdf_page: int
    mini_page: Optional[int] = None
    heading: Optional[str] = None
    chunk_text: str


class TraceRecord(BaseModel):
    """A single QA trace (question, answer, retrieved chunks)."""

    trace_id: UUID
    question: str
    answer: str
    confidence: float
    top_k: int
    created_at: datetime
    source_path: Optional[str] = None
    chunks: list[ChunkRecord] = Field(default_factory=list)


class TraceLogger:
    """Handles logging and retrieval of QA traces for debugging and review."""

    def __init__(self, use_db: bool = True):
        """Initialize the trace logger.

        Args:
            use_db: If True, attempt to persist traces to PostgreSQL when available.
                File-based last-trace is always written for the last-chunks CLI.
        """
        self._use_db = use_db
        self._db = None

    def _get_db(self):
        """Lazy-load database; return None if not available."""
        if not self._use_db:
            return None
        if self._db is None:
            try:
                from src.db.database import get_database
                self._db = get_database()
            except Exception:
                return None
        return self._db

    def log_qa_trace(
        self,
        question: str,
        answer: str,
        confidence: float,
        top_k: int,
        retrieved: list[tuple[Chunk, float, list[Block]]],
        document_id: Optional[UUID] = None,
        source_path: Optional[str] = None,
    ) -> UUID:
        """Log a QA run (question, answer, retrieved chunks).

        Always writes the last trace to a file so the last-chunks CLI works.
        If DB is configured and available, also inserts into query_traces and query_trace_chunks.

        Args:
            question: The user question.
            answer: The generated answer.
            confidence: Answer confidence score.
            top_k: Number of chunks requested.
            retrieved: List of (chunk, score, context_blocks) from the retriever.
            document_id: Optional document UUID if the rulebook is in the DB.
            source_path: Optional path to the rulebook JSON.

        Returns:
            trace_id for the logged trace.
        """
        trace_id = uuid4()
        created_at = datetime.now(timezone.utc)

        chunk_records = []
        for rank, (chunk, score, _) in enumerate(retrieved, start=1):
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    similarity_score=score,
                    pdf_page=chunk.pdf_page,
                    mini_page=chunk.mini_page,
                    heading=chunk.heading,
                    chunk_text=chunk.text,
                )
            )

        record = TraceRecord(
            trace_id=trace_id,
            question=question,
            answer=answer,
            confidence=confidence,
            top_k=top_k,
            created_at=created_at,
            source_path=source_path,
            chunks=chunk_records,
        )

        self._write_last_trace_file(record)
        self._write_trace_to_db(record, document_id)
        return trace_id

    def _write_last_trace_file(self, record: TraceRecord) -> None:
        """Write the trace to the last-query file."""
        path = Path(settings.trace_last_query_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trace_id": str(record.trace_id),
            "question": record.question,
            "answer": record.answer,
            "confidence": record.confidence,
            "top_k": record.top_k,
            "created_at": record.created_at.isoformat(),
            "source_path": record.source_path,
            "chunks": [
                {
                    "chunk_id": str(c.chunk_id),
                    "rank": c.rank,
                    "similarity_score": c.similarity_score,
                    "pdf_page": c.pdf_page,
                    "mini_page": c.mini_page,
                    "heading": c.heading,
                    "chunk_text": c.chunk_text,
                }
                for c in record.chunks
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _write_trace_to_db(
        self,
        record: TraceRecord,
        document_id: Optional[UUID] = None,
    ) -> None:
        """Insert trace into PostgreSQL if DB is available."""
        db = self._get_db()
        if db is None:
            return
        try:
            from src.db.models import QueryTraceChunkModel, QueryTraceModel

            with db.session() as session:
                trace_model = QueryTraceModel(
                    trace_id=record.trace_id,
                    question=record.question,
                    answer=record.answer,
                    top_k=record.top_k,
                    confidence=record.confidence,
                    document_id=document_id,
                    source_path=record.source_path,
                )
                session.add(trace_model)
                for c in record.chunks:
                    chunk_model = QueryTraceChunkModel(
                        trace_id=record.trace_id,
                        chunk_id=c.chunk_id,
                        rank=c.rank,
                        similarity_score=c.similarity_score,
                        pdf_page=c.pdf_page,
                        mini_page=c.mini_page,
                        heading=c.heading,
                        chunk_text=c.chunk_text,
                    )
                    session.add(chunk_model)
        except Exception:
            pass

    def get_last_trace(self) -> Optional[TraceRecord]:
        """Return the most recent trace (from file, then DB if file missing).

        The file is the primary source so last-chunks works without PostgreSQL.
        """
        record = self._read_last_trace_file()
        if record is not None:
            return record
        return self._get_last_trace_from_db()

    def _read_last_trace_file(self) -> Optional[TraceRecord]:
        """Read the last trace from the JSON file."""
        path = Path(settings.trace_last_query_path)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            chunks = [
                ChunkRecord(
                    chunk_id=UUID(c["chunk_id"]),
                    rank=c["rank"],
                    similarity_score=c["similarity_score"],
                    pdf_page=c["pdf_page"],
                    mini_page=c.get("mini_page"),
                    heading=c.get("heading"),
                    chunk_text=c["chunk_text"],
                )
                for c in data["chunks"]
            ]
            return TraceRecord(
                trace_id=UUID(data["trace_id"]),
                question=data["question"],
                answer=data["answer"],
                confidence=data["confidence"],
                top_k=data["top_k"],
                created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
                source_path=data.get("source_path"),
                chunks=chunks,
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

    def _get_last_trace_from_db(self) -> Optional[TraceRecord]:
        """Load the most recent trace from the database."""
        db = self._get_db()
        if db is None:
            return None
        try:
            from sqlalchemy import desc, select
            from src.db.models import QueryTraceModel

            with db.session() as session:
                stmt = (
                    select(QueryTraceModel)
                    .order_by(desc(QueryTraceModel.created_at))
                    .limit(1)
                )
                trace_model = session.execute(stmt).scalars().first()
                if trace_model is None:
                    return None
                chunk_models = sorted(trace_model.chunk_rows, key=lambda r: r.rank)
                chunks = [
                    ChunkRecord(
                        chunk_id=cm.chunk_id,
                        rank=cm.rank,
                        similarity_score=cm.similarity_score,
                        pdf_page=cm.pdf_page,
                        mini_page=cm.mini_page,
                        heading=cm.heading,
                        chunk_text=cm.chunk_text,
                    )
                    for cm in chunk_models
                ]
                return TraceRecord(
                    trace_id=trace_model.trace_id,
                    question=trace_model.question,
                    answer=trace_model.answer,
                    confidence=trace_model.confidence,
                    top_k=trace_model.top_k,
                    created_at=trace_model.created_at,
                    source_path=trace_model.source_path,
                    chunks=chunks,
                )
        except Exception:
            return None

    def get_trace(self, trace_id: UUID) -> Optional[TraceRecord]:
        """Load a specific trace by ID from the database."""
        db = self._get_db()
        if db is None:
            return None
        try:
            from src.db.models import QueryTraceChunkModel, QueryTraceModel

            with db.session() as session:
                trace_model = session.get(QueryTraceModel, trace_id)
                if trace_model is None:
                    return None
                chunk_models = sorted(
                    trace_model.chunk_rows,
                    key=lambda r: r.rank,
                )
                chunks = [
                    ChunkRecord(
                        chunk_id=cm.chunk_id,
                        rank=cm.rank,
                        similarity_score=cm.similarity_score,
                        pdf_page=cm.pdf_page,
                        mini_page=cm.mini_page,
                        heading=cm.heading,
                        chunk_text=cm.chunk_text,
                    )
                    for cm in chunk_models
                ]
                return TraceRecord(
                    trace_id=trace_model.trace_id,
                    question=trace_model.question,
                    answer=trace_model.answer,
                    confidence=trace_model.confidence,
                    top_k=trace_model.top_k,
                    created_at=trace_model.created_at,
                    source_path=trace_model.source_path,
                    chunks=chunks,
                )
        except Exception:
            return None
