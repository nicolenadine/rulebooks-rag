#!/usr/bin/env python3
"""CLI for the Rulebook RAG system."""

import hashlib
import json
import sys
from pathlib import Path
from uuid import UUID

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.chunking.chunk_builder import ChunkBuilder
from src.config import settings  # noqa: F401 - loads .env on import
from src.db.database import get_database
from src.embeddings.embedder import Embedder
from src.graph.graph_builder import GraphBuilder
from src.ingestion.mini_page_detector import MiniPageDetector
from src.ingestion.parse_loader import ParseLoader
from src.models.schema import Chunk, Document, DocumentGraph
from src.qa.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.vector_index import VectorIndex
from src.tracing import TraceLogger

console = Console()
trace_logger = TraceLogger()


@click.group()
def cli():
    """Rulebook RAG - Question answering for board game rulebooks."""
    pass


def _normalize_text(text: str) -> str:
    """Normalize text for stable hashing: strip and collapse whitespace."""
    return " ".join(text.strip().split())


def _stable_chunk_key(chunk: Chunk) -> str:
    """Stable hash key for matching in-memory chunks to DB chunks.

    Uses sha256(normalized_text + pdf_page + mini_page + chunking_version)
    so that minor whitespace or normalization changes do not break reuse,
    and chunking algorithm changes can be detected by bumping chunking_version.
    """
    normalized = _normalize_text(chunk.text)
    mini = chunk.mini_page if chunk.mini_page is not None else -1
    payload = f"{normalized}\n{chunk.pdf_page}\n{mini}\n{settings.chunking_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _embed_reuse_log(msg: str) -> None:
    """Temporary debug logging for embedding reuse (stderr)."""
    print(f"[embed-reuse] {msg}", file=sys.stderr)


def _prepare_embeddings_with_reuse(
    document: Document,
    graph: DocumentGraph,
    chunks: list[Chunk],
    embedder: Embedder,
) -> tuple[list[Chunk], int, int]:
    """Load existing embeddings from DB where possible; embed only chunks with NULL embedding; persist.

    Uses canonical source_path (Path(...).resolve()) for lookup and save so relative vs
    absolute paths match. Returns (chunks with embeddings filled, n_reused, n_newly_generated).
    """
    n_reused = 0
    n_new = 0
    lookup_path = str(Path(document.source_path).resolve())
    _embed_reuse_log(f"lookup source_path (canonical): {lookup_path!r}")

    try:
        db = get_database()
        doc_id = db.get_document_by_source_path(document.source_path)
    except Exception as e:
        _embed_reuse_log(f"get_document_by_source_path raised: {e}")
        doc_id = None

    _embed_reuse_log(f"get_document_by_source_path found document_id: {doc_id is not None} ({doc_id})")

    if doc_id is not None:
        # Document exists: load chunks, match by stable hash key, fill embeddings.
        _embed_reuse_log("branch: document exists")
        try:
            db_chunks = db.get_chunks_by_document(doc_id)
            n_db_with_embedding = sum(1 for c in db_chunks if c.embedding is not None)
            _embed_reuse_log(f"DB chunks loaded: {len(db_chunks)} (with embedding: {n_db_with_embedding})")

            key_to_db: dict[str, Chunk] = {
                _stable_chunk_key(c): c for c in db_chunks
            }
            _embed_reuse_log(f"stable chunk keys from DB: {len(key_to_db)}")

            matched_chunk_ids: set[UUID] = set()
            n_matched = 0
            for ch in chunks:
                key = _stable_chunk_key(ch)
                if key in key_to_db:
                    db_ch = key_to_db[key]
                    ch.chunk_id = db_ch.chunk_id
                    ch.embedding = db_ch.embedding
                    matched_chunk_ids.add(ch.chunk_id)
                    n_matched += 1
            _embed_reuse_log(f"in-memory chunks matched DB keys: {n_matched}")

            chunks_to_embed = [c for c in chunks if c.embedding is None]
            _embed_reuse_log(f"chunks_to_embed (need API call): {len(chunks_to_embed)}")

            n_reused = len(chunks) - len(chunks_to_embed)
            if chunks_to_embed:
                embedder.embed_chunks(chunks_to_embed)
                n_new = len(chunks_to_embed)
                for c in chunks_to_embed:
                    if c.chunk_id in matched_chunk_ids:
                        db.update_chunk_embedding(c.chunk_id, c.embedding or [])
            return (chunks, n_reused, n_new)
        except Exception as e:
            _embed_reuse_log(f"document-exists path raised: {e}")
            pass
        # Fallback: embed all
        embedder.embed_chunks(chunks)
        return (chunks, 0, len(chunks))

    # Document does not exist: persist graph, embed all, save all chunks.
    _embed_reuse_log("branch: document does not exist (will insert)")
    saved_source = str(Path(graph.document.source_path).resolve())
    _embed_reuse_log(f"saving new document with source_path (canonical): {saved_source!r}")
    try:
        db = get_database()
        db.save_document_graph(graph)
        embedder.embed_chunks(chunks)
        for c in chunks:
            db.save_chunk(c, document.document_id)
        return (chunks, 0, len(chunks))
    except Exception as e:
        _embed_reuse_log(f"new-document path raised: {e}")
        embedder.embed_chunks(chunks)
        return (chunks, 0, len(chunks))


@cli.command()
@click.option(
    "--rulebook",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Path to the parsed rulebook JSON file.",
)
@click.option(
    "--question",
    "-q",
    required=True,
    help="Question to ask about the rulebook.",
)
@click.option(
    "--top-k",
    "-k",
    default=5,
    help="Number of chunks to retrieve.",
)
@click.option(
    "--index-path",
    "-i",
    type=click.Path(),
    help="Path to save/load the vector index.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Rebuild the index even if cached.",
)
def ask(rulebook: str, question: str, top_k: int, index_path: str, no_cache: bool):
    """Ask a question about a rulebook."""
    rulebook_path = Path(rulebook)

    # Step 1: Verify raw JSON shape before ParseLoader touches it
    try:
        data = json.loads(rulebook_path.read_text())
    except Exception as e:
        console.print(f"[red]Failed to load JSON from {rulebook_path}: {e}[/red]")
        raise
    console.print("[dim]Raw JSON shape:[/dim]")
    console.print(f"  TOP LEVEL TYPE: {type(data).__name__}")
    if isinstance(data, dict):
        console.print(f"  TOP LEVEL KEYS: {list(data.keys())[:20]}")
        if "result" in data and isinstance(data["result"], dict):
            r = data["result"]
            console.print(f"  result KEYS: {list(r.keys())[:20]}")
            if "blocks" in r:
                console.print(f"  result.blocks LEN: {len(r['blocks'])}")
                if r["blocks"]:
                    b0 = r["blocks"][0]
                    console.print(f"  FIRST BLOCK KEYS: {list(b0.keys()) if isinstance(b0, dict) else type(b0).__name__}")
            if "pages" in r:
                console.print(f"  result.pages LEN: {len(r['pages'])}")
        if "pages" in data:
            console.print(f"  pages LEN: {len(data['pages'])}")
    elif isinstance(data, list):
        console.print(f"  TOP LEVEL LIST LEN: {len(data)}")
        if data:
            console.print(f"  FIRST ITEM TYPE: {type(data[0]).__name__}")
            if isinstance(data[0], dict):
                console.print(f"  FIRST ITEM KEYS: {list(data[0].keys())[:20]}")

    if index_path:
        index_file = Path(index_path)
    else:
        index_file = rulebook_path.with_suffix(".index")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if index_file.with_suffix(".faiss").exists() and not no_cache:
            progress.add_task("Loading cached index...", total=None)
            vector_index = VectorIndex.load(index_file)

            loader = ParseLoader(rulebook_path)
            document, blocks = loader.load()
            document.source_path = str(rulebook_path.resolve())
            if not blocks:
                raise ValueError(
                    f"No blocks were loaded from {rulebook_path}. "
                    "The ParseLoader assumptions may not match the Reducto JSON format."
                )
            console.print(f"  [dim]Blocks after ParseLoader: {len(blocks)}[/dim]")
            console.print("  Sample block:", blocks[0] if blocks else "None")
            builder = GraphBuilder(document, blocks)
            graph = builder.build()
            console.print(f"  [dim]Graph blocks (nodes): {len(graph.blocks)}[/dim]")
        else:
            task = progress.add_task("Loading rulebook...", total=None)
            loader = ParseLoader(rulebook_path)
            document, blocks = loader.load()
            document.source_path = str(rulebook_path.resolve())
            if not blocks:
                raise ValueError(
                    f"No blocks were loaded from {rulebook_path}. "
                    "The ParseLoader assumptions may not match the Reducto JSON format."
                )
            console.print(f"  [dim]Blocks after ParseLoader: {len(blocks)}[/dim]")
            console.print("  Sample block:", blocks[0] if blocks else "None")
            progress.update(task, description=f"Loaded {len(blocks)} blocks")

            progress.add_task("Detecting mini-pages...", total=None)
            detector = MiniPageDetector()
            page_dims = _estimate_page_dimensions(blocks)
            blocks = detector.process_document(blocks, page_dims)
            console.print(f"  [dim]Blocks after mini-page assignment: {len(blocks)}[/dim]")

            progress.add_task("Building document graph...", total=None)
            builder = GraphBuilder(document, blocks)
            graph = builder.build()
            console.print(f"  [dim]Graph blocks (nodes): {len(graph.blocks)}[/dim]")

            progress.add_task("Creating chunks...", total=None)
            chunk_builder = ChunkBuilder()
            chunks = chunk_builder.build_chunks(graph)
            chunks = chunk_builder.merge_small_chunks(chunks)
            if not chunks:
                raise ValueError(
                    "Chunking produced zero chunks. Check block extraction and chunk settings."
                )
            console.print(f"  [dim]Chunks created: {len(chunks)}[/dim]")

            progress.add_task("Generating embeddings (reuse from DB when present)...", total=None)
            embedder = Embedder()
            chunks, n_reused, n_new = _prepare_embeddings_with_reuse(
                document, graph, chunks, embedder
            )
            console.print(
                f"  [dim]Chunks: {len(chunks)} | "
                f"Embeddings reused: {n_reused} | Newly generated: {n_new}[/dim]"
            )

            progress.add_task("Building vector index...", total=None)
            vector_index = VectorIndex()
            if chunks:
                vector_index.add_chunks(chunks)
            else:
                console.print("Skipping vector index build: no chunks available.")

            if index_path or not no_cache:
                vector_index.save(index_file)
                console.print(f"  Saved index to {index_file}")

        progress.add_task("Retrieving relevant context...", total=None)
        embedder = Embedder()
        retriever = Retriever(vector_index, embedder, graph)

        progress.add_task("Generating answer...", total=None)
        pipeline = RAGPipeline(retriever, graph=graph)
        last_retrieved = []

        def on_retrieved(retrieved):
            last_retrieved.append(retrieved)

        response = pipeline.answer(
            question, top_k=top_k, on_retrieved=on_retrieved
        )

        if last_retrieved:
            trace_logger.log_qa_trace(
                question=question,
                answer=response.answer,
                confidence=response.confidence,
                top_k=top_k,
                retrieved=last_retrieved[0],
                source_path=str(rulebook_path.resolve()),
            )

    console.print()
    console.print(Panel(f"[bold]Question:[/bold] {question}", border_style="blue"))
    console.print()
    console.print(Panel(response.answer, title="Answer", border_style="green"))

    if response.citations:
        console.print()
        console.print("[bold]Citations:[/bold]")
        for i, citation in enumerate(response.citations, 1):
            page_ref = citation.format_reference()
            snippet = citation.text_snippet[:100] + "..." if len(citation.text_snippet) > 100 else citation.text_snippet
            console.print(f"  [{i}] {page_ref}")
            console.print(f"      [dim]\"{snippet}\"[/dim]")

    console.print()
    console.print(f"[dim]Confidence: {response.confidence:.0%} | Chunks: {len(response.retrieved_chunks)}[/dim]")


@cli.command()
@click.option(
    "--rulebook",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Path to the parsed rulebook JSON file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for the vector index.",
)
def index(rulebook: str, output: str):
    """Build and save a vector index for a rulebook."""
    rulebook_path = Path(rulebook)
    output_path = Path(output) if output else rulebook_path.with_suffix(".index")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading rulebook...", total=None)
        loader = ParseLoader(rulebook_path)
        document, blocks = loader.load()
        document.source_path = str(rulebook_path.resolve())
        console.print(f"  Loaded {len(blocks)} blocks from {document.name}")

        progress.add_task("Detecting mini-pages...", total=None)
        detector = MiniPageDetector()
        page_dims = _estimate_page_dimensions(blocks)
        blocks = detector.process_document(blocks, page_dims)

        progress.add_task("Building document graph...", total=None)
        builder = GraphBuilder(document, blocks)
        graph = builder.build()

        progress.add_task("Creating chunks...", total=None)
        chunk_builder = ChunkBuilder()
        chunks = chunk_builder.build_chunks(graph)
        chunks = chunk_builder.merge_small_chunks(chunks)
        console.print(f"  Created {len(chunks)} chunks")

        progress.add_task("Generating embeddings (reuse from DB when present)...", total=None)
        embedder = Embedder()
        chunks, n_reused, n_new = _prepare_embeddings_with_reuse(
            document, graph, chunks, embedder
        )
        console.print(
            f"  Chunks: {len(chunks)} | Embeddings reused: {n_reused} | Newly generated: {n_new}"
        )

        progress.add_task("Saving index...", total=None)
        vector_index = VectorIndex()
        vector_index.add_chunks(chunks)
        vector_index.save(output_path)

    console.print()
    console.print(f"[green]Index saved to {output_path}[/green]")
    console.print(f"  Chunks: {len(chunks)}")
    console.print(f"  Pages: {document.total_pages}")


@cli.command()
@click.option(
    "--rulebook",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Path to the parsed rulebook JSON file.",
)
def info(rulebook: str):
    """Display information about a parsed rulebook."""
    rulebook_path = Path(rulebook)

    loader = ParseLoader(rulebook_path)
    document, blocks = loader.load()

    detector = MiniPageDetector()
    page_dims = _estimate_page_dimensions(blocks)
    blocks = detector.process_document(blocks, page_dims)

    console.print(Panel(f"[bold]{document.name}[/bold]", border_style="blue"))
    console.print()
    console.print(f"[bold]Source:[/bold] {document.source_path}")
    console.print(f"[bold]Pages:[/bold] {document.total_pages}")
    console.print(f"[bold]Total blocks:[/bold] {len(blocks)}")

    type_counts = {}
    for block in blocks:
        type_counts[block.block_type.value] = type_counts.get(block.block_type.value, 0) + 1

    console.print()
    console.print("[bold]Block types:[/bold]")
    for block_type, count in sorted(type_counts.items()):
        console.print(f"  {block_type}: {count}")

    pages_with_mini = sum(1 for b in blocks if b.mini_page is not None and b.mini_page > 0)
    if pages_with_mini:
        console.print()
        console.print(f"[bold]Blocks in mini-pages:[/bold] {pages_with_mini}")


@cli.command("last-chunks")
@click.option(
    "--full",
    is_flag=True,
    help="Print full chunk text instead of a preview.",
)
@click.option(
    "--trace-id",
    type=click.UUID,
    default=None,
    help="Show a specific trace by ID (from DB). Default is the last query.",
)
def last_chunks(full: bool, trace_id):
    """Show the chunks retrieved for the last query (for debugging and review)."""
    if trace_id is not None:
        record = trace_logger.get_trace(trace_id)
    else:
        record = trace_logger.get_last_trace()

    if record is None:
        console.print("[yellow]No trace found. Run [bold]ask[/bold] first.[/yellow]")
        return

    preview_len = 400 if not full else None
    created = record.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    console.print()
    console.print(Panel(record.question, title="Question", border_style="blue"))
    console.print()
    console.print(
        Panel(
            record.answer if len(record.answer) <= 500 else record.answer[:500] + "...",
            title="Answer (excerpt)" if len(record.answer) > 500 else "Answer",
            border_style="green",
        )
    )
    console.print()
    console.print(
        f"[dim]Trace: {record.trace_id} | {created} | "
        f"confidence {record.confidence:.0%} | top_k={record.top_k}[/dim]"
    )
    if record.source_path:
        console.print(f"[dim]Source: {record.source_path}[/dim]")
    console.print()
    console.print(Panel("Retrieved chunks", border_style="cyan"))
    console.print()

    for c in record.chunks:
        page_ref = f"p.{c.pdf_page + 1}"
        if c.mini_page is not None:
            page_ref += f" (section {c.mini_page + 1})"
        heading = f" — [bold]{c.heading}[/bold]" if c.heading else ""
        console.print(f"  [bold]#{c.rank}[/bold]  {page_ref}  score={c.similarity_score:.3f}{heading}")
        text = c.chunk_text
        if preview_len and len(text) > preview_len:
            text = text[:preview_len] + " [...]"
        console.print(f"  [dim]{text}[/dim]")
        console.print()

    console.print(f"[dim]{len(record.chunks)} chunk(s)[/dim]")


def _estimate_page_dimensions(blocks: list) -> dict:
    """Estimate page dimensions from blocks."""
    page_dims = {}

    for block in blocks:
        if block.pdf_page not in page_dims:
            page_dims[block.pdf_page] = (0.0, 0.0)

        current_w, current_h = page_dims[block.pdf_page]
        page_dims[block.pdf_page] = (
            max(current_w, block.bbox.x1),
            max(current_h, block.bbox.y1),
        )

    for page, (w, h) in page_dims.items():
        page_dims[page] = (w * 1.1, h * 1.1)

    return page_dims


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
