# Rulebook RAG

A system that converts board game rulebooks (PDFs) into a structured knowledge base that supports a "rules assistant" capable of answering questions about the rules with accurate citations.

## Pipeline Overview

```
PDF → Parsed Layout JSON → Document Graph → Semantic Chunks → Vector Index → LLM Question Answering
```

### Pipeline Stages

1. **PDF Parsing** (external): Use [Reducto](https://reducto.ai) to parse PDFs into structured JSON with text blocks, bounding boxes, and reading order.

2. **Ingestion**: Load Reducto JSON output and convert into internal Block objects. Detect mini-pages (multiple logical pages per PDF page).

3. **Graph Building**: Create a document graph with edges representing reading order, figure-text relationships, and chunk membership.

4. **Chunking**: Create semantic chunks that respect headings, mini-page boundaries, and token limits.

5. **Embedding**: Generate vector embeddings for each chunk using OpenAI's embedding models.

6. **Retrieval**: Find relevant chunks using similarity search, with context expansion from the document graph.

7. **Question Answering**: Generate answers using an LLM with retrieved context, including citations with page numbers and bounding boxes.

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- PostgreSQL (optional, for persistent storage)
- OpenAI API key

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd rulebook-rag
```

2. Install dependencies with uv:

```bash
uv sync
```

3. Install development dependencies:

```bash
uv sync --dev
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

5. Install pre-commit hooks:

```bash
uv run pre-commit install
```

### Database Setup (Optional)

For persistent storage, run PostgreSQL using Docker:

```bash
cd docker
docker-compose up -d
```

This starts PostgreSQL on port 5432 with the schema automatically initialized.

To also run pgAdmin for database management:

```bash
docker-compose --profile tools up -d
```

pgAdmin will be available at http://localhost:5050

## Usage

### Quick Start

Ask a question about a parsed rulebook:

```bash
uv run python main.py ask --rulebook data/parsed/flip7.json --question "What happens if two players tie?"
```

### CLI Commands

#### `ask` - Answer a question

```bash
uv run python main.py ask \
  --rulebook data/parsed/game.json \
  --question "How do you win the game?" \
  --top-k 5
```

Options:
- `--rulebook, -r`: Path to parsed rulebook JSON (required)
- `--question, -q`: Question to ask (required)
- `--top-k, -k`: Number of chunks to retrieve (default: 5)
- `--index-path, -i`: Path to save/load vector index
- `--no-cache`: Rebuild index even if cached

#### `index` - Build a vector index

```bash
uv run python main.py index \
  --rulebook data/parsed/game.json \
  --output data/processed/game.index
```

Pre-building the index speeds up subsequent queries.

#### `info` - Display rulebook information

```bash
uv run python main.py info --rulebook data/parsed/game.json
```

Shows document statistics, block types, and mini-page detection results.

## Project Structure

```
rulebook-rag/
├── src/
│   ├── ingestion/          # JSON loading and mini-page detection
│   │   ├── parse_loader.py
│   │   └── mini_page_detector.py
│   │
│   ├── graph/              # Document graph building and traversal
│   │   ├── document_graph.py
│   │   └── graph_builder.py
│   │
│   ├── chunking/           # Semantic chunking
│   │   └── chunk_builder.py
│   │
│   ├── embeddings/         # Vector embedding generation
│   │   └── embedder.py
│   │
│   ├── retrieval/          # Vector search and context expansion
│   │   ├── vector_index.py
│   │   └── retriever.py
│   │
│   ├── qa/                 # RAG pipeline and answer generation
│   │   └── rag_pipeline.py
│   │
│   ├── db/                 # PostgreSQL storage
│   │   ├── models.py
│   │   ├── schema.sql
│   │   └── database.py
│   │
│   ├── models/             # Pydantic data models
│   │   └── schema.py
│   │
│   └── utils/              # Utility functions
│       └── bbox_utils.py
│
├── data/
│   ├── raw/                # Original PDF files
│   ├── parsed/             # Reducto JSON output
│   └── processed/          # Vector indices and processed data
│
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks for exploration
├── docker/                 # Docker configuration
│
├── main.py                 # CLI entry point
├── pyproject.toml          # Project configuration
└── README.md
```

## Data Models

### Block

Represents a text unit extracted from the PDF:

```python
Block(
    block_id: UUID,
    text: str,
    block_type: BlockType,  # text, title, figure, table, etc.
    bbox: BoundingBox,      # x0, y0, x1, y1 coordinates
    pdf_page: int,          # Zero-indexed PDF page
    mini_page: int | None,  # Logical page within PDF page
    reading_order: int,
    confidence: float,
)
```

### Chunk

A semantic unit for embedding and retrieval:

```python
Chunk(
    chunk_id: UUID,
    text: str,
    block_ids: list[UUID],
    pdf_page: int,
    mini_page: int | None,
    bbox_union: BoundingBox,
    heading: str | None,
    token_count: int,
    embedding: list[float] | None,
)
```

### DocumentGraph

Graph representation with blocks and edges:

- **NEXT**: Reading order connections
- **ILLUSTRATES**: Figure-to-text relationships
- **CAPTION_OF**: Caption-to-figure relationships
- **IN_CHUNK**: Block-to-chunk membership

## Mini-Page Detection

Board game rulebooks often have multiple logical pages printed on a single PDF page. The mini-page detector:

1. Clusters blocks into quadrant regions (top-left, top-right, bottom-left, bottom-right)
2. Detects if multiple regions contain significant content
3. Identifies printed page numbers in each region
4. Assigns blocks to their logical mini-page

## Future Extensions

The architecture supports:

- **Figure embeddings**: Embed images for multimodal retrieval
- **Diagram understanding**: Extract structured info from game diagrams
- **UI highlighting**: Display source regions in a viewer
- **Incremental updates**: Add new rulebooks without reindexing everything

## Development

### Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src --cov-report=html
```

### Linting and Formatting

```bash
# Check with ruff
uv run ruff check .

# Format with black
uv run black .

# Type check with mypy
uv run mypy src/
```

### Pre-commit Hooks

Hooks run automatically on commit. To run manually:

```bash
uv run pre-commit run --all-files
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM | Required |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://localhost:5432/rulebook_rag` |

## License

MIT
