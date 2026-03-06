-- Schema for the rulebook RAG document graph database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(512) NOT NULL,
    source_path VARCHAR(1024) NOT NULL,
    total_pages INTEGER NOT NULL CHECK (total_pages >= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_name ON documents(name);

-- Blocks table
CREATE TABLE IF NOT EXISTS blocks (
    block_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    block_type VARCHAR(50) NOT NULL,
    bbox_x0 FLOAT NOT NULL,
    bbox_y0 FLOAT NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    pdf_page INTEGER NOT NULL CHECK (pdf_page >= 0),
    mini_page INTEGER,
    reading_order INTEGER NOT NULL CHECK (reading_order >= 0),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    raw_block_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_blocks_document ON blocks(document_id);
CREATE INDEX idx_blocks_page ON blocks(document_id, pdf_page);
CREATE INDEX idx_blocks_mini_page ON blocks(document_id, pdf_page, mini_page);
CREATE INDEX idx_blocks_reading_order ON blocks(document_id, reading_order);
CREATE INDEX idx_blocks_type ON blocks(block_type);

-- Edges table
CREATE TABLE IF NOT EXISTS edges (
    edge_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    source_block_id UUID NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE,
    target_block_id UUID NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_block_id, target_block_id, edge_type)
);

CREATE INDEX idx_edges_document ON edges(document_id);
CREATE INDEX idx_edges_source ON edges(source_block_id);
CREATE INDEX idx_edges_target ON edges(target_block_id);
CREATE INDEX idx_edges_type ON edges(edge_type);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    pdf_page INTEGER NOT NULL CHECK (pdf_page >= 0),
    mini_page INTEGER,
    bbox_x0 FLOAT NOT NULL,
    bbox_y0 FLOAT NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    heading VARCHAR(512),
    token_count INTEGER DEFAULT 0,
    embedding FLOAT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_page ON chunks(document_id, pdf_page);

-- Junction table for chunk-block relationships
CREATE TABLE IF NOT EXISTS chunk_blocks (
    chunk_id UUID NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    block_id UUID NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (chunk_id, block_id)
);

CREATE INDEX idx_chunk_blocks_chunk ON chunk_blocks(chunk_id);
CREATE INDEX idx_chunk_blocks_block ON chunk_blocks(block_id);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for documents table
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- View for block neighbors (useful for graph traversal)
CREATE OR REPLACE VIEW block_neighbors AS
SELECT 
    e.source_block_id,
    e.target_block_id,
    e.edge_type,
    e.weight,
    sb.text AS source_text,
    sb.block_type AS source_type,
    tb.text AS target_text,
    tb.block_type AS target_type
FROM edges e
JOIN blocks sb ON e.source_block_id = sb.block_id
JOIN blocks tb ON e.target_block_id = tb.block_id;

-- View for chunk details with aggregated block info
CREATE OR REPLACE VIEW chunk_details AS
SELECT 
    c.chunk_id,
    c.document_id,
    c.text,
    c.pdf_page,
    c.mini_page,
    c.heading,
    c.token_count,
    COUNT(cb.block_id) AS block_count,
    ARRAY_AGG(cb.block_id ORDER BY cb.position) AS block_ids
FROM chunks c
LEFT JOIN chunk_blocks cb ON c.chunk_id = cb.chunk_id
GROUP BY c.chunk_id;
