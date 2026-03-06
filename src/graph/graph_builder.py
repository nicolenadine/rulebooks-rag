"""Build document graphs from blocks."""

from typing import Optional
from uuid import UUID

from src.models.schema import Block, BlockType, Document, DocumentGraph, Edge, EdgeType


class GraphBuilder:
    """Build a document graph from extracted blocks."""

    def __init__(self, document: Document, blocks: list[Block]):
        """Initialize the graph builder.

        Args:
            document: The source document.
            blocks: Extracted blocks from the document.
        """
        self.document = document
        self.blocks = sorted(blocks, key=lambda b: (b.pdf_page, b.mini_page or 0, b.reading_order))
        self._block_index = {b.block_id: b for b in self.blocks}

    def build(self) -> DocumentGraph:
        """Build the complete document graph.

        Returns:
            DocumentGraph with all edges constructed.
        """
        edges = []

        reading_order_edges = self._build_reading_order_edges()
        edges.extend(reading_order_edges)

        illustration_edges = self._build_illustration_edges()
        edges.extend(illustration_edges)

        caption_edges = self._build_caption_edges()
        edges.extend(caption_edges)

        return DocumentGraph(
            document=self.document,
            blocks=self.blocks,
            edges=edges,
        )

    def _build_reading_order_edges(self) -> list[Edge]:
        """Build NEXT edges based on reading order."""
        edges = []

        for i in range(len(self.blocks) - 1):
            current = self.blocks[i]
            next_block = self.blocks[i + 1]

            if current.pdf_page == next_block.pdf_page and current.mini_page == next_block.mini_page:
                weight = 1.0
            elif current.pdf_page == next_block.pdf_page:
                weight = 0.8
            else:
                weight = 0.5

            edge = Edge(
                source_block_id=current.block_id,
                target_block_id=next_block.block_id,
                edge_type=EdgeType.NEXT,
                weight=weight,
            )
            edges.append(edge)

        return edges

    def _build_illustration_edges(self) -> list[Edge]:
        """Build ILLUSTRATES edges linking figures to nearby text."""
        edges = []

        figure_blocks = [b for b in self.blocks if b.block_type == BlockType.FIGURE]

        for figure in figure_blocks:
            nearby_text = self._find_nearby_text_blocks(figure)

            for text_block, score in nearby_text[:3]:
                edge = Edge(
                    source_block_id=figure.block_id,
                    target_block_id=text_block.block_id,
                    edge_type=EdgeType.ILLUSTRATES,
                    weight=score,
                    metadata={"relationship": "spatial_proximity"},
                )
                edges.append(edge)

        return edges

    def _build_caption_edges(self) -> list[Edge]:
        """Build CAPTION_OF edges linking captions to figures/tables."""
        edges = []

        caption_blocks = [b for b in self.blocks if b.block_type == BlockType.CAPTION]

        for caption in caption_blocks:
            target = self._find_caption_target(caption)
            if target:
                edge = Edge(
                    source_block_id=caption.block_id,
                    target_block_id=target.block_id,
                    edge_type=EdgeType.CAPTION_OF,
                    weight=1.0,
                )
                edges.append(edge)

        return edges

    def _find_nearby_text_blocks(
        self,
        figure: Block,
        max_distance: float = 100.0,
    ) -> list[tuple[Block, float]]:
        """Find text blocks near a figure.

        Args:
            figure: The figure block.
            max_distance: Maximum distance to consider.

        Returns:
            List of (block, score) tuples, sorted by score descending.
        """
        candidates = []

        for block in self.blocks:
            if block.block_type not in (BlockType.TEXT, BlockType.TITLE):
                continue
            if block.pdf_page != figure.pdf_page:
                continue
            if block.mini_page != figure.mini_page:
                continue

            distance = self._bbox_distance(figure, block)
            if distance < max_distance:
                score = 1.0 - (distance / max_distance)
                candidates.append((block, score))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def _find_caption_target(self, caption: Block) -> Optional[Block]:
        """Find the figure or table that a caption describes."""
        best_target = None
        best_distance = float("inf")

        for block in self.blocks:
            if block.block_type not in (BlockType.FIGURE, BlockType.TABLE):
                continue
            if block.pdf_page != caption.pdf_page:
                continue
            if block.mini_page != caption.mini_page:
                continue

            distance = self._bbox_distance(caption, block)

            if distance < best_distance:
                best_distance = distance
                best_target = block

        return best_target

    def _bbox_distance(self, block1: Block, block2: Block) -> float:
        """Calculate distance between two blocks based on bounding boxes."""
        cx1, cy1 = block1.bbox.center
        cx2, cy2 = block2.bbox.center

        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

    def add_chunk_edges(
        self,
        graph: DocumentGraph,
        chunk_id: UUID,
        block_ids: list[UUID],
    ) -> DocumentGraph:
        """Add IN_CHUNK edges for a chunk.

        Args:
            graph: The document graph to modify.
            chunk_id: ID of the chunk.
            block_ids: IDs of blocks in the chunk.

        Returns:
            Updated document graph.
        """
        new_edges = list(graph.edges)

        for block_id in block_ids:
            edge = Edge(
                source_block_id=block_id,
                target_block_id=block_id,
                edge_type=EdgeType.IN_CHUNK,
                metadata={"chunk_id": str(chunk_id)},
            )
            new_edges.append(edge)

        return DocumentGraph(
            document=graph.document,
            blocks=graph.blocks,
            edges=new_edges,
        )
