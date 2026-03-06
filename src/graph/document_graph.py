"""Operations on the document graph."""

from collections import defaultdict
from typing import Optional
from uuid import UUID

from src.models.schema import Block, DocumentGraph, Edge, EdgeType


class DocumentGraphOps:
    """Operations for traversing and querying the document graph."""

    def __init__(self, graph: DocumentGraph):
        """Initialize with a document graph.

        Args:
            graph: The DocumentGraph to operate on.
        """
        self.graph = graph
        self._block_index: dict[UUID, Block] = {}
        self._outgoing_edges: dict[UUID, list[Edge]] = defaultdict(list)
        self._incoming_edges: dict[UUID, list[Edge]] = defaultdict(list)
        self._build_indices()

    def _build_indices(self) -> None:
        """Build internal indices for fast lookups."""
        for block in self.graph.blocks:
            self._block_index[block.block_id] = block

        for edge in self.graph.edges:
            self._outgoing_edges[edge.source_block_id].append(edge)
            self._incoming_edges[edge.target_block_id].append(edge)

    def get_block(self, block_id: UUID) -> Optional[Block]:
        """Get a block by ID."""
        return self._block_index.get(block_id)

    def get_next_blocks(self, block_id: UUID) -> list[Block]:
        """Get blocks that follow this one in reading order."""
        return self._get_neighbors(block_id, EdgeType.NEXT, outgoing=True)

    def get_previous_blocks(self, block_id: UUID) -> list[Block]:
        """Get blocks that precede this one in reading order."""
        return self._get_neighbors(block_id, EdgeType.NEXT, outgoing=False)

    def get_illustrated_blocks(self, figure_block_id: UUID) -> list[Block]:
        """Get text blocks that a figure illustrates."""
        return self._get_neighbors(figure_block_id, EdgeType.ILLUSTRATES, outgoing=True)

    def get_illustrating_figures(self, text_block_id: UUID) -> list[Block]:
        """Get figures that illustrate a text block."""
        return self._get_neighbors(text_block_id, EdgeType.ILLUSTRATES, outgoing=False)

    def get_context_window(
        self,
        block_id: UUID,
        before: int = 2,
        after: int = 2,
    ) -> list[Block]:
        """Get a window of blocks around the given block.

        Args:
            block_id: Center block ID.
            before: Number of preceding blocks to include.
            after: Number of following blocks to include.

        Returns:
            List of blocks in reading order.
        """
        center_block = self.get_block(block_id)
        if not center_block:
            return []

        result = [center_block]

        current = block_id
        for _ in range(before):
            prev_blocks = self.get_previous_blocks(current)
            if not prev_blocks:
                break
            prev_block = prev_blocks[0]
            result.insert(0, prev_block)
            current = prev_block.block_id

        current = block_id
        for _ in range(after):
            next_blocks = self.get_next_blocks(current)
            if not next_blocks:
                break
            next_block = next_blocks[0]
            result.append(next_block)
            current = next_block.block_id

        return result

    def get_blocks_in_chunk(self, chunk_id: UUID) -> list[Block]:
        """Get all blocks belonging to a chunk.

        Note: This requires IN_CHUNK edges to be present in the graph.
        """
        blocks = []
        for edge in self.graph.edges:
            if edge.edge_type == EdgeType.IN_CHUNK and edge.metadata.get("chunk_id") == chunk_id:
                block = self.get_block(edge.source_block_id)
                if block:
                    blocks.append(block)
        return sorted(blocks, key=lambda b: b.reading_order)

    def get_reading_order_sequence(
        self,
        start_block_id: Optional[UUID] = None,
        max_blocks: int = 100,
    ) -> list[Block]:
        """Get blocks in reading order starting from a given block.

        Args:
            start_block_id: Block to start from. If None, starts from first block.
            max_blocks: Maximum number of blocks to return.

        Returns:
            List of blocks in reading order.
        """
        if start_block_id is None:
            sorted_blocks = sorted(self.graph.blocks, key=lambda b: b.reading_order)
            return sorted_blocks[:max_blocks]

        start_block = self.get_block(start_block_id)
        if not start_block:
            return []

        result = [start_block]
        current = start_block_id

        while len(result) < max_blocks:
            next_blocks = self.get_next_blocks(current)
            if not next_blocks:
                break
            next_block = next_blocks[0]
            result.append(next_block)
            current = next_block.block_id

        return result

    def find_section_blocks(self, heading_block_id: UUID) -> list[Block]:
        """Find all blocks under a heading until the next heading.

        Args:
            heading_block_id: ID of the heading block.

        Returns:
            All blocks in this section including the heading.
        """
        heading = self.get_block(heading_block_id)
        if not heading:
            return []

        result = [heading]
        current = heading_block_id

        while True:
            next_blocks = self.get_next_blocks(current)
            if not next_blocks:
                break

            next_block = next_blocks[0]
            if next_block.block_type in (BlockType.TITLE, BlockType.HEADER):
                break

            result.append(next_block)
            current = next_block.block_id

        return result

    def _get_neighbors(
        self,
        block_id: UUID,
        edge_type: EdgeType,
        outgoing: bool = True,
    ) -> list[Block]:
        """Get neighboring blocks by edge type."""
        if outgoing:
            edges = self._outgoing_edges.get(block_id, [])
            target_attr = "target_block_id"
        else:
            edges = self._incoming_edges.get(block_id, [])
            target_attr = "source_block_id"

        neighbors = []
        for edge in edges:
            if edge.edge_type == edge_type:
                neighbor_id = getattr(edge, target_attr)
                neighbor = self.get_block(neighbor_id)
                if neighbor:
                    neighbors.append(neighbor)

        return neighbors

    def get_blocks_by_page(
        self,
        pdf_page: int,
        mini_page: Optional[int] = None,
    ) -> list[Block]:
        """Get all blocks on a specific page."""
        return self.graph.get_blocks_by_page(pdf_page, mini_page)

    def get_all_headings(self) -> list[Block]:
        """Get all heading/title blocks in the document."""
        return [
            b for b in self.graph.blocks
            if b.block_type in (BlockType.TITLE, BlockType.HEADER)
        ]


from src.models.schema import BlockType  # noqa: E402
