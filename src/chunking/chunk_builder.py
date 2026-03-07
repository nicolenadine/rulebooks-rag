"""Build semantic chunks from document blocks."""

from typing import Optional
from uuid import uuid4

import tiktoken

from src.config import settings
from src.models.schema import Block, BlockType, BoundingBox, Chunk, DocumentGraph


class ChunkBuilder:
    """Build semantic chunks from document blocks.

    Chunks respect:
    - Heading boundaries (new chunk on new heading)
    - Mini-page boundaries
    - Maximum token limits
    - Reading order

    All configuration is loaded from settings.
    """

    def __init__(self):
        """Initialize the chunk builder using settings."""
        self.max_tokens = settings.chunk_max_tokens
        self.min_tokens = settings.chunk_min_tokens
        self.overlap_tokens = settings.chunk_overlap_tokens

        try:
            self._tokenizer = tiktoken.encoding_for_model(settings.openai_embedding_model)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def build_chunks(self, graph: DocumentGraph) -> list[Chunk]:
        """Build chunks from a document graph.

        Args:
            graph: The document graph to chunk.

        Returns:
            List of Chunk objects.
        """
        chunks = []
        sorted_blocks = sorted(
            graph.blocks,
            key=lambda b: (b.pdf_page, b.mini_page or 0, b.reading_order),
        )

        current_blocks: list[Block] = []
        current_tokens = 0
        current_heading: Optional[str] = None
        current_page: Optional[int] = None
        current_mini_page: Optional[int] = None

        for block in sorted_blocks:
            if block.block_type in (BlockType.PAGE_NUMBER, BlockType.HEADER, BlockType.FOOTER):
                continue

            block_tokens = self._count_tokens(block.text)

            should_break = self._should_break_chunk(
                block=block,
                current_tokens=current_tokens,
                block_tokens=block_tokens,
                current_page=current_page,
                current_mini_page=current_mini_page,
            )

            if should_break and current_blocks:
                chunk = self._create_chunk(current_blocks, current_heading)
                chunks.append(chunk)

                overlap_blocks = self._get_overlap_blocks(current_blocks)
                current_blocks = overlap_blocks
                current_tokens = sum(self._count_tokens(b.text) for b in overlap_blocks)

                if block.block_type == BlockType.TITLE:
                    current_heading = block.text

            current_blocks.append(block)
            current_tokens += block_tokens
            current_page = block.pdf_page
            current_mini_page = block.mini_page

            if block.block_type == BlockType.TITLE and current_heading is None:
                current_heading = block.text

        if current_blocks:
            chunk = self._create_chunk(current_blocks, current_heading)
            chunks.append(chunk)

        return chunks

    def _should_break_chunk(
        self,
        block: Block,
        current_tokens: int,
        block_tokens: int,
        current_page: Optional[int],
        current_mini_page: Optional[int],
    ) -> bool:
        """Determine if we should break and start a new chunk."""
        if current_tokens + block_tokens > self.max_tokens:
            return True

        if current_page is not None and block.pdf_page != current_page:
            if current_tokens >= self.min_tokens:
                return True

        if current_mini_page is not None and block.mini_page != current_mini_page:
            if current_tokens >= self.min_tokens:
                return True

        if block.block_type == BlockType.TITLE:
            if current_tokens >= self.min_tokens:
                return True

        return False

    def _get_overlap_blocks(self, blocks: list[Block]) -> list[Block]:
        """Get blocks to overlap into the next chunk."""
        if self.overlap_tokens <= 0:
            return []

        overlap_blocks = []
        overlap_tokens = 0

        for block in reversed(blocks):
            tokens = self._count_tokens(block.text)
            if overlap_tokens + tokens > self.overlap_tokens:
                break
            overlap_blocks.insert(0, block)
            overlap_tokens += tokens

        return overlap_blocks

    def _create_chunk(self, blocks: list[Block], heading: Optional[str]) -> Chunk:
        """Create a Chunk from a list of blocks."""
        text = "\n\n".join(b.text for b in blocks)
        block_ids = [b.block_id for b in blocks]

        bbox_union = blocks[0].bbox
        for block in blocks[1:]:
            bbox_union = bbox_union.union(block.bbox)

        pdf_page = blocks[0].pdf_page
        mini_page = blocks[0].mini_page

        return Chunk(
            chunk_id=uuid4(),
            text=text,
            block_ids=block_ids,
            pdf_page=pdf_page,
            mini_page=mini_page,
            bbox_union=bbox_union,
            heading=heading,
            token_count=self._count_tokens(text),
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._tokenizer.encode(text))

    def merge_small_chunks(
        self,
        chunks: list[Chunk],
        min_tokens: Optional[int] = None,
    ) -> list[Chunk]:
        """Merge chunks that are too small.

        Args:
            chunks: List of chunks to potentially merge.
            min_tokens: Minimum token count (defaults to self.min_tokens).

        Returns:
            List of chunks with small ones merged.
        """
        if not chunks:
            return []

        min_tokens = min_tokens or self.min_tokens
        result = []
        current: Optional[Chunk] = None

        for chunk in chunks:
            if current is None:
                current = chunk
                continue

            if current.pdf_page != chunk.pdf_page or current.mini_page != chunk.mini_page:
                result.append(current)
                current = chunk
                continue

            if current.token_count < min_tokens:
                current = self._merge_chunks(current, chunk)
            else:
                result.append(current)
                current = chunk

        if current:
            result.append(current)

        return result

    def _merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """Merge two chunks into one."""
        text = chunk1.text + "\n\n" + chunk2.text
        block_ids = chunk1.block_ids + chunk2.block_ids
        bbox_union = chunk1.bbox_union.union(chunk2.bbox_union)

        return Chunk(
            chunk_id=uuid4(),
            text=text,
            block_ids=block_ids,
            pdf_page=chunk1.pdf_page,
            mini_page=chunk1.mini_page,
            bbox_union=bbox_union,
            heading=chunk1.heading or chunk2.heading,
            token_count=self._count_tokens(text),
        )
