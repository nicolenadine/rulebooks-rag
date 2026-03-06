"""Detect mini-pages within PDF pages (multiple logical pages per physical page)."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.models.schema import Block, BlockType


class MiniPageRegion(Enum):
    """Quadrant regions for mini-page detection."""

    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


@dataclass
class PageLayout:
    """Detected page layout information."""

    num_columns: int = 1
    num_rows: int = 1
    mini_page_count: int = 1
    page_numbers: dict[int, Optional[int]] = None

    def __post_init__(self):
        if self.page_numbers is None:
            self.page_numbers = {}


class MiniPageDetector:
    """Detect and assign mini-pages within PDF pages.

    Board game rulebooks often have multiple logical pages printed on a single
    PDF page. This detector identifies these regions and assigns blocks to
    their respective mini-pages.
    """

    def __init__(
        self,
        min_blocks_per_region: int = 3,
        page_number_margin: float = 0.1,
    ):
        """Initialize the mini-page detector.

        Args:
            min_blocks_per_region: Minimum blocks to consider a region occupied.
            page_number_margin: Margin (as fraction) to look for page numbers.
        """
        self.min_blocks_per_region = min_blocks_per_region
        self.page_number_margin = page_number_margin

    def detect_layout(self, blocks: list[Block], page_width: float, page_height: float) -> PageLayout:
        """Detect the page layout from block positions.

        Args:
            blocks: All blocks on a single PDF page.
            page_width: Width of the PDF page.
            page_height: Height of the PDF page.

        Returns:
            Detected PageLayout with column/row information.
        """
        if not blocks:
            return PageLayout()

        region_blocks = self._assign_blocks_to_quadrants(blocks, page_width, page_height)

        top_occupied = (
            len(region_blocks[MiniPageRegion.TOP_LEFT]) >= self.min_blocks_per_region
            or len(region_blocks[MiniPageRegion.TOP_RIGHT]) >= self.min_blocks_per_region
        )
        bottom_occupied = (
            len(region_blocks[MiniPageRegion.BOTTOM_LEFT]) >= self.min_blocks_per_region
            or len(region_blocks[MiniPageRegion.BOTTOM_RIGHT]) >= self.min_blocks_per_region
        )
        left_occupied = (
            len(region_blocks[MiniPageRegion.TOP_LEFT]) >= self.min_blocks_per_region
            or len(region_blocks[MiniPageRegion.BOTTOM_LEFT]) >= self.min_blocks_per_region
        )
        right_occupied = (
            len(region_blocks[MiniPageRegion.TOP_RIGHT]) >= self.min_blocks_per_region
            or len(region_blocks[MiniPageRegion.BOTTOM_RIGHT]) >= self.min_blocks_per_region
        )

        num_columns = 2 if (left_occupied and right_occupied) else 1
        num_rows = 2 if (top_occupied and bottom_occupied) else 1
        mini_page_count = num_columns * num_rows

        page_numbers = self._detect_page_numbers(region_blocks, page_width, page_height)

        return PageLayout(
            num_columns=num_columns,
            num_rows=num_rows,
            mini_page_count=mini_page_count,
            page_numbers=page_numbers,
        )

    def assign_mini_pages(
        self,
        blocks: list[Block],
        page_width: float,
        page_height: float,
        layout: Optional[PageLayout] = None,
    ) -> list[Block]:
        """Assign mini-page numbers to blocks.

        Args:
            blocks: Blocks to process (should be from a single PDF page).
            page_width: Width of the PDF page.
            page_height: Height of the PDF page.
            layout: Pre-detected layout (will detect if not provided).

        Returns:
            Blocks with mini_page field populated.
        """
        if layout is None:
            layout = self.detect_layout(blocks, page_width, page_height)

        if layout.mini_page_count == 1:
            for block in blocks:
                block.mini_page = 0
            return blocks

        mid_x = page_width / 2
        mid_y = page_height / 2

        for block in blocks:
            center_x, center_y = block.bbox.center

            if layout.num_columns == 2 and layout.num_rows == 2:
                col = 0 if center_x < mid_x else 1
                row = 0 if center_y < mid_y else 1
                block.mini_page = row * 2 + col
            elif layout.num_columns == 2:
                block.mini_page = 0 if center_x < mid_x else 1
            elif layout.num_rows == 2:
                block.mini_page = 0 if center_y < mid_y else 1
            else:
                block.mini_page = 0

        return blocks

    def process_document(
        self,
        blocks: list[Block],
        page_dimensions: dict[int, tuple[float, float]],
    ) -> list[Block]:
        """Process all blocks in a document, assigning mini-pages.

        Args:
            blocks: All blocks in the document.
            page_dimensions: Dict mapping page number to (width, height).

        Returns:
            All blocks with mini_page field populated.
        """
        pages: dict[int, list[Block]] = {}
        for block in blocks:
            if block.pdf_page not in pages:
                pages[block.pdf_page] = []
            pages[block.pdf_page].append(block)

        for page_num, page_blocks in pages.items():
            if page_num in page_dimensions:
                width, height = page_dimensions[page_num]
            else:
                width, height = self._estimate_page_dimensions(page_blocks)

            layout = self.detect_layout(page_blocks, width, height)
            self.assign_mini_pages(page_blocks, width, height, layout)

        return blocks

    def _assign_blocks_to_quadrants(
        self,
        blocks: list[Block],
        page_width: float,
        page_height: float,
    ) -> dict[MiniPageRegion, list[Block]]:
        """Assign blocks to quadrant regions."""
        mid_x = page_width / 2
        mid_y = page_height / 2

        regions: dict[MiniPageRegion, list[Block]] = {
            MiniPageRegion.TOP_LEFT: [],
            MiniPageRegion.TOP_RIGHT: [],
            MiniPageRegion.BOTTOM_LEFT: [],
            MiniPageRegion.BOTTOM_RIGHT: [],
        }

        for block in blocks:
            center_x, center_y = block.bbox.center

            if center_x < mid_x:
                if center_y < mid_y:
                    regions[MiniPageRegion.TOP_LEFT].append(block)
                else:
                    regions[MiniPageRegion.BOTTOM_LEFT].append(block)
            else:
                if center_y < mid_y:
                    regions[MiniPageRegion.TOP_RIGHT].append(block)
                else:
                    regions[MiniPageRegion.BOTTOM_RIGHT].append(block)

        return regions

    def _detect_page_numbers(
        self,
        region_blocks: dict[MiniPageRegion, list[Block]],
        page_width: float,
        page_height: float,
    ) -> dict[int, Optional[int]]:
        """Detect page numbers in each region."""
        page_numbers: dict[int, Optional[int]] = {}

        region_to_mini_page = {
            MiniPageRegion.TOP_LEFT: 0,
            MiniPageRegion.TOP_RIGHT: 1,
            MiniPageRegion.BOTTOM_LEFT: 2,
            MiniPageRegion.BOTTOM_RIGHT: 3,
        }

        for region, blocks in region_blocks.items():
            page_num = self._find_page_number(blocks, region, page_width, page_height)
            mini_page_idx = region_to_mini_page[region]
            page_numbers[mini_page_idx] = page_num

        return page_numbers

    def _find_page_number(
        self,
        blocks: list[Block],
        region: MiniPageRegion,
        page_width: float,
        page_height: float,
    ) -> Optional[int]:
        """Find a page number in a list of blocks."""
        for block in blocks:
            if block.block_type == BlockType.PAGE_NUMBER:
                match = re.search(r"\d+", block.text)
                if match:
                    return int(match.group())

        margin_x = page_width * self.page_number_margin
        margin_y = page_height * self.page_number_margin

        candidate_blocks = []
        for block in blocks:
            is_near_edge = (
                block.bbox.x0 < margin_x
                or block.bbox.x1 > page_width - margin_x
                or block.bbox.y0 < margin_y
                or block.bbox.y1 > page_height - margin_y
            )
            if is_near_edge:
                text = block.text.strip()
                if re.match(r"^\d{1,3}$", text):
                    candidate_blocks.append((block, int(text)))

        if candidate_blocks:
            return candidate_blocks[0][1]

        return None

    def _estimate_page_dimensions(self, blocks: list[Block]) -> tuple[float, float]:
        """Estimate page dimensions from block positions."""
        if not blocks:
            return (612.0, 792.0)  # Default US Letter

        max_x = max(b.bbox.x1 for b in blocks)
        max_y = max(b.bbox.y1 for b in blocks)

        padding_factor = 1.1
        return (max_x * padding_factor, max_y * padding_factor)
