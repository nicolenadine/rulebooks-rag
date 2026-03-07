"""Load and convert Reducto parsed JSON output into Block objects."""

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.models.schema import Block, BlockType, BoundingBox, Document


def _normalize_raw_type(raw_type: str) -> str:
    """Normalize Reducto-style types (e.g. 'Page Number') to our keys (e.g. 'page_number')."""
    if not isinstance(raw_type, str):
        return "text"
    return raw_type.strip().lower().replace(" ", "_")


class ParseLoader:
    """Load Reducto parse output and convert to internal Block format."""

    BLOCK_TYPE_MAP = {
        "text": BlockType.TEXT,
        "title": BlockType.TITLE,
        "heading": BlockType.TITLE,
        "figure": BlockType.FIGURE,
        "image": BlockType.FIGURE,
        "table": BlockType.TABLE,
        "list": BlockType.LIST,
        "list_item": BlockType.LIST,
        "caption": BlockType.CAPTION,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
        "page_number": BlockType.PAGE_NUMBER,
        "page number": BlockType.PAGE_NUMBER,
    }

    def __init__(self, json_path: str | Path):
        """Initialize the loader with a path to the Reducto JSON output.

        Args:
            json_path: Path to the parsed JSON file from Reducto.
        """
        self.json_path = Path(json_path)
        self._raw_data: dict[str, Any] | None = None

    @property
    def raw_data(self) -> dict[str, Any]:
        """Lazily load and cache the raw JSON data."""
        if self._raw_data is None:
            with open(self.json_path) as f:
                self._raw_data = json.load(f)
        return self._raw_data

    def load(self) -> tuple[Document, list[Block]]:
        """Load the parsed JSON and convert to Document and Block objects.

        Returns:
            Tuple of (Document, list of Blocks)
        """
        data = self.raw_data

        document = self._create_document(data)
        blocks = self._extract_blocks(data)

        if not blocks:
            raise ValueError(
                "ParseLoader produced 0 blocks. Check input schema and block extraction logic."
            )

        return document, blocks

    def _create_document(self, data: dict[str, Any]) -> Document:
        """Create a Document from the parsed data."""
        result = data.get("result") or data
        metadata = result.get("metadata", data.get("metadata", {}))
        pages = result.get("pages", data.get("pages", []))
        usage = data.get("usage", {})
        num_pages = result.get("num_pages") or usage.get("num_pages") or data.get("num_pages")

        if pages:
            total_pages = len(pages)
        elif num_pages is not None:
            total_pages = int(num_pages)
        else:
            total_pages = 1

        return Document(
            document_id=uuid4(),
            name=metadata.get("title", self.json_path.stem),
            source_path=str(self.json_path),
            total_pages=total_pages,
            metadata={
                "source_format": "reducto",
                "original_metadata": metadata,
            },
        )

    def _extract_blocks(self, data: dict[str, Any]) -> list[Block]:
        """Extract all blocks from the parsed data."""
        blocks = []
        global_reading_order = 0
        kept = 0
        skipped_no_text = 0
        skipped_no_bbox = 0

        # Support Reducto API shape: data.result.blocks (flat list) or data.result.pages / data.pages
        result = data.get("result") or data
        raw_blocks_by_page: list[tuple[int, list[dict[str, Any]]]] = []

        if "blocks" in result and isinstance(result["blocks"], list):
            # Flat list; get page from each block's bbox.page or block.page
            raw_list = result["blocks"]
            for b in raw_list:
                if not isinstance(b, dict):
                    continue
                bbox_obj = b.get("bbox") or b.get("bounding_box") or {}
                page = bbox_obj.get("page") if isinstance(bbox_obj, dict) else None
                if page is None:
                    page = b.get("page", 0)
                try:
                    page = int(page)
                except (TypeError, ValueError):
                    page = 0
                raw_blocks_by_page.append((page, [b]))
            # Sort by page then preserve order
            raw_blocks_by_page.sort(key=lambda x: x[0])
        elif "chunks" in result and isinstance(result["chunks"], list):
            # Reducto: result.chunks[].blocks (one chunk per doc or per page)
            for chunk in result["chunks"]:
                if not isinstance(chunk, dict):
                    continue
                raw_list = chunk.get("blocks") or []
                for b in raw_list:
                    if not isinstance(b, dict):
                        continue
                    bbox_obj = b.get("bbox") or b.get("bounding_box") or {}
                    page = bbox_obj.get("page") if isinstance(bbox_obj, dict) else None
                    if page is None:
                        page = b.get("page", 0)
                    try:
                        page = int(page)
                    except (TypeError, ValueError):
                        page = 0
                    raw_blocks_by_page.append((page, [b]))
            raw_blocks_by_page.sort(key=lambda x: x[0])
        else:
            pages = result.get("pages", data.get("pages", []))
            if not pages and "blocks" in data:
                pages = [{"blocks": data["blocks"], "page_number": 0}]
            for page_idx, page in enumerate(pages):
                page_blocks = page.get("blocks", []) if isinstance(page, dict) else []
                if isinstance(page_blocks, dict):
                    page_blocks = list(page_blocks.values())
                page_num = page.get("page_number", page_idx) if isinstance(page, dict) else page_idx
                raw_blocks_by_page.append((page_num, page_blocks))

        raw_total = sum(len(blist) for _, blist in raw_blocks_by_page)
        print(f"  [ParseLoader] RAW BLOCK COUNT: {raw_total}")
        if raw_total > 0:
            first_block = raw_blocks_by_page[0][1][0]
            print(f"  [ParseLoader] SAMPLE RAW BLOCK KEYS: {list(first_block.keys()) if isinstance(first_block, dict) else type(first_block)}")

        for pdf_page, page_blocks in raw_blocks_by_page:
            for block_data in page_blocks:
                if not isinstance(block_data, dict):
                    continue
                text = self._extract_text(block_data)
                if not text or not text.strip():
                    skipped_no_text += 1
                    continue
                bbox = self._extract_bbox(block_data)
                if not bbox:
                    skipped_no_bbox += 1
                    continue
                block = self._convert_block(
                    block_data,
                    pdf_page=pdf_page,
                    reading_order=global_reading_order,
                )
                if block:
                    blocks.append(block)
                    kept += 1
                    global_reading_order += 1

        print(f"  [ParseLoader] KEPT: {kept}  SKIPPED NO TEXT: {skipped_no_text}  SKIPPED NO BBOX: {skipped_no_bbox}")

        return blocks

    def _convert_block(
        self, block_data: dict[str, Any], pdf_page: int, reading_order: int
    ) -> Block | None:
        """Convert a raw block dict to a Block object."""
        text = self._extract_text(block_data)
        if not text or not text.strip():
            return None

        bbox = self._extract_bbox(block_data)
        if not bbox:
            return None

        block_type = self._determine_block_type(block_data)
        raw_confidence = block_data.get("confidence", block_data.get("score", 1.0))
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 1.0 if str(raw_confidence).lower() in ("high", "medium", "true") else 0.5

        return Block(
            block_id=uuid4(),
            text=text.strip(),
            block_type=block_type,
            bbox=bbox,
            pdf_page=pdf_page,
            mini_page=None,
            reading_order=reading_order,
            confidence=min(1.0, max(0.0, confidence)),
            raw_block_id=block_data.get("id", block_data.get("block_id")),
        )

    def _extract_text(self, block_data: dict[str, Any]) -> str:
        """Extract text content from a block (supports text, content, ocr_text, value)."""
        text = (
            block_data.get("text")
            or block_data.get("content")
            or block_data.get("ocr_text")
            or block_data.get("value")
        )
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            return text.get("text", text.get("content", "")) or ""
        return str(text)

    def _extract_bbox(self, block_data: dict[str, Any]) -> BoundingBox | None:
        """Extract bounding box from a block."""
        bbox = block_data.get("bbox", block_data.get("bounding_box"))

        if bbox is None:
            return None

        if isinstance(bbox, list) and len(bbox) == 4:
            return BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])

        if isinstance(bbox, dict):
            if all(k in bbox for k in ["x0", "y0", "x1", "y1"]):
                return BoundingBox(
                    x0=bbox["x0"],
                    y0=bbox["y0"],
                    x1=bbox["x1"],
                    y1=bbox["y1"],
                )
            if all(k in bbox for k in ["left", "top", "right", "bottom"]):
                return BoundingBox(
                    x0=bbox["left"],
                    y0=bbox["top"],
                    x1=bbox["right"],
                    y1=bbox["bottom"],
                )
            if all(k in bbox for k in ["left", "top", "width", "height"]):
                return BoundingBox(
                    x0=bbox["left"],
                    y0=bbox["top"],
                    x1=bbox["left"] + bbox["width"],
                    y1=bbox["top"] + bbox["height"],
                )
            if all(k in bbox for k in ["x", "y", "width", "height"]):
                return BoundingBox(
                    x0=bbox["x"],
                    y0=bbox["y"],
                    x1=bbox["x"] + bbox["width"],
                    y1=bbox["y"] + bbox["height"],
                )

        return None

    def _determine_block_type(self, block_data: dict[str, Any]) -> BlockType:
        """Determine the block type from the raw data (supports Reducto 'Page Number', 'Text', etc.)."""
        raw_type = block_data.get("type") or block_data.get("block_type") or block_data.get("label") or "text"
        raw_type = _normalize_raw_type(str(raw_type))
        if raw_type in self.BLOCK_TYPE_MAP:
            return self.BLOCK_TYPE_MAP[raw_type]

        text = self._extract_text(block_data)
        if self._looks_like_page_number(text):
            return BlockType.PAGE_NUMBER
        if self._looks_like_heading(text, block_data):
            return BlockType.TITLE

        return BlockType.TEXT

    def _looks_like_page_number(self, text: str) -> bool:
        """Check if text looks like a page number."""
        text = text.strip()
        if re.match(r"^\d{1,3}$", text):
            return True
        if re.match(r"^(page\s*)?\d{1,3}(\s*of\s*\d{1,3})?$", text, re.IGNORECASE):
            return True
        return False

    def _looks_like_heading(self, text: str, block_data: dict[str, Any]) -> bool:
        """Check if text looks like a heading."""
        if len(text) > 200:
            return False
        if text.endswith((".", ":", ",", ";")):
            return False
        font_size = block_data.get("font_size", block_data.get("size"))
        if font_size and font_size > 14:
            return True
        if block_data.get("bold") or block_data.get("is_bold"):
            return True
        return False
