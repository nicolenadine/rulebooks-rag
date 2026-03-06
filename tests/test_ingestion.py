"""Tests for ingestion modules."""

import json
import tempfile
from pathlib import Path

import pytest

from src.ingestion.mini_page_detector import MiniPageDetector, PageLayout
from src.ingestion.parse_loader import ParseLoader
from src.models.schema import Block, BlockType, BoundingBox


class TestParseLoader:
    """Tests for ParseLoader."""

    def test_load_simple_json(self):
        data = {
            "metadata": {"title": "Test Game Rules"},
            "pages": [
                {
                    "page_number": 0,
                    "blocks": [
                        {
                            "id": "block1",
                            "text": "Introduction",
                            "type": "title",
                            "bbox": [10, 20, 200, 50],
                        },
                        {
                            "id": "block2",
                            "text": "This is the game introduction.",
                            "type": "text",
                            "bbox": [10, 60, 200, 100],
                        },
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = ParseLoader(temp_path)
            document, blocks = loader.load()

            assert document.name == "Test Game Rules"
            assert document.total_pages == 1
            assert len(blocks) == 2
            assert blocks[0].block_type == BlockType.TITLE
            assert blocks[1].block_type == BlockType.TEXT
        finally:
            Path(temp_path).unlink()

    def test_load_with_bbox_dict(self):
        data = {
            "pages": [
                {
                    "page_number": 0,
                    "blocks": [
                        {
                            "text": "Sample text",
                            "type": "text",
                            "bbox": {"x0": 10, "y0": 20, "x1": 200, "y1": 50},
                        },
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = ParseLoader(temp_path)
            document, blocks = loader.load()

            assert len(blocks) == 1
            assert blocks[0].bbox.x0 == 10
            assert blocks[0].bbox.y1 == 50
        finally:
            Path(temp_path).unlink()

    def test_empty_text_blocks_filtered(self):
        data = {
            "pages": [
                {
                    "page_number": 0,
                    "blocks": [
                        {"text": "Valid text", "type": "text", "bbox": [10, 20, 200, 50]},
                        {"text": "", "type": "text", "bbox": [10, 60, 200, 100]},
                        {"text": "   ", "type": "text", "bbox": [10, 110, 200, 150]},
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = ParseLoader(temp_path)
            _, blocks = loader.load()

            assert len(blocks) == 1
            assert blocks[0].text == "Valid text"
        finally:
            Path(temp_path).unlink()


class TestMiniPageDetector:
    """Tests for MiniPageDetector."""

    def test_single_page_layout(self):
        detector = MiniPageDetector()

        blocks = [
            Block(
                text="Content",
                block_type=BlockType.TEXT,
                bbox=BoundingBox(x0=50, y0=100, x1=550, y1=700),
                pdf_page=0,
                reading_order=0,
            )
        ]

        layout = detector.detect_layout(blocks, page_width=612, page_height=792)

        assert layout.num_columns == 1
        assert layout.num_rows == 1
        assert layout.mini_page_count == 1

    def test_two_column_layout(self):
        detector = MiniPageDetector(min_blocks_per_region=2)

        blocks = []
        for i in range(3):
            blocks.append(
                Block(
                    text=f"Left content {i}",
                    block_type=BlockType.TEXT,
                    bbox=BoundingBox(x0=50, y0=100 + i * 50, x1=280, y1=140 + i * 50),
                    pdf_page=0,
                    reading_order=i,
                )
            )
        for i in range(3):
            blocks.append(
                Block(
                    text=f"Right content {i}",
                    block_type=BlockType.TEXT,
                    bbox=BoundingBox(x0=330, y0=100 + i * 50, x1=560, y1=140 + i * 50),
                    pdf_page=0,
                    reading_order=i + 3,
                )
            )

        layout = detector.detect_layout(blocks, page_width=612, page_height=792)

        assert layout.num_columns == 2
        assert layout.mini_page_count >= 2

    def test_assign_mini_pages(self):
        detector = MiniPageDetector(min_blocks_per_region=1)

        blocks = [
            Block(
                text="Top left",
                block_type=BlockType.TEXT,
                bbox=BoundingBox(x0=50, y0=50, x1=250, y1=100),
                pdf_page=0,
                reading_order=0,
            ),
            Block(
                text="Top right",
                block_type=BlockType.TEXT,
                bbox=BoundingBox(x0=350, y0=50, x1=550, y1=100),
                pdf_page=0,
                reading_order=1,
            ),
            Block(
                text="Bottom left",
                block_type=BlockType.TEXT,
                bbox=BoundingBox(x0=50, y0=450, x1=250, y1=500),
                pdf_page=0,
                reading_order=2,
            ),
            Block(
                text="Bottom right",
                block_type=BlockType.TEXT,
                bbox=BoundingBox(x0=350, y0=450, x1=550, y1=500),
                pdf_page=0,
                reading_order=3,
            ),
        ]

        layout = PageLayout(num_columns=2, num_rows=2, mini_page_count=4)
        detector.assign_mini_pages(blocks, page_width=612, page_height=792, layout=layout)

        assert blocks[0].mini_page == 0  # top left
        assert blocks[1].mini_page == 1  # top right
        assert blocks[2].mini_page == 2  # bottom left
        assert blocks[3].mini_page == 3  # bottom right
