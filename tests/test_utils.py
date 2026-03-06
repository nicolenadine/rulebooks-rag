"""Tests for utility functions."""

import pytest

from src.models.schema import BoundingBox
from src.utils.bbox_utils import (
    bbox_area,
    bbox_center,
    bbox_contains,
    bbox_distance,
    bbox_intersection,
    bbox_iou,
    bbox_union,
    normalize_bbox,
)


class TestBboxUtils:
    """Tests for bounding box utility functions."""

    def test_bbox_union_multiple(self):
        boxes = [
            BoundingBox(x0=0, y0=0, x1=50, y1=50),
            BoundingBox(x0=25, y0=25, x1=75, y1=75),
            BoundingBox(x0=50, y0=50, x1=100, y1=100),
        ]

        union = bbox_union(boxes)

        assert union is not None
        assert union.x0 == 0
        assert union.y0 == 0
        assert union.x1 == 100
        assert union.y1 == 100

    def test_bbox_union_empty(self):
        assert bbox_union([]) is None

    def test_bbox_intersection_overlap(self):
        box1 = BoundingBox(x0=0, y0=0, x1=50, y1=50)
        box2 = BoundingBox(x0=25, y0=25, x1=75, y1=75)

        intersection = bbox_intersection(box1, box2)

        assert intersection is not None
        assert intersection.x0 == 25
        assert intersection.y0 == 25
        assert intersection.x1 == 50
        assert intersection.y1 == 50

    def test_bbox_intersection_no_overlap(self):
        box1 = BoundingBox(x0=0, y0=0, x1=50, y1=50)
        box2 = BoundingBox(x0=60, y0=60, x1=100, y1=100)

        intersection = bbox_intersection(box1, box2)

        assert intersection is None

    def test_bbox_iou(self):
        box1 = BoundingBox(x0=0, y0=0, x1=100, y1=100)
        box2 = BoundingBox(x0=0, y0=0, x1=100, y1=100)

        assert bbox_iou(box1, box2) == 1.0

        box3 = BoundingBox(x0=50, y0=50, x1=150, y1=150)
        iou = bbox_iou(box1, box3)
        assert 0 < iou < 1

    def test_bbox_iou_no_overlap(self):
        box1 = BoundingBox(x0=0, y0=0, x1=50, y1=50)
        box2 = BoundingBox(x0=60, y0=60, x1=100, y1=100)

        assert bbox_iou(box1, box2) == 0.0

    def test_bbox_contains(self):
        outer = BoundingBox(x0=0, y0=0, x1=100, y1=100)
        inner = BoundingBox(x0=25, y0=25, x1=75, y1=75)

        assert bbox_contains(outer, inner) is True
        assert bbox_contains(inner, outer) is False

    def test_bbox_distance(self):
        box1 = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        box2 = BoundingBox(x0=10, y0=0, x1=20, y1=10)

        distance = bbox_distance(box1, box2)

        assert distance == 10.0

    def test_bbox_area(self):
        box = BoundingBox(x0=0, y0=0, x1=10, y1=20)
        assert bbox_area(box) == 200.0

    def test_bbox_center(self):
        box = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        center = bbox_center(box)

        assert center == (50.0, 25.0)

    def test_normalize_bbox(self):
        box = BoundingBox(x0=100, y0=200, x1=300, y1=400)
        normalized = normalize_bbox(box, page_width=600, page_height=800)

        assert normalized.x0 == pytest.approx(100 / 600)
        assert normalized.y0 == pytest.approx(200 / 800)
        assert normalized.x1 == pytest.approx(300 / 600)
        assert normalized.y1 == pytest.approx(400 / 800)
