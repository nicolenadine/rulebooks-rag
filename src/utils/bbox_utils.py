"""Bounding box utility functions."""

from typing import Optional

from src.models.schema import BoundingBox


def bbox_union(boxes: list[BoundingBox]) -> Optional[BoundingBox]:
    """Compute the union of multiple bounding boxes.

    Args:
        boxes: List of bounding boxes.

    Returns:
        Bounding box containing all input boxes, or None if empty.
    """
    if not boxes:
        return None

    result = boxes[0]
    for box in boxes[1:]:
        result = result.union(box)

    return result


def bbox_intersection(box1: BoundingBox, box2: BoundingBox) -> Optional[BoundingBox]:
    """Compute the intersection of two bounding boxes.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        Intersection bounding box, or None if no overlap.
    """
    x0 = max(box1.x0, box2.x0)
    y0 = max(box1.y0, box2.y0)
    x1 = min(box1.x1, box2.x1)
    y1 = min(box1.y1, box2.y1)

    if x0 >= x1 or y0 >= y1:
        return None

    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def bbox_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        IoU score between 0 and 1.
    """
    intersection = bbox_intersection(box1, box2)
    if intersection is None:
        return 0.0

    intersection_area = intersection.area
    union_area = box1.area + box2.area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def bbox_contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    """Check if one bounding box fully contains another.

    Args:
        outer: The potentially containing box.
        inner: The potentially contained box.

    Returns:
        True if outer fully contains inner.
    """
    return (
        outer.x0 <= inner.x0
        and outer.y0 <= inner.y0
        and outer.x1 >= inner.x1
        and outer.y1 >= inner.y1
    )


def bbox_distance(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute the Euclidean distance between bounding box centers.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        Distance between centers.
    """
    c1 = box1.center
    c2 = box2.center

    return ((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2) ** 0.5


def bbox_area(box: BoundingBox) -> float:
    """Compute the area of a bounding box.

    Args:
        box: The bounding box.

    Returns:
        Area of the box.
    """
    return box.area


def bbox_center(box: BoundingBox) -> tuple[float, float]:
    """Get the center point of a bounding box.

    Args:
        box: The bounding box.

    Returns:
        Tuple of (x, y) center coordinates.
    """
    return box.center


def normalize_bbox(
    box: BoundingBox,
    page_width: float,
    page_height: float,
) -> BoundingBox:
    """Normalize bounding box coordinates to 0-1 range.

    Args:
        box: The bounding box to normalize.
        page_width: Width of the page.
        page_height: Height of the page.

    Returns:
        Normalized bounding box.
    """
    return BoundingBox(
        x0=box.x0 / page_width,
        y0=box.y0 / page_height,
        x1=box.x1 / page_width,
        y1=box.y1 / page_height,
    )


def scale_bbox(
    box: BoundingBox,
    scale_x: float,
    scale_y: float,
) -> BoundingBox:
    """Scale a bounding box by given factors.

    Args:
        box: The bounding box to scale.
        scale_x: Horizontal scale factor.
        scale_y: Vertical scale factor.

    Returns:
        Scaled bounding box.
    """
    return BoundingBox(
        x0=box.x0 * scale_x,
        y0=box.y0 * scale_y,
        x1=box.x1 * scale_x,
        y1=box.y1 * scale_y,
    )


def expand_bbox(
    box: BoundingBox,
    margin: float,
    page_width: Optional[float] = None,
    page_height: Optional[float] = None,
) -> BoundingBox:
    """Expand a bounding box by a margin.

    Args:
        box: The bounding box to expand.
        margin: Margin to add on each side.
        page_width: Optional max width to clip to.
        page_height: Optional max height to clip to.

    Returns:
        Expanded bounding box.
    """
    x0 = max(0, box.x0 - margin)
    y0 = max(0, box.y0 - margin)
    x1 = box.x1 + margin
    y1 = box.y1 + margin

    if page_width is not None:
        x1 = min(x1, page_width)
    if page_height is not None:
        y1 = min(y1, page_height)

    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
