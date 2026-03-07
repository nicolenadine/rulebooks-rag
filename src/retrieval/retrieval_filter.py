"""Post-retrieval filtering: threshold, cap, and deduplication before answer generation."""

import sys
from src.models.schema import Block, Chunk

# Type for retrieved items: (chunk, score, context_blocks)
RetrievedItem = tuple[Chunk, float, list[Block]]


def _normalize_text(text: str) -> str:
    """Normalize for dedup: strip and collapse whitespace."""
    return " ".join(text.strip().split())


def _is_duplicate_of_any(chunk: Chunk, kept: list[RetrievedItem]) -> bool:
    """True if chunk is redundant with any already-kept chunk (same page + mini_page + similar text)."""
    norm = _normalize_text(chunk.text)
    page_key = (chunk.pdf_page, chunk.mini_page if chunk.mini_page is not None else -1)
    for kept_chunk, _, _ in kept:
        if (kept_chunk.pdf_page, kept_chunk.mini_page if kept_chunk.mini_page is not None else -1) != page_key:
            continue
        kept_norm = _normalize_text(kept_chunk.text)
        if norm == kept_norm:
            return True
        if len(norm) > 0 and len(kept_norm) > 0 and (norm in kept_norm or kept_norm in norm):
            return True
    return False


def filter_retrieved(
    retrieved: list[RetrievedItem],
    *,
    similarity_floor: float,
    relative_margin: float,
    max_final_chunks: int,
    debug_log: bool = True,
) -> list[RetrievedItem]:
    """Filter and deduplicate retrieved chunks before answer generation.

    - Always keeps the top-ranked chunk.
    - Keeps others only if score >= similarity_floor and score >= best_score - relative_margin.
    - Caps at max_final_chunks.
    - Deduplicates: same page + same mini_page + (identical or one text contains the other) -> keep higher-scoring.

    When debug_log is True, logs to stderr: initial count, each score, removed by threshold,
    removed by dedup, final count.
    """
    def log(msg: str) -> None:
        if debug_log:
            print(f"[retrieval-filter] {msg}", file=sys.stderr)

    n_initial = len(retrieved)
    log(f"initial retrieved chunk count: {n_initial}")

    if not retrieved:
        return []

    best_score = retrieved[0][1]
    for i, (chunk, score, _) in enumerate(retrieved):
        log(f"  rank {i + 1} score={score:.4f}")

    # Threshold: keep rank 1 always; others only if score >= floor and score >= best - margin
    after_threshold: list[RetrievedItem] = []
    for i, item in enumerate(retrieved):
        chunk, score, context_blocks = item
        if i == 0:
            after_threshold.append(item)
            continue
        if score < similarity_floor:
            continue
        if score < best_score - relative_margin:
            continue
        after_threshold.append(item)

    n_after_threshold = len(after_threshold)
    n_removed_threshold = n_initial - n_after_threshold
    log(f"removed by threshold: {n_removed_threshold} (kept {n_after_threshold})")

    # Cap at max_final_chunks
    after_cap = after_threshold[:max_final_chunks]

    # Deduplicate: same page + same mini_page + very similar text -> keep first (higher score)
    deduped: list[RetrievedItem] = []
    for item in after_cap:
        chunk, score, context_blocks = item
        if _is_duplicate_of_any(chunk, deduped):
            continue
        deduped.append(item)

    n_removed_dedup = len(after_cap) - len(deduped)
    log(f"removed by deduplication: {n_removed_dedup}")
    log(f"final chunk count passed to answer model: {len(deduped)}")

    return deduped
