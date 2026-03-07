"""RAG pipeline for question answering with citations."""

from typing import Callable, Optional

from openai import OpenAI

from src.config import settings
from src.models.schema import Block, BoundingBox, Chunk, Citation, DocumentGraph, QAResponse
from src.retrieval.retrieval_filter import filter_retrieved
from src.retrieval.retriever import Retriever


class RAGPipeline:
    """RAG pipeline for answering questions about rulebooks.

    All configuration is loaded from settings.
    """

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions about board game rules.

You will be given relevant excerpts from a rulebook. Answer the user's question using the provided context.

Hard rule: If the retrieved context directly answers the question, answer directly from the text and do not add a remaining-uncertainty section unless the text actually leaves a relevant part unresolved. Many questions are direct factual lookups and should not have uncertainty at all.

Two answer modes:

(1) Direct factual lookup — If the context directly answers the question, answer directly from the text. Do not add "remaining uncertainty" unless the text truly leaves something unresolved. Prefer restating the rule plainly and accurately; do not paraphrase in a way that changes who or what the rule applies to. Do not contradict the cited rule. Do not restate the rule in a way that changes who or what the rule applies to (e.g. if the rule says "Action or Modifier cards do not count", do not say or imply that Modifier cards do count).

(2) Scenario / edge case — Only when a general rule resolves the main scenario but leaves a narrower edge case unresolved, your answer MUST:
1. First state the resolved main outcome (what the rule says or implies for the situation asked about).
2. Then, only if there is a narrower unresolved edge case, state that edge case separately (e.g. "The excerpt does not specify what happens if [narrower case only]").
3. Do NOT describe the full scenario as unspecified after you have already answered it. Do not restate the main scenario as unspecified once you have stated the main outcome.

You may reason internally with "main outcome" and "remaining uncertainty" but do NOT output the labels "Main outcome:" or "Remaining uncertainty:" in your final answer. Final answers must be natural prose only.

Concrete examples:

Example A — Direct factual (no uncertainty needed):
Rule: "Action or Modifier cards do not count toward the seven card bonus. The only way to achieve the Flip 7 bonus is by having seven unique Number cards face up in front of you."
Question: "Do action cards count when determining if I get a seven card bonus?"
Good answer: "No. Action cards do not count toward the seven card bonus. The rule says that Action or Modifier cards do not count, and that the bonus requires seven unique Number cards."
Bad answer: "Action cards do not count, as only Number cards and Modifier cards are mentioned..." (Wrong: this changes the meaning—the rule says Action or Modifier cards do not count; do not imply Modifier cards are the only ones excluded.)

Example B — Scenario with edge case:
Rule: "At the end of the round when at least one player reaches 200 points, the player with the most points wins."
Question: "What happens if two players reach 200 in the same round?"
Good answer: "If two players reach 200 in the same round, the player with the most points wins at the end of that round. The excerpt does not specify what happens if those players are tied on points."
Bad answer: "The player with the most points wins. However, the excerpt does not specify what happens if two players reach 200 in the same round." (Wrong: the main scenario is resolved; only the tie sub-case is unspecified.)

Other guidelines:
- Prefer explicit rules; apply general rules to specific cases only when the inference is a single logical step from the rule text. Do not chain multiple assumptions.
- Signal when you are inferring: "Based on the rule...", "This implies...". For truly unspecified cases (no rule applies): "The context does not specify...".
- Stay conservative: do not invent rules. Be concise; include page references when available."""

    def __init__(
        self,
        retriever: Retriever,
        graph: Optional[DocumentGraph] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            retriever: The retriever for finding relevant chunks.
            graph: Document graph for block lookups.
        """
        self.retriever = retriever
        self.model = settings.openai_chat_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._graph = graph

    def answer(
        self,
        question: str,
        top_k: int = 5,
        expand_context: bool = True,
        on_retrieved: Optional[Callable[[list[tuple[Chunk, float, list[Block]]]], None]] = None,
    ) -> QAResponse:
        """Answer a question about the rulebook.

        Args:
            question: The user's question.
            top_k: Ignored; initial retrieval size is from settings.retrieval_initial_top_k (then filtered).
            expand_context: Whether to expand context using the graph.
            on_retrieved: Optional callback invoked with the retrieved list
                (chunk, score, context_blocks) for each result. Used for traceability.

        Returns:
            QAResponse with answer and citations.
        """
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=settings.retrieval_initial_top_k,
            expand_context=expand_context,
        )

        retrieved = filter_retrieved(
            retrieved,
            similarity_floor=settings.retrieval_similarity_floor,
            relative_margin=settings.retrieval_relative_margin,
            max_final_chunks=settings.retrieval_max_final_chunks,
            debug_log=settings.debug or settings.environment == "dev",
        )

        if on_retrieved is not None:
            on_retrieved(retrieved)

        context = self._build_context(retrieved)

        answer_text = self._generate_answer(question, context)

        citations = self._extract_citations(retrieved)

        chunks = [chunk for chunk, _, _ in retrieved]

        confidence = self._estimate_confidence(retrieved)

        return QAResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            retrieved_chunks=chunks,
        )

    def _build_context(
        self,
        retrieved: list[tuple[Chunk, float, list[Block]]],
    ) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, (chunk, score, context_blocks) in enumerate(retrieved, 1):
            page_ref = f"Page {chunk.pdf_page + 1}"
            if chunk.mini_page is not None:
                page_ref += f", Section {chunk.mini_page + 1}"

            header = f"[Excerpt {i}] ({page_ref})"
            if chunk.heading:
                header += f" - {chunk.heading}"

            context_text = chunk.text

            if context_blocks:
                before_text = "\n".join(
                    b.text for b in context_blocks if b.reading_order < min(
                        self._get_block_reading_order(chunk, b.block_id) for b in context_blocks
                        if self._get_block_reading_order(chunk, b.block_id) is not None
                    ) if self._get_block_reading_order(chunk, b.block_id) is not None
                )
                after_text = "\n".join(
                    b.text for b in context_blocks if b.reading_order > max(
                        self._get_block_reading_order(chunk, b.block_id) for b in context_blocks
                        if self._get_block_reading_order(chunk, b.block_id) is not None
                    ) if self._get_block_reading_order(chunk, b.block_id) is not None
                )

                if before_text:
                    context_text = f"[...] {before_text}\n\n{context_text}"
                if after_text:
                    context_text = f"{context_text}\n\n{after_text} [...]"

            context_parts.append(f"{header}\n{context_text}")

        return "\n\n---\n\n".join(context_parts)

    def _get_block_reading_order(self, chunk: Chunk, block_id) -> Optional[int]:
        """Get reading order for a block in the chunk."""
        if self._graph is None:
            return None
        block = self._graph.get_block(block_id)
        return block.reading_order if block else None

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context from the rulebook:\n\n{context}\n\nQuestion: {question}",
            },
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )

        return response.choices[0].message.content or ""

    def _extract_citations(
        self,
        retrieved: list[tuple[Chunk, float, list[Block]]],
    ) -> list[Citation]:
        """Extract citations from retrieved chunks."""
        citations = []

        for chunk, score, _ in retrieved:
            snippet = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text

            citation = Citation(
                chunk_id=chunk.chunk_id,
                block_ids=chunk.block_ids,
                pdf_page=chunk.pdf_page,
                mini_page=chunk.mini_page,
                bbox=chunk.bbox_union,
                text_snippet=snippet,
            )
            citations.append(citation)

        return citations

    def _estimate_confidence(
        self,
        retrieved: list[tuple[Chunk, float, list[Block]]],
    ) -> float:
        """Estimate confidence based on retrieval scores."""
        if not retrieved:
            return 0.0

        top_score = retrieved[0][1]

        if len(retrieved) > 1:
            avg_score = sum(score for _, score, _ in retrieved) / len(retrieved)
            confidence = (top_score + avg_score) / 2
        else:
            confidence = top_score

        return min(1.0, max(0.0, confidence))

    def format_response(self, response: QAResponse) -> str:
        """Format a QAResponse for display.

        Args:
            response: The QA response to format.

        Returns:
            Formatted string for terminal output.
        """
        lines = []

        lines.append("=" * 60)
        lines.append("ANSWER")
        lines.append("=" * 60)
        lines.append("")
        lines.append(response.answer)
        lines.append("")

        if response.citations:
            lines.append("-" * 60)
            lines.append("CITATIONS")
            lines.append("-" * 60)
            lines.append("")

            for i, citation in enumerate(response.citations, 1):
                page_ref = citation.format_reference()
                lines.append(f"[{i}] {page_ref}")
                lines.append(f"    \"{citation.text_snippet}\"")
                lines.append("")

        lines.append("-" * 60)
        lines.append(f"Confidence: {response.confidence:.2%}")
        lines.append(f"Chunks retrieved: {len(response.retrieved_chunks)}")
        lines.append("=" * 60)

        return "\n".join(lines)
